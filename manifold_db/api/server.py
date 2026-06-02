"""
REST API server for Manifold Database using FastAPI.
Provides HTTP endpoints for all database operations.

Endpoints
----------
POST   /api/v1/insert            — insert a single data point
POST   /api/v1/batch-insert      — insert multiple data points
POST   /api/v1/query             — execute a ManifoldQuery (JSON)
POST   /api/v1/geodesic-query    — geodesic ball query
POST   /api/v1/cross-modal-query — cross-modal retrieval
GET    /api/v1/charts            — list all charts
GET    /api/v1/charts/{id}       — get chart details
POST   /api/v1/atlas/build       — trigger atlas building
GET    /api/v1/stats             — database statistics
GET    /api/v1/health            — health check with version and status
POST   /api/v1/save              — save database to disk
POST   /api/v1/load              — load database from disk
GET    /api/v1/query/explain     — explain a query plan
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from manifold_db.api.middleware import (
    ErrorHandlerMiddleware,
    RateLimitMiddleware,
    RequestLoggingMiddleware,
)
from manifold_db.api.routes import (
    router_atlas,
    router_charts,
    router_insert,
    router_query,
    router_system,
)
from manifold_db.atlas.atlas_manager import AtlasManager
from manifold_db.query.dsl import ManifoldQuery
from manifold_db.query.engine import ExecutionPlan, QueryEngine, QueryResult
from manifold_db.storage.backend import StorageManager
from manifold_db.storage.data_store import DataPoint, DataStore
from manifold_db.utils.config import ManifoldConfig, default_config, load_config

logger = logging.getLogger(__name__)

__VERSION__ = "0.1.0"


# ═══════════════════════════════════════════════════════════════
# Pydantic Request / Response Models
# ═══════════════════════════════════════════════════════════════


class InsertData(BaseModel):
    """Schema for a single data-point to be inserted."""

    vector: list[float] = Field(..., description="Embedding vector")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Key-value metadata"
    )
    modality: str = Field(
        default="default", description="Modality tag (text, image, audio, etc.)"
    )
    chart_id: str | None = Field(default=None, description="Optional chart assignment")


class InsertResponse(BaseModel):
    """Response after inserting a single data point."""

    point_id: str
    status: str = "ok"


class BatchInsertRequest(BaseModel):
    """Schema for inserting multiple data points at once."""

    points: list[InsertData]
    modality: str = "default"
    chart_id: str | None = None


class BatchInsertResponse(BaseModel):
    """Response after batch insertion."""

    inserted_count: int
    point_ids: list[str]
    status: str = "ok"


class QueryRequest(BaseModel):
    """General query request accepting a full ManifoldQuery JSON."""

    query: dict[str, Any] = Field(..., description="ManifoldQuery parameters")
    use_explain: bool = Field(
        default=False, description="If True, return plan instead of results"
    )


class GeodesicQueryRequest(BaseModel):
    """Request for a geodesic ball query (all points within epsilon)."""

    center: list[float] = Field(..., description="Center point of the geodesic ball")
    epsilon: float = Field(default=1.0, description="Radius of the geodesic ball")
    metric: str = Field(default="geodesic", description="Distance metric to use")
    modality: str | None = Field(default=None, description="Filter by modality")
    chart_id: str | None = Field(default=None, description="Restrict to chart")
    max_results: int = Field(default=100, description="Maximum number of results")


class CrossModalQueryRequest(BaseModel):
    """Request for cross-modal retrieval with parallel transport."""

    source_modality: str = Field(..., description="Source modality (e.g. 'text')")
    target_modality: str = Field(..., description="Target modality (e.g. 'image')")
    query_point: list[float] = Field(..., description="Query embedding in source space")
    k: int = Field(default=10, description="Number of results to return")
    metric: str = Field(default="geodesic", description="Distance metric")
    transport_via: str = Field(
        default="overlap_region", description="Transport strategy"
    )
    source_chart: str | None = Field(default=None, description="Explicit source chart")
    target_chart: str | None = Field(default=None, description="Explicit target chart")


class ChartInfo(BaseModel):
    """Summary information about a single chart."""

    chart_id: str
    name: str
    dim: int
    ambient_dim: int
    has_bounds: bool
    n_anchor_points: int
    metadata: dict[str, Any] = {}
    modality: str | None = None


class DatabaseStats(BaseModel):
    """Database-level statistics."""

    version: str
    total_points: int = 0
    n_charts: int = 0
    n_transitions: int = 0
    modalities: dict[str, int] = {}
    charts_list: list[str] = []
    atlas_name: str = "default"
    storage_backend: str = "memory"
    uptime_seconds: float = 0.0


class HealthResponse(BaseModel):
    """Health-check response."""

    status: str = "ok"
    version: str = __VERSION__
    uptime_seconds: float = 0.0
    total_points: int = 0


class QueryResultResponse(BaseModel):
    """Response for a query execution result."""

    point_ids: list[int] = []
    distances: list[float] = []
    metadata: list[dict[str, Any]] = []
    execution_time_ms: float = 0.0
    chart_id: str | None = None
    query_type: str | None = None
    count: int = 0


class ExplainResponse(BaseModel):
    """Response for query explanation."""

    query_type: str
    steps: list[dict[str, Any]] = []
    total_estimated_ms: float = 0.0
    plan_text: str = ""


class SaveRequest(BaseModel):
    """Request body to save the database to disk."""

    path: str = "./manifold_data"


class LoadRequest(BaseModel):
    """Request body to load the database from disk."""

    path: str = "./manifold_data"


class AtlasBuildRequest(BaseModel):
    """Request to trigger atlas building from current data."""

    modality: str | None = None
    overlap_ratio: float = 0.3
    min_chart_size: int = 50
    max_charts: int = 100


# ═══════════════════════════════════════════════════════════════
# Application State
# ═══════════════════════════════════════════════════════════════


class AppState:
    """Shared application state accessible from route handlers.

    Holds references to the data store, query engine, atlas manager,
    and other runtime objects needed across endpoints.
    """

    def __init__(self) -> None:
        self.config: ManifoldConfig = default_config()
        self.data_store: DataStore | None = None
        self.query_engine: QueryEngine | None = None
        self.atlas_manager: AtlasManager | None = None
        self.storage_manager: StorageManager | None = None
        self._start_time: float = time.time()
        self._atlas_building: bool = False

    @property
    def uptime_seconds(self) -> float:
        """Seconds since the application was started."""
        return time.time() - self._start_time


# Module-level singleton — uvicorn --reload recreates this on each reload.
app_state = AppState()


def get_app_state() -> AppState:
    """FastAPI dependency that returns the shared application state."""
    return app_state


# ═══════════════════════════════════════════════════════════════
# Lifespan (startup / shutdown)
# ═══════════════════════════════════════════════════════════════


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    """Initialise database on startup, clean up resources on shutdown."""
    logger.info("Manifold DB API starting up (v%s) ...", __VERSION__)

    # Determine config path from environment or use defaults
    config_path = os.environ.get("MANIFOLD_CONFIG", None)
    if config_path and Path(config_path).exists():
        try:
            app_state.config = load_config(config_path)
            logger.info("Loaded configuration from %s", config_path)
        except Exception as exc:
            logger.warning(
                "Failed to load config from %s: %s — using defaults",
                config_path,
                exc,
            )

    # Initialise storage backend
    storage_cfg = app_state.config.storage
    app_state.storage_manager = StorageManager.create(
        backend_type=storage_cfg.backend_type,
        config={
            "base_path": storage_cfg.path,
            "db_path": str(Path(storage_cfg.path) / "manifold.db"),
        },
    )

    # Initialise data store
    app_state.data_store = DataStore(backend=app_state.storage_manager)

    # Initialise atlas manager
    app_state.atlas_manager = AtlasManager(name="default_atlas")

    # Initialise query engine
    app_state.query_engine = QueryEngine(
        atlas_manager=app_state.atlas_manager,
    )

    logger.info("Manifold DB API ready.")
    yield

    # ── Shutdown ──────────────────────────────────────────────
    logger.info("Manifold DB API shutting down ...")
    if app_state.data_store is not None:
        await app_state.data_store.close()
    if app_state.storage_manager is not None:
        await app_state.storage_manager.close()
    logger.info("Manifold DB API stopped.")


# ═══════════════════════════════════════════════════════════════
# FastAPI Application Factory
# ═══════════════════════════════════════════════════════════════


def create_app() -> FastAPI:
    """Build and return the fully configured FastAPI application.

    The returned ``app`` is ready to be passed to ``uvicorn.run``.
    Middleware, routes, and lifespan hooks are all attached here.
    """
    fastapi_app = FastAPI(
        title="Manifold Database API",
        description=(
            "REST API for a manifold-structured database supporting geodesic "
            "queries, tangent-space search, cross-modal retrieval, and atlas "
            "management on Riemannian manifolds."
        ),
        version=__VERSION__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Middleware (outermost first) ──────────────────────────

    # CORS
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom middleware (inner to outer)
    fastapi_app.add_middleware(ErrorHandlerMiddleware)
    fastapi_app.add_middleware(RateLimitMiddleware, max_requests=100, window_seconds=60)
    fastapi_app.add_middleware(RequestLoggingMiddleware)

    # ── Include route groups ─────────────────────────────────
    fastapi_app.include_router(router_insert, prefix="/api/v1", tags=["insert"])
    fastapi_app.include_router(router_query, prefix="/api/v1", tags=["query"])
    fastapi_app.include_router(router_charts, prefix="/api/v1", tags=["charts"])
    fastapi_app.include_router(router_atlas, prefix="/api/v1", tags=["atlas"])
    fastapi_app.include_router(router_system, prefix="/api/v1", tags=["system"])

    return fastapi_app


# Convenience: the default app instance (used when imported directly)
app: FastAPI = create_app()


# ═══════════════════════════════════════════════════════════════
# Helper Utilities (shared across route modules)
# ═══════════════════════════════════════════════════════════════


async def _insert_single(
    vector: list[float],
    metadata: dict[str, Any],
    modality: str = "default",
    chart_id: str | None = None,
) -> str:
    """Insert a single data point into the data store and return its id."""
    if app_state.data_store is None:
        raise RuntimeError("DataStore not initialised")
    point_id = str(uuid.uuid4())
    dp = DataPoint(
        id=point_id,
        vector=np.asarray(vector, dtype=np.float64),
        metadata=metadata,
        modality=modality,
        chart_id=chart_id,
    )
    await app_state.data_store.insert(dp)
    return point_id


async def _run_query(query: ManifoldQuery) -> QueryResult:
    """Execute a ManifoldQuery via the query engine."""
    if app_state.query_engine is None:
        raise RuntimeError("QueryEngine not initialised")
    return await app_state.query_engine.execute(query)


def _result_to_response(result: QueryResult) -> dict[str, Any]:
    """Convert a QueryResult to a JSON-serialisable response dict."""
    return {
        "point_ids": [int(x) for x in result.point_ids.tolist()],
        "distances": [float(x) for x in result.distances.tolist()],
        "metadata": result.metadata or [],
        "execution_time_ms": round(result.execution_time * 1000, 4),
        "chart_id": result.chart_id,
        "query_type": result.query_type,
        "count": len(result),
    }


def _plan_to_response(plan: ExecutionPlan, query_type: str) -> dict[str, Any]:
    """Convert an ExecutionPlan to a JSON-serialisable response dict."""
    steps = [
        {
            "name": s.name,
            "description": s.description,
            "estimated_cost_ms": s.estimated_cost_ms,
            "depends_on": s.depends_on,
        }
        for s in plan.steps
    ]
    return {
        "query_type": query_type,
        "steps": steps,
        "total_estimated_ms": plan.total_estimated_ms,
        "plan_text": plan.visualize(),
    }


def _chart_to_info(chart: Any) -> dict[str, Any]:
    """Convert a Chart object to a ChartInfo-compatible dict."""
    summary = chart.summary() if hasattr(chart, "summary") else {}
    return {
        "chart_id": chart.chart_id,
        "name": chart.name,
        "dim": chart.dim,
        "ambient_dim": chart.ambient_dim,
        "has_bounds": summary.get("has_bounds", False),
        "n_anchor_points": summary.get("n_anchor_points", 0),
        "metadata": chart.metadata if hasattr(chart, "metadata") else {},
        "modality": (chart.metadata or {}).get("modality"),
    }
