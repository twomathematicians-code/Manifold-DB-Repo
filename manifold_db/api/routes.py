"""
API route definitions — separated for clean architecture.

Each ``router_*`` is an :class:`fastapi.APIRouter` with a logical grouping:

- **router_insert** — data ingestion (single and batch)
- **router_query** — query execution, geodesic ball, cross-modal, explain
- **router_charts** — chart introspection
- **router_atlas** — atlas construction
- **router_system** — health, stats, save/load

All routers import helpers from ``server.py`` which owns the :class:`AppState`
singleton and Pydantic models.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi import Query as QueryParam

from manifold_db.api.server import (
    AppState,  # noqa: F401 — re-exports
    AtlasBuildRequest,
    BatchInsertRequest,
    BatchInsertResponse,
    ChartInfo,
    CrossModalQueryRequest,
    DatabaseStats,
    GeodesicQueryRequest,
    HealthResponse,
    InsertData,
    InsertResponse,
    LoadRequest,
    QueryRequest,
    SaveRequest,
    _chart_to_info,
    _insert_single,
    _plan_to_response,
    _result_to_response,
    _run_query,
    app_state,
)
from manifold_db.query.dsl import ManifoldQuery, MetricType, QueryType

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# router_insert — data ingestion
# ═══════════════════════════════════════════════════════════════

router_insert = APIRouter()


@router_insert.post(
    "/insert",
    response_model=InsertResponse,
    summary="Insert a single data point",
    description="Insert one data point (vector + metadata + modality) into the manifold database.",
    responses={
        400: {"description": "Invalid input"},
        500: {"description": "Internal error"},
    },
)
async def insert_data_point(body: InsertData) -> InsertResponse:
    """Insert a single data point and return its generated ID."""
    try:
        point_id = await _insert_single(
            vector=body.vector,
            metadata=body.metadata,
            modality=body.modality,
            chart_id=body.chart_id,
        )
        return InsertResponse(point_id=point_id, status="ok")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Insert failed")
        raise HTTPException(status_code=500, detail=f"Insert failed: {exc}")


@router_insert.post(
    "/batch-insert",
    response_model=BatchInsertResponse,
    summary="Insert multiple data points",
    description="Insert many data points in a single request.",
    responses={
        400: {"description": "Invalid input"},
        500: {"description": "Internal error"},
    },
)
async def batch_insert(body: BatchInsertRequest) -> BatchInsertResponse:
    """Insert multiple data points and return their IDs."""
    if not body.points:
        raise HTTPException(status_code=400, detail="No points provided in the request")

    point_ids: list[str] = []
    try:
        for pt in body.points:
            pid = await _insert_single(
                vector=pt.vector,
                metadata=pt.metadata,
                modality=body.modality or pt.modality,
                chart_id=body.chart_id or pt.chart_id,
            )
            point_ids.append(pid)

        return BatchInsertResponse(
            inserted_count=len(point_ids),
            point_ids=point_ids,
            status="ok",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Batch insert failed")
        raise HTTPException(status_code=500, detail=f"Batch insert failed: {exc}")


# ═══════════════════════════════════════════════════════════════
# router_query — query execution
# ═══════════════════════════════════════════════════════════════

router_query = APIRouter()


@router_query.post(
    "/query",
    summary="Execute a ManifoldQuery",
    description=(
        "Accept a ManifoldQuery JSON object and execute it against the database. "
        "Set use_explain=true to return the execution plan instead of results."
    ),
    responses={
        400: {"description": "Invalid query"},
        500: {"description": "Execution error"},
    },
)
async def execute_query(body: QueryRequest) -> dict[str, Any]:
    """Parse, validate, and execute a ManifoldQuery from JSON."""
    try:
        manifold_query = ManifoldQuery.from_dict(body.query)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid query format: {exc}")

    valid, msg = manifold_query.validate()
    if not valid:
        raise HTTPException(status_code=400, detail=f"Query validation failed: {msg}")

    # Explain mode
    if body.use_explain:
        if app_state.query_engine is None:
            raise HTTPException(status_code=503, detail="QueryEngine not initialised")
        plan = await app_state.query_engine.explain(manifold_query)
        return _plan_to_response(plan, manifold_query.query_type.value)

    # Execute mode
    try:
        result = await _run_query(manifold_query)
        return _result_to_response(result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Query execution failed")
        raise HTTPException(status_code=500, detail=f"Query execution failed: {exc}")


@router_query.post(
    "/geodesic-query",
    summary="Geodesic ball query",
    description="Find all data points within a geodesic distance epsilon of a center point.",
    responses={
        400: {"description": "Invalid parameters"},
        500: {"description": "Query error"},
    },
)
async def geodesic_query(body: GeodesicQueryRequest) -> dict[str, Any]:
    """Execute a geodesic ball query using the query engine."""
    try:
        manifold_query = ManifoldQuery(
            query_type=QueryType.SELECT,
            query_point=np.asarray(body.center, dtype=np.float64),
            epsilon=body.epsilon,
            metric_type=MetricType(body.metric),
            k=body.max_results,
            modality=body.modality,
            chart_id=body.chart_id,
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid parameters: {exc}")

    try:
        result = await _run_query(manifold_query)
        return _result_to_response(result)
    except Exception as exc:
        logger.exception("Geodesic query failed")
        raise HTTPException(status_code=500, detail=f"Geodesic query failed: {exc}")


@router_query.post(
    "/cross-modal-query",
    summary="Cross-modal retrieval",
    description=(
        "Retrieve data from one modality given a query in another modality. "
        "Uses parallel transport to bridge the modality gap."
    ),
    responses={
        400: {"description": "Invalid parameters"},
        500: {"description": "Query error"},
    },
)
async def cross_modal_query(body: CrossModalQueryRequest) -> dict[str, Any]:
    """Execute a cross-modal query with parallel transport."""
    try:
        manifold_query = ManifoldQuery(
            query_type=QueryType.CROSS_MODAL,
            query_point=np.asarray(body.query_point, dtype=np.float64),
            modality=body.source_modality,
            target_modality=body.target_modality,
            k=body.k,
            metric_type=MetricType(body.metric),
            transport_via=body.transport_via,
            source_chart=body.source_chart,
            target_chart=body.target_chart,
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid parameters: {exc}")

    try:
        result = await _run_query(manifold_query)
        return _result_to_response(result)
    except Exception as exc:
        logger.exception("Cross-modal query failed")
        raise HTTPException(status_code=500, detail=f"Cross-modal query failed: {exc}")


@router_query.get(
    "/query/explain",
    summary="Explain a query plan",
    description="Return the execution plan for a given ManifoldQuery without running it.",
)
async def explain_query(
    query_type: str = QueryParam("select", description="Type of query"),
    metric: str = QueryParam("geodesic", description="Distance metric"),
    k: int = QueryParam(10, description="Number of results"),
    epsilon: float = QueryParam(1.0, description="Geodesic ball radius"),
    modality: str | None = QueryParam(None, description="Modality filter"),
    chart_id: str | None = QueryParam(None, description="Chart ID"),
) -> dict[str, Any]:
    """Return an execution plan for a query without executing it."""
    try:
        manifold_query = ManifoldQuery(
            query_type=QueryType(query_type),
            metric_type=MetricType(metric),
            k=k,
            epsilon=epsilon,
            modality=modality,
            chart_id=chart_id,
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid parameters: {exc}")

    if app_state.query_engine is None:
        raise HTTPException(status_code=503, detail="QueryEngine not initialised")

    plan = await app_state.query_engine.explain(manifold_query)
    return _plan_to_response(plan, manifold_query.query_type.value)


# ═══════════════════════════════════════════════════════════════
# router_charts — chart introspection
# ═══════════════════════════════════════════════════════════════

router_charts = APIRouter()


@router_charts.get(
    "/charts",
    response_model=list[ChartInfo],
    summary="List all charts",
    description="Return a summary of every chart in the atlas.",
)
async def list_charts() -> list[dict[str, Any]]:
    """Return a list of all charts with summary information."""
    if app_state.atlas_manager is None:
        raise HTTPException(status_code=503, detail="AtlasManager not initialised")
    charts = app_state.atlas_manager.get_all_charts()
    return [_chart_to_info(c) for c in charts]


@router_charts.get(
    "/charts/{chart_id}",
    response_model=ChartInfo,
    summary="Get chart details",
    description="Return detailed information about a specific chart.",
    responses={404: {"description": "Chart not found"}},
)
async def get_chart(chart_id: str) -> dict[str, Any]:
    """Return details for a single chart identified by its ID."""
    if app_state.atlas_manager is None:
        raise HTTPException(status_code=503, detail="AtlasManager not initialised")
    try:
        chart = app_state.atlas_manager.get_chart(chart_id)
        return _chart_to_info(chart)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Chart '{chart_id}' not found")


# ═══════════════════════════════════════════════════════════════
# router_atlas — atlas construction
# ═══════════════════════════════════════════════════════════════

router_atlas = APIRouter()


async def _build_atlas_background(
    modality: str | None = None,
    overlap_ratio: float = 0.3,
    min_chart_size: int = 50,
    max_charts: int = 100,
) -> None:
    """Background task to build the atlas from stored data."""
    if app_state._atlas_building:
        logger.warning("Atlas build already in progress; skipping.")
        return

    app_state._atlas_building = True
    try:
        if app_state.atlas_manager is None or app_state.data_store is None:
            logger.error(
                "Cannot build atlas: atlas_manager or data_store not initialised"
            )
            return

        # Gather all vectors from the data store
        if app_state.data_store._chart_index:
            # Collect vectors from in-memory chart index
            all_vectors: list[np.ndarray] = []
            all_ids: list[str] = []
            for chart_vecs in app_state.data_store._chart_index.values():
                for pid, vec in chart_vecs.items():
                    all_ids.append(pid)
                    all_vectors.append(vec)
            if not all_vectors:
                logger.info("No data points to build atlas from.")
                return
            data_matrix = np.vstack(all_vectors)
        else:
            logger.info("No data points in chart index to build atlas from.")
            return

        logger.info(
            "Building atlas from %d data points (modality=%s, overlap=%.2f, max_charts=%d)",
            len(data_matrix),
            modality,
            overlap_ratio,
            max_charts,
        )

        app_state.atlas_manager.build_atlas(
            data_matrix,
            modality=modality,
            overlap_ratio=overlap_ratio,
            min_chart_size=min_chart_size,
            max_charts=max_charts,
        )

        logger.info(
            "Atlas build complete: %d charts, %d transitions",
            len(app_state.atlas_manager.get_all_charts()),
            len(app_state.atlas_manager.get_all_transition_maps()),
        )
    except Exception:
        logger.exception("Atlas build failed")
    finally:
        app_state._atlas_building = False


@router_atlas.post(
    "/atlas/build",
    summary="Trigger atlas building",
    description="Build the atlas from current data. This runs as a background task.",
    responses={
        202: {"description": "Atlas build started"},
        503: {"description": "Service unavailable"},
    },
)
async def build_atlas(
    body: AtlasBuildRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    """Trigger atlas building as a background task."""
    if app_state.atlas_manager is None:
        raise HTTPException(status_code=503, detail="AtlasManager not initialised")

    if app_state._atlas_building:
        raise HTTPException(
            status_code=202,
            detail="Atlas build is already in progress",
        )

    background_tasks.add_task(
        _build_atlas_background,
        modality=body.modality,
        overlap_ratio=body.overlap_ratio,
        min_chart_size=body.min_chart_size,
        max_charts=body.max_charts,
    )

    return {"status": "accepted", "message": "Atlas build started in background"}


# ═══════════════════════════════════════════════════════════════
# router_system — health, stats, save, load
# ═══════════════════════════════════════════════════════════════

router_system = APIRouter()


@router_system.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Return the API version, uptime, and basic status.",
)
async def health_check() -> dict[str, Any]:
    """Return health status of the API."""
    total_points = 0
    if app_state.data_store is not None:
        try:
            stats = await app_state.data_store.stats()
            total_points = stats.get("total_points", 0)
        except Exception:
            pass

    return {
        "status": "ok",
        "version": __import__(
            "manifold_db.api.server", fromlist=["__VERSION__"]
        ).__VERSION__,
        "uptime_seconds": round(app_state.uptime_seconds, 2),
        "total_points": total_points,
    }


@router_system.get(
    "/stats",
    response_model=DatabaseStats,
    summary="Database statistics",
    description="Return comprehensive statistics about the database.",
)
async def database_stats() -> dict[str, Any]:
    """Return full database statistics."""
    import importlib

    server_mod = importlib.import_module("manifold_db.api.server")

    total_points = 0
    modalities: dict[str, int] = {}
    charts_list: list[str] = []
    n_charts = 0
    n_transitions = 0

    if app_state.data_store is not None:
        try:
            ds_stats = await app_state.data_store.stats()
            total_points = ds_stats.get("total_points", 0)
            modalities = ds_stats.get("modalities", {})
            charts_list = ds_stats.get("charts_list", [])
        except Exception:
            pass

    if app_state.atlas_manager is not None:
        charts = app_state.atlas_manager.get_all_charts()
        n_charts = len(charts)
        n_transitions = len(app_state.atlas_manager.get_all_transition_maps())
        charts_list = [c.chart_id for c in charts]

    storage_backend = "unknown"
    if app_state.storage_manager is not None:
        storage_backend = type(app_state.storage_manager._backend).__name__

    return {
        "version": server_mod.__VERSION__,
        "total_points": total_points,
        "n_charts": n_charts,
        "n_transitions": n_transitions,
        "modalities": modalities,
        "charts_list": charts_list,
        "atlas_name": (
            app_state.atlas_manager.name if app_state.atlas_manager else "default"
        ),
        "storage_backend": storage_backend,
        "uptime_seconds": round(app_state.uptime_seconds, 2),
    }


@router_system.post(
    "/save",
    summary="Save database to disk",
    description="Persist the atlas and data store to the given path.",
    responses={500: {"description": "Save failed"}},
)
async def save_database(body: SaveRequest) -> dict[str, str]:
    """Save the database (atlas + data store) to disk."""
    try:
        save_path = body.path
        from pathlib import Path as FilePath

        FilePath(save_path).mkdir(parents=True, exist_ok=True)

        # Save atlas
        if app_state.atlas_manager is not None:
            atlas_file = str(FilePath(save_path) / "atlas.json")
            app_state.atlas_manager.save(atlas_file)

        # Export data store
        if app_state.data_store is not None:
            data_file = str(FilePath(save_path) / "data.json")
            await app_state.data_store.export(format="json", path=data_file)

        return {"status": "ok", "message": f"Database saved to {save_path}"}
    except Exception as exc:
        logger.exception("Database save failed")
        raise HTTPException(status_code=500, detail=f"Save failed: {exc}")


@router_system.post(
    "/load",
    summary="Load database from disk",
    description="Restore the atlas and data store from the given path.",
    responses={
        400: {"description": "Path not found"},
        500: {"description": "Load failed"},
    },
)
async def load_database(body: LoadRequest) -> dict[str, str]:
    """Load the database (atlas + data store) from disk."""
    from pathlib import Path as FilePath

    load_path = FilePath(body.path)
    atlas_file = load_path / "atlas.json"
    data_file = load_path / "data.json"

    if not load_path.exists() or (not atlas_file.exists() and not data_file.exists()):
        raise HTTPException(
            status_code=400,
            detail=f"No database found at {body.path}",
        )

    try:
        # Load atlas
        if atlas_file.exists() and app_state.atlas_manager is not None:
            app_state.atlas_manager.load(str(atlas_file))

        # Import data
        if data_file.exists() and app_state.data_store is not None:
            count = await app_state.data_store.import_data(
                str(data_file), format="json"
            )
            logger.info("Loaded %d data points from %s", count, data_file)

        return {"status": "ok", "message": f"Database loaded from {body.path}"}
    except Exception as exc:
        logger.exception("Database load failed")
        raise HTTPException(status_code=500, detail=f"Load failed: {exc}")
