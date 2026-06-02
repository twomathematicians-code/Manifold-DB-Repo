"""
Query Execution Engine - executes parsed queries against the manifold database.
Orchestrates atlas lookup, tangent space indexing, geodesic computation,
parallel transport, and cross-modal retrieval.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union,
)

import numpy as np

from manifold_db.query.dsl import (
    CostTier, ManifoldQuery, MetricType, QueryType,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Lightweight stubs for collaborators (replace with real imports
# when atlas_manager, metric_store, etc. are available).
# ──────────────────────────────────────────────────────────────

class _Stub:
    """Placeholder for external collaborators."""
    async def lookup_chart(self, point: np.ndarray) -> Optional[Dict[str, Any]]:
        return {"chart_id": "default_chart", "dimension": point.size}

    async def chart_size(self, chart_id: str) -> int:
        return 1000

    async def tangent_search(self, chart_id: str, query: np.ndarray,
                            k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (ids, distances) arrays of shape (k,)."""
        ids = np.arange(k, dtype=np.int64)
        dists = np.linspace(0.01, 0.1, k)
        return ids, dists

    async def get_embeddings(self, point_ids: np.ndarray) -> np.ndarray:
        return np.random.randn(len(point_ids), 3)

    async def geodesic_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        return float(np.linalg.norm(p - q))

    async def parallel_transport(self, vector: np.ndarray,
                                source: str, target: str) -> np.ndarray:
        return vector.copy()

    async def list_charts(self) -> List[Dict[str, Any]]:
        return [{"chart_id": "default_chart", "size": 1000}]


# ──────────────────────────────────────────────────────────────
# Query Result
# ──────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    """
    Container for query execution results.

    Supports iteration (streaming large results), and conversion
    to list / dict / DataFrame representations.
    """
    point_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    distances: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    metadata: Optional[List[Dict[str, Any]]] = None
    execution_time: float = 0.0
    chart_id: Optional[str] = None
    query_type: Optional[str] = None

    def __len__(self) -> int:
        return len(self.point_ids)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for i in range(len(self)):
            row: Dict[str, Any] = {
                "point_id": int(self.point_ids[i]),
                "distance": float(self.distances[i]),
            }
            if self.metadata and i < len(self.metadata):
                row["metadata"] = self.metadata[i]
            yield row

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "point_id": int(self.point_ids[index]),
            "distance": float(self.distances[index]),
        }
        if self.metadata and index < len(self.metadata):
            row["metadata"] = self.metadata[index]
        return row

    # ── conversions ──────────────────────────────────────────

    def to_list(self) -> List[Dict[str, Any]]:
        return list(self)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "point_ids": self.point_ids.tolist(),
            "distances": self.distances.tolist(),
            "metadata": self.metadata,
            "execution_time": self.execution_time,
            "chart_id": self.chart_id,
            "query_type": self.query_type,
            "count": len(self),
        }

    def to_dataframe(self):
        """Return a pandas DataFrame (imported lazily to avoid hard dep)."""
        import pandas as pd  # type: ignore
        records = self.to_list()
        if not records:
            return pd.DataFrame(columns=["point_id", "distance"])
        return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────
# Execution Plan
# ──────────────────────────────────────────────────────────────

@dataclass
class ExecutionStep:
    """Single step in a query execution plan."""
    name: str
    description: str
    estimated_cost_ms: float = 0.0
    depends_on: List[int] = field(default_factory=list)


@dataclass
class ExecutionPlan:
    """
    Ordered list of execution steps with cost estimates.
    Can be visualised as an ASCII diagram.
    """
    steps: List[ExecutionStep] = field(default_factory=list)
    total_estimated_ms: float = 0.0

    def add_step(self, name: str, description: str,
                 estimated_cost_ms: float = 0.0,
                 depends_on: Optional[List[int]] = None) -> int:
        idx = len(self.steps)
        self.steps.append(ExecutionStep(
            name=name,
            description=description,
            estimated_cost_ms=estimated_cost_ms,
            depends_on=depends_on or [],
        ))
        self.total_estimated_ms += estimated_cost_ms
        return idx

    def visualize(self, width: int = 72) -> str:
        """Return an ASCII execution plan."""
        lines: List[str] = []
        lines.append("=" * width)
        lines.append("  QUERY EXECUTION PLAN")
        lines.append("=" * width)
        lines.append(f"  Total estimated cost: {self.total_estimated_ms:.2f} ms")
        lines.append(f"  Steps: {len(self.steps)}")
        lines.append("-" * width)

        for i, step in enumerate(self.steps):
            arrow = "├──" if i < len(self.steps) - 1 else "└──"
            cost_str = f"{step.estimated_cost_ms:.1f} ms"
            dep_str = ""
            if step.depends_on:
                dep_str = f"  (after step {step.depends_on})"
            lines.append(f"  {arrow} Step {i}: {step.name}")
            lines.append(f"  │      {step.description}")
            lines.append(f"  │      est. {cost_str}{dep_str}")
            if i < len(self.steps) - 1:
                lines.append(f"  │")

        lines.append("=" * width)
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Query Engine
# ──────────────────────────────────────────────────────────────

class QueryEngine:
    """
    Executes ManifoldQuery objects against the manifold database.

    Orchestrates:
      1. Atlas lookup – locate the chart for a query point
      2. Tangent-space projection and search
      3. Geodesic distance refinement
      4. Parallel transport for cross-chart / cross-modal queries
      5. Result assembly
    """

    def __init__(
        self,
        atlas_manager: Optional[Any] = None,
        metric_store: Optional[Any] = None,
        tangent_index: Optional[Any] = None,
        geodesic_solver: Optional[Any] = None,
        connection: Optional[Any] = None,
    ) -> None:
        self.atlas_manager = atlas_manager or _Stub()
        self.metric_store = metric_store or _Stub()
        self.tangent_index = tangent_index or _Stub()
        self.geodesic_solver = geodesic_solver or _Stub()
        self.connection = connection
        logger.info("QueryEngine initialised (stubs=%s)", isinstance(self.atlas_manager, _Stub))

    # ── public interface ──────────────────────────────────────

    async def execute(self, query: ManifoldQuery) -> QueryResult:
        """Execute a single ManifoldQuery and return a QueryResult."""
        valid, msg = query.validate()
        if not valid:
            raise ValueError(f"Cannot execute invalid query: {msg}")

        t0 = time.perf_counter()

        if query.query_type == QueryType.SELECT:
            result = await self._execute_geodesic_query(query)
        elif query.query_type == QueryType.TANGENT:
            result = await self._execute_tangent_query(query)
        elif query.query_type == QueryType.CROSS_MODAL:
            result = await self._execute_cross_modal_query(query)
        elif query.query_type == QueryType.TRANSPORT:
            result = await self._execute_transport_query(query)
        elif query.query_type == QueryType.RANGE:
            result = await self._execute_range_query(query)
        else:
            raise ValueError(f"Unknown query type: {query.query_type}")

        elapsed = time.perf_counter() - t0
        result.execution_time = elapsed
        result.query_type = query.query_type.value
        logger.info(
            "Query executed: type=%s results=%d time=%.4fs",
            query.query_type.value, len(result), elapsed,
        )
        return result

    async def batch_execute(self, queries: Sequence[ManifoldQuery]) -> List[QueryResult]:
        """Execute multiple queries concurrently using asyncio.gather."""
        logger.info("Batch executing %d queries", len(queries))
        tasks = [self.execute(q) for q in queries]
        return list(await asyncio.gather(*tasks))

    async def explain(self, query: ManifoldQuery) -> ExecutionPlan:
        """Return an execution plan without actually running the query."""
        plan = ExecutionPlan()

        if query.query_type == QueryType.SELECT:
            plan.add_step("locate_chart",
                          "Locate chart containing the query point via atlas lookup",
                          estimated_cost_ms=0.5)
            plan.add_step("tangent_project",
                          "Project query point into chart's tangent space",
                          estimated_cost_ms=0.2, depends_on=[0])
            plan.add_step("tangent_search",
                          "Search candidates in tangent-space index",
                          estimated_cost_ms=1.0, depends_on=[1])
            plan.add_step("geodesic_refine",
                          "Refine top-k with true geodesic distances",
                          estimated_cost_ms=5.0, depends_on=[2])
            plan.add_step("assemble",
                          "Assemble result set with metadata",
                          estimated_cost_ms=0.1, depends_on=[3])

        elif query.query_type == QueryType.TANGENT:
            plan.add_step("tangent_search",
                          "Search directly in chart's tangent space",
                          estimated_cost_ms=1.0)
            plan.add_step("assemble",
                          "Assemble result set",
                          estimated_cost_ms=0.1, depends_on=[0])

        elif query.query_type == QueryType.CROSS_MODAL:
            plan.add_step("locate_source",
                          "Locate source chart for modality",
                          estimated_cost_ms=0.5)
            plan.add_step("transport_vector",
                          "Parallel-transport query vector to target chart",
                          estimated_cost_ms=2.0, depends_on=[0])
            plan.add_step("target_search",
                          "Search in target chart's tangent space",
                          estimated_cost_ms=1.0, depends_on=[1])
            plan.add_step("assemble",
                          "Assemble cross-modal results",
                          estimated_cost_ms=0.1, depends_on=[2])

        elif query.query_type == QueryType.TRANSPORT:
            plan.add_step("locate_charts",
                          "Resolve source and target chart IDs",
                          estimated_cost_ms=0.2)
            plan.add_step("compute_transport",
                          "Compute parallel transport along overlap",
                          estimated_cost_ms=3.0, depends_on=[0])
            plan.add_step("assemble",
                          "Return transported vector",
                          estimated_cost_ms=0.05, depends_on=[1])

        elif query.query_type == QueryType.RANGE:
            plan.add_step("locate_chart",
                          "Locate chart for query point",
                          estimated_cost_ms=0.5)
            plan.add_step("candidate_scan",
                          "Scan candidates within epsilon ball in tangent space",
                          estimated_cost_ms=2.0, depends_on=[0])
            plan.add_step("geodesic_refine",
                          "Filter by true geodesic distance",
                          estimated_cost_ms=5.0, depends_on=[1])
            plan.add_step("assemble",
                          "Assemble range query results",
                          estimated_cost_ms=0.1, depends_on=[2])

        return plan

    # ── internal executors ────────────────────────────────────

    async def _execute_geodesic_query(self, query: ManifoldQuery) -> QueryResult:
        """
        Full geodesic query pipeline:
          1. Locate chart for query point
          2. Project to tangent space
          3. Search candidates in tangent space
          4. Refine with true geodesic distance
          5. Return top-k with distances
        """
        if query.query_point is None:
            return QueryResult()

        # Step 1: locate chart
        chart = await self.atlas_manager.lookup_chart(query.query_point)
        chart_id = query.chart_id or (chart["chart_id"] if chart else "default")

        # Step 2: project (identity in euclidean charts)
        tangent_point = query.query_point

        # Step 3: search candidates
        ids, tang_dists = await self.tangent_index.tangent_search(
            chart_id, tangent_point, query.k * 3  # oversample
        )

        if len(ids) == 0:
            return QueryResult(chart_id=chart_id)

        # Step 4: geodesic refinement
        embeddings = await self.metric_store.get_embeddings(ids)
        geodesic_dists = np.zeros(len(ids), dtype=np.float64)
        for i in range(len(ids)):
            geodesic_dists[i] = await self.geodesic_solver.geodesic_distance(
                tangent_point, embeddings[i]
            )

        # Sort by geodesic distance, take top-k
        order = np.argsort(geodesic_dists)[: query.k]
        top_ids = ids[order]
        top_dists = geodesic_dists[order]

        return QueryResult(
            point_ids=top_ids,
            distances=top_dists,
            chart_id=chart_id,
        )

    async def _execute_tangent_query(self, query: ManifoldQuery) -> QueryResult:
        """Fast tangent-space-only query (no geodesic refinement)."""
        if query.query_point is None:
            return QueryResult()

        chart_id = query.chart_id or "default"
        ids, dists = await self.tangent_index.tangent_search(
            chart_id, query.query_point, query.k
        )

        # Filter by epsilon if specified
        if query.epsilon < float("inf"):
            mask = dists <= query.epsilon
            ids = ids[mask]
            dists = dists[mask]

        return QueryResult(
            point_ids=ids,
            distances=dists,
            chart_id=chart_id,
        )

    async def _execute_cross_modal_query(self, query: ManifoldQuery) -> QueryResult:
        """
        Cross-modal retrieval:
          1. Locate source chart
          2. Parallel transport query vector to target chart
          3. Search in target chart's tangent space
          4. Return results
        """
        source_chart = query.source_chart or query.chart_id or "default"
        target_chart = query.target_chart or f"{query.target_modality}_chart"

        if query.query_point is None:
            return QueryResult()

        # Step 2: parallel transport
        transported = await self.geodesic_solver.parallel_transport(
            query.query_point, source_chart, target_chart
        )

        # Step 3: search in target
        ids, dists = await self.tangent_index.tangent_search(
            target_chart, transported, query.k
        )

        return QueryResult(
            point_ids=ids,
            distances=dists,
            chart_id=target_chart,
            metadata=[
                {"source_chart": source_chart, "target_chart": target_chart}
                for _ in range(len(ids))
            ],
        )

    async def _execute_transport_query(self, query: ManifoldQuery) -> QueryResult:
        """Pure parallel transport – returns the transported vector as a result."""
        if query.transport_vector is None:
            return QueryResult()

        transported = await self.geodesic_solver.parallel_transport(
            query.transport_vector, query.source_chart, query.target_chart
        )

        # Pack the single transported vector as a result
        return QueryResult(
            point_ids=np.array([0], dtype=np.int64),
            distances=np.array([0.0], dtype=np.float64),
            chart_id=query.target_chart,
            metadata=[{"transported_vector": transported.tolist()}],
        )

    async def _execute_range_query(self, query: ManifoldQuery) -> QueryResult:
        """Geodesic ball query – all points within epsilon distance."""
        if query.query_point is None:
            return QueryResult()

        chart_id = query.chart_id or "default"

        # Oversample from tangent index
        oversample = max(query.k * 5, 100)
        ids, tang_dists = await self.tangent_index.tangent_search(
            chart_id, query.query_point, oversample
        )

        if len(ids) == 0:
            return QueryResult(chart_id=chart_id)

        # Refine with geodesic distance
        embeddings = await self.metric_store.get_embeddings(ids)
        geodesic_dists = np.zeros(len(ids), dtype=np.float64)
        for i in range(len(ids)):
            geodesic_dists[i] = await self.geodesic_solver.geodesic_distance(
                query.query_point, embeddings[i]
            )

        # Filter by epsilon
        mask = geodesic_dists <= query.epsilon
        result_ids = ids[mask]
        result_dists = geodesic_dists[mask]

        # Sort by distance
        order = np.argsort(result_dists)
        result_ids = result_ids[order]
        result_dists = result_dists[order]

        return QueryResult(
            point_ids=result_ids,
            distances=result_dists,
            chart_id=chart_id,
        )
