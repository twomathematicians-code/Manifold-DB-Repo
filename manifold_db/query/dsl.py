"""
Manifold Query DSL - SQL-like query language for manifold databases.
Supports geodesic queries, tangent-space queries, cross-chart queries,
parallel transport, and cross-modal retrieval across manifold atlases.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────


class MetricType(str, Enum):
    """Supported distance metrics on the manifold."""

    GEODESIC = "geodesic"
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    WASSERSTEIN_RIEMANNIAN = "wasserstein_riemannian"
    FISCHER_RAOC = "fischer_rao"
    LOG_EUCLIDEAN = "log_euclidean"


class QueryType(str, Enum):
    """Classification of query types."""

    SELECT = "select"
    TANGENT = "tangent"
    CROSS_MODAL = "cross_modal"
    TRANSPORT = "transport"
    RANGE = "range"


class CostTier(str, Enum):
    """Rough cost tiers for query planning."""

    CHEAP = "cheap"  # < 1 ms tangent-only lookup
    MODERATE = "moderate"  # ~10 ms with geodesic refinement
    EXPENSIVE = "expensive"  # ~100 ms cross-modal + transport
    VERY_EXPENSIVE = "very_expensive"  # > 100 ms full geodesic scan


# ──────────────────────────────────────────────────────────────
# Query AST Nodes
# ──────────────────────────────────────────────────────────────


@dataclass
class GeodesicWhereClause:
    """WHERE clause using geodesic distance on the manifold."""

    embedding_field: str = "embedding"
    query_point: np.ndarray | None = None
    epsilon: float = 1.0
    operator: str = "<"
    metric: MetricType = MetricType.GEODESIC

    def evaluate(self, point: np.ndarray) -> bool:
        """Evaluate this clause against a point vector."""
        dist = self._compute_distance(point)
        if self.operator == "<":
            return dist < self.epsilon
        elif self.operator == "<=":
            return dist <= self.epsilon
        elif self.operator == ">":
            return dist > self.epsilon
        elif self.operator == ">=":
            return dist >= self.epsilon
        else:
            raise ValueError(f"Unsupported operator: {self.operator}")

    def _compute_distance(self, point: np.ndarray) -> float:
        if self.query_point is None:
            return 0.0
        if self.metric == MetricType.EUCLIDEAN:
            return float(np.linalg.norm(point - self.query_point))
        elif self.metric == MetricType.COSINE:
            norm_p = np.linalg.norm(point)
            norm_q = np.linalg.norm(self.query_point)
            if norm_p == 0 or norm_q == 0:
                return 1.0
            return 1.0 - float(np.dot(point, self.query_point) / (norm_p * norm_q))
        else:
            # Default to euclidean as approximation
            return float(np.linalg.norm(point - self.query_point))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "geodesic_where",
            "embedding_field": self.embedding_field,
            "query_point": (
                self.query_point.tolist() if self.query_point is not None else None
            ),
            "epsilon": self.epsilon,
            "operator": self.operator,
            "metric": self.metric.value,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GeodesicWhereClause:
        qp = d.get("query_point")
        return cls(
            embedding_field=d.get("embedding_field", "embedding"),
            query_point=np.array(qp) if qp is not None else None,
            epsilon=d.get("epsilon", 1.0),
            operator=d.get("operator", "<"),
            metric=MetricType(d.get("metric", "geodesic")),
        )


@dataclass
class SelectQuery:
    """SELECT query with optional geodesic WHERE clause."""

    fields: list[str] = field(default_factory=lambda: ["*"])
    source: str = "observations"
    where: GeodesicWhereClause | None = None
    atlas_name: str | None = None
    metric: MetricType = MetricType.GEODESIC
    order_by: str | None = None
    limit: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "select",
            "fields": self.fields,
            "source": self.source,
            "where": self.where.to_dict() if self.where else None,
            "atlas_name": self.atlas_name,
            "metric": self.metric.value,
            "order_by": self.order_by,
            "limit": self.limit,
        }


@dataclass
class TangentQuery:
    """Query operating entirely in a single chart's tangent space."""

    chart_id: str
    query_point: np.ndarray
    epsilon: float = 1.0
    metric: MetricType = MetricType.EUCLIDEAN
    top_k: int | None = None
    fields: list[str] = field(default_factory=lambda: ["*"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "tangent",
            "chart_id": self.chart_id,
            "query_point": self.query_point.tolist(),
            "epsilon": self.epsilon,
            "metric": self.metric.value,
            "top_k": self.top_k,
            "fields": self.fields,
        }


@dataclass
class CrossModalQuery:
    """Cross-modal retrieval with parallel transport between charts."""

    source_modality: str
    target_modality: str
    query_point: np.ndarray | None = None
    transport_via: str = "overlap_region"
    top_k: int = 10
    metric: MetricType = MetricType.GEODESIC
    source_chart: str | None = None
    target_chart: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "cross_modal",
            "source_modality": self.source_modality,
            "target_modality": self.target_modality,
            "query_point": (
                self.query_point.tolist() if self.query_point is not None else None
            ),
            "transport_via": self.transport_via,
            "top_k": self.top_k,
            "metric": self.metric.value,
            "source_chart": self.source_chart,
            "target_chart": self.target_chart,
        }


@dataclass
class TransportQuery:
    """Pure parallel transport of a vector between charts."""

    vector: np.ndarray
    source_chart: str
    target_chart: str
    path_points: list[np.ndarray] | None = None
    via_overlap: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "transport",
            "vector": self.vector.tolist(),
            "source_chart": self.source_chart,
            "target_chart": self.target_chart,
            "path_points": (
                [p.tolist() for p in self.path_points] if self.path_points else None
            ),
            "via_overlap": self.via_overlap,
        }


# ──────────────────────────────────────────────────────────────
# Executed Query
# ──────────────────────────────────────────────────────────────


@dataclass
class ManifoldQuery:
    """
    Fully-compiled query ready for execution by the QueryEngine.
    Carries all parameters the engine needs: chart, metric, point, k, etc.
    """

    query_type: QueryType = QueryType.SELECT
    chart_id: str | None = None
    metric_type: MetricType = MetricType.GEODESIC
    query_point: np.ndarray | None = None
    epsilon: float = 1.0
    k: int = 10
    modality: str | None = None
    target_modality: str | None = None
    fields: list[str] = field(default_factory=lambda: ["*"])
    atlas_name: str | None = None
    transport_via: str | None = None
    source_chart: str | None = None
    target_chart: str | None = None
    transport_vector: np.ndarray | None = None
    order_by: str | None = None
    limit: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── validation ────────────────────────────────────────────

    def validate(self) -> tuple[bool, str]:
        """Check whether this query is valid and return (ok, message)."""
        if self.query_point is not None:
            if not isinstance(self.query_point, np.ndarray):
                try:
                    self.query_point = np.asarray(self.query_point, dtype=np.float64)
                except Exception as e:
                    return False, f"query_point is not convertible to ndarray: {e}"
            if self.query_point.ndim != 1:
                return (
                    False,
                    f"query_point must be 1-D, got ndim={self.query_point.ndim}",
                )
            if self.query_point.size == 0:
                return False, "query_point must not be empty"

        if self.epsilon <= 0:
            return False, f"epsilon must be positive, got {self.epsilon}"

        if self.k < 1:
            return False, f"k must be >= 1, got {self.k}"

        if self.query_type == QueryType.CROSS_MODAL:
            if not self.modality or not self.target_modality:
                return False, "cross_modal query requires modality and target_modality"

        if self.query_type == QueryType.TRANSPORT:
            if self.transport_vector is None:
                return False, "transport query requires transport_vector"
            if not self.source_chart or not self.target_chart:
                return False, "transport query requires source_chart and target_chart"

        return True, "valid"

    # ── cost estimation ──────────────────────────────────────

    def estimate_cost(self, chart_sizes: dict[str, int] | None = None) -> CostTier:
        """Return a rough cost tier based on query type and assumed chart sizes."""
        if self.query_type == QueryType.TANGENT:
            return CostTier.CHEAP
        if self.query_type == QueryType.TRANSPORT:
            return CostTier.MODERATE
        if self.query_type == QueryType.CROSS_MODAL:
            return CostTier.EXPENSIVE
        # SELECT / RANGE — depends on chart size
        n = (chart_sizes or {}).get(self.chart_id or "", 10_000)
        if n < 1_000:
            return CostTier.MODERATE
        elif n < 100_000:
            return CostTier.EXPENSIVE
        return CostTier.VERY_EXPENSIVE

    # ── serialisation ────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "query_type": self.query_type.value,
            "chart_id": self.chart_id,
            "metric_type": self.metric_type.value,
            "query_point": (
                self.query_point.tolist() if self.query_point is not None else None
            ),
            "epsilon": self.epsilon,
            "k": self.k,
            "modality": self.modality,
            "target_modality": self.target_modality,
            "fields": self.fields,
            "atlas_name": self.atlas_name,
            "transport_via": self.transport_via,
            "source_chart": self.source_chart,
            "target_chart": self.target_chart,
            "transport_vector": (
                self.transport_vector.tolist()
                if self.transport_vector is not None
                else None
            ),
            "order_by": self.order_by,
            "limit": self.limit,
            "metadata": self.metadata,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ManifoldQuery:
        qp = d.get("query_point")
        tv = d.get("transport_vector")
        return cls(
            query_type=QueryType(d.get("query_type", "select")),
            chart_id=d.get("chart_id"),
            metric_type=MetricType(d.get("metric_type", "geodesic")),
            query_point=np.array(qp, dtype=np.float64) if qp is not None else None,
            epsilon=d.get("epsilon", 1.0),
            k=d.get("k", 10),
            modality=d.get("modality"),
            target_modality=d.get("target_modality"),
            fields=d.get("fields", ["*"]),
            atlas_name=d.get("atlas_name"),
            transport_via=d.get("transport_via"),
            source_chart=d.get("source_chart"),
            target_chart=d.get("target_chart"),
            transport_vector=np.array(tv, dtype=np.float64) if tv is not None else None,
            order_by=d.get("order_by"),
            limit=d.get("limit"),
            metadata=d.get("metadata", {}),
        )


# ──────────────────────────────────────────────────────────────
# Query Parser
# ──────────────────────────────────────────────────────────────


class QueryParser:
    """
    Parses SQL-like strings into ManifoldQuery objects.

    Supported syntax:
        SELECT * FROM observations WHERE geodesic_distance(embedding, [1,2,3]) < 0.5
        SELECT * FROM observations ALONG manifold 'climate_model_atlas'
        SELECT * FROM observations USING metric 'wasserstein_riemannian'
        TANGENT_QUERY FROM chart 'text_embeddings' WHERE distance < 0.5
        CROSS_MODAL FROM 'text' TO 'image' TRANSPORT VIA 'overlap_region'
        PARALLEL_TRANSPORT vector FROM chart_a TO chart_b
    """

    _TOKEN_RE = re.compile(
        r"""(?x)
        ('[^']*')          # quoted string
        |(\[[^\]]*\])      # array literal like [1,2,3]
        |(\d+\.\d+)        # float
        |(\d+)              # int
        |([A-Za-z_]\w*)    # identifier / keyword
        |([<>=!]+)          # comparison operator
        |(\*)               # wildcard
        """,
        re.IGNORECASE,
    )

    def parse(self, text: str) -> ManifoldQuery:
        """Parse a SQL-like query string into a ManifoldQuery."""
        tokens = self._tokenize(text)
        tokens_upper = [
            t.upper() if not t.startswith(("[", "'", '"')) else t for t in tokens
        ]

        # Determine query type from first token
        if "TANGENT_QUERY" in tokens_upper:
            return self._parse_tangent(tokens, tokens_upper)
        elif "CROSS_MODAL" in tokens_upper:
            return self._parse_cross_modal(tokens, tokens_upper)
        elif "PARALLEL_TRANSPORT" in tokens_upper:
            return self._parse_transport(tokens, tokens_upper)
        else:
            return self._parse_select(tokens, tokens_upper)

    # ── tokenizer ─────────────────────────────────────────────

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        matches = cls._TOKEN_RE.findall(text)
        tokens = []
        for group in matches:
            for tok in group:
                if tok:
                    tokens.append(tok)
                    break
        return tokens

    # ── SELECT ────────────────────────────────────────────────

    def _parse_select(self, tokens: list[str], upper: list[str]) -> ManifoldQuery:
        fields = ["*"]
        atlas_name = None
        metric = MetricType.GEODESIC
        where_clause = None
        epsilon = 1.0
        query_point = None
        limit = None

        i = 0
        while i < len(tokens):
            tok = upper[i]
            if tok == "SELECT":
                fields = self._parse_fields(tokens, upper, i + 1)
                # Advance past the SELECT fields until we hit FROM
                i = i + 1
                while i < len(tokens) and upper[i] != "FROM":
                    i += 1
                continue
            elif tok == "FROM":
                i += 1  # skip source name
            elif tok == "ALONG":
                i += 1
                if i < len(tokens):
                    atlas_name = tokens[i].strip("'\"")
                i += 1
                continue
            elif tok == "USING":
                i += 1
                if i < len(tokens) and upper[i] == "METRIC":
                    i += 1
                if i < len(tokens):
                    metric = MetricType(tokens[i].strip("'\"").lower())
                i += 1
                continue
            elif tok == "WHERE":
                where_clause, query_point, epsilon = self._parse_where(
                    tokens, upper, i + 1
                )
                i = len(tokens)
                continue
            elif tok == "LIMIT":
                i += 1
                if i < len(tokens):
                    limit = int(tokens[i])
                i += 1
                continue
            else:
                i += 1

        mq = ManifoldQuery(
            query_type=QueryType.SELECT,
            metric_type=metric,
            query_point=query_point,
            epsilon=epsilon,
            atlas_name=atlas_name,
            fields=fields,
            limit=limit,
        )
        valid, msg = mq.validate()
        if not valid:
            logger.warning("Query validation warning: %s", msg)
        return mq

    def _parse_fields(self, tokens, upper, start):
        fields = []
        i = start
        while i < len(tokens) and upper[i] not in ("FROM", "WHERE", "LIMIT"):
            f = tokens[i].strip(",").strip()
            if f:
                fields.append(f)
            i += 1
        return fields if fields else ["*"]

    def _parse_where(self, tokens, upper, start):
        query_point = None
        epsilon = 1.0
        i = start
        while i < len(tokens):
            tok = tokens[i]
            # Detect array literal as query point
            if tok.startswith("["):
                try:
                    vals = tok.strip("[]").split(",")
                    query_point = np.array([float(v.strip()) for v in vals])
                except ValueError:
                    pass
                i += 1
                continue
            if tok.startswith("'") or tok.startswith('"'):
                try:
                    vals = tok.strip("'\"").split(",")
                    query_point = np.array([float(v.strip()) for v in vals])
                except ValueError:
                    pass
                i += 1
                continue
            # Detect numeric epsilon
            try:
                epsilon = float(tok)
                i += 1
                continue
            except ValueError:
                pass
            i += 1
        return None, query_point, epsilon

    # ── TANGENT_QUERY ─────────────────────────────────────────

    def _parse_tangent(self, tokens, upper) -> ManifoldQuery:
        chart_id = ""
        query_point = None
        epsilon = 1.0
        top_k = None
        i = 0
        while i < len(tokens):
            if upper[i] == "FROM":
                i += 1
                if i < len(tokens) and upper[i] == "CHART":
                    i += 1
                if i < len(tokens):
                    chart_id = tokens[i].strip("'\"")
                i += 1
            elif upper[i] == "WHERE":
                i += 1
                # skip to distance
                while i < len(tokens) and upper[i] != "DISTANCE":
                    i += 1
                i += 1  # skip "distance"
                # next should be operator then value
                if i < len(tokens):
                    i += 1  # skip operator
                if i < len(tokens):
                    try:
                        epsilon = float(tokens[i])
                    except ValueError:
                        pass
                    i += 1
            elif upper[i] == "TOP_K":
                i += 1
                if i < len(tokens):
                    top_k = int(tokens[i])
                i += 1
            elif upper[i] == "LIMIT":
                i += 1
                if i < len(tokens):
                    top_k = int(tokens[i])
                i += 1
            else:
                i += 1
        return ManifoldQuery(
            query_type=QueryType.TANGENT,
            chart_id=chart_id,
            query_point=query_point,
            epsilon=epsilon,
            k=top_k or 10,
        )

    # ── CROSS_MODAL ───────────────────────────────────────────

    def _parse_cross_modal(self, tokens, upper) -> ManifoldQuery:
        src = "text"
        tgt = "image"
        transport_via = "overlap_region"
        top_k = 10
        i = 0
        while i < len(tokens):
            if upper[i] == "FROM":
                i += 1
                if i < len(tokens):
                    src = tokens[i].strip("'\"")
                i += 1
            elif upper[i] == "TO":
                i += 1
                if i < len(tokens):
                    tgt = tokens[i].strip("'\"")
                i += 1
            elif upper[i] == "VIA":
                i += 1
                if i < len(tokens):
                    transport_via = tokens[i].strip("'\"")
                i += 1
            elif upper[i] == "TOP_K":
                i += 1
                if i < len(tokens):
                    top_k = int(tokens[i])
                i += 1
            elif upper[i] == "LIMIT":
                i += 1
                if i < len(tokens):
                    top_k = int(tokens[i])
                i += 1
            else:
                i += 1
        return ManifoldQuery(
            query_type=QueryType.CROSS_MODAL,
            modality=src,
            target_modality=tgt,
            transport_via=transport_via,
            k=top_k,
        )

    # ── PARALLEL_TRANSPORT ───────────────────────────────────

    def _parse_transport(self, tokens, upper) -> ManifoldQuery:
        vector = np.zeros(1)
        src = ""
        tgt = ""
        i = 0
        while i < len(tokens):
            if tokens[i].startswith("["):
                try:
                    vals = tokens[i].strip("[]").split(",")
                    vector = np.array([float(v.strip()) for v in vals])
                except ValueError:
                    pass
                i += 1
            elif upper[i] == "FROM":
                i += 1
                if i < len(tokens):
                    src = tokens[i].strip("'\"")
                i += 1
            elif upper[i] == "TO":
                i += 1
                if i < len(tokens):
                    tgt = tokens[i].strip("'\"")
                i += 1
            else:
                i += 1
        return ManifoldQuery(
            query_type=QueryType.TRANSPORT,
            source_chart=src,
            target_chart=tgt,
            transport_vector=vector,
        )


# ──────────────────────────────────────────────────────────────
# Fluent Query Builder
# ──────────────────────────────────────────────────────────────


class QueryBuilder:
    """
    Fluent builder API for constructing ManifoldQuery objects programmatically.

    Example:
        QueryBuilder().select('*').from_chart('text')
                      .where_geodesic(query_vec, epsilon=0.5)
                      .top_k(10).build()

        QueryBuilder().cross_modal('text', 'image')
                      .with_transport('overlap_region')
                      .top_k(10).build()
    """

    def __init__(self) -> None:
        self._q: ManifoldQuery = ManifoldQuery()

    def select(self, *fields: str) -> QueryBuilder:
        self._q.fields = list(fields) if fields else ["*"]
        return self

    def from_chart(self, chart_id: str) -> QueryBuilder:
        self._q.chart_id = chart_id
        return self

    def along_manifold(self, atlas_name: str) -> QueryBuilder:
        self._q.atlas_name = atlas_name
        return self

    def using_metric(self, metric: str | MetricType) -> QueryBuilder:
        if isinstance(metric, str):
            metric = MetricType(metric)
        self._q.metric_type = metric
        return self

    def where_geodesic(
        self, query_point: np.ndarray, epsilon: float = 1.0
    ) -> QueryBuilder:
        self._q.query_type = QueryType.SELECT
        self._q.query_point = np.asarray(query_point, dtype=np.float64)
        self._q.epsilon = epsilon
        return self

    def tangent_query(
        self, chart_id: str, query_point: np.ndarray, epsilon: float = 1.0
    ) -> QueryBuilder:
        self._q.query_type = QueryType.TANGENT
        self._q.chart_id = chart_id
        self._q.query_point = np.asarray(query_point, dtype=np.float64)
        self._q.epsilon = epsilon
        return self

    def cross_modal(self, source: str, target: str) -> QueryBuilder:
        self._q.query_type = QueryType.CROSS_MODAL
        self._q.modality = source
        self._q.target_modality = target
        return self

    def with_transport(self, via: str = "overlap_region") -> QueryBuilder:
        self._q.transport_via = via
        return self

    def parallel_transport(
        self, vector: np.ndarray, source: str, target: str
    ) -> QueryBuilder:
        self._q.query_type = QueryType.TRANSPORT
        self._q.transport_vector = np.asarray(vector, dtype=np.float64)
        self._q.source_chart = source
        self._q.target_chart = target
        return self

    def top_k(self, k: int) -> QueryBuilder:
        self._q.k = max(1, k)
        return self

    def limit(self, n: int) -> QueryBuilder:
        self._q.limit = max(1, n)
        return self

    def order_by(self, field: str) -> QueryBuilder:
        self._q.order_by = field
        return self

    def with_metadata(self, **kwargs: Any) -> QueryBuilder:
        self._q.metadata.update(kwargs)
        return self

    def build(self) -> ManifoldQuery:
        """Build and return the ManifoldQuery. Runs validate() first."""
        valid, msg = self._q.validate()
        if not valid:
            raise ValueError(f"Invalid query: {msg}")
        return self._q
