# API Reference

> Complete API documentation for all public classes and methods.

## Atlas Layer

### `Chart`

```python
class Chart:
    """Local coordinate patch on the manifold."""
    
    def __init__(
        self,
        name: str,
        dim: int,                    # Intrinsic (chart) dimension
        ambient_dim: int,             # Ambient (embedding) dimension
        embedding_fn: Callable = None,  # (N, D) → (N, dim)
        inverse_fn: Callable = None,    # (N, dim) → (N, D)
        bounds: tuple = None,        # (min_coords, max_coords) each (dim,)
        anchor_points: ndarray = None,
        chart_id: str = None,        # Auto-generated UUID
        metadata: dict = None,
    )
```

| Method | Returns | Description |
|--------|---------|-------------|
| `embed(data)` | `ndarray (N, dim)` | Project ambient → chart coords |
| `inverse(coords)` | `ndarray (N, D)` | Lift chart → ambient coords |
| `contains(coords, margin=0.0)` | `ndarray (N,) bool` | Check bounds containment |
| `to_dict()` | `dict` | Serialize to JSON dict |
| `from_dict(d)` | `Chart` | Deserialize from dict |
| `summary()` | `dict` | Human-readable summary |

### `TransitionMap` (Abstract)

```python
class TransitionMap:
    """Diffeomorphism between chart coordinate systems."""
    
    def forward(self, coords: np.ndarray) -> np.ndarray
    def inverse(self, coords: np.ndarray) -> np.ndarray
    def to_dict(self) -> dict
```

### `AffineTransition`

```python
class AffineTransition(TransitionMap):
    """y = M @ x + b transition."""
    
    def __init__(
        self,
        source_chart_id: str,
        target_chart_id: str,
        dim: int,
        matrix: np.ndarray,  # (d, d)
        bias: np.ndarray,    # (d,)
        overlap_region: tuple = None,
    )
```

### `LinearTransition`

```python
class LinearTransition(TransitionMap):
    """y = M @ x transition (no bias)."""
    
    def __init__(
        self,
        source_chart_id: str,
        target_chart_id: str,
        dim: int,
        matrix: np.ndarray,  # (d, d)
    )
```

### `NeuralTransition`

```python
class NeuralTransition(TransitionMap):
    """Invertible MLP coupling layer transition."""
    
    def __init__(
        self,
        source_chart_id: str,
        target_chart_id: str,
        dim: int,
        hidden_dim: int = 64,
        n_layers: int = 3,
    )
    
    def fit(self, source, target, epochs=300) -> None
```

### `AtlasManager`

```python
class AtlasManager:
    """Central orchestrator for charts and transitions."""
    
    def __init__(self, name: str = "default_atlas", metadata: dict = None)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `add_chart(chart)` | `None` | Register a chart |
| `remove_chart(chart_id)` | `None` | Remove chart + transitions |
| `get_chart(chart_id)` | `Chart` | Get chart by ID |
| `get_all_charts()` | `List[Chart]` | List all charts |
| `add_transition_map(tmap)` | `None` | Register transition |
| `get_transition(src, tgt)` | `TransitionMap` | Get transition |
| `get_all_transition_maps()` | `List[TransitionMap]` | List all |
| `find_chart(data, modality=None)` | `Chart or None` | Best chart for data |
| `build_atlas(data, **kwargs)` | `None` | Auto-build from data |
| `rebuild_overlaps()` | `None` | Recompute overlaps |
| `atlas_summary()` | `dict` | Statistics dictionary |
| `save(filepath)` | `None` | Write JSON file |
| `load(filepath)` | `None` | Read JSON file |
| `serialize()` | `dict` | JSON-compatible dict |
| `deserialize(d)` | `None` | Restore from dict |

### `AtlasBuilder`

```python
class AtlasBuilder:
    """Automatic manifold-learning atlas constructor."""
    
    def __init__(
        self,
        k_neighbors: int = 15,
        pca_variance_threshold: float = 0.95,
        min_chart_size: int = 50,
        overlap_margin: float = 0.05,
        random_state: int = None,
    )
```

| Method | Returns | Description |
|--------|---------|-------------|
| `estimate_intrinsic_dimension(data, n_samples=500)` | `int` | Auto-detect dim |
| `compute_knn_graph(data, k=None)` | `sparse.csr_matrix` | KNN adjacency |
| `detect_chart_boundaries(graph, ...)` | `List[ndarray]` | Chart assignments |
| `build(data, n_charts_hint=None)` | `AtlasManager` | Full pipeline |
| `analyze_quality(atlas, data)` | `dict` | Quality metrics |

---

## Tangent Index

### `TangentSpace`

```python
class TangentSpace:
    """Local linear approximation of the manifold at a point."""
    
    def __init__(
        self,
        base_point: np.ndarray,       # (ambient_dim,)
        data: np.ndarray = None,       # (n, ambient_dim) local neighbourhood
        intrinsic_dim: int = None,      # Override or auto-detect
    )
```

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `project` | `(data) → coords` | `ndarray (n, d)` | Ambient → tangent |
| `lift` | `(coords) → ambient` | `ndarray (n, D)` | Tangent → ambient |
| `log_map` | `(point) → vec` | `ndarray (d,)` | Point → tangent vector |
| `exp_map` | `(vec) → point` | `ndarray (D,)` | Tangent vector → point |
| `compute_metric` | `(v1, v2) → float` | `float` | Riemannian inner product |
| `compute_distance` | `(v1, v2) → float` | `float` | Geodesic distance |
| `parallel_transport` | `(vec, target_ts)` | `ndarray` | Transport vector |
| `update_basis` | `(batch, lr=0.1)` | `None` | Online basis update |
| `to_dict` | `→ dict` | `dict` | Serialize |
| `from_dict` | `(d) → TangentSpace` | `TangentSpace` | Deserialize |

**Properties:**
- `base_point` — Anchor point on manifold (D,)
- `basis` — Orthonormal basis matrix (D, d)
- `dimension` — Intrinsic dimension d
- `metric_tensor` — Riemannian metric G (d, d)
- `christoffel_symbols` — Γ^i_{jk} (d, d, d)

### `TangentBundle`

```python
class TangentBundle:
    """Collection of tangent spaces covering the manifold."""
    
    def __init__(self, intrinsic_dim=None, metric_eps=0.1)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `add_tangent_space(ts)` | `None` | Add a TS (rebuilds KDTree) |
| `remove_tangent_space(id)` | `None` | Remove a TS |
| `nearest_anchor(point, k)` | `List[(id, dist)]` | k nearest anchors |
| `get_optimal_tangent_space(point, k)` | `(ts, weight)` | Best TS combination |
| `reindex(data, n_anchors)` | `None` | Full rebuild (FPS) |
| `coverage_analysis(data)` | `dict` | Projection error stats |
| `serialize()` | `dict` | Serialize |
| `deserialize(d)` | `TangentBundle` | Deserialize |

### `TangentSpaceIndex`

```python
class TangentSpaceIndex:
    """Main index for approximate nearest-neighbour search on manifolds."""
    
    def __init__(
        self,
        intrinsic_dim: int = None,
        metric_eps: float = 0.1,
        leaf_size: int = 40,
    )
```

| Method | Returns | Description |
|--------|---------|-------------|
| `build_from_data(ids, data, n_anchors)` | `dict` | Build from scratch |
| `insert(point_id, data_point)` | `None` | Insert single point |
| `batch_insert(ids, data)` | `None` | Insert multiple points |
| `search(query, k, n_candidates, search_k_anchors)` | `List[(id, dist)]` | k-NN search |
| `delete(point_id)` | `bool` | Delete point |
| `update(point_id, new_data)` | `bool` | Update point |
| `to_dict()` | `dict` | Serialize |
| `from_dict(d)` | `TangentSpaceIndex` | Deserialize |

**Properties:**
- `size` — Total indexed points
- `stats` — Summary dict
- `chart_id` — Unique index identifier

---

## Query Layer

### `QueryParser`

```python
class QueryParser:
    """Parse SQL-like strings into ManifoldQuery objects."""
    
    def parse(self, text: str) -> ManifoldQuery
```

### `QueryBuilder`

```python
class QueryBuilder:
    """Fluent builder for ManifoldQuery objects."""
    
    def select(*fields) -> QueryBuilder
    def from_chart(chart_id) -> QueryBuilder
    def along_manifold(atlas_name) -> QueryBuilder
    def using_metric(metric) -> QueryBuilder
    def where_geodesic(query_point, epsilon) -> QueryBuilder
    def tangent_query(chart_id, query_point, epsilon) -> QueryBuilder
    def cross_modal(source, target) -> QueryBuilder
    def with_transport(via) -> QueryBuilder
    def parallel_transport(vector, source, target) -> QueryBuilder
    def top_k(k) -> QueryBuilder
    def limit(n) -> QueryBuilder
    def order_by(field) -> QueryBuilder
    def with_metadata(**kwargs) -> QueryBuilder
    def build() -> ManifoldQuery
```

### `ManifoldQuery`

```python
@dataclass
class ManifoldQuery:
    query_type: QueryType = QueryType.SELECT
    chart_id: str = None
    metric_type: MetricType = MetricType.GEODESIC
    query_point: np.ndarray = None
    epsilon: float = 1.0
    k: int = 10
    modality: str = None
    target_modality: str = None
    fields: List[str] = field(default_factory=lambda: ["*"])
    atlas_name: str = None
    transport_via: str = None
    source_chart: str = None
    target_chart: str = None
    transport_vector: np.ndarray = None
    order_by: str = None
    limit: int = None
    metadata: dict = field(default_factory=dict)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `validate()` | `(bool, str)` | Check query validity |
| `estimate_cost(chart_sizes)` | `CostTier` | Cost estimation |
| `to_dict()` | `dict` | Serialize |
| `from_dict(d)` | `ManifoldQuery` | Deserialize |

### `QueryEngine`

```python
class QueryEngine:
    """Execute ManifoldQuery objects."""
    
    def __init__(
        self,
        atlas_manager=None,
        metric_store=None,
        tangent_index=None,
        geodesic_solver=None,
        connection=None,
    )
```

| Method | Returns | Description |
|--------|---------|-------------|
| `execute(query)` | `QueryResult` | Execute single query (async) |
| `batch_execute(queries)` | `List[QueryResult]` | Concurrent batch (async) |
| `explain(query)` | `ExecutionPlan` | Execution plan (async) |

### `QueryResult`

```python
@dataclass
class QueryResult:
    point_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    distances: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    metadata: List[Dict] = None
    execution_time: float = 0.0
    chart_id: str = None
    query_type: str = None
```

| Method | Returns |
|--------|---------|
| `to_list()` | `List[Dict]` |
| `to_dict()` | `Dict` |
| `to_dataframe()` | `pd.DataFrame` |

### Enums

```python
class QueryType(str, Enum):
    SELECT = "select"
    TANGENT = "tangent"
    CROSS_MODAL = "cross_modal"
    TRANSPORT = "transport"
    RANGE = "range"

class MetricType(str, Enum):
    GEODESIC = "geodesic"
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    WASSERSTEIN_RIEMANNIAN = "wasserstein_riemannian"
    FISCHER_RAOC = "fischer_rao"
    LOG_EUCLIDEAN = "log_euclidean"

class CostTier(str, Enum):
    CHEAP = "cheap"
    MODERATE = "moderate"
    EXPENSIVE = "expensive"
    VERY_EXPENSIVE = "very_expensive"
```

---

## Metric Layer

### `MetricTensor` (Abstract)

```python
class MetricTensor:
    """Base class for Riemannian metric tensors."""
    
    def evaluate(self, x: np.ndarray) -> np.ndarray     # → (d, d)
    def inverse(self, x: np.ndarray) -> np.ndarray       # → (d, d)
    def determinant(self, x: np.ndarray) -> float
    def log_det(self, x: np.ndarray) -> float
    def christoffel_symbols(self, x) -> np.ndarray        # → (d, d, d)
    def sectional_curvature(self, x, v1, v2) -> float
    def ricci_curvature(self, x) -> np.ndarray            # → (d, d)
    def scalar_curvature(self, x) -> float
    def to_dict() -> dict
    def from_dict(d) -> MetricTensor
```

### `EuclideanMetric`

```python
class EuclideanMetric(MetricTensor):
    def __init__(self, dimension: int)
    # All curvatures = 0 (flat metric)
```

### `DiagonalMetric`

```python
class DiagonalMetric(MetricTensor):
    def __init__(
        self,
        dimension: int,
        weights: np.ndarray = None,    # (d,) diagonal values
        variation_fn: Callable = None,  # position-dependent scaling
    )
```

### `MetricTensorStore`

```python
class MetricTensorStore:
    """Per-chart metric tensor registry."""
    
    def register(self, chart_id: str, metric: MetricTensor) -> None
    def get(self, chart_id: str) -> MetricTensor  # Falls back to Euclidean
    def list_charts(self) -> List[str]
    def unregister(self, chart_id: str) -> None
    def serialize() -> dict
    def deserialize(d) -> None
```

---

## Connection Layer

### `LeviCivitaConnection`

```python
class LeviCivitaConnection:
    """Torsion-free, metric-compatible connection."""
    
    def connection_coefficients(point, metric_fn) -> np.ndarray
    def parallel_transport(vector, path, metric_fn) -> np.ndarray
    def parallel_transport_along_geodesic(vector, start, end, metric_fn) -> np.ndarray
    def covariant_derivative(V, direction, point, metric_fn) -> np.ndarray
    def transport_across_charts(vector, src, tgt, transition_map) -> np.ndarray
    def transport_batch(vectors, paths, metric_fn) -> np.ndarray
```

### `SchemaTransport`

```python
class SchemaTransport:
    """Cross-schema-version manifold transport."""
    
    def register_schema(schema_id, reference_point, metric) -> None
    def transport_query(query_vector, source_schema, target_schema) -> np.ndarray
    def get_schema_info(schema_id) -> dict
```

### `TemporalTransport`

```python
class TemporalTransport:
    """Time-evolution transport on temporal manifolds."""
    
    def register_temporal_path(times, reference_curve, metric_fn) -> None
    def transport(vector, source_time, target_time) -> np.ndarray
```

### `TransportRegistry`

```python
class TransportRegistry:
    """LRU cache for transport functions."""
    
    def __init__(self, max_size: int = 256)
    def register_transport(chart_id_a, chart_id_b, transport_fn) -> None
    def get_transport(chart_id_a, chart_id_b) -> Callable
    def get_cached(chart_id_a, chart_id_b, vector) -> np.ndarray
    def put_cached(chart_id_a, chart_id_b, vector, result) -> None
    def has_transport(chart_id_a, chart_id_b) -> bool
    def compute_chain(transport_chain, vector) -> np.ndarray
    def invalidate(chart_id) -> None
    def precompute_heatmap(chart_ids) -> np.ndarray
    def serialize() -> dict
    def deserialize(d) -> None
```

---

## Geodesic Layer

### `RiemannianDistance`

```python
class RiemannianDistance:
    def __init__(self, metric_tensor_fn=None, christoffel_fn=None)
    
    def geodesic_distance(p, q) -> float          # Energy minimisation
    def tangent_approx_distance(p, q) -> float    # Tangent-space approx
    def curvature_corrected_distance(p, q) -> float  # 2nd-order correction
```

### `WassersteinDistance`

```python
class WassersteinDistance:
    def __init__(self, reg=0.1, max_iter=500, tol=1e-8)
    
    def sinkhorn_distance(mu, nu, cost_matrix, reg=None) -> float
    def batch_sinkhorn_distance(mu_batch, nu_batch, costs) -> np.ndarray
```

### `FisherRaoDistance`

```python
class FisherRaoDistance:
    def fisher_rao_distance(p_dist, q_dist) -> float    # ∈ [0, π]
    def fisher_rao_metric_tensor(distribution) -> np.ndarray
```

### `DistanceComputer`

```python
class DistanceComputer:
    def __init__(self, metric_tensor_fn=None, christoffel_fn=None)
    
    def compute(p, q, metric_type="geodesic", **kwargs) -> float
    def batch_compute(points_a, points_b, metric_type="geodesic") -> np.ndarray
```
