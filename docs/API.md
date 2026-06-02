# ManifoldDB — API Reference

> Complete reference for all public classes, methods, and functions in the
> ManifoldDB Python and C++ APIs.

---

## Table of Contents

1. [ManifoldDB (Python)](#manifolddb-python)
2. [ManifoldPoint (Python)](#manifoldpoint-python)
3. [GeodesicPath (Python)](#geodesicpath-python)
4. [NeighborResult (Python)](#neighborresult-python)
5. [Chart (Python / C++)](#chart-python--c)
6. [LinearChart (Python / C++)](#linearchart-python--c)
7. [ParametricChart (Python / C++)](#parametricchart-python--c)
8. [MetricTensor (Python / C++)](#metrictensor-python--c)
9. [MetricStore (Python / C++)](#metricstore-python--c)
10. [GeodesicSolver (Python / C++)](#geodesicsolver-python--c)
11. [TangentSpaceIndex (Python / C++)](#tangentspaceindex-python--c)
12. [Atlas (Python / C++)](#atlas-python--c)
13. [TransitionMap / LinearTransitionMap (Python / C++)](#transitionmap--lineartransitionmap-python--c)
14. [Config / Stats / SolverConfig (Python / C++)](#config--stats--solverconfig-python--c)
15. [torch_compat Utilities](#torch_compat-utilities)
16. [IO Utilities](#io-utilities)
17. [Enums](#enums)
18. [Exceptions](#exceptions)

---

## ManifoldDB (Python)

High-level Python wrapper around the C++ `ManifoldDB` engine. Accepts numpy arrays and PyTorch tensors, returns Python-native dicts and lists.

### Constructor

```python
manifolddb.ManifoldDB(
    storage_path: str = "./manifolddb_data",
    intrinsic_dim: int = 10,
    enable_cuda: bool = False,
    geodesic_tolerance: float = 1e-6,
    max_charts: int = 20,
    rbf_bandwidth: float = 1.0,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `storage_path` | `str` | `"./manifolddb_data"` | Filesystem path for persistent storage (metrics, indexes) |
| `intrinsic_dim` | `int` | `10` | Default intrinsic (manifold) dimension `d` |
| `enable_cuda` | `bool` | `False` | Enable CUDA-accelerated geodesic solvers |
| `geodesic_tolerance` | `float` | `1e-6` | Convergence tolerance for the geodesic solver |
| `max_charts` | `int` | `20` | Maximum number of charts for atlas discovery |
| `rbf_bandwidth` | `float` | `1.0` | Bandwidth parameter for RBF metric interpolation |

**Raises**: `ValueError` if `intrinsic_dim < 1`, `geodesic_tolerance <= 0`, `max_charts < 1`, or `rbf_bandwidth <= 0`.

### Methods

#### `insert(points, modality_id=0)`

Insert data points into the database.

```python
db.insert(points, modality_id=0)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `points` | `array-like`, shape `(N, D)` or `(D,)` | Ambient-space data. Accepts numpy (float32/float64), PyTorch tensors, or list-of-lists. 1-D input is reshaped to `(1, D)`. |
| `modality_id` | `int` | Modality identifier (default 0). Use different values for multi-modal data. |

**Raises**: `ValueError` if points are empty, not 1-D/2-D, or have zero dimensions.

**Example**:
```python
db.insert(np.random.randn(1000, 768), modality_id=0)
db.insert(torch.randn(500, 768, dtype=torch.float64), modality_id=1)
```

---

#### `build(method='linear')`

Build the atlas from inserted data.

```python
db.build(method="linear")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | `str` | `"linear"` | Atlas construction method. Supported: `"linear"`, `"pca"` (alias for linear). |

**Raises**: `ValueError` if method is not recognised.

---

#### `build_atlas_linear(intrinsic_dim=None)`

Build the atlas using PCA-based linear charts.

```python
db.build_atlas_linear(intrinsic_dim=16)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `intrinsic_dim` | `int` or `None` | `None` | Target intrinsic dimension. If `None`, uses the value from the constructor. |

---

#### `query_knn(query, k=10, max_distance=inf)`

K-nearest neighbours by geodesic distance.

```python
results = db.query_knn(query, k=10, max_distance=float('inf'))
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `array-like`, shape `(D,)` or `(1, D)` | — | Query point in ambient space |
| `k` | `int` | `10` | Number of nearest neighbours |
| `max_distance` | `float` | `inf` | Exclude neighbours farther than this geodesic distance |

**Returns**: `list[dict]` — Each dict contains:
- `"id"` (`int`): Unique point identifier
- `"chart_id"` (`int`): Home chart identifier
- `"timestamp"` (`float`): Insertion time
- `"distance"` (`float`): True geodesic distance d_g(p, q)
- `"euclidean_residual"` (`float`): |d_g(p,q) - ||y_p - y_q|||
- `"local_coords"` (`np.ndarray`): Local chart coordinates
- `"ambient_coords"` (`np.ndarray`): Ambient coordinates

**Raises**: `ValueError` if `k < 1`.

**Example**:
```python
results = db.query_knn(query_vector, k=5)
for r in results:
    print(f"id={r['id']} dist={r['distance']:.4f}")
```

---

#### `query_ball(center, radius)`

All points within a geodesic ball.

```python
points = db.query_ball(center, radius=0.5)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `center` | `array-like`, shape `(D,)` or `(1, D)` | — | Centre of the ball in ambient space |
| `radius` | `float` | — | Geodesic radius |

**Returns**: `list[dict]` — Each dict contains `id`, `chart_id`, `timestamp`, `local_coords`, `ambient_coords`.

**Raises**: `ValueError` if `radius < 0`.

---

#### `cross_modal_query(query, source_modality, target_modality, k=10)`

Search one modality using a query from another modality.

```python
results = db.cross_modal_query(
    query_text, source_modality=0, target_modality=1, k=10
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `array-like`, shape `(D,)` | — | Query point from source modality |
| `source_modality` | `int` | — | Modality ID of the query |
| `target_modality` | `int` | — | Modality ID to search in |
| `k` | `int` | `10` | Number of results |

**Returns**: `list[dict]` — Same format as `query_knn()`.

---

#### `geodesic_path(start, end, tolerance=1e-6)`

Compute the geodesic path between two points.

```python
path = db.geodesic_path(start_pt, end_pt, tolerance=1e-6)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start` | `array-like`, shape `(D,)` | — | Start point in ambient space |
| `end` | `array-like`, shape `(D,)` | — | End point in ambient space |
| `tolerance` | `float` | `1e-6` | Solver tolerance for BVP |

**Returns**: `dict` containing:
- `"total_length"` (`float`): Total geodesic arc length
- `"converged"` (`bool`): Whether the solver converged
- `"num_steps"` (`int`): Number of integration steps
- `"points"` (`list[dict]`): Sampled points along the path
- `"arc_lengths"` (`list[float]`): Cumulative arc lengths

---

#### `evolve(new_data)`

Extend the manifold to accommodate new data points.

```python
db.evolve(new_data)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `new_data` | `array-like`, shape `(N, D)` or `(D,)` | New data points to incorporate |

Internally inserts as a new virtual modality and rebuilds the atlas.

---

#### `stats()`

Return database statistics.

```python
stats = db.stats()
```

**Returns**: `dict` containing:
- `"num_charts"` (`int`): Number of charts in the atlas
- `"total_points"` (`int`): Total points across all modalities
- `"index_size"` (`int`): Total points indexed across all charts
- `"build_time_ms"` (`float`): Last atlas build time (ms)
- `"avg_geodesic_time_ms"` (`float`): Average geodesic solve time (ms)

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `core` | `_core.ManifoldDB` | Direct access to the underlying C++ engine |
| `atlas` | `_core.Atlas` | Access the underlying Atlas |
| `metric_store` | `_core.MetricStore` | Access the underlying MetricStore |
| `solver` | `_core.GeodesicSolver` | Access the underlying GeodesicSolver |
| `num_charts` | `int` | Number of charts in the atlas |
| `total_points` | `int` | Total number of points in the database |
| `storage_path` | `str` | Filesystem storage path |

---

## ManifoldPoint (Python)

A point on the Riemannian manifold with dual representation.

```python
pt = manifolddb.ManifoldPoint(
    local_coords=np.array([0.1, 0.2]),
    ambient_coords=np.array([1.0, 2.0, 3.0]),
    chart_id=0,
    global_id=42,
    timestamp=0.0,
)
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `chart_id` | `int` | Home chart identifier |
| `global_id` | `int` | Unique identifier across the database |
| `local_coords` | `np.ndarray` | Local chart coordinates x ∈ R^d (read/write) |
| `ambient_coords` | `np.ndarray` | Ambient coordinates y ∈ R^D (read/write) |
| `timestamp` | `float` | Insertion time (read/write) |

### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `local_norm()` | `()` | `float` | Euclidean norm of local coordinate vector |
| `ambient_norm()` | `()` | `float` | Euclidean norm of ambient coordinate vector |
| `to_dict()` | `()` | `dict` | Convert to plain dictionary |

---

## GeodesicPath (Python)

Discrete approximation of a geodesic curve.

### Constructor

```python
path = manifolddb.GeodesicPath(
    points=[...],           # list of dicts
    arc_lengths=[...],      # list of floats
    total_length=1.23,
    converged=True,
    num_steps=42,
)
```

### Class Method

```python
path = manifolddb.GeodesicPath.from_cpp(cpp_geodesic_path)
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `total_length` | `float` | Total geodesic arc length |
| `converged` | `bool` | Whether the solver converged |
| `num_steps` | `int` | Number of integration steps |
| `points` | `list[dict]` | Sampled points along the geodesic |
| `arc_lengths` | `list[float]` | Cumulative arc lengths |
| `is_empty` | `bool` | True if path has no points |
| `size` | `int` | Number of sample points |

### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `points_array()` | `()` | `np.ndarray`, shape `(N, D)` | All ambient coordinates as numpy array |
| `arc_lengths_array()` | `()` | `np.ndarray`, shape `(N,)` | Arc lengths as numpy array |
| `to_dict()` | `()` | `dict` | Convert to plain dictionary |

---

## NeighborResult (Python)

Result entry from a geodesic k-nearest-neighbour query.

### Constructor

```python
nr = manifolddb.NeighborResult(
    point={...},
    distance=0.1234,
    euclidean_residual=0.005,
)
```

### Class Method

```python
nr = manifolddb.NeighborResult.from_cpp(cpp_neighbor_result)
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `point` | `dict` | The neighbour point data |
| `id` | `int` | Unique identifier of the neighbour |
| `chart_id` | `int` | Chart identifier |
| `distance` | `float` | True geodesic distance d_g(p, q) |
| `euclidean_residual` | `float` | |d_g(p,q) - ||y_p - y_q|| |
| `local_coords` | `np.ndarray` | Local coordinates of the neighbour |
| `ambient_coords` | `np.ndarray` | Ambient coordinates of the neighbour |

### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `to_dict()` | `()` | `dict` | Convert to plain dictionary |

---

## Chart (Python / C++)

Abstract base class for a coordinate chart (U, φ) on the manifold.

### C++ Methods (exposed to Python via PyBind11)

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `embed` | `(local_coords: np.ndarray)` | `np.ndarray`, shape `(D,)` | Embed local coords into ambient space: φ(x) → R^D |
| `project` | `(ambient_coords: np.ndarray)` | `np.ndarray`, shape `(d,)` | Project ambient coords to local: φ⁻¹(y) → R^d |
| `jacobian` | `(local_coords: np.ndarray)` | `np.ndarray`, shape `(D, d)` | Pushforward J_ij = ∂y^i/∂x^j |
| `compute_local_metric` | `(local_coords: np.ndarray)` | `np.ndarray`, shape `(d, d)` | Induced metric g_ij = J^T J |
| `compute_inverse_metric` | `(local_coords: np.ndarray)` | `np.ndarray`, shape `(d, d)` | Inverse metric g^{ij} |
| `christoffel_first_kind` | `(local_coords, h=1e-5)` | `np.ndarray`, shape `(d, d, d)` | Γ_{ijk} via central differences |
| `christoffel_second_kind` | `(local_coords, h=1e-5)` | `np.ndarray`, shape `(d, d, d)` | Γ^k_{ij} |
| `sectional_curvature` | `(local_coords, u, v, h=1e-5)` | `float` | Sectional curvature K(u, v) |
| `exponential_map` | `(base, tangent_vec, step_size=1e-3, max_steps=1000)` | `ManifoldPoint` | exp_p(v) |
| `log_map` | `(base, target, tolerance=1e-8, max_iterations=100)` | `np.ndarray` | log_p(q) |
| `contains` | `(local_coords)` | `bool` | Point within chart domain? |
| `id` | `()` | `int` | Chart identifier |
| `intrinsic_dim` | `()` | `int` | Intrinsic dimension d |
| `ambient_dim` | `()` | `int` | Ambient dimension D |
| `type` | `()` | `ChartType` | Chart type enum |

### Python Subclassing

```python
from manifolddb import Chart, ChartType
import numpy as np

class SphereChart(Chart):
    """A chart for the unit sphere S² in R³."""

    def embed(self, local_coords):
        theta, phi = local_coords[0], local_coords[1]
        return np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ])

    def project(self, ambient_coords):
        x, y, z = ambient_coords
        theta = np.arccos(np.clip(z, -1, 1))
        phi = np.arctan2(y, x)
        return np.array([theta, phi])

    def jacobian(self, local_coords):
        theta, phi = local_coords[0], local_coords[1]
        st, ct = np.sin(theta), np.cos(theta)
        sp, cp = np.sin(phi), np.cos(phi)
        return np.array([
            [ct * cp, -st * sp],
            [ct * sp,  st * cp],
            [-st,     0.0     ],
        ])

    def type(self):
        return ChartType.CUSTOM
```

---

## LinearChart (Python / C++)

Affine (PCA) chart: φ(x) = origin + B·x.

### Constructor

```python
chart = manifolddb.LinearChart(
    id=0,                    # Chart identifier
    basis=basis_matrix,      # B ∈ R^{D×d} (orthonormal columns)
    origin=origin_vector,    # origin ∈ R^D
)
```

### Additional Methods (beyond Chart)

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `basis` | `()` | `np.ndarray`, shape `(D, d)` | Basis matrix B (D × d) |
| `origin` | `()` | `np.ndarray`, shape `(D,)` | Origin vector (D) |
| `projection_residual` | `(ambient_coords)` | `float` | Distance from point to affine plane |

### Example

```python
import numpy as np
from manifolddb import LinearChart

# Construct a linear chart from PCA
B = np.random.randn(768, 16)  # 768-dim ambient, 16-dim intrinsic
Q, _ = np.linalg.qr(B)          # Orthonormalise
origin = np.zeros(768)

chart = LinearChart(id=0, basis=Q, origin=origin)
local = chart.project(some_768d_vector)
reembedded = chart.embed(local)
metric = chart.compute_local_metric(local)  # ≈ Identity (orthonormal basis)
```

---

## ParametricChart (Python / C++)

Chart with user-supplied Python callbacks.

### Constructor

```python
chart = manifolddb.ParametricChart(
    id=0,
    intrinsic_dim=2,
    ambient_dim=3,
    embed_fn=embed_func,
    project_fn=project_func,
    jacobian_fn=jacobian_func,
)
```

### Callback Types

| Callback | Signature | Description |
|----------|-----------|-------------|
| `EmbedFunc` | `(local_coords: np.ndarray) → np.ndarray` | φ(x): local → ambient |
| `ProjectFunc` | `(ambient_coords: np.ndarray) → np.ndarray` | φ⁻¹(y): ambient → local |
| `JacobianFunc` | `(local_coords: np.ndarray) → np.ndarray` | J(x): D × d Jacobian matrix |

---

## MetricTensor (Python / C++)

Riemannian metric tensor field g_ij(x) on a single chart.

### Constructor

```python
metric = manifolddb.MetricTensor(
    chart_id=0,   # Chart identifier
    dim=16,        # Intrinsic dimension d
)
```

Initially set to identity metric g_ij = δ_ij.

### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `evaluate` | `(local_coords: np.ndarray)` | `np.ndarray`, shape `(d, d)` | Evaluate g_ij(x) |
| `inverse` | `(local_coords: np.ndarray)` | `np.ndarray`, shape `(d, d)` | Evaluate g^{ij}(x) |
| `christoffel_symbols` | `(local_coords, h=1e-5)` | `np.ndarray`, shape `(d, d, d)` | Γ^k_{ij} via finite differences |
| `sectional_curvature` | `(u, v)` | `float` | Sectional curvature K(u, v) |
| `scalar_curvature` | `(local_coords, h=1e-5)` | `float` | Scalar curvature S = g^{ij} R_{ij} |
| `update` | `(local_coords, local_metric, weight=1.0)` | `None` | Add RBF anchor point |
| `set_constant` | `(metric: np.ndarray)` | `None` | Set constant position-independent metric |
| `set_identity` | `()` | `None` | Reset to identity matrix |
| `clear` | `()` | `None` | Clear all anchors, reset to identity |
| `is_constant` | `()` | `bool` | True if metric has no anchors |
| `chart_id` | `()` | `int` | Chart this metric belongs to |
| `dim` | `()` | `int` | Intrinsic dimension d |
| `num_anchors` | `()` | `int` | Number of RBF interpolation anchors |
| `serialize` | `()` | `bytes` | Serialize to binary |
| `deserialize` | `(data: bytes)` | `None` | Deserialize from binary |

### Example: RBF Interpolated Metric

```python
from manifolddb import MetricTensor
import numpy as np

metric = MetricTensor(chart_id=0, dim=3)

# Add anchor points with local metric values
for i in range(10):
    anchor_pos = np.random.randn(3)
    local_g = np.eye(3) + 0.1 * np.random.randn(3, 3)
    local_g = 0.5 * (local_g + local_g.T)  # Ensure SPD
    metric.update(anchor_pos, local_g, weight=1.0)

# Evaluate at an arbitrary point
g = metric.evaluate(np.array([0.5, -0.3, 1.0]))
Gamma = metric.christoffel_symbols(np.array([0.5, -0.3, 1.0]))
K = metric.sectional_curvature(np.array([1, 0, 0]), np.array([0, 1, 0]))
```

---

## MetricStore (Python / C++)

Thread-safe persistent storage and caching layer for MetricTensors.

### Constructor

```python
store = manifolddb.MetricStore(db_path="./metrics")
```

### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `get_metric` | `(chart_id: int)` | `MetricTensor` or `None` | Get cached or loaded metric |
| `create_metric` | `(chart_id: int, dim: int)` | `MetricTensor` | Create new identity metric |
| `commit` | `(chart_id: int, metric: MetricTensor)` | `None` | Persist metric to disk and update cache |
| `batch_evaluate` | `(chart_id: int, points: list[np.ndarray])` | `list[np.ndarray]` | Evaluate metric at multiple points |
| `num_charts` | `()` | `int` | Number of cached charts |
| `has_chart` | `(chart_id: int)` | `bool` | Check if metric is cached |
| `flush` | `()` | `None` | Flush all metrics to disk |

---

## GeodesicSolver (Python / C++)

Geodesic equation solver supporting IVP, BVP, parallel transport, and distance computation.

### Constructor

```python
solver = manifolddb.GeodesicSolver(
    metric_store=store,
    config=solver_config,
)
```

### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `solve_ivp` | `(start: ManifoldPoint, initial_velocity: np.ndarray, t_max=1.0, method=SolverType.RK45)` | `GeodesicPath` | Solve geodesic IVP |
| `solve_bvp` | `(start: ManifoldPoint, end: ManifoldPoint, method=SolverType.SHOOTING)` | `GeodesicPath` | Solve geodesic BVP |
| `parallel_transport` | `(path: GeodesicPath, vector_at_start: np.ndarray)` | `list[np.ndarray]` | Levi-Civita parallel transport |
| `geodesic_distance` | `(p: ManifoldPoint, q: ManifoldPoint)` | `float` | Geodesic distance d_g(p, q) |
| `batch_geodesic_distance` | `(chart_id: int, query: np.ndarray, candidates: np.ndarray)` | `list[float]` | Batch distances |
| `config` | `()` | `SolverConfig` | Get mutable solver config reference |
| `set_config` | `(cfg: SolverConfig)` | `None` | Update solver configuration |

---

## TangentSpaceIndex (Python / C++)

R-tree spatial index over the tangent bundle for fast local search.

### Constructor

```python
index = manifolddb.TangentSpaceIndex(
    chart_id=0,
    intrinsic_dim=16,
    max_leaf_size=16,
)
```

### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `insert` | `(point: ManifoldPoint)` | `None` | Insert a single point |
| `build` | `(points: list[ManifoldPoint])` | `None` | Bulk build via STR packing |
| `knn` | `(query_local: np.ndarray, k: int, max_radius=float('inf'))` | `list[NeighborResult]` | K-NN in local coordinates |
| `knn_tangent` | `(query_local: np.ndarray, k: int)` | `list[NeighborResult]` | Alias for `knn()` |
| `range_search` | `(query_local: np.ndarray, radius: float)` | `list[ManifoldPoint]` | Ball query in local coordinates |
| `clear` | `()` | `None` | Remove all points and reset |
| `size` | `()` | `int` | Number of indexed points |
| `is_built` | `()` | `bool` | Whether index has been built |
| `chart_id` | `()` | `int` | Chart identifier |
| `save` | `(path: str)` | `None` | Serialize to binary file |
| `load` | `(path: str)` | `None` | Load and rebuild from binary file |

---

## Atlas (Python / C++)

Manages a collection of charts covering the manifold M with transition maps.

### Constructor

```python
atlas = manifolddb.Atlas()
```

### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `add_chart` | `(chart: Chart)` | `None` | Register a chart in the atlas |
| `add_transition` | `(transition: TransitionMap)` | `None` | Register a transition map |
| `locate_chart` | `(ambient_coords: np.ndarray)` | `Chart` or `None` | Find best chart for ambient point |
| `transport` | `(point: ManifoldPoint, target_chart_id: int)` | `ManifoldPoint` | Multi-hop coordinate transport |
| `charts_overlap` | `(id_a: int, id_b: int)` | `bool` | Check chart overlap |
| `get_chart` | `(chart_id: int)` | `Chart` or `None` | Get chart by ID |
| `get_transition` | `(from_id: int, to_id: int)` | `TransitionMap` or `None` | Get transition map |
| `find_path` | `(from_id: int, to_id: int)` | `list[int]` | BFS shortest multi-hop path |
| `num_charts` | `()` | `int` | Number of charts |
| `charts` | `()` | `list[Chart]` | All chart references |
| `discover_charts_linear` | `(data: np.ndarray, target_intrinsic_dim: int, num_charts_target=0, overlap_threshold=0.1)` | `None` | Auto-discover PCA charts |

---

## TransitionMap / LinearTransitionMap (Python / C++)

### LinearTransitionMap Constructor

```python
tmap = manifolddb.LinearTransitionMap(
    from_chart=0,          # Source chart ID
    to_chart=1,            # Target chart ID
    rotation=R_matrix,    # R ∈ R^{d×d}
    translation=t_vector,  # t ∈ R^d
)
```

### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `forward` | `(coords_a: np.ndarray)` | `np.ndarray` | x_β = R · x_α + t |
| `inverse` | `(coords_b: np.ndarray)` | `np.ndarray` | x_α = R⁻¹ · (x_β - t) |
| `jacobian` | `(coords_a: np.ndarray)` | `np.ndarray` | J = R (constant) |
| `in_overlap` | `(coords_a: np.ndarray)` | `bool` | Point in overlap region? |

---

## Config / Stats / SolverConfig (Python / C++)

### Config (ManifoldDB::Config)

```python
from manifolddb import Config
cfg = Config()
cfg.storage_path = "./my_db"
cfg.default_intrinsic_dim = 16
cfg.enable_cuda = False
cfg.geodesic_tolerance = 1e-6
cfg.solver_config = SolverConfig()
cfg.index_max_leaf_size = 16
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `storage_path` | `str` | `"./manifolddb_data"` | Persistent storage directory |
| `default_intrinsic_dim` | `uint32` | `10` | Default intrinsic dimension d |
| `enable_cuda` | `bool` | `false` | GPU-accelerated geodesics |
| `geodesic_tolerance` | `double` | `1e-6` | Solver convergence tolerance |
| `solver_config` | `SolverConfig` | — | Solver tuning parameters |
| `index_max_leaf_size` | `size_t` | `16` | R-tree leaf capacity |

### Stats (ManifoldDB::Stats)

```python
stats = db.stats()
# or from C++: db.core.stats()
```

| Field | Type | Description |
|-------|------|-------------|
| `num_charts` | `size_t` | Number of charts in the atlas |
| `total_points` | `size_t` | Total points across all modalities |
| `avg_geodesic_time_ms` | `double` | Average geodesic solve time (ms) |
| `index_size` | `size_t` | Total indexed points |
| `build_time_ms` | `double` | Last atlas build time (ms) |

### SolverConfig

```python
from manifolddb import SolverConfig
cfg = SolverConfig()
cfg.initial_step = 1e-3
cfg.min_step = 1e-8
cfg.max_step = 0.1
cfg.tolerance = 1e-8
cfg.max_iterations = 10000
cfg.max_bvp_iterations = 50
cfg.bvp_tolerance = 1e-6
cfg.adaptive_step = True
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `initial_step` | `double` | `1e-3` | Default integration step size |
| `min_step` | `double` | `1e-8` | Minimum step (prevent division by zero) |
| `max_step` | `double` | `0.1` | Maximum step |
| `tolerance` | `double` | `1e-8` | Convergence tolerance |
| `max_iterations` | `int` | `10000` | Max IVP steps |
| `max_bvp_iterations` | `int` | `50` | Max Newton shooting iterations |
| `bvp_tolerance` | `double` | `1e-6` | BVP residual tolerance |
| `adaptive_step` | `bool` | `true` | Enable adaptive RK45 stepping |

---

## torch_compat Utilities

Module: `manifolddb.torch_compat`

```python
from manifolddb.torch_compat import (
    ensure_float64,
    torch_to_eigen,
    eigen_to_torch,
    eigen_to_numpy,
    dlpack_export,
    batch_geodesic_distances,
)
```

### `ensure_float64(tensor)`

Ensure a tensor/array is float64 and on CPU.

```python
import torch
from manifolddb.torch_compat import ensure_float64

t = torch.randn(4, 3, dtype=torch.float32)
t64 = ensure_float64(t)  # float64 CPU tensor
```

| Parameter | Type | Returns | Description |
|-----------|------|---------|-------------|
| `tensor` | `torch.Tensor` or `np.ndarray` | Same type | Float64, CPU |

### `torch_to_eigen(tensor)`

Convert a torch.Tensor to an Eigen-compatible numpy array.

```python
import torch
from manifolddb.torch_compat import torch_to_eigen

t = torch.randn(100, 768, dtype=torch.float64)
arr = torch_to_eigen(t)  # np.ndarray, contiguous float64
```

| Parameter | Type | Returns | Description |
|-----------|------|---------|-------------|
| `tensor` | `torch.Tensor` | `np.ndarray` | Contiguous float64 numpy array |

### `eigen_to_torch(vectors, requires_grad=False)`

Convert Eigen output (numpy) back to torch.

```python
from manifolddb.torch_compat import eigen_to_torch

t = eigen_to_torch(some_numpy_array)
```

| Parameter | Type | Default | Returns | Description |
|-----------|------|---------|-------------|
| `vectors` | `np.ndarray` | — | 1-D or 2-D float64 numpy array |
| `requires_grad` | `bool` | `False` | Track gradients? |

### `batch_geodesic_distances(query, candidates, metric_store, chart_id=None)`

Compute batch geodesic distances.

```python
from manifolddb.torch_compat import batch_geodesic_distances

query = np.array([0.1, 0.2, 0.3])
candidates = np.random.randn(50, 3).astype(np.float64)
dists = batch_geodesic_distances(query, candidates, db.metric_store)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `array-like` | — | Query point(s), shape `(Q, d)` or `(d,)` |
| `candidates` | `array-like` | — | Candidate points, shape `(N, d)` |
| `metric_store` | `MetricStore` or `ManifoldDB` | — | Metric provider |
| `chart_id` | `int` or `None` | `None` | Chart ID for metric eval |

**Returns**: `np.ndarray`, shape `(Q, N)` or `(N,)` — Pairwise geodesic distances.

---

## IO Utilities

Module: `manifolddb.io`

```python
from manifolddb.io import (
    save_manifold,
    load_manifold,
    export_charts_to_json,
    import_charts_from_json,
    export_metrics_to_hdf5,
    import_metrics_from_hdf5,
)
```

### `save_manifold(db, path)`

Persist the full database to a directory.

### `load_manifold(path)`

Load a previously saved database. Returns a `ManifoldDB` wrapper.

### `export_charts_to_json(db, path)`

Export atlas chart info (basis, origin, transitions) to JSON.

### `import_charts_from_json(path)`

Import chart information from JSON. Returns `{"charts": [...], "transitions": [...]}`.

### `export_metrics_to_hdf5(db, path)`

Export metric tensors to HDF5. Requires `h5py`.

### `import_metrics_from_hdf5(path, db=None)`

Import metric tensors from HDF5. Optionally populates a database.

---

## Enums

```python
from manifolddb import SolverType, DistanceType, ChartType

# SolverType: ODE integration methods
SolverType.RK4          # Classical 4th-order Runge-Kutta
SolverType.RK45         # Adaptive Dormand-Prince (RK4(5))
SolverType.RK4_CUDA     # GPU-accelerated RK4 (requires CUDA build)
SolverType.SYMPLECTIC   # Symplectic (Störmer-Verlet) integrator
SolverType.SHOOTING     # Newton shooting for BVP

# DistanceType: Distance metrics
DistanceType.GEODESIC    # True Riemannian distance d_g(p,q)
DistanceType.EUCLIDEAN   # Ambient Euclidean ||y_p - y_q||
DistanceType.LOG_CHORDAL # Chordal distance in log-map coordinates

# ChartType: Chart types
ChartType.LINEAR      # Affine (PCA) chart: φ(x) = origin + B·x
ChartType.NEURAL      # Neural network chart (ONNX)
ChartType.PARAMETRIC  # User-supplied callback chart
ChartType.CUSTOM      # User-defined chart subclass
```

---

## Exceptions

```python
from manifolddb import (
    ChartNotFoundError,
    GeodesicSolverError,
    IndexBuildError,
    SerializationError,
    DimensionMismatchError,
)
```

| Exception | Base | Thrown When |
|-----------|------|-------------|
| `ChartNotFoundError` | `DBException` | Requested chart ID not found in atlas |
| `GeodesicSolverError` | `DBException` | Geodesic solver fails to converge |
| `IndexBuildError` | `DBException` | Index build step fails |
| `SerializationError` | `DBException` | Serialisation/deserialisation error |
| `DimensionMismatchError` | `DBException` | Matrix/vector dimensions inconsistent |
