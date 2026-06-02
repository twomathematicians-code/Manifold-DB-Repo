# ManifoldDB — Architecture Document

> A detailed technical reference for the internal design, data flow, and
> implementation decisions of the ManifoldDB geometric inference engine.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [C++ Class Hierarchy](#c-class-hierarchy)
3. [Data Flow](#data-flow)
4. [Storage Format](#storage-format)
5. [Query Execution](#query-execution)
6. [Cross-Modal Query](#cross-modal-query)
7. [Parallel Transport for Schema Evolution](#parallel-transport-for-schema-evolution)
8. [GPU Acceleration Strategy](#gpu-acceleration-strategy)
9. [Python/C++ Interop](#python-c-interop)
10. [Future Directions](#future-directions)

---

## System Overview

ManifoldDB is organized as a **three-tier system**: a Python API layer for user-facing operations, a C++ geodesic computation engine for performance-critical geometry, and a persistent storage layer for metric tensors and spatial indexes.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           Python API Layer                                  │
│                                                                             │
│  manifolddb/__init__.py   → ManifoldDB (high-level wrapper)                 │
│  manifolddb/torch_compat.py → torch_to_eigen, eigen_to_torch, batch ops     │
│  manifolddb/io.py          → save_manifold, load_manifold, JSON/HDF5        │
└───────────────────────────────────┬────────────────────────────────────────┘
                                    │ PyBind11
┌───────────────────────────────────┼────────────────────────────────────────┐
│                           C++ Geodesic Engine                               │
│                                                                             │
│  ManifoldDB  ←→  Atlas  ←→  Chart (LinearChart, NeuralChart, Parametric) │
│       ↕              ↕           ↕                                           │
│  TangentSpaceIndex   MetricStore   GeodesicSolver                          │
│  (R-tree)           (g_ij cache)  (RK4/RK45/Shooting)                      │
└───────────────────────────────────┬────────────────────────────────────────┘
                                    │ File I/O
┌───────────────────────────────────┼────────────────────────────────────────┐
│                           Storage Layer                                     │
│                                                                             │
│  {storage_path}/metrics/metric_{id}.bin   (per-chart metric tensors)       │
│  {storage_path}/indexes/index_{id}.bin      (per-chart R-tree data)        │
│  {storage_path}/meta.json                  (atlas metadata)                  │
│  [Optional] HDF5 export, JSON chart dump                                       │
└────────────────────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Geometric Fidelity First**: Every query is grounded in Riemannian geometry. Approximations (Euclidean proxies for candidate generation) are always followed by exact geodesic re-ranking.

2. **Zero-Copy Bridges**: PyTorch tensors and numpy arrays are passed directly to C++ Eigen types via PyBind11's `py::array_t<double>` with `Eigen::Map`, avoiding memory copies where possible.

3. **Lazy Construction**: Tangent-space indexes are built on first access (double-checked locking pattern). Atlas and metric tensors are created only when `build()` is called.

4. **Thread-Safe Persistence**: MetricStore uses `std::shared_mutex` for reader-writer locking, allowing concurrent reads during query execution while serialising writes.

5. **Modular Extensibility**: New chart types (NeuralChart, custom subclasses), storage backends (TileDB), and solver methods can be added without modifying the top-level API.

---

## C++ Class Hierarchy

```
namespace manifold {

// ─── Fundamental Types ───────────────────────────────────────────────────
Scalar        = double
Vector        = Eigen::Matrix<double, Dynamic, 1>
Matrix        = Eigen::Matrix<double, Dynamic, Dynamic>
Tensor3D      = Eigen::Tensor<double, 3>
SparseMatrix  = Eigen::SparseMatrix<double>

// ─── Data Structures ────────────────────────────────────────────────────
ManifoldPoint {
    uint32_t chart_id;
    Vector   local_coords;      // x ∈ R^d
    Vector   ambient_coords;    // y ∈ R^D
    uint64_t global_id;
    double   timestamp;
}

GeodesicPath {
    vector<ManifoldPoint> points;
    vector<Scalar>        arc_lengths;
    Scalar   total_length;
    bool     converged;
    int      num_steps;
}

NeighborResult {
    ManifoldPoint point;
    Scalar geodesic_distance;
    Scalar euclidean_residual;
}

// ─── Charts ────────────────────────────────────────────────────────────
Chart {                          // Abstract base
    embed(local_coords) → Vector
    project(ambient_coords) → Vector
    jacobian(local_coords) → Matrix
    compute_local_metric(local_coords) → Matrix
    compute_inverse_metric(local_coords) → Matrix
    christoffel_first_kind(local_coords, h) → Tensor3D
    christoffel_second_kind(local_coords, h) → Tensor3D
    sectional_curvature(local_coords, u, v, h) → Scalar
    exponential_map(base, tangent_vec, step_size, max_steps) → ManifoldPoint
    log_map(base, target, tolerance, max_iterations) → Vector
    contains(local_coords) → bool
    type() → ChartType
    id(), intrinsic_dim(), ambient_dim()
}
  ├── LinearChart : Chart {    // φ(x) = origin + B·x
  │     basis() → Matrix        // B ∈ R^{D×d}
  │     origin() → Vector      // origin ∈ R^D
  │     projection_residual(ambient_coords) → Scalar
  │   }
  ├── NeuralChart : Chart {     // φ_θ(x) = ONNX forward pass
  │     model_path() → string
  │   }
  └── ParametricChart : Chart { // User-supplied callbacks
        EmbedFunc, ProjectFunc, JacobianFunc
      }

// ─── Atlas ───────────────────────────────────────────────────────────────
TransitionMap {               // Abstract coordinate transformation
    from_chart, to_chart
    forward(coords_a) → Vector
    inverse(coords_b) → Vector
    jacobian(coords_a) → Matrix
    in_overlap(coords_a) → bool
}
  └── LinearTransitionMap : TransitionMap {
        rotation() → Matrix       // R ∈ R^{d×d}
        translation() → Vector    // t ∈ R^d
      }

Atlas {
    add_chart(chart)
    add_transition(transition)
    locate_chart(ambient_coords) → Chart*
    transport(point, target_chart_id) → ManifoldPoint
    charts_overlap(id_a, id_b) → bool
    get_chart(chart_id) → shared_ptr<Chart>
    get_transition(from_id, to_id) → TransitionMap*
    find_path(from_id, to_id) → vector<uint32_t>
    discover_charts_linear(data, dim, num_charts, threshold)
    num_charts() → size_t
    charts() → vector<shared_ptr<Chart>>
}

// ─── Metric ──────────────────────────────────────────────────────────────
MetricTensor {
    evaluate(local_coords) → Matrix          // g_ij(x)
    inverse(local_coords) → Matrix           // g^{ij}(x)
    christoffel_symbols(local_coords, h) → Tensor3D  // Γ^k_{ij}
    sectional_curvature(u, v) → Scalar       // K(u,v)
    scalar_curvature(local_coords, h) → Scalar // S = g^{ij}R_{ij}
    update(local_coords, local_metric, weight) // Add RBF anchor
    set_constant(metric)
    set_identity()
    clear()
    serialize() → vector<uint8_t>
    deserialize(data)
    chart_id(), dim(), num_anchors(), is_constant()
}

MetricStore {                   // Thread-safe cache + persistence
    get_metric(chart_id) → shared_ptr<MetricTensor>
    create_metric(chart_id, dim) → shared_ptr<MetricTensor>
    commit(chart_id, metric)
    batch_evaluate(chart_id, points) → vector<Matrix>
    num_charts() → size_t
    has_chart(chart_id) → bool
    flush()
}

// ─── Solver ───────────────────────────────────────────────────────────────
SolverConfig {
    initial_step, min_step, max_step, tolerance
    max_iterations, max_bvp_iterations, bvp_tolerance
    adaptive_step
}

GeodesicSolver {
    solve_ivp(start, initial_velocity, t_max, method) → GeodesicPath
    solve_bvp(start, end, method) → GeodesicPath
    parallel_transport(path, vector_at_start) → vector<Vector>
    geodesic_distance(p, q) → Scalar
    batch_geodesic_distance(chart_id, query, candidates) → vector<Scalar>
    config() → SolverConfig&
}

// ─── Index ──────────────────────────────────────────────────────────────
RTreeNode {                      // AABB node for R-tree
    min_corner, max_corner       // Axis-aligned bounding box
    points                        // Leaf data (is_leaf = true)
    children                      // Internal children (is_leaf = false)
    intersects(query_min, query_max) → bool
    min_distance(query) → Scalar   // For branch-and-bound pruning
    volume() → Scalar
}

TangentSpaceIndex {
    insert(point)
    build(points)                 // STR packing bulk build
    knn(query_local, k, max_radius) → vector<NeighborResult>
    knn_tangent(query_local, k) → vector<NeighborResult>
    range_search(query_local, radius) → vector<ManifoldPoint>
    clear()
    size(), chart_id(), is_built()
    save(path) / load(path)
}

// ─── Top-Level API ──────────────────────────────────────────────────────
ManifoldDB {
    Config {
        storage_path, default_intrinsic_dim, enable_cuda
        geodesic_tolerance, solver_config, index_max_leaf_size
    }
    Stats {
        num_charts, total_points, avg_geodesic_time_ms
        index_size, build_time_ms
    }

    insert(ambient_points, modality_id)
    insert(ambient_matrix, modality_id)
    build_atlas(target_intrinsic_dim)
    build_atlas_linear(intrinsic_dim)

    query_geodesic_knn(query_ambient, k, max_distance) → vector<NeighborResult>
    query_geodesic_ball(center_ambient, radius) → vector<ManifoldPoint>
    query_geodesic_path(start_ambient, end_ambient) → GeodesicPath
    query_cross_modal(query_ambient, source_mod, target_mod, k) → vector<NeighborResult>

    evolve_schema(new_ambient_points)

    stats() → Stats
    atlas(), metric_store(), solver()
}
}
```

---

## Data Flow

### Ingestion → Atlas Building → Metric Learning → Indexing → Querying

The complete lifecycle of data through ManifoldDB:

```
┌──────────────┐     ┌────────────────┐     ┌───────────────┐
│ 1. INGESTION │────▶│ 2. ATLAS BUILD │────▶│ 3. METRIC INIT │
│              │     │                │     │               │
│ insert(pts,  │     │ PCA/SVD on     │     │ Identity g_ij  │
│ modality_id) │     │ ambient data   │     │ per chart      │
│              │     │                │     │               │
│ Assign       │     │ k-means for    │     │ Store to disk  │
│ global IDs   │     │ multi-chart    │     │ (MetricStore)  │
│              │     │ decomposition  │     │               │
│ Assign chart  │     │                │     │ Create metric  │
│ if atlas exists│    │ LinearCharts   │     │ for each chart │
│              │     │ Auto-discover  │     │               │
│ Rebuild       │     │ transitions   │     └───────┬───────┘
│ ambient       │     │ (LinearTrans-  │             │
│ matrix        │     │  itionMap)     │             │
└──────────────┘     └───────┬────────┘             │
                             │                      │
                    ┌────────▼────────┐             │
                    │ 4. INDEX BUILD  │◀────────────┘
                    │                │
                    │ Per-chart      │
                    │ TangentSpace   │
                    │ Index (R-tree) │
                    │                │
                    │ STR packing    │
                    │ from assigned  │
                    │ local coords   │
                    └───────┬────────┘
                            │
                    ┌────────▼────────┐
                    │ 5. QUERY       │
                    │                │
                    │ Locate chart   │
                    │ → Project to   │
                    │   local coords │
                    │ → R-tree k-NN  │
                    │   (candidates) │
                    │ → Geodesic     │
                    │   re-ranking   │
                    │ → Return top-k │
                    └────────────────┘
```

### Step-by-Step Detail

#### 1. Data Ingestion (`insert()`)

- Input: A matrix of N ambient-space vectors in R^D (as numpy array, torch tensor, or list of vectors)
- Each point receives a monotonically increasing `global_id`
- If the atlas already exists, each point is tentatively assigned to the best chart (minimum projection residual)
- Points are stored in a per-modality `ModalityData` structure with both the raw ambient matrix and individual `ManifoldPoint` records
- The ambient matrix is rebuilt in column-major format (D × N) for SVD-based PCA

#### 2. Atlas Construction (`build_atlas_linear()`)

- **PCA via SVD**: The ambient data matrix is centered and decomposed via JacobiSVD. The top-d right singular vectors form the orthonormal basis B. The mean becomes the chart origin.
- **Residual analysis**: If the residual variance (fraction of unexplained variance) exceeds the `overlap_threshold`, the data is split into multiple charts via k-means clustering in PCA space.
- **k-means++ initialisation**: Cluster centroids are initialised using the k-means++ algorithm for robustness.
- **Per-cluster PCA**: Each cluster gets its own LinearChart with independent PCA basis and origin.
- **Transition map discovery**: For nearby LinearCharts, affine transition maps are computed analytically:
  ```
  R = B_β^T · B_α        (rotation in local coordinates)
  t = B_β^T · (origin_α - origin_β)  (translation)
  ```
  Forward + reverse transitions are registered for bidirectional transport.

#### 3. Metric Initialisation

- Each chart receives an **identity metric tensor** g_ij = δ_ij (flat space)
- Metrics are created in the MetricStore and flushed to `{storage_path}/metrics/metric_{id}.bin`
- The identity metric yields zero Christoffel symbols, making initial geodesics straight lines in local coordinates (Euclidean baseline)

#### 4. Index Building

- For each chart, a `TangentSpaceIndex` is constructed and bulk-loaded via STR packing
- STR packing sorts points along cyclic axes (axis = depth % d) for balanced tree structure
- Each leaf node holds at most `max_leaf_size` (default 16) points
- The R-tree is built over the **local coordinates** (x ∈ R^d), providing fast Euclidean proxy search

#### 5. Query Execution

See the dedicated [Query Execution](#query-execution) section below.

---

## Storage Format

### Directory Layout

```
{storage_path}/
├── metrics/
│   ├── metric_0.bin         # Chart 0's metric tensor
│   ├── metric_1.bin         # Chart 1's metric tensor
│   └── ...
├── indexes/
│   ├── index_0.bin          # Chart 0's R-tree index
│   ├── index_1.bin          # Chart 1's R-tree index
│   └── ...
└── meta.json                # Atlas metadata
```

### Metric Tensor Binary Format

Each `metric_{id}.bin` file contains:

```
┌────────────────┬───────────┬───────────┬──────────────────────────┐
│ chart_id: u32  │ dim: u32  │ flags: u32 │ [constant_metric data]  │
│                │           │  bit 0:    │  OR                      │
│                │           │  is_const  │ [anchor_count: u32]      │
│                │           │           │  [anchor_0 data]         │
│                │           │           │  [anchor_1 data]         │
│                │           │           │  ...                     │
└────────────────┴───────────┴───────────┴──────────────────────────┘
```

#### Constant Mode (flags bit 0 = 1)

The metric is a single d×d SPD matrix stored as d² `double` values in row-major order.

#### Interpolated Mode (flags bit 0 = 0)

Each anchor point stores:

```
┌───────────────────┬──────────────────────┬───────────┬──────────────┬───────────────┐
│ coord_dim: u32    │ coords: d doubles     │ metric:   │ weight:      │ sigma:        │
│                   │                       │ d² doubles │ double       │ double        │
└───────────────────┴──────────────────────┴───────────┴──────────────┴───────────────┘
```

- `coords`: Anchor location x_k ∈ R^d
- `metric`: Metric value g_k ∈ R^{d×d} at this anchor
- `weight`: Scalar weight α_k for the RBF kernel
- `sigma`: Gaussian bandwidth σ_k (estimated from k-nearest existing anchors)

### R-tree Index Binary Format

Each `index_{id}.bin` file contains:

```
┌───────────┬───────────┬───────────┬────────────────────────────┐
│ chart_id  │ dim: u32  │ n: size_t │ [point_0 data]            │
│ : u32     │           │           │ [point_1 data]            │
│           │           │           │ ...                       │
└───────────┴───────────┴───────────┴────────────────────────────┘
```

Each point stores:

```
┌───────────┬────────────┬────────────────┬──────────────────────┬───────────┐
│ chart_id  │ global_id  │ local_dim: u32  │ local_coords: d dbl │ ambient:  │
│ : u32     │ : u64      │ ambient_dim: u32 │                    │ D dbl     │
└───────────┴────────────┴────────────────┴──────────────────────┴───────────┘
```

**Note**: The R-tree structure is NOT serialised — only the point data. On `load()`, the tree is rebuilt from scratch via STR packing. This trades reload time for serialisation simplicity. Future versions may persist the tree structure for faster restore.

### Atlas Metadata (meta.json)

```json
{
  "version": "0.1.0",
  "num_charts": 3,
  "total_points": 10000,
  "chart_ids": [0, 1, 2],
  "chart_dims": {
    "0": {"intrinsic": 10, "ambient": 768},
    "1": {"intrinsic": 10, "ambient": 768},
    "2": {"intrinsic": 10, "ambient": 768}
  }
}
```

### Chart JSON Export (charts.json)

For `LinearChart` instances, the basis and origin are serialised:

```json
{
  "version": "0.1.0",
  "charts": [
    {
      "id": 0,
      "type": "LINEAR",
      "intrinsic_dim": 10,
      "ambient_dim": 768,
      "basis": [[...], [...], ...],
      "origin": [...]
    }
  ],
  "transitions": [
    {
      "from_chart": 0,
      "to_chart": 1,
      "is_identity": false,
      "rotation": [[...], [...], ...],
      "translation": [...]
    }
  ]
}
```

---

## Query Execution

### Geodesic k-NN Query

The geodesic k-NN query is the primary query primitive. It uses a **two-phase strategy** to balance the efficiency of flat-space indexing with the geometric fidelity of Riemannian distance computation.

```
Query: query_geodesic_knn(query_ambient, k, max_distance)
  │
  ├── Phase 1: Candidate Generation (fast, approximate)
  │     │
  │     ├── 1. Locate chart: find chart with minimum projection residual
  │     │     for query_ambient → project to local coordinates
  │     │
  │     ├── 2. R-tree k-NN: retrieve 3k candidates by local Euclidean distance
  │     │     (uses branch-and-bound pruning with AABB min-distance)
  │     │
  │     └── 3. Return candidate set {c₁, c₂, ..., c₃ₖ}
  │
  ├── Phase 2: Geodesic Re-ranking (accurate, exact)
  │     │
  │     ├── 4. For each candidate cᵢ:
  │     │     solve geodesic BVP from query to cᵢ
  │     │     → geodesic_distance(query, cᵢ)
  │     │
  │     ├── 5. Sort all candidates by ascending geodesic distance
  │     │
  │     ├── 6. Return top-k results
  │     │
  │     └── 7. Filter by max_distance if specified
  │
  └── Return: vector<NeighborResult> sorted by geodesic distance
```

#### Phase 1: Tangent Space Candidate Generation

The key insight is that **local Euclidean distance in chart coordinates is a reasonable proxy for geodesic distance** when the manifold is approximately flat at the scale of the query neighbourhood. The R-tree provides O(log n + k) candidate retrieval.

Candidate expansion factor: we retrieve `max(3k, 50)` candidates to account for the fact that the Euclidean proxy ranking may differ from the geodesic ranking.

#### Phase 2: Geodesic Re-ranking

Each candidate is re-ranked using the **exact geodesic distance** computed by the `GeodesicSolver`:

1. **Same-chart candidates**: Direct BVP solve via Newton shooting
2. **Cross-chart candidates**: Not currently supported in k-NN (fall back to Euclidean). Future: multi-chart geodesic via concatenated chart paths.

The `NeighborResult` struct captures both the geodesic distance and the Euclidean residual, enabling downstream analysis of how much the manifold geometry affects the ranking.

### Geodesic Ball Query

```
Query: query_geodesic_ball(center_ambient, radius)
  │
  ├── 1. Locate chart → project to local coordinates
  │
  ├── 2. R-tree range search with 1.5× radius enlargement
  │     (accounts for Euclidean underestimation of geodesic distance)
  │
  ├── 3. For each candidate: compute true geodesic distance
  │     Filter: keep only points with d_g ≤ radius
  │
  └── Return: vector<ManifoldPoint>
```

### Geodesic Path Query

```
Query: query_geodesic_path(start_ambient, end_ambient)
  │
  ├── Same chart?
  │   ├── YES: Solve BVP via Newton shooting
  │   │     Initial guess: v₀ = (end_local - start_local)
  │   │     Newton iteration: vₙ₊₁ = vₙ - J⁻¹(γ_{vₙ}(T) - target)
  │   │     With backtracking line search
  │   │
  │   └── NO: Cross-chart geodesic
  │         ├── Parallel transport end point to start chart
  │         ├── Solve BVP on start chart
  │         └── Fallback: straight line in ambient space
  │
  └── Return: GeodesicPath {points, arc_lengths, total_length, converged}
```

---

## Cross-Modal Query

Cross-modal retrieval enables searching one data modality using a query from another modality. Both modalities share a **common atlas**, enabling geometric comparison.

```
Query: query_cross_modal(query_ambient, source_modality, target_modality, k)
  │
  ├── 1. Locate the chart for the query point
  │     (regardless of source modality — the atlas is shared)
  │
  ├── 2. Project query to local chart coordinates
  │
  ├── 3. Filter target modality points on the same chart
  │     (points from target_modality where chart_id matches)
  │
  ├── 4. k-NN in local coordinates (Euclidean proxy)
  │     partial_sort by ||local_coords - query_local||
  │
  ├── 5. Return top-k results with distances
  │
  └── Result: vector<NeighborResult> from target_modality
```

### Chart Transport Mechanism

When a query and target points reside on different charts, the atlas uses **multi-hop coordinate transport** via BFS:

```
transport(point, target_chart_id):
  │
  ├── 1. BFS to find shortest path: [chart_a → chart_b → ... → chart_target]
  │
  ├── 2. For each hop (i → i+1):
  │     Apply transition map: local_coords = ψ_{i→i+1}(local_coords)
  │     Re-embed in new chart: ambient_coords = φ_{i+1}(local_coords)
  │
  └── 3. Return transported point on target chart
```

For `LinearTransitionMap`, the transition is affine:
```
x_β = R · x_α + t
```
where R = B_β^T · B_α and t = B_β^T · (origin_α - origin_β).

---

## Parallel Transport for Schema Evolution

When new data arrives that doesn't fit the existing manifold structure, ManifoldDB uses **schema evolution** to extend the manifold:

```
evolve_schema(new_ambient_points):
  │
  ├── 1. Assign new points to a fresh modality ID
  │
  ├── 2. Insert into the database (standard insert)
  │
  ├── 3. Rebuild the atlas with the combined dataset
  │     → New charts may be discovered
  │     → Existing charts may shift (PCA re-computation)
  │     → Transition maps recomputed
  │
  ├── 4. Re-initialise metric tensors (identity)
  │
  └── 5. Rebuild tangent-space indexes
```

### Parallel Transport in Geodesic Context

The `GeodesicSolver::parallel_transport()` method implements Levi-Civita transport along a geodesic path:

```
Dv^i/dt = −Γ^i_{jk}(x) v^j (dx^k/dt) = 0
```

This is integrated using RK4 steps alongside the geodesic itself:

1. At each path point, compute the Christoffel symbols Γ^i_{jk}
2. Evaluate the transport derivative: DV^i = -Γ^i_{jk} V^j v^k
3. Advance V using RK4 with the geodesic velocity v as the path parameter

This ensures tangent vectors maintain their geometric relationship as they are "moved" across the manifold — essential for cross-chart queries and consistent metric comparison.

---

## GPU Acceleration Strategy

### Current State (CPU)

All geodesic computation runs on CPU via Eigen's vectorised operations. The geodesic equation is a tight loop over dimension indices:

```cpp
for (int i = 0; i < d; ++i) {
    Scalar sum = 0.0;
    for (int j = 0; j < d; ++j)
        for (int k = 0; k < d; ++k)
            sum -= Gamma(i, j, k) * velocity(j) * velocity(k);
    accel(i) = sum;
}
```

### Planned CUDA Acceleration

The `SolverType::RK4_CUDA` enum value is reserved for GPU-accelerated geodesic integration. The strategy:

1. **Batch Christoffel evaluation**: Compute Γ^i_{jk} for multiple points simultaneously on GPU (one thread per point, shared memory for the Christoffel tensor)
2. **Batch RK4 step**: Advance multiple geodesics in parallel (one CUDA block per geodesic, warp-level synchronisation)
3. **Batch BVP shooting**: Run Newton iterations for multiple BVPs in parallel on GPU

```cuda
// Pseudo-code for CUDA geodesic kernel
__global__ void geodesic_rk4_step_kernel(
    const double* __restrict__ gamma,  // Christoffel symbols (d³ per point)
    const double* __restrict__ pos,     // Current positions (batch_size × d)
    const double* __restrict__ vel,     // Current velocities (batch_size × d)
    double* __restrict__ new_pos,       // Updated positions
    double* __restrict__ new_vel,       // Updated velocities
    double dt, int d
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    // ... RK4 stage computation ...
}
```

### Memory Transfer Strategy

- **Metric tensors** stay on CPU (small, evaluated per-chart)
- **Christoffel symbols** are computed on CPU and transferred to GPU as read-only constant memory
- **Position/velocity arrays** are allocated on GPU for the duration of the solve
- **DLPack** is used for zero-copy transfer between PyTorch CUDA tensors and the C++ solver

---

## Python/C++ Interop

### PyBind11 Bridge

The C++ extension module `manifolddb_core` is built via PyBind11 with the following interop mechanisms:

| C++ Type | Python Representation | Mechanism |
|----------|----------------------|-----------|
| `Eigen::VectorXd` | `numpy.ndarray` (1-D, float64) | `py:: eigen.h` direct mapping |
| `Eigen::MatrixXd` | `numpy.ndarray` (2-D, float64) | `py:: eigen.h` direct mapping |
| `torch::Tensor` | `torch.Tensor` | `torch::Tensor` → `tensor_to_vectors()` → `std::vector<Vector>` |
| `std::vector<ManifoldPoint>` | `list[dict]` | Manual conversion in Python wrapper |
| `std::vector<NeighborResult>` | `list[dict]` | Manual conversion in Python wrapper |
| `GeodesicPath` | `dict` | Manual conversion with arc_lengths as list |

### Data Flow for a Typical Query

```
Python:  query = torch.randn(768, dtype=torch.float64)
         results = db.query_knn(query, k=10)
                    │
                    ▼
Python Wrapper (_db.query_geodesic_knn):
    q = _ensure_numpy_float64(query).ravel()  # torch → numpy float64
    results_cpp = self._db.query_geodesic_knn(q, k, max_distance)
    return [self._nr_to_dict(nr) for nr in results_cpp]
                    │
                    ▼
PyBind11: py::array_t<double> → Eigen::Ref<Eigen::VectorXd>
           (zero-copy view via Eigen::Map)
                    │
                    ▼
C++ ManifoldDB::query_geodesic_knn():
    Chart* chart = atlas_->locate_chart(query_ambient)
    Vector query_local = chart->project(query_ambient)
    auto candidates = index->knn(query_local, n_candidates)
    // Re-rank by geodesic distance...
    return results
                    │
                    ▼
Python:  results = [{'id': 0, 'distance': 0.1234, ...}, ...]
```

### DLPack Integration (Future)

The `dlpack_export()` function in `torch_compat.py` is a placeholder for future zero-copy tensor sharing:

```
C++ MetricTensor anchor data ──DLPack──▶ PyTorch tensor (no copy)
  or
C++ geodesic results       ──DLPack──▶ JAX array (no copy)
```

This would enable seamless integration with any DLPack-compatible framework (PyTorch, TensorFlow, JAX).

### Trampoline Classes for Python Subclassing

PyBind11 trampoline classes enable Python users to subclass abstract C++ classes:

```python
from manifolddb import Chart

class CustomChart(Chart):
    def embed(self, local_coords):
        return np.sin(local_coords) * 2.0

    def project(self, ambient_coords):
        return np.arcsin(np.clip(ambient_coords / 2.0, -1, 1))

    def jacobian(self, local_coords):
        return np.diag(np.cos(local_coords) * 2.0)

    def type(self):
        return ChartType.CUSTOM
```

The trampoline classes (`PyChart`, `PyTransitionMap`) override all pure virtual methods with `PYBIND11_OVERRIDE_PURE` calls that dispatch to Python.

---

## Future Directions

### TileDB Integration

Replace the file-based storage layer with TileDB, a multi-dimensional array database:

- **Benefits**: Cloud-native storage, compression, multi-threaded reads, array slicing
- **Schema**: Charts as TileDB arrays with dimension = (chart_id, intrinsic_dim)
- **Metric tensors**: Dense array `(chart_id × dim × dim)` with real-time updates
- **Indexes**: Sparse array representation of R-tree structure

```
TileDB Schema:
  charts/
    points/    → (chart_id, point_idx, dim)   dense float64
    metadata/  → (chart_id, field)             dense float64
  metrics/
    tensors/   → (chart_id, i, j)              dense float64
    anchors/   → (chart_id, anchor_idx, dim)   dense float64
  indexes/
    rtree/     → (chart_id, node_id, ...)       sparse float64
```

### CUDA Kernels

Custom CUDA kernels for batch geodesic computation:

1. **`christoffel_kernel`**: Batch compute Γ^i_{jk} at multiple points
2. **`geodesic_rk4_kernel`**: Advance multiple geodesics in parallel
3. **`batch_distance_kernel`**: Compute pairwise geodesic distances
4. **`christoffel_numerical_kernel`**: Finite-difference Christoffel computation on GPU

### Learned Manifolds

Replace static PCA charts with **neural network embeddings** learned end-to-end:

- **Chart**: NeuralChart with ONNX Runtime (current) → torch.nn.Module (future)
- **Metric**: Neural metric tensor field g_ij(x; θ) parameterised by a neural network
- **Training**: Minimise geodesic reconstruction loss + metric consistency loss
- **Benefits**: Capture highly non-linear manifold structure, differentiable end-to-end

### Distributed Atlas

Shard the atlas across multiple nodes for datasets exceeding single-node memory:

- **Chart sharding**: Each node manages a subset of charts
- **Geodesic routing**: Query coordinator routes to relevant shards
- **Transition map replication**: Cross-shard transition maps cached locally
- **Consistent hashing**: Points assigned to shards by chart_id hash
