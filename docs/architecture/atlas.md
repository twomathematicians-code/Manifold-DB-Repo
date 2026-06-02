# Atlas Layer

> Charts, transition maps, dimensionality estimation, and atlas construction.

## The Atlas Concept

An **atlas** `A = {(U_α, φ_α)}` is the mathematical foundation of a
manifold database.  Each chart `(U_α, φ_α)` represents:

- **U_α** — An open patch of the manifold (a region of data space)
- **φ_α** — A coordinate map `φ_α: U_α → R^d` into d-dimensional Euclidean space

Where charts overlap (`U_α ∩ U_β ≠ ∅`), a **transition map**
`ψ_{αβ}: φ_α(U_α ∩ U_β) → φ_β(U_α ∩ U_β)` ensures coordinate consistency.

```
  Manifold M
  ┌───────────────────────────────────────┐
  │                                       │
  │    ┌─────────┐    ┌─────────┐        │
  │    │ Chart A │    │ Chart B │        │
  │    │  U_A    │◄──►│  U_B    │        │
  │    │ φ_A:    │    │ φ_B:    │        │
  │    │ U_A→R^d │    │ U_B→R^d │        │
  │    └────┬────┘    └────┬────┘        │
  │         │              │              │
  │         │  overlap      │              │
  │         │  region       │              │
  │         └── ψ_AB ──────┘              │
  │         (transition map)               │
  │                                       │
  │              ┌─────────┐              │
  │              │ Chart C │              │
  │              │  U_C    │              │
  │              └─────────┘              │
  └───────────────────────────────────────┘
```

## Chart Class

### Construction

```python
from manifold_db.atlas import Chart

# Simple chart with default embedding (first-dim projection)
chart = Chart(name="cluster_0", dim=5, ambient_dim=20)

# Chart with custom PCA-based embedding
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
pca.fit(data)

def embed(d):
    return pca.transform(d)

def inverse(c):
    return pca.inverse_transform(c)

chart = Chart(
    name="pca_chart",
    dim=5,
    ambient_dim=20,
    embedding_fn=embed,
    inverse_fn=inverse,
)
```

### Core Operations

| Method | Signature | Description |
|--------|-----------|-------------|
| `embed(data)` | `(N, D) → (N, d)` | Project ambient → chart coordinates |
| `inverse(coords)` | `(N, d) → (N, D)` | Lift chart → ambient coordinates |
| `contains(coords)` | `(N, d) → (N,) bool` | Check if points are within bounds |
| `to_dict()` | → dict | Serialize to JSON-friendly dict |
| `from_dict(d)` | dict → Chart | Deserialize from dict |

### Bounds

Charts maintain axis-aligned bounding boxes for containment checks.
Bounds are either set explicitly or inferred from embedded data.

```python
# Explicit bounds
chart.bounds = (
    np.array([-5.0, -3.0, -1.0, -2.0, -4.0]),  # min per dim
    np.array([ 5.0,  3.0,  1.0,  2.0,  4.0]),  # max per dim
)

# Check containment
mask = chart.contains(tangent_coords, margin=0.1)
```

## Transition Maps

Transition maps are diffeomorphisms (smooth, invertible maps) between
chart coordinate systems.

### Available Types

| Type | Class | Formula | Use Case |
|------|-------|---------|----------|
| Linear | `LinearTransition` | `y = M @ x` | Rigid rotations |
| Affine | `AffineTransition` | `y = M @ x + b` | Most common |
| Neural | `NeuralTransition` | MLP coupling layer | Complex non-linear transitions |

### Affine Transition

```python
from manifold_db.atlas import AffineTransition

# y = M @ x + b
transition = AffineTransition(
    source_chart_id=chart_a.chart_id,
    target_chart_id=chart_b.chart_id,
    dim=5,
    matrix=np.random.randn(5, 5) * 0.1 + np.eye(5),  # near-identity
    bias=np.zeros(5),
)

# Apply
output = transition.forward(input_coords)
input_recovered = transition.inverse(output_coords)
```

### Overlap Regions

Transition maps have an associated `overlap_region` — the bounding box in
source coordinates where the transition is valid.

```python
transition.overlap_region = (
    np.array([-2.0, -1.0, -0.5, -1.5, -1.0]),  # min
    np.array([ 2.0,  1.0,  0.5,  1.5,  1.0]),  # max
)
```

## AtlasBuilder Algorithm

The `AtlasBuilder` automatically discovers the chart structure of a dataset.
The full pipeline has six stages:

```
  Raw Data (N × D)
       │
       ▼
  ┌─────────────────────┐   Stage 1: Dimensionality Estimation
  │ Local PCA Analysis   │   For each sample, fit PCA on its k-NN
  │                      │   neighbourhood.  Median of local dim estimates.
  └──────────┬──────────┘
             │ intrinsic_dim (d)
             ▼
  ┌─────────────────────┐   Stage 2: KNN Graph
  │ K-Nearest-Neighbour  │   Build symmetric adjacency with
  │ Graph                 │   exp(-distance) weights
  └──────────┬──────────┘
             │ sparse adjacency (N × N)
             ▼
  ┌─────────────────────┐   Stage 3: Community Detection
  │ Louvain Algorithm     │   Partition graph into communities
  │                      │   (chart assignments)
  └──────────┬──────────┘
             │ list of index arrays
             ▼
  ┌─────────────────────┐   Stage 4: Chart Creation
  │ Local PCA Projection  │   Per-cluster PCA defines chart
  │                      │   embedding/inverse functions
  └──────────┬──────────┘
             │ Chart objects
             ▼
  ┌─────────────────────┐   Stage 5: Overlap Detection
  │ Graph-based Overlap  │   Points in chart A with neighbours
  │ Detection             │   in chart B → overlap region
  └──────────┬──────────┘
             │ overlap index arrays
             ▼
  ┌─────────────────────┐   Stage 6: Transition Fitting
  │ Affine Fitting        │   Least-squares y = Mx + b on
  │                      │   overlap data
  └──────────┬──────────┘
             │ TransitionMap objects
             ▼
       AtlasManager (complete)
```

### Dimensionality Estimation

```python
builder = AtlasBuilder(pca_variance_threshold=0.95)
intrinsic_dim = builder.estimate_intrinsic_dimension(data, n_samples=500)
```

Algorithm:
1. Sub-sample `n_samples` points
2. For each sample, fit PCA on its k-NN neighbourhood
3. Find smallest d where cumulative variance > threshold
4. Return median of local estimates

### KNN Graph Construction

```python
adj = builder.compute_knn_graph(data, k=15)
# Returns scipy.sparse.csr_matrix (N × N)
```

- Symmetric adjacency: if i→j then j→i
- Weights: `exp(-distance)` for soft weighting
- Used by both Louvain and overlap detection

### Community Detection (Chart Boundaries)

```python
clusters = builder.detect_chart_boundaries(
    knn_graph=adj,
    min_chart_size=50,
    resolution=1.0,  # Louvain resolution parameter
)
# Returns list of np.ndarray (index arrays per chart)
```

- **Resolution** > 1.0 → more (smaller) charts
- **Resolution** < 1.0 → fewer (larger) charts
- Small clusters (< `min_chart_size`) are merged into nearest neighbour

### Transition Fitting

```python
transition = builder.fit_transition_map(
    overlap_data=ambient_data,
    source_coords=chart_a_coords,
    target_coords=chart_b_coords,
    map_type="affine",  # or "neural"
)
```

Affine fitting via least squares:
```
  [x_1 | 1]              [y_1]
  [x_2 | 1]  →  [M | b]  [y_2]
  [  ...  ]              [ ...]
```

## AtlasManager

The `AtlasManager` orchestrates charts and transitions.

### CRUD Operations

```python
atlas = AtlasManager(name="my_atlas")

# Charts
atlas.add_chart(chart)
chart = atlas.get_chart(chart_id)
charts = atlas.get_all_charts()
atlas.remove_chart(chart_id)

# Transitions
atlas.add_transition_map(tmap)
tmap = atlas.get_transition(src_id, tgt_id)
transitions = atlas.get_all_transition_maps()

# Chart lookup (find best chart for a data point)
best_chart = atlas.find_chart(data_point, modality="text")
```

### Chart Lookup Strategy

1. Filter by modality if specified
2. Try containment (embed → check bounds)
3. Fallback: nearest centroid distance

### Auto-Build

```python
atlas.build_atlas(
    data,                          # (N, D) array
    modality_labels=None,          # optional per-point labels
    modality="default",            # tag for all charts
    n_charts_hint=5,              # approximate chart count
)
```

### Persistence

```python
atlas.save("my_atlas.json")
atlas.load("my_atlas.json")

# Or in-memory
d = atlas.serialize()
atlas.deserialize(d)
```

## Quality Analysis

```python
metrics = builder.analyze_quality(atlas, data)
print(metrics)
# {
#   "coverage": 0.98,           # fraction of data covered by charts
#   "avg_overlap_ratio": 0.15,  # mean overlap / chart size
#   "dim_estimates": [3, 3, 2], # per-chart intrinsic dims
#   "n_isolated_charts": 0,     # charts with zero transitions
#   "n_charts": 5,
#   "n_transitions": 8,
# }
```

## Configuration Reference

### AtlasBuilder Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k_neighbors` | int | 15 | KNN graph connectivity |
| `pca_variance_threshold` | float | 0.95 | Cumulative variance for dim estimation |
| `min_chart_size` | int | 50 | Minimum points per chart |
| `overlap_margin` | float | 0.05 | Fractional margin for overlap bounds |
| `random_state` | int | None | Seed for reproducibility |
