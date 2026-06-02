# Getting Started Tutorial

> Installation, quick start, configuration, CLI usage, and API reference.

## Installation

### From PyPI (when published)

```bash
pip install manifold-db
```

### From Source

```bash
git clone https://github.com/your-org/manifold-db.git
cd manifold-db
pip install -e .

# Optional: torch for LearnedMetric
pip install torch
```

### Dependencies

| Package | Version | Required | Notes |
|---------|---------|----------|-------|
| numpy | ≥ 1.20 | Yes | Core numerics |
| scipy | ≥ 1.7 | Yes | KDTree, sparse, optimization |
| scikit-learn | ≥ 1.0 | Yes | PCA, BallTree, NearestNeighbors |
| networkx | ≥ 2.6 | Yes | Louvain community detection |
| torch | ≥ 1.9 | No | Optional: `LearnedMetric` |

## Quick Start (5 Minutes)

### Step 1: Generate Data

```python
import numpy as np

# Swiss roll: 2D manifold in 3D space
rng = np.random.default_rng(42)
t = rng.uniform(0, 4 * np.pi, 1000)
x = t * np.cos(t) + rng.normal(0, 0.1, 1000)
y = t * np.sin(t) + rng.normal(0, 0.1, 1000)
z = rng.uniform(-0.5, 0.5, 1000)
data = np.column_stack([x, y, z])

point_ids = [f"pt_{i:04d}" for i in range(len(data))]
```

### Step 2: Build Atlas

```python
from manifold_db.atlas import AtlasBuilder

builder = AtlasBuilder(
    k_neighbors=15,
    pca_variance_threshold=0.95,
    min_chart_size=50,
    random_state=42,
)
atlas = builder.build(data, n_charts_hint=3)

print(f"Charts: {len(atlas.get_all_charts())}")
print(f"Transitions: {len(atlas.get_all_transition_maps())}")
```

### Step 3: Build Index

```python
from manifold_db.tangent_index import TangentSpaceIndex

index = TangentSpaceIndex(intrinsic_dim=2)
stats = index.build_from_data(
    point_ids=point_ids,
    data_points=data,
    n_anchors=30,
)
print(f"Points indexed: {stats['n_points']}")
print(f"Anchors: {stats['n_anchors']}")
print(f"Intrinsic dim: {stats['intrinsic_dim']}")
```

### Step 4: Search

```python
# Find 10 nearest neighbours
query = data[0]
results = index.search(query, k=10)

for pid, dist in results:
    print(f"  {pid}: distance = {dist:.6f}")
```

### Step 5: Save/Load

```python
# Save atlas
atlas.save("my_atlas.json")

# Load atlas
from manifold_db.atlas import AtlasManager
loaded = AtlasManager("loaded")
loaded.load("my_atlas.json")
```

## Configuration

### AtlasBuilder Configuration

```python
builder = AtlasBuilder(
    k_neighbors=15,                # KNN graph connectivity
    pca_variance_threshold=0.95,   # Dim estimation threshold
    min_chart_size=50,             # Min points per chart
    overlap_margin=0.05,          # Overlap bounding box margin
    random_state=42,               # Reproducibility
)
```

**Tuning guidelines:**
- `k_neighbors`: 10-20 is typical.  Higher = smoother graph but slower build.
- `pca_variance_threshold`: 0.90-0.98.  Higher = higher intrinsic dim.
- `min_chart_size`: 30-100.  Lower = more charts, finer granularity.
- `n_charts_hint`: Set via `builder.build(data, n_charts_hint=N)`.
  The builder auto-tunes the Louvain resolution parameter.

### TangentSpaceIndex Configuration

```python
index = TangentSpaceIndex(
    intrinsic_dim=None,   # Auto-detect from data, or set explicitly
    metric_eps=0.1,        # Epsilon for coverage analysis
    leaf_size=40,          # BallTree leaf size (sklearn parameter)
)
```

**Tuning guidelines:**
- `intrinsic_dim`: Set if you know the manifold's dimension.  Auto-detect
  uses 95% variance threshold.
- `n_anchors` (in `build_from_data`): 20-50 for 1K points, 50-100 for 10K+.
  More anchors = better coverage but slower search.
- `search_k_anchors` (in `search`): 1-3.  Search multiple anchors for
  better recall at the cost of speed.

### Custom Embedding Functions

```python
from sklearn.decomposition import PCA
from manifold_db.atlas import Chart, AtlasBuilder, AtlasManager

# Build custom PCA-based chart
pca = PCA(n_components=5)
pca.fit(data)

chart = Chart(
    name="custom_pca",
    dim=5,
    ambient_dim=data.shape[1],
    embedding_fn=lambda d: pca.transform(d),
    inverse_fn=lambda c: pca.inverse_transform(c),
)

atlas = AtlasManager("custom")
atlas.add_chart(chart)
```

## API Usage

### Direct TangentSpaceIndex API

For most use cases, the `TangentSpaceIndex` is the primary interface:

```python
from manifold_db.tangent_index import TangentSpaceIndex

# Build
index = TangentSpaceIndex(intrinsic_dim=8)
index.build_from_data(ids, data, n_anchors=30)

# Insert
index.insert("new_point", new_data)
index.batch_insert(new_ids, new_data_batch)

# Search
results = index.search(query_point, k=10, search_k_anchors=3)
# Returns: [(point_id, distance), ...]

# Update
index.update("existing_point", new_data)

# Delete
success = index.delete("point_to_remove")

# Stats
print(index.stats)
# {
#   "chart_id": "...",
#   "size": 1000,
#   "n_anchors": 30,
#   "intrinsic_dim": 8,
#   "build_time_sec": 0.45,
#   "points_per_anchor": {"min": 20, "max": 50, "mean": 33.3},
# }
```

### TangentSpace API

For low-level operations:

```python
from manifold_db.tangent_index import TangentSpace

ts = TangentSpace(
    base_point=data[0],
    data=data[:100],     # local neighbourhood
    intrinsic_dim=5,
)

# Project / lift
tangent_coords = ts.project(data)       # (N, ambient) → (N, intrinsic)
recovered = ts.lift(tangent_coords)     # (N, intrinsic) → (N, ambient)

# Log / exp maps
log_vec = ts.log_map(point_on_manifold)  # ambient → tangent vector
exp_point = ts.exp_map(log_vec)           # tangent vector → ambient

# Distances
dist = ts.compute_distance(log_a, log_b)  # geodesic distance

# Metric
inner = ts.compute_metric(v1, v2)  # Riemannian inner product

# Parallel transport
transported = ts.parallel_transport(vec, other_ts)

# Online update
ts.update_basis(new_data_batch, lr=0.1)
```

### QueryBuilder API

```python
from manifold_db.query import QueryBuilder

# Build and validate a query
query = (
    QueryBuilder()
    .select("id", "embedding", "label")
    .from_chart("chart_0")
    .where_geodesic(query_vector, epsilon=0.5)
    .using_metric("geodesic")
    .top_k(10)
    .order_by("distance")
    .with_metadata(source="my_app")
    .build()
)

# Inspect
print(query.to_dict())
valid, msg = query.validate()
print(f"Cost tier: {query.estimate_cost()}")
```

### QueryParser API

```python
from manifold_db.query import QueryParser

parser = QueryParser()

# Parse SQL-like query
query = parser.parse(
    "SELECT * FROM observations "
    "WHERE geodesic_distance(embedding, [1,2,3]) < 0.5 "
    "LIMIT 10"
)

print(query.query_type)    # QueryType.SELECT
print(query.epsilon)       # 0.5
print(query.query_point)   # array([1, 2, 3])
```

### MetricTensorStore API

```python
from manifold_db.metric import (
    EuclideanMetric, DiagonalMetric, MetricTensorStore
)

store = MetricTensorStore()

# Register metrics per chart
store.register("chart_0", EuclideanMetric(dimension=10))
store.register("chart_1", DiagonalMetric(
    dimension=10,
    weights=np.array([1.0, 2.0, 0.5, 1.0, 1.5, 1.0, 0.8, 1.2, 1.0, 1.0]),
))

# Get metric (falls back to Euclidean if not registered)
metric = store.get("chart_0")
g = metric.evaluate(point)  # (d, d) metric tensor
g_inv = metric.inverse(point)
det = metric.determinant(point)
log_det = metric.log_det(point)
```

## Common Patterns

### Pattern 1: Basic Similarity Search

```python
# Build once, query many times
index = TangentSpaceIndex(intrinsic_dim=8)
index.build_from_data(ids, data, n_anchors=30)

# Batch queries
for query in queries:
    results = index.search(query, k=5)
    top_ids = [pid for pid, _ in results]
```

### Pattern 2: Multi-Modal Setup

```python
# Separate indices per modality
text_index = TangentSpaceIndex(intrinsic_dim=32)
text_index.build_from_data(text_ids, text_data, n_anchors=20)

image_index = TangentSpaceIndex(intrinsic_dim=32)
image_index.build_from_data(image_ids, image_data, n_anchors=20)

# Cross-modal via tangent space transport
transported = text_ts.parallel_transport(text_coords, image_ts)
image_results = image_index.search(image_ts.lift(transported), k=10)
```

### Pattern 3: Incremental Updates

```python
# Insert new points as they arrive
for new_id, new_point in stream:
    index.insert(new_id, new_point)

# Periodically update basis with new data
ts.update_basis(recent_batch, lr=0.05)
```

## Next Steps

- [Advanced Queries Tutorial](advanced-queries.md) — Geodesic, cross-modal, schema evolution
- [Architecture Overview](architecture/overview.md) — System design and data flow
- [API Reference](api/api-reference.md) — Complete method signatures
- [Quickstart Example](../examples/scripts/quickstart.py) — Full runnable script
