# Advanced Queries Tutorial

> Geodesic queries, cross-modal retrieval, schema evolution, and temporal analysis.

## Geodesic Queries

### What are Geodesic Queries?

Geodesic queries use the true shortest-path distance on the manifold,
not the flat Euclidean distance in ambient space.  This is crucial when
the manifold is curved — Euclidean distance can be misleading.

### Using the Geodesic Solver

```python
from manifold_db.geodesic.distance import RiemannianDistance

# Define a metric tensor function
def my_metric(x: np.ndarray) -> np.ndarray:
    """Position-dependent Riemannian metric."""
    d = len(x)
    scale = 1.0 + 0.3 * np.sin(x[0])
    return np.eye(d) * scale

# Compute geodesic distance
riemann = RiemannianDistance(metric_tensor_fn=my_metric)
distance = riemann.geodesic_distance(point_a, point_b)
```

### Curvature-Corrected Distance

For higher accuracy, use the second-order curvature correction:

```python
def my_christoffel(x: np.ndarray) -> np.ndarray:
    """Christoffel symbols Γ^i_{jk}(x)."""
    d = len(x)
    return np.zeros((d, d, d))  # flat example

riemann = RiemannianDistance(
    metric_tensor_fn=my_metric,
    christoffel_fn=my_christoffel,
)
distance = riemann.curvature_corrected_distance(p, q)
```

### DistanceComputer (Unified Interface)

```python
from manifold_db.geodesic.distance import DistanceComputer

comp = DistanceComputer(
    metric_tensor_fn=my_metric,
    christoffel_fn=my_christoffel_fn,
)

# Multiple distance types
eucl = comp.compute(p, q, metric_type="euclidean")
tangent = comp.compute(p, q, metric_type="tangent")
curved = comp.compute(p, q, metric_type="curvature")

# Batch computation
dists = comp.batch_compute(points_a, points_b, metric_type="tangent")
```

### Geodesic Ball Query

Find all points within a geodesic radius:

```python
# Oversample from tangent index, then filter by true geodesic distance
candidates = index.search(query_point, k=100, search_k_anchors=3)

results = []
for pid, tangent_dist in candidates:
    point = data_lookup[pid]
    geo_dist = riemann.geodesic_distance(query_point, point)
    if geo_dist <= epsilon:
        results.append((pid, geo_dist))

results.sort(key=lambda x: x[1])
```

### Full Example: Potential Energy Surface

```python
import numpy as np
from scipy.optimize import minimize
from manifold_db.geodesic.distance import RiemannianDistance

# Mexican hat potential
def metric(x):
    g = np.eye(len(x))
    grad = 4.0 * (np.sum(x**2) - 1.0) * x
    g *= (1.0 + 0.5 * np.dot(grad, grad))
    return g

riemann = RiemannianDistance(metric_tensor_fn=metric)

# Find minimum-energy path between two states
p = np.array([1.0, 0.0])   # on the minimum ring
q = np.array([-1.0, 0.0])  # opposite side

geo_dist = riemann.geodesic_distance(p, q)
euc_dist = np.linalg.norm(p - q)

print(f"Geodesic: {geo_dist:.4f}")
print(f"Euclidean: {euc_dist:.4f}")
print(f"Ratio: {geo_dist / euc_dist:.4f}")
# Geodesic > Euclidean because the path must follow the curved surface
```

## Cross-Modal Retrieval

### Concept

Cross-modal retrieval finds data in one modality (e.g., images) that
matches a query in another modality (e.g., text).  The bridge is
**parallel transport** through the overlap region between modal charts.

```
  Text Query ──→ Text Chart ──→ [transport] ──→ Image Chart ──→ Image Results
                    φ_text       ψ_{text→image}       φ_image
```

### Setup

```python
from manifold_db.tangent_index import TangentSpace, TangentSpaceIndex
from manifold_db.connection import TransportRegistry

# Build separate indices for each modality
text_index = TangentSpaceIndex(intrinsic_dim=32)
text_index.build_from_data(text_ids, text_data, n_anchors=20)

image_index = TangentSpaceIndex(intrinsic_dim=32)
image_index.build_from_data(image_ids, image_data, n_anchors=20)

# Build tangent spaces at the overlap region
text_ts = TangentSpace(
    base_point=text_overlap.mean(axis=0),
    data=text_overlap,
    intrinsic_dim=32,
)
image_ts = TangentSpace(
    base_point=image_overlap.mean(axis=0),
    data=image_overlap,
    intrinsic_dim=32,
)

# Set up transport registry
registry = TransportRegistry(max_size=128)
registry.register_transport(
    "text_chart", "image_chart",
    transport_fn=lambda v: text_ts.parallel_transport(v, image_ts),
)
```

### Cross-Modal Search

```python
def cross_modal_search(
    text_query: np.ndarray,
    text_ts: TangentSpace,
    image_ts: TangentSpace,
    image_index: TangentSpaceIndex,
    k: int = 10,
) -> list:
    """Find images matching a text query via parallel transport."""
    
    # 1. Project text query to text tangent space
    text_coords = text_ts.project(text_query.reshape(1, -1))
    
    # 2. Parallel transport to image tangent space
    transported = text_ts.parallel_transport(text_coords[0], image_ts)
    
    # 3. Lift to image ambient space
    transported_ambient = image_ts.lift(transported.reshape(1, -1))
    
    # 4. Search in image index
    results = image_index.search(transported_ambient[0], k=k)
    
    return results
```

### Using QueryBuilder

```python
from manifold_db.query import QueryBuilder

query = (
    QueryBuilder()
    .cross_modal("text", "image")
    .with_transport("overlap_region")
    .top_k(10)
    .build()
)
```

### Comparison: Manifold vs Naive Euclidean

The key advantage of manifold-based cross-modal retrieval:

1. **Geometry preservation**: Parallel transport preserves the geometric
   structure of the query, not just the raw vector values.
2. **Manifold-awareness**: The transport accounts for the different shapes
   of the text and image manifolds.
3. **Overlap learning**: The transition map is learned from real paired
   data (image-caption pairs), not arbitrary padding.

Without manifold geometry, you would need to:
- Pad/truncate dimensions (lossy)
- Use a shared embedding space (requires joint training)
- Apply heuristic dimension reduction (no geometric guarantee)

## Schema Evolution

### Concept

In a manifold database, schema changes (adding/removing columns) are
smooth geometric deformations of the data manifold.  This avoids the
traditional "stop-the-world" migration.

### Setup: Two Schema Versions

```python
from manifold_db.connection import SchemaTransport

# Old schema: 10 features
old_data = generate_data(n=500, dim=10)

# New schema: 15 features (5 new derived columns)
new_data = add_features(old_data, new_cols=5)

# Build tangent spaces for each schema
old_ts = TangentSpace(base_point=old_data.mean(0), data=old_data[:100], intrinsic_dim=5)
new_ts = TangentSpace(base_point=new_data.mean(0), data=new_data[:100], intrinsic_dim=7)
```

### Register Schema Transport

```python
from manifold_db.metric import EuclideanMetric

schema_transport = SchemaTransport()
schema_transport.register_schema(
    schema_id="v1",
    reference_point=old_data.mean(axis=0),
    metric=EuclideanMetric(dimension=10),
)
schema_transport.register_schema(
    schema_id="v2",
    reference_point=new_data.mean(axis=0),
    metric=EuclideanMetric(dimension=15),
)
```

### Transport a Query

```python
# Query in old schema space
old_query = old_data[0]  # 10D

# Transport to new schema space
new_query = schema_transport.transport_query(
    query_vector=old_query,
    source_schema="v1",
    target_schema="v2",
)
# new_query is now 15D (old features preserved, new features interpolated)
```

### Zero-Downtime Migration

Both schemas coexist during migration:

```python
# Old queries → old index (unchanged)
old_index.search(old_query, k=10)

# New queries → new index (uses new features)
new_index.search(new_query, k=10)

# Cross-schema queries → transport bridge
transported = schema_transport.transport_query(old_query, "v1", "v2")
new_index.search(transported, k=10)
```

## Temporal Analysis

### Concept

Time-series data on manifolds often exhibits **concept drift** — the
underlying distribution changes over time.  Temporal parallel transport
enables queries across time steps.

### Setup: Time-Indexed Atlas

```python
from manifold_db.connection import TemporalTransport

# Generate time-varying data
for t in range(n_time_steps):
    data_t = generate_data_at_time(t)
    ts_t = TangentSpace(base_point=data_t.mean(0), data=data_t[:50], intrinsic_dim=5)
    tangent_spaces[t] = ts_t
    indices[t] = build_index(data_t)
```

### Register Temporal Transport

```python
temporal_transport = TemporalTransport()

# Reference curve: sequence of centroids
reference_curve = np.array([
    data_t.mean(axis=0) for t in range(n_time_steps)
])

temporal_transport.register_temporal_path(
    times=np.arange(n_time_steps, dtype=float),
    reference_curve=reference_curve,
    metric_fn=lambda t, x: np.eye(len(x)),  # or time-dependent metric
)
```

### Cross-Time Query

```python
# Search at t=5 using a query from t=0
query_t0 = data[0]
coords_t0 = tangent_spaces[0].project(query_t0.reshape(1, -1))

# Transport along the time path
transported = temporal_transport.transport(
    vector=coords_t0[0],
    source_time=0.0,
    target_time=5.0,
)

# Lift and search
transported_ambient = tangent_spaces[5].lift(transported.reshape(1, -1))
results = indices[5].search(transported_ambient[0], k=10)
```

### Anomaly Detection

Points far from the manifold are anomalies:

```python
def is_anomaly(point: np.ndarray, ts: TangentSpace, threshold: float) -> bool:
    """Check if a point deviates significantly from the manifold."""
    coords = ts.project(point.reshape(1, -1))
    recovered = ts.lift(coords)
    deviation = np.linalg.norm(point - recovered[0])
    return deviation > threshold
```

## Wasserstein Distance

For distributional data, use the optimal transport distance:

```python
from manifold_db.geodesic.distance import WassersteinDistance

wass = WassersteinDistance(reg=0.1, max_iter=500)

# Source and target distributions
mu = np.array([0.3, 0.5, 0.2])  # discrete distribution
nu = np.array([0.4, 0.3, 0.3])

# Cost matrix
cost = np.array([
    [0.0, 1.0, 2.0],
    [1.0, 0.0, 1.0],
    [2.0, 1.0, 0.0],
])

distance = wass.sinkhorn_distance(mu, nu, cost)
print(f"Wasserstein distance: {distance:.6f}")
```

## Fisher-Rao Distance

For information-geometric applications:

```python
from manifold_db.geodesic.distance import FisherRaoDistance

fr = FisherRaoDistance()

p = np.array([0.3, 0.5, 0.2])
q = np.array([0.4, 0.3, 0.3])

distance = fr.fisher_rao_distance(p, q)
print(f"Fisher-Rao distance: {distance:.6f}")  # ∈ [0, π]
```

## Putting It All Together

A complete cross-modal pipeline:

```python
# 1. Build per-modality atlases
text_atlas = builder.build(text_data, modality="text")
image_atlas = builder.build(image_data, modality="image")

# 2. Build per-modality indices
text_index = TangentSpaceIndex(intrinsic_dim=32)
text_index.build_from_data(text_ids, text_data, n_anchors=20)

image_index = TangentSpaceIndex(intrinsic_dim=32)
image_index.build_from_data(image_ids, image_data, n_anchors=20)

# 3. Set up cross-modal transport
registry = TransportRegistry()
registry.register_transport(
    "text", "image",
    lambda v: text_ts.parallel_transport(v, image_ts),
)

# 4. Execute cross-modal queries
query = text_data[0]
transported = text_ts.parallel_transport(
    text_ts.project(query.reshape(1, -1))[0], image_ts
)
results = image_index.search(image_ts.lift(transported.reshape(1, -1))[0], k=10)

# 5. Compare with geodesic refinement
for pid, tangent_dist in results[:5]:
    point = image_data[int(pid.split("_")[1])]
    geo_dist = riemann.geodesic_distance(
        image_ts.lift(transported.reshape(1, -1))[0], point
    )
    print(f"  {pid}: tangent={tangent_dist:.4f}, geodesic={geo_dist:.4f}")
```
