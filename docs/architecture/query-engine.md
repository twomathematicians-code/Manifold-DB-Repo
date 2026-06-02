# Query Engine

> Query DSL, execution strategies, cost model, and parallel transport in queries.

## Query DSL Grammar

The Manifold Database supports both a SQL-like query language and a
programmatic fluent builder API.

### SQL-like Syntax

```sql
-- Geodesic query: find nearest neighbours within epsilon
SELECT * FROM observations
WHERE geodesic_distance(embedding, [1.0, 2.0, 3.0]) < 0.5
LIMIT 10;

-- Specify manifold and metric
SELECT * FROM observations
ALONG manifold 'climate_model_atlas'
USING metric 'wasserstein_riemannian';

-- Tangent-space query: fast search in a specific chart
TANGENT_QUERY FROM chart 'text_embeddings'
WHERE distance < 0.5 TOP_K 10;

-- Cross-modal retrieval: text → image
CROSS_MODAL FROM 'text' TO 'image'
TRANSPORT VIA 'overlap_region' TOP_K 20;

-- Parallel transport: move a vector between charts
PARALLEL_TRANSPORT [1.0, 2.0, 3.0] FROM chart_a TO chart_b;
```

### Supported Keywords

| Keyword | Meaning |
|---------|---------|
| `SELECT` | Begin a SELECT query |
| `FROM` | Data source or chart reference |
| `WHERE` | Geodesic distance predicate |
| `ALONG manifold` | Specify atlas name |
| `USING metric` | Distance metric type |
| `LIMIT` | Max results |
| `TANGENT_QUERY` | Fast tangent-space-only query |
| `CROSS_MODAL` | Cross-modal retrieval |
| `TRANSPORT VIA` | Transport strategy |
| `PARALLEL_TRANSPORT` | Pure vector transport |

### Fluent Builder API

```python
from manifold_db.query import QueryBuilder

# Geodesic query
query = (
    QueryBuilder()
    .select("*")
    .from_chart("chart_0")
    .where_geodesic(query_vector, epsilon=0.5)
    .using_metric("geodesic")
    .top_k(10)
    .build()
)

# Tangent query
query = (
    QueryBuilder()
    .tangent_query("chart_0", query_point, epsilon=1.0)
    .top_k(5)
    .build()
)

# Cross-modal query
query = (
    QueryBuilder()
    .cross_modal("text", "image")
    .with_transport("overlap_region")
    .top_k(10)
    .build()
)

# Parallel transport
query = (
    QueryBuilder()
    .parallel_transport(vector, source="chart_a", target="chart_b")
    .build()
)
```

## Query Types

### SELECT (Geodesic)

Full geodesic query pipeline with refinement.

```python
query = ManifoldQuery(
    query_type=QueryType.SELECT,
    query_point=np.array([1.0, 2.0, 3.0]),
    epsilon=0.5,
    k=10,
    metric_type=MetricType.GEODESIC,
)
```

### TANGENT

Fast tangent-space-only search (no geodesic refinement).

```python
query = ManifoldQuery(
    query_type=QueryType.TANGENT,
    chart_id="chart_0",
    query_point=np.array([1.0, 2.0]),
    epsilon=1.0,
    k=10,
)
```

### CROSS_MODAL

Cross-chart retrieval with parallel transport.

```python
query = ManifoldQuery(
    query_type=QueryType.CROSS_MODAL,
    modality="text",
    target_modality="image",
    query_point=text_embedding,
    transport_via="overlap_region",
    k=10,
)
```

### TRANSPORT

Pure parallel transport of a vector.

```python
query = ManifoldQuery(
    query_type=QueryType.TRANSPORT,
    transport_vector=np.array([1.0, 2.0]),
    source_chart="chart_a",
    target_chart="chart_b",
)
```

### RANGE

Geodesic ball query — all points within epsilon.

```python
query = ManifoldQuery(
    query_type=QueryType.RANGE,
    query_point=np.array([1.0, 2.0]),
    epsilon=0.5,
    k=100,  # oversample
)
```

## Execution Strategies

### Geodesic Query Pipeline

```
  ManifoldQuery (SELECT)
         │
         ▼
  ┌──────────────────┐   Step 1: Chart Lookup
  │ locate_chart     │   Find chart containing query point
  │ (O(1) or KDTree) │   Cost: ~0.5ms
  └────────┬─────────┘
           │ chart_id
           ▼
  ┌──────────────────┐   Step 2: Tangent Projection
  │ tangent_project  │   ambient → tangent coordinates
  │ (matrix multiply) │   Cost: ~0.2ms
  └────────┬─────────┘
           │ tangent_point
           ▼
  ┌──────────────────┐   Step 3: Candidate Search
  │ tangent_search    │   BallTree query in low-D
  │ (oversample 3×)    │   Cost: ~1ms
  └────────┬─────────┘
           │ candidate IDs + tangent distances
           ▼
  ┌──────────────────┐   Step 4: Geodesic Refinement
  │ geodesic_refine   │   Energy minimisation per candidate
  │ (N_cand × N_wp)  │   Cost: ~5ms (10 candidates)
  └────────┬─────────┘
           │ geodesic distances
           ▼
  ┌──────────────────┐   Step 5: Result Assembly
  │ sort + truncate   │   Sort by distance, take top-k
  │ (O(k log k))     │   Cost: ~0.1ms
  └──────────────────┘
```

### Tangent Query Pipeline

```
  ManifoldQuery (TANGENT)
         │
         ▼
  ┌──────────────────┐   Step 1: Direct tangent search
  │ tangent_search    │   No chart lookup needed
  │ (single BallTree) │   Cost: ~1ms
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐   Step 2: Epsilon filter
  │ filter ≤ epsilon │   Cost: ~0ms
  └──────────────────┘
```

### Cross-Modal Query Pipeline

```
  ManifoldQuery (CROSS_MODAL)
         │
         ▼
  ┌──────────────────┐   Step 1: Locate source chart
  │ locate_chart     │   Find chart for source modality
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐   Step 2: Parallel Transport
  │ parallel_transport│ Move query vector to target chart
  │ (SVD rotation)    │ Cost: ~2ms
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐   Step 3: Target Search
  │ tangent_search    │ Search in target chart
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐   Step 4: Assembly
  │ result assembly   │
  └──────────────────┘
```

## Cost Model

### Cost Tiers

| Tier | Query Types | Estimated Latency | Notes |
|------|-------------|------------------|-------|
| `CHEAP` | TANGENT | < 1 ms | Single BallTree query |
| `MODERATE` | TRANSPORT | ~10 ms | Transport + single search |
| `EXPENSIVE` | SELECT, CROSS_MODAL | ~50 ms | Multi-step with refinement |
| `VERY_EXPENSIVE` | RANGE (large) | > 100 ms | Full geodesic scan |

### Cost Estimation

```python
query = QueryBuilder().tangent_query("c0", point).build()
print(query.estimate_cost())  # CostTier.CHEAP

query = QueryBuilder().cross_modal("text", "image").top_k(20).build()
print(query.estimate_cost())  # CostTier.EXPENSIVE
```

### Execution Plans

```python
engine = QueryEngine(atlas_manager=atlas)
plan = await engine.explain(query)
print(plan.visualize())
```

Output:
```
  ========================================================================
    QUERY EXECUTION PLAN
  ========================================================================
    Total estimated cost: 6.80 ms
    Steps: 5
  ------------------------------------------------------------------------
    ├── Step 0: locate_chart
    │      Locate chart containing the query point via atlas lookup
    │      est. 0.5 ms
    │
    ├── Step 1: tangent_project
    │      Project query point into chart's tangent space
    │      est. 0.2 ms  (after step 0)
    │
    ├── Step 2: tangent_search
    │      Search candidates in tangent-space index
    │      est. 1.0 ms  (after step 1)
    │
    ├── Step 3: geodesic_refine
    │      Refine top-k with true geodesic distances
    │      est. 5.0 ms  (after step 2)
    │
    └── Step 4: assemble
           Assemble result set with metadata
           est. 0.1 ms  (after step 3)
  ========================================================================
```

## Parallel Transport in Queries

### Cross-Modal Transport

When querying across modalities (e.g., text → image), the query vector
must be parallel transported from the source chart to the target chart:

```
  Text Chart          Overlap Region        Image Chart
  ┌──────────┐       ┌──────────┐        ┌──────────┐
  │           │       │          │        │           │
  │  q_text   │──────▶│  ψ_AB    │──────▶ │ q_image  │
  │  (768D)   │       │ transport│        │ (512D)    │
  │           │       │          │        │           │
  └──────────┘       └──────────┘        └──────────┘

  1. Project q_text into text tangent space
  2. SVD-optimal rotation: B_image^T @ B_text
  3. Apply rotation + magnitude correction
  4. Search in image tangent space
```

### Schema Transport

For queries that need to cross schema versions:

```
  Old Schema (10D)     Transition     New Schema (15D)
  ┌──────────┐                      ┌──────────┐
  │ q_old    │────── pad/adjust ──▶│ q_new    │
  │ (10D)    │      + geometric    │ (15D)    │
  └──────────┘        mapping        └──────────┘
```

SchemaTransport handles dimensionality mismatch by padding/truncating
and applying geodesic interpolation.

### Temporal Transport

For time-evolving data:

```
  t=0           t=1           t=2           t=3
  ┌────┐       ┌────┐       ┌────┐       ┌────┐
  │ TS │──────▶│ TS │──────▶│ TS │──────▶│ TS │
  │  0 │       │  1 │       │  2 │       │  3 │
  └────┘       └────┘       └────┘       └────┘
         chain transport along time path
```

## Metric Types

| Metric | Enum Value | Description | Cost |
|--------|-----------|-------------|------|
| Geodesic | `geodesic` | Energy-minimisation path | High |
| Euclidean | `euclidean` | Standard L2 distance | Low |
| Cosine | `cosine` | 1 - cosine similarity | Low |
| Wasserstein | `wasserstein_riemannian` | Optimal transport | Very High |
| Fisher-Rao | `fischer_rao` | Information distance | High |
| Log-Euclidean | `log_euclidean` | Log-domain Euclidean | Medium |

## QueryResult

```python
result = await engine.execute(query)

# Properties
len(result)           # number of results
result.point_ids      # np.ndarray of IDs
result.distances      # np.ndarray of distances
result.execution_time # float (seconds)
result.chart_id       # which chart was used
result.query_type     # query type string

# Conversion
result.to_list()      # List[Dict]
result.to_dict()      # Dict
result.to_dataframe() # pandas DataFrame

# Iteration
for row in result:
    print(row["point_id"], row["distance"])
```

## Async Batch Execution

```python
queries = [query1, query2, query3]
results = await engine.batch_execute(queries)
# List[QueryResult] — executed concurrently
```
