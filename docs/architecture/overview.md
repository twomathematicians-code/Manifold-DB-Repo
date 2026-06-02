# Architecture Overview

> System architecture, data flow, and design decisions for Manifold Database.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Manifold Database System                         │
│                                                                     │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐   │
│  │   Query DSL  │──▶│ Query Engine │──▶│    Result Assembly    │   │
│  │  (Parser +   │   │ (Orchestrator│   │  (QueryResult / DF)    │   │
│  │   Builder)   │   │              │   │                        │   │
│  └──────────────┘   └──────┬───────┘   └────────────────────────┘   │
│                           │                                         │
│              ┌────────────┼────────────┐                             │
│              ▼            ▼            ▼                             │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                │
│  │ Atlas Manager│ │ Tangent Index│ │  Geodesic    │                │
│  │ (Charts +    │ │ (Tangent     │ │  Solver      │                │
│  │  Transitions)│ │  Bundle +    │ │  (Distance   │                │
│  │              │ │  BallTrees)  │ │   Computer)  │                │
│  └──────┬───────┘ └──────┬───────┘ └──────────────┘                │
│         │                │                                          │
│  ┌──────▼────────────────▼───────┐                                  │
│  │      Metric Tensor Store      │                                  │
│  │  (Euclidean, Diagonal, etc.)  │                                  │
│  └──────────────────────────────┘                                  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              Connection & Transport Layer                      │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌─────────────────────┐  │   │
│  │  │ Levi-Civita  │ │    Schema    │ │    Temporal          │  │   │
│  │  │ Connection   │ │  Transport   │ │    Transport         │  │   │
│  │  └──────────────┘ └──────────────┘ └─────────────────────┘  │   │
│  │                                                              │   │
│  │  ┌──────────────────────────────────────────────────────┐   │   │
│  │  │           Transport Registry (LRU Cache)              │   │   │
│  │  └──────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Storage Layer                              │   │
│  │  Data Store  │  Atlas Persistence  │  Index Serialization    │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## Module Interaction Diagram

```
                    User Application
                          │
                    ┌─────▼─────┐
                    │  Query    │  SQL-like DSL  /  Fluent Builder
                    │  Parser   │  ─────────────────────────────────
                    └─────┬─────┘
                          │ ManifoldQuery
                    ┌─────▼─────┐
                    │  Query    │  Orchestrates all components
                    │  Engine   │  ─────────────────────────────────
                    └──┬───┬──┬─┘
                       │   │  │
          ┌────────────┘   │  └────────────┐
          ▼                ▼                ▼
    ┌───────────┐   ┌───────────┐   ┌──────────────┐
    │  Atlas     │   │  Tangent   │   │  Connection   │
    │  Manager   │   │  Index     │   │  (Transport)  │
    │            │   │            │   │                │
    │ • Charts   │   │ • Bundle   │   │ • Levi-Civita │
    │ • Transition│   │ • BallTree │   │ • Schema      │
    │ • Builder  │   │ • Search   │   │ • Temporal    │
    └──────┬─────┘   └──────┬────┘   └──────┬────────┘
           │                │                │
           └────────┬───────┘                │
                    ▼                        │
           ┌─────────────────┐               │
           │  Metric Store   │◀──────────────┘
           │  (per-chart)    │
           └─────────────────┘
                    │
                    ▼
           ┌─────────────────┐
           │  Geodesic       │
           │  Solver         │
           └─────────────────┘
```

## Data Flows

### Insert Flow

```
  Raw Data Point
       │
       ▼
  ┌──────────────┐    1. Locate nearest chart via atlas
  │ Atlas Manager │───────┐
  └──────────────┘       │
                          ▼
  ┌──────────────┐    2. Find nearest anchor
  │ Tangent Bundle│
  │ (KDTree)     │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐    3. Project to tangent space
  │ Tangent Space │    (ambient → low-dim)
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐    4. Insert into BallTree
  │ Local Index   │    (fast low-dim NN search)
  └──────────────┘
```

### Query Flow (Geodesic)

```
  Query Point
       │
       ▼
  ┌──────────────┐    1. Locate chart
  │ Atlas Manager │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐    2. Project to tangent space
  │ Tangent Space │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐    3. Search candidates (oversample 3×)
  │ BallTree      │    (fast in low-dim tangent space)
  └──────┬───────┘
         │ candidate IDs + tangent distances
         ▼
  ┌──────────────┐    4. Refine with geodesic distance
  │ Geodesic      │    (energy minimisation)
  │ Solver        │
  └──────┬───────┘
         │ geodesic distances
         ▼
  ┌──────────────┐    5. Sort, truncate to top-k
  │ Result Assembly │
  └──────────────┘
```

### Cross-Modal Query Flow

```
  Query Point (modality A)
       │
       ▼
  ┌──────────────┐    1. Locate source chart (modality A)
  │ Atlas Manager │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐    2. Project to source tangent space
  │ Tangent Space │    (modality A)
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐    3. Parallel transport to target chart
  │ Connection    │    via overlap region
  │ (Transport)   │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐    4. Search in target tangent space
  │ Target Index  │    (modality B)
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐    5. Return cross-modal results
  │ Result Assembly │
  └──────────────┘
```

## Component Details

### Atlas Layer (`manifold_db/atlas/`)

| Component | File | Responsibility |
|-----------|------|---------------|
| `Chart` | `chart.py` | Local coordinate patch: embed/invert data |
| `TransitionMap` | `transition_map.py` | Diffeomorphisms between chart coordinates |
| `AtlasManager` | `atlas_manager.py` | CRUD for charts and transitions |
| `AtlasBuilder` | `atlas_builder.py` | Auto-discovery: PCA → KNN → Louvain → charts |

### Tangent Index (`manifold_db/tangent_index/`)

| Component | File | Responsibility |
|-----------|------|---------------|
| `TangentSpace` | `tangent_space.py` | Basis, metric, log/exp maps, transport |
| `TangentBundle` | `tangent_bundle.py` | Collection of TS with KDTree routing |
| `TangentSpaceIndex` | `index.py` | Per-anchor BallTrees for fast search |

### Query Layer (`manifold_db/query/`)

| Component | File | Responsibility |
|-----------|------|---------------|
| `QueryParser` | `dsl.py` | SQL-like string → ManifoldQuery AST |
| `QueryBuilder` | `dsl.py` | Fluent Python API → ManifoldQuery |
| `QueryEngine` | `engine.py` | Async execution with plan + cost estimation |
| `QueryResult` | `engine.py` | Typed results: list/dict/DataFrame |

### Metric Layer (`manifold_db/metric/`)

| Metric | Description |
|--------|-------------|
| `EuclideanMetric` | Flat δ_ij metric (zero curvature) |
| `DiagonalMetric` | Weighted diagonal with spatial variation |
| `LearnedMetric` | Torch MLP Cholesky factor (parametric) |
| `FisherRaoMetric` | Information-geometric metric |
| `WassersteinMetric` | Optimal transport metric (Sinkhorn) |

### Connection Layer (`manifold_db/connection/`)

| Component | Description |
|-----------|-------------|
| `LeviCivitaConnection` | Torsion-free, metric-compatible connection |
| `SchemaTransport` | Cross-schema-version transport |
| `TemporalTransport` | Time-evolution transport |
| `TransportRegistry` | LRU cache for transport functions |

## Design Decisions

### 1. Local PCA for Charts

**Decision:** Use local PCA (not global) for chart coordinate systems.

**Rationale:** Global PCA captures the dominant variance direction across the
entire dataset, but manifold structure is *local*.  A Swiss roll has global
variance along the roll axis, but locally the intrinsic dimension is 2D.
Local PCA correctly estimates this.

**Trade-off:** Local PCA is more expensive (O(N × k × D²) per chart) but
produces more accurate charts.

### 2. Louvain Community Detection

**Decision:** Use Louvain algorithm for chart boundary discovery.

**Rationale:** Chart boundaries should follow the natural topology of the
data.  Louvain maximises graph modularity, producing communities that align
with the manifold's connected components.

**Trade-off:** Louvain may produce too many or too few charts depending on
the resolution parameter.  We expose `n_charts_hint` with automatic tuning.

### 3. BallTree per Anchor

**Decision:** Use sklearn BallTree in tangent space (not ambient space).

**Rationale:** BallTrees in ambient space are expensive for high-D data
(curse of dimensionality).  In tangent space (low-D), BallTrees are very
efficient with excellent recall.

**Trade-off:** Requires building and maintaining per-anchor BallTrees,
adding memory overhead proportional to n_anchors.

### 4. SVD-based Parallel Transport

**Decision:** Use SVD-optimal rotation for parallel transport.

**Rationale:** The SVD of the transition matrix `B_target^T @ B_source`
gives the closest orthogonal matrix to the true transition, preserving
vector magnitudes and minimizing distortion.

**Trade-off:** SVD is O(d³) per transport.  For very high intrinsic
dimensions, this can be expensive.  The TransportRegistry cache mitigates
this by caching repeated transports.

### 5. Async Query Engine

**Decision:** Use `asyncio` for the query engine with stub collaborators.

**Rationale:** Query execution involves multiple I/O-bound steps (chart
lookup, index search, geodesic refinement).  Async enables concurrent
batch execution and non-blocking operation.

**Trade-off:** Adds complexity vs. synchronous execution.  In practice,
most users will use the synchronous TangentSpaceIndex API directly.

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Atlas build | O(N × k × D² + N × k log k) | KNN + PCA + Louvain |
| Index build | O(N × k × D × d) | FPS anchors + projection |
| Insert | O(k × D × d) | Find anchor + project |
| Search (tangent) | O(d × log(N/k)) | BallTree query |
| Search (geodesic) | O(k × N_wp × D) | Energy minimisation |
| Parallel transport | O(d³) | SVD of transition matrix |

Where N = total points, D = ambient dim, d = intrinsic dim, k = neighbours.

## Error Handling

- **Invalid queries**: `QueryBuilder.build()` raises `ValueError` with
  descriptive message.
- **Chart not found**: `AtlasManager.get_chart()` raises `KeyError`.
- **Dimension mismatch**: `Chart.embed()` validates input dimensions.
- **Missing metric**: `MetricTensorStore.get()` falls back to Euclidean.
- **Transport chain broken**: `TransportRegistry.compute_chain()` raises
  with the missing link identity.
