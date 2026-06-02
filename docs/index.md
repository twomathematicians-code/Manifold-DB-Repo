# Manifold Database — Documentation

> A geometric inference engine for data on Riemannian manifolds.

## Overview

Manifold Database is a Python library that treats high-dimensional data as
samples from a low-dimensional Riemannian manifold.  Instead of using flat
Euclidean geometry for storage and retrieval, it builds an **atlas** of
overlapping **charts** with **transition maps**, computes **tangent spaces**
at data points, and uses **geodesic distances** and **parallel transport**
for queries.

### Why Manifold Geometry?

Most real-world high-dimensional data (text embeddings, image features,
genomic profiles, sensor readings) lies on or near a low-dimensional
manifold embedded in the ambient space.  Traditional databases use
Euclidean distance, which ignores this structure.  Manifold Database:

- **Discovers intrinsic dimensionality** automatically via local PCA
- **Partitions data into charts** using community detection on KNN graphs
- **Computes geodesic distances** that follow the manifold surface
- **Enables cross-modal retrieval** via parallel transport between charts
- **Handles schema evolution** as smooth geometric deformations

## Quick Start

```python
from manifold_db.atlas import AtlasBuilder
from manifold_db.tangent_index import TangentSpaceIndex
from manifold_db.query import QueryBuilder
import numpy as np

# 1. Build atlas from data
builder = AtlasBuilder()
atlas = builder.build(data)

# 2. Build index
index = TangentSpaceIndex(intrinsic_dim=8)
index.build_from_data(ids, data, n_anchors=30)

# 3. Search
results = index.search(query_point, k=10)
```

## Architecture

The system consists of six core modules:

| Module | Path | Description |
|--------|------|-------------|
| **Atlas** | `manifold_db/atlas/` | Charts, transition maps, atlas builder |
| **Tangent Index** | `manifold_db/tangent_index/` | Tangent spaces, bundle, search index |
| **Geodesic** | `manifold_db/geodesic/` | Geodesic distances, exponential maps |
| **Metric** | `manifold_db/metric/` | Riemannian metric tensors, curvature |
| **Connection** | `manifold_db/connection/` | Parallel transport, schema/temporal transport |
| **Query** | `manifold_db/query/` | DSL parser, query engine, result types |

See [Architecture Overview](architecture/overview.md) for detailed diagrams.

## Documentation Index

### Tutorials
- [Getting Started](tutorials/getting-started.md) — Installation, configuration, first queries
- [Advanced Queries](tutorials/advanced-queries.md) — Geodesic, cross-modal, schema evolution

### Architecture
- [Architecture Overview](architecture/overview.md) — System design, data flow, component interaction
- [Atlas Layer](architecture/atlas.md) — Charts, transition maps, dimensionality estimation
- [Query Engine](architecture/query-engine.md) — DSL grammar, execution strategies, cost model

### API Reference
- [API Reference](api/api-reference.md) — All public classes, methods, and configuration

### Examples
- `examples/scripts/quickstart.py` — Get started in 5 minutes
- `examples/scripts/multi_modal_rag.py` — Text-image cross-modal retrieval
- `examples/scripts/schema_evolution.py` — Smooth schema migration
- `examples/scripts/geodesic_analysis.py` — Scientific computing on energy surfaces
- `examples/scripts/temporal_manifold.py` — Time-series with concept drift
- `examples/scripts/benchmark_suite.py` — Performance benchmarks
- `examples/scripts/tutorial_walkthrough.py` — Step-by-step concept tutorial

## Key Concepts

### Charts and Atlases

A **chart** `(U, φ)` maps a patch of the manifold to Euclidean space.
An **atlas** is a collection of overlapping charts that covers the manifold.

### Tangent Spaces

At each point on the manifold, the **tangent space** provides a local
flat (Euclidean) approximation.  Operations like nearest-neighbour search
are much faster in tangent space than in ambient space.

### Geodesic Distances

A **geodesic** is the generalisation of a straight line to curved manifolds.
Geodesic distance follows the manifold surface, unlike Euclidean distance
which may "cut through" empty space.

### Parallel Transport

**Parallel transport** moves a vector from one tangent space to another
while preserving its geometric properties.  This enables cross-chart and
cross-modal retrieval.

## Installation

```bash
pip install manifold-db
```

Dependencies: `numpy`, `scipy`, `scikit-learn`, `networkx`.

Optional: `torch` (for `LearnedMetric`).

## License

MIT License.
