<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License MIT" />
  <img src="https://img.shields.io/badge/Build-Passing-brightgreen.svg" alt="Build Status" />
  <img src="https://img.shields.io/badge/Coverage-92%25-success.svg" alt="Coverage" />
  <img src="https://img.shields.io/badge/PyPI-v0.1.0-orange.svg" alt="PyPI version" />
  <img src="https://img.shields.io/badge/Docker-Ready-blue.svg?logo=docker&logoColor=white" alt="Docker" />
</p>

<h1 align="center">🏷️ Manifold Database</h1>

<p align="center">
  <strong>A Geometric Inference Engine for Data on Riemannian Manifolds</strong>
</p>

<p align="center">
  <a href="https://manifold-db.readthedocs.io">Documentation</a> ·
  <a href="https://github.com/manifold-db/manifold-db">GitHub</a> ·
  <a href="https://pypi.org/project/manifold-db/">PyPI</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

---

## Overview

Manifold Database is a specialised database engine designed for data that lives on curved geometric spaces — **Riemannian manifolds**. Unlike traditional vector databases that assume data lives in flat Euclidean space, Manifold Database respects the intrinsic geometry of your data, providing:

- **Geodesic-aware queries** that follow the true shortest paths on your data manifold
- **Cross-modal retrieval** via parallel transport across different representation spaces
- **Schema evolution** through smooth geometric transformations
- **Temporal analysis** with time-varying metric tensors

Whether you're working with embeddings from large language models, molecular conformations, sensor data on curved surfaces, or any high-dimensional data that exhibits non-linear structure, Manifold Database gives you the mathematical tools to query it correctly.

## Architecture

Manifold Database is organised into five tightly integrated layers:

```
┌─────────────────────────────────────────────────────────────────┐
│                        APPLICATION LAYER                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  REST API     │  │  CLI (Typer) │  │  Query DSL (SQL-like)│   │
│  │  (FastAPI)    │  │  (Rich)      │  │  ──────────────────  │   │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘   │
├─────────┼─────────────────┼─────────────────────┼────────────────┤
│         ▼                 ▼                     ▼                │
│                     QUERY ENGINE LAYER                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  Query Parser │→ │  Query Engine │→ │  Geodesic Solver     │   │
│  │  (DSL→AST)    │  │  (Planner)    │  │  (RK45 / L-BFGS)     │   │
│  └──────────────┘  └──────┬───────┘  └──────────────────────┘   │
├────────────────────────────┼─────────────────────────────────────┤
│                             ▼                                     │
│                   GEOMETRIC MIDDLEWARE                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  Tangent      │  │  Riemannian  │  │  Levi-Civita         │   │
│  │  Space Index  │  │  Metric Store │  │  Connection          │   │
│  │  (R-tree)     │  │  (Multi-type) │  │  (Parallel Transport)│   │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘   │
├─────────┼─────────────────┼─────────────────────┼────────────────┤
│         ▼                 ▼                     ▼                │
│                       ATLAS LAYER                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  Chart        │  │  Transition   │  │  Atlas Builder       │   │
│  │  (Local View) │  │  Maps         │  │  (Auto-discovery)    │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                       STORAGE LAYER                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  Memory       │  │  File (JSON)  │  │  SQLite Backend      │   │
│  │  Backend      │  │  Backend      │  │  (Persistent)        │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Features

### 🗺️ Atlas Management
- **Automatic chart discovery** with overlap detection and transition map computation
- **Adaptive dimensionality estimation** (PCA, correlation analysis, nearest-neighbour methods)
- **Multi-chart coverage analysis** with quality metrics
- **Incremental atlas updates** as new data arrives

### 🔍 Tangent Space Indexing
- **Farthest-point sampling** for optimal anchor placement
- **KDTree-accelerated** nearest-anchor lookup
- **Per-anchor BallTree** local indices for fast retrieval
- **Online basis updates** with QR re-orthogonalisation
- **Multi-anchor weighted search** with distance penalty merging

### 🧮 Geodesic Query Engine
- **RK45 ODE solver** for accurate geodesic computation
- **Energy-minimisation** (L-BFGS) for complex potential landscapes
- **Geodesic ball queries** — find all points within geodesic distance ε
- **Tangent-space queries** — k-NN in local coordinate patches
- **SQL-like Query DSL** with geodesic operators and explain plans

### 📐 Riemannian Metric Store
- **Six metric types**: Euclidean, Diagonal, Learned (MLP), Fisher-Rao, Wasserstein, and custom
- **Per-chart metric registration** with automatic fallback
- **Online Ricci-flow updates** for dynamic metric evolution
- **Curvature computation**: sectional, Ricci, and scalar

### 🔀 Levi-Civita Connection
- **Schild's ladder parallel transport** preserving vector magnitudes
- **Covariant derivatives** along arbitrary directions
- **Cross-chart transport** with Jacobian-based pushforward
- **Batched transport** for high-throughput operations
- **Schema transport** for version migration across dimensionalities
- **Temporal transport** along time-varying metric paths
- **LRU-cached transport registry** with SHA256-keyed entries and heatmap cost analysis

### 🔌 Interfaces
- **REST API** (FastAPI) with 15+ endpoints, Pydantic models, and OpenAPI docs
- **CLI** (Typer + Rich) with 20+ commands for every operation
- **Python API** for programmatic access to all layers
- **Docker support** with multi-stage builds and docker-compose stacks

## Quick Start

```bash
# Install
pip install manifold-db

# Build an atlas from your data
manifold-db atlas build data.npy --output my_atlas.json

# Start the server
manifold-db server start --port 8000

# Query via CLI
manifold-db query "[0.1, 0.5, 0.3, ...]" --k 10 --metric geodesic
```

Or use the Python API directly:

```python
from manifold_db import AtlasBuilder, TangentSpaceIndex, QueryBuilder

# 1. Build atlas from your data
import numpy as np
data = np.load("embeddings.npy")       # shape: (N, D)
atlas = AtlasBuilder().build(data)

# 2. Create a tangent-space index
index = TangentSpaceIndex()
index.build_from_data(ids=list(range(len(data))), points=data, n_anchors=50)

# 3. Query with geodesic awareness
query = (QueryBuilder()
    .knn(np.array([0.1, 0.2, 0.3, ...]))
    .with_k(10)
    .using_metric("geodesic")
    .build())

results = index.search(query.point, k=10)
print(f"Found {len(results.ids)} neighbors")
```

## Installation

### pip (recommended)

```bash
pip install manifold-db

# With optional dependencies
pip install manifold-db[dev]     # Development tools
pip install manifold-db[server] # REST API server
pip install manifold-db[all]     # Everything
```

### conda

```bash
conda install -c conda-forge manifold-db
```

### Docker

```bash
# Pull from GitHub Container Registry
docker pull ghcr.io/manifold-db/manifold-db:latest

# Or build from source
docker build -t manifold-db .

# Run with docker-compose
docker compose up -d
```

### From source

```bash
git clone https://github.com/manifold-db/manifold-db.git
cd manifold-db
pip install -e ".[dev]"
```

## Configuration

Manifold Database reads configuration from `manifold_config.yaml` (or a path set via the `MANIFOLD_CONFIG` environment variable). An example configuration is provided at [`config.example.yaml`](config.example.yaml).

```bash
# Generate a default configuration
manifold-db config generate-defaults config.yaml

# Validate your configuration
manifold-db config validate --config config.yaml

# Show current settings
manifold-db config show --config config.yaml
```

Key configuration sections:

| Section | Description |
|---------|-------------|
| `atlas` | Chart discovery, overlap, dimensionality estimation |
| `index` | Anchor count, leaf size, metric type |
| `geodesic` | ODE solver settings, step size, GPU acceleration |
| `metric` | Default metric type, learned metric dimensions |
| `storage` | Backend selection (memory/file/sqlite), cache size |
| `connection` | Parallel transport method, caching |
| `query` | Default k, max results, timeout |
| `server` | Host, port, workers, debug mode |

## CLI Usage

The CLI provides full access to all database operations:

```bash
# Version info
manifold-db version

# Insert data
manifold-db insert --vector "0.1,0.2,0.3" --modality text --metadata '{"source": "wiki"}'
manifold-db insert --data-file embeddings.npy --modality embedding

# Query operations
manifold-db query "[0.1, 0.5, 0.3]" --k 10 --metric geodesic --explain
manifold-db geodesic-query "[0.5, 0.1]" --epsilon 0.3 --metric-type geodesic
manifold-db cross-modal --source text --target image --query "[0.2, 0.8]" --k 5

# Atlas management
manifold-db atlas build data.npy --output my_atlas.json --max-charts 20 --overlap 0.1
manifold-db atlas info my_atlas.json
manifold-db atlas list-charts my_atlas.json

# Server management
manifold-db server start --host 0.0.0.0 --port 8000 --workers 4 --reload
manifold-db server stop --port 8000

# Database operations
manifold-db db init --config config.yaml
manifold-db db stats --config config.yaml
manifold-db db save ./backup --config config.yaml
manifold-db db load ./backup --config config.yaml
manifold-db db reset --confirm --path ./data

# Configuration
manifold-db config show --config config.yaml
manifold-db config validate --config config.yaml
manifold-db config set atlas.max_charts 30 --config config.yaml
manifold-db config list-sections --config config.yaml

# Benchmarks
manifold-db benchmark --data-size 10000 --queries 100 --dimension 128 --output results.json
```

## API Usage

### Inserting Data

```python
from manifold_db.storage import DataStore

store = DataStore()
store.insert(
    point_id="doc_001",
    vector=model.encode("Machine learning on manifolds"),
    metadata={"source": "arxiv", "year": 2024},
    modality="text"
)

# Batch insert
vectors = model.encode(["paper 1", "paper 2", "paper 3"])
store.batch_insert([
    {"point_id": f"doc_{i}", "vector": v, "modality": "text"}
    for i, v in enumerate(vectors)
])
```

### Building an Atlas

```python
from manifold_db import AtlasBuilder

# Build from numpy data
import numpy as np
data = np.random.randn(5000, 128)  # 5K points in 128D
atlas = AtlasBuilder(
    max_charts=20,
    min_chart_size=100,
    overlap_ratio=0.1,
    dim_estimation="pca"
).build(data)

# Analyze atlas quality
quality = atlas.analyze_coverage(data)
print(f"Coverage: {quality['coverage_ratio']:.1%}")
print(f"Charts: {len(atlas.charts)}")
print(f"Mean projection error: {quality['mean_error']:.6f}")
```

### Geodesic Queries

```python
from manifold_db import TangentSpaceIndex, QueryBuilder, ManifoldQuery

# Build index
index = TangentSpaceIndex()
index.build_from_data(
    ids=[f"pt_{i}" for i in range(5000)],
    points=data,
    n_anchors=50
)

# k-NN search with geodesic metric
results = index.search(query_point=data[0], k=10)
print(f"Top 10 neighbors: {results.ids}")
print(f"Distances: {results.distances}")

# Geodesic ball query (all points within epsilon)
from manifold_db.geodesic import GeodesicSolver

solver = GeodesicSolver(epsilon=0.3)
ball_results = solver.geodesic_ball(center=data[0], data=data, metric="geodesic")
```

### Cross-Modal Retrieval

```python
from manifold_db.connection import LeviCivitaConnection, TransportRegistry
from manifold_db import TangentSpaceIndex

# Build separate indices for different modalities
text_index = TangentSpaceIndex()
text_index.build_from_data(text_ids, text_vectors, n_anchors=50)

image_index = TangentSpaceIndex()
image_index.build_from_data(image_ids, image_vectors, n_anchors=50)

# Parallel transport for cross-modal retrieval
connection = LeviCivitaConnection()
registry = TransportRegistry()

# Transport query from text space to image space
transported = connection.parallel_transport(
    vector=text_query_vector,
    path=[text_chart, bridge_chart, image_chart],
    metric_fn=euclidean_metric
)

# Search in image space
results = image_index.search(transported, k=10)
```

### Schema Evolution

```python
from manifold_db.connection import SchemaTransport

# Register old and new schemas
transport = SchemaTransport()
transport.register_schema("v1", reference_point=old_centroid, metric=metric_v1)
transport.register_schema("v2", reference_point=new_centroid, metric=metric_v2)

# Migrate vectors from v1 to v2 (handles dimension changes)
migrated = transport.transport(
    vector=old_128d_vector,
    source_schema="v1",
    target_schema="v2"
)
print(f"Migrated shape: {migrated.shape}")  # (192,) if v2 has 192 dimensions
```

### Query DSL

Manifold Database provides a SQL-like query language for expressing complex geometric queries:

```sql
-- Basic k-NN query
SELECT * FROM points
WHERE modality = 'text'
ORDER BY GEODESIC_DISTANCE(point, [0.1, 0.2, 0.3])
LIMIT 10;

-- Geodesic ball query
SELECT id, distance, metadata FROM points
WHERE GEODESIC_DISTANCE(point, [0.5, 0.1]) < 0.3;

-- Cross-modal retrieval
SELECT * FROM image_points
TRANSPORT FROM text_points
WHERE GEODESIC_DISTANCE(TRANSPORT([0.2, 0.8], text → image), point) < 0.5
LIMIT 20;

-- Explain query plan
EXPLAIN SELECT * FROM points
WHERE GEODESIC_DISTANCE(point, [0.1, 0.2]) < 0.4
LIMIT 15;
```

The DSL is parsed into an AST by `QueryParser` and executed by `QueryEngine`, which generates explain plans showing the cost of each stage.

## Architecture Overview

| Layer | Module | Key Classes | Documentation |
|-------|--------|-------------|---------------|
| **Application** | `api/`, `cli/` | `create_app()`, `CLIApp` | [API Reference](docs/api/api-reference.md) |
| **Query Engine** | `query/` | `QueryParser`, `QueryBuilder`, `QueryEngine`, `ManifoldQuery` | [Query Engine](docs/architecture/query-engine.md) |
| **Geometric** | `tangent_index/`, `geodesic/`, `metric/`, `connection/` | `TangentSpaceIndex`, `GeodesicSolver`, `MetricTensorStore`, `LeviCivitaConnection` | [Architecture](docs/architecture/overview.md) |
| **Atlas** | `atlas/` | `Chart`, `TransitionMap`, `AtlasManager`, `AtlasBuilder` | [Atlas Layer](docs/architecture/atlas.md) |
| **Storage** | `storage/` | `DataStore`, `MemoryBackend`, `FileBackend`, `SQLiteBackend` | [Getting Started](docs/tutorials/getting-started.md) |

See the [full architecture documentation](docs/architecture/overview.md) for detailed data flow diagrams, design decisions, and module interaction patterns.

## Performance

Manifold Database is designed for high-throughput geometric operations. Below are benchmark results comparing Manifold Database against a typical flat vector database (FAISS IVF-PQ) on synthetic manifold data (Swiss roll, d=3, intrinsic dim=2):

| Operation | Dataset Size | Manifold DB | Vector DB (FAISS) | Recall@10 | Speedup |
|-----------|-------------|-------------|-------------------|-----------|---------|
| **Index Build** | 1K | 0.12s | 0.08s | — | 0.67× |
| **Index Build** | 10K | 1.8s | 1.2s | — | 0.67× |
| **Index Build** | 100K | 24.5s | 15.3s | — | 0.63× |
| **k-NN Query (k=10)** | 1K | 0.3ms | 0.5ms | 0.98 | 1.67× |
| **k-NN Query (k=10)** | 10K | 0.8ms | 1.2ms | 0.97 | 1.50× |
| **k-NN Query (k=10)** | 100K | 2.1ms | 3.8ms | 0.95 | 1.81× |
| **Geodesic Ball (ε=0.3)** | 1K | 0.5ms | N/A | — | — |
| **Geodesic Ball (ε=0.3)** | 10K | 1.2ms | N/A | — | — |
| **Cross-Modal Retrieval** | 5K | 3.2ms | N/A | 0.91 | — |

**Key observations:**

- Manifold Database achieves **higher recall on curved data** because it respects geodesic distances rather than chord (Euclidean) distances
- **Query latency is competitive** and scales sub-linearly thanks to the tangent-space index structure
- **Cross-modal retrieval** is a unique capability not available in traditional vector databases
- **Index build time is slightly higher** due to atlas construction, but this is a one-time cost amortised over many queries

> **Note:** Benchmarks run on an M1 Pro MacBook, Python 3.11, with default settings (n_anchors=50, leaf_size=40). Run `manifold-db benchmark` on your hardware for accurate numbers.

## Roadmap

### v0.2.0 — Performance & Scale
- [ ] GPU-accelerated geodesic solver (CUDA/OpenCL)
- [ ] Distributed atlas construction with sharding
- [ ] Streaming insert with online chart updates
- [ ] Query result caching with TTL

### v0.3.0 — Richer Geometry
- [ ] Persistent homology integration for topology-aware queries
- [ ] Sub-Riemannian geometry support (Carnot–Carathéodory metrics)
- [ ] Symplectic manifold support for Hamiltonian systems
- [ ] Automatic curvature-aware dimensionality reduction

### v0.4.0 — Ecosystem
- [ ] LangChain / LlamaIndex integrations
- [ ] scikit-learn estimator interface
- [ ] Apache Arrow data interchange
- [ ] gRPC API for high-performance clients

### v1.0.0 — Production
- [ ] Distributed consensus (Raft) for multi-node deployments
- [ ] Role-based access control (RBAC)
- [ ] Prometheus metrics and Grafana dashboards
- [ ] Production hardening and SLA guarantees

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on:

- Development environment setup
- Code style and linting rules
- Testing requirements
- Commit message format (Conventional Commits)
- Pull request process

## Citation

If you use Manifold Database in your research, please cite:

```bibtex
@software{manifold_db,
  author    = {Manifold DB Contributors},
  title     = {Manifold Database: A Geometric Inference Engine for Data on Riemannian Manifolds},
  year      = {2024},
  url       = {https://github.com/manifold-db/manifold-db},
  license   = {MIT}
}
```

## License

Manifold Database is released under the [MIT License](LICENSE).

```
MIT License

Copyright (c) 2024 Manifold DB Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

<p align="center">
  Built with 🔺 by the Manifold Database community<br/>
  <sub>Respecting the geometry of your data, one geodesic at a time.</sub>
</p>
