<p align="center">
  <img src="https://img.shields.io/badge/build-passing-brightgreen" alt="Build Status" />
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License: MIT" />
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/C%2B%2B-20-blue" alt="C++20" />
  <img src="https://img.shields.io/badge/pypi-v0.1.0-orange" alt="PyPI" />
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen" alt="PRs Welcome" />
</p>

<h1 align="center">ManifoldDB</h1>

<p align="center">
  <strong>A Geometric Inference Engine for Riemannian Data</strong>
</p>

<p align="center">
  <a href="#what-is-manifolddb">What is ManifoldDB?</a> &bull;
  <a href="#key-features">Features</a> &bull;
  <a href="#architecture">Architecture</a> &bull;
  <a href="#installation">Installation</a> &bull;
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#api-reference">API Reference</a> &bull;
  <a href="#mathematical-background">Mathematics</a> &bull;
  <a href="#applications">Applications</a>
</p>

---

## Table of Contents

- [What is ManifoldDB?](#what-is-manifolddb)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
  - [From PyPI](#from-pypi)
  - [From Source (CMake)](#from-source-cmake)
  - [Dependencies](#dependencies)
- [Quick Start](#quick-start)
  - [Basic Workflow](#basic-workflow)
  - [Geodesic k-NN](#geodesic-k-nn)
  - [Cross-Modal Retrieval](#cross-modal-retrieval)
  - [Schema Evolution](#schema-evolution)
  - [PyTorch Integration](#pytorch-integration)
- [API Reference](#api-reference)
  - [ManifoldDB](#manifolddb-class)
  - [Chart Classes](#chart-classes)
  - [MetricTensor](#metrictensor)
  - [GeodesicSolver](#geodesicsolver)
  - [TangentSpaceIndex](#tangentspaceindex)
  - [Atlas](#atlas)
  - [torch_compat Utilities](#torch_compat-utilities)
  - [IO Utilities](#io-utilities)
- [Mathematical Background](#mathematical-background)
  - [Riemannian Manifolds](#riemannian-manifolds)
  - [Charts and Atlas](#charts-and-atlas)
  - [Metric Tensor](#metric-tensor)
  - [Geodesics](#geodesics)
  - [Parallel Transport](#parallel-transport)
  - [Exponential and Logarithmic Maps](#exponential-and-logarithmic-maps)
- [Applications](#applications)
- [Project Structure](#project-structure)
- [MVP Roadmap](#mvp-roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## What is ManifoldDB?

ManifoldDB is a **geometric inference engine** that treats high-dimensional data as if it lives on a **Riemannian manifold** rather than in flat Euclidean space. Instead of using straight-line Euclidean distances for nearest-neighbor search, clustering, and retrieval, ManifoldDB computes **geodesic distances** — the true shortest paths along the curved surface of the data manifold.

### The Core Idea

In many real-world datasets — image embeddings, molecular descriptors, sensor readings, language model vectors — the data is **not uniformly distributed** through the ambient space. Instead, it concentrates on or near a low-dimensional manifold **M** embedded in a high-dimensional ambient space **R^D**. Classic nearest-neighbor methods ignore this structure and measure distances through the ambient space, which can yield semantically meaningless results when the manifold "folds back" on itself.

ManifoldDB addresses this by combining three pillars:

1. **Manifold Learning**: Automatically decompose the dataset into an atlas of local coordinate charts via PCA and clustering. Each chart provides a low-dimensional (d ≪ D) local coordinate system that faithfully represents the manifold geometry in a neighbourhood.

2. **Riemannian Geometry**: Learn a position-dependent Riemannian metric tensor **g_ij(x)** from the data. This metric encodes how distances stretch and curve across the manifold, enabling geodesic equations to be solved for true shortest-path distances.

3. **Database Indexing**: Build spatial indexes (R-trees) in the tangent space of each chart, then re-rank candidates using exact geodesic distances. This provides the query performance of flat-space indexing with the geometric fidelity of Riemannian distance computation.

### Why Does This Matter?

Consider a Swiss Roll embedded in R³: two points on opposite layers of the roll may be close in Euclidean distance but far apart along the manifold surface. A standard k-NN search would retrieve the "wrong" neighbours — the closest point through the ambient space, not along the manifold. ManifoldDB correctly navigates the manifold's intrinsic geometry to find semantically meaningful neighbours.

This principle extends to any domain where data has geometric structure: trajectory data in robotics, embedding spaces in machine learning, conformational landscapes in computational chemistry, and more.

---

## Key Features

### Geodesic k-NN and Ball Queries 🔍
- Replace flat Euclidean nearest-neighbor search with **geodesic distance queries**
- `query_knn()` finds the *k* closest points along the manifold surface
- `query_ball()` retrieves all points within a geodesic radius
- Two-phase strategy: R-tree candidate generation → geodesic re-ranking

### Cross-Modal Retrieval 🔗
- Search across different data modalities (e.g., text → image) using a shared atlas
- Both modalities are mapped to a common chart space, enabling geodesic transport
- `cross_modal_query()` bridges embedding spaces with geometric fidelity

### Riemannian Metric Learning 📐
- Position-dependent metric tensor field **g_ij(x)** with RBF interpolation
- Online metric refinement as new data arrives
- Full Christoffel symbol computation (1st and 2nd kind) via central differences
- Sectional and scalar curvature estimation

### Schema Evolution via Parallel Transport 🔄
- Extend the manifold structure when new data points arrive
- Parallel transport preserves geometric consistency across chart boundaries
- Incremental atlas reconstruction without full recomputation

### GPU-Accelerated Geodesic Solver ⚡
- CUDA-enabled RK4 geodesic integration (optional, compile-time flag)
- Dormand-Prince RK45 adaptive step-size control for efficiency
- Symplectic (Störmer-Verlet) integrator for energy-preserving long-range geodesics
- Newton shooting method for boundary value problems

### PyTorch Native Tensor Interop 🔥
- Zero-copy interop between PyTorch tensors and the C++ Eigen backend via DLPack
- `torch_to_eigen()` / `eigen_to_torch()` conversion utilities
- Batch geodesic distance computation directly on torch tensors
- `insert()` accepts `torch.Tensor` of shape `(N, D)`

### Thread-Safe Persistent Storage 💾
- Metric tensors persisted to binary `.bin` files per chart
- R-tree spatial indexes serialised with bulk-load on restore
- Reader-writer locking (`std::shared_mutex`) for concurrent access
- JSON export/import for charts and transition maps
- Optional HDF5 export for metric tensors

---

## Architecture

ManifoldDB uses a three-tier architecture: a high-level Python API, a C++ geodesic computation engine, and a persistent storage layer.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Python API Layer                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ ManifoldDB   │  │ torch_compat │  │    io.py     │  │  Python users  │  │
│  │ (high-level) │  │ (tensor I/O) │  │ (persist/    │  │  & scripts    │  │
│  │              │  │              │  │  export)     │  │               │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └───────┬───────┘  │
│         │                 │                  │                  │          │
├─────────┼─────────────────┼──────────────────┼──────────────────┼──────────┤
│         ▼                 ▼                  ▼                  ▼          │
│                    PyBind11 Bindings (bindings.cpp)                        │
│              torch::Tensor ↔ Eigen::MatrixXd ↔ numpy.ndarray              │
├─────────────────────────────────────────────────────────────────────────────┤
│                         C++ Geodesic Engine                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │    Atlas     │  │ MetricStore  │  │GeodesicSolver│  │ TangentSpace │  │
│  │  (charts +   │  │ (g_ij field) │  │ (IVP/BVP/    │  │  Index       │  │
│  │ transitions) │  │              │  │  transport)   │  │  (R-tree)    │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                  │                  │          │
├─────────┼─────────────────┼──────────────────┼──────────────────┼──────────┤
│         ▼                 ▼                  ▼                  ▼          │
│                         Storage Layer                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                     │
│  │  {db_path}/  │  │  metric_N.bin│  │ index_N.bin  │  JSON / HDF5     │
│  │  metrics/    │  │  (per-chart) │  │ (per-chart)  │  (export only)    │
│  └──────────────┘  └──────────────┘  └──────────────┘                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Overview

| Component | Description |
|-----------|-------------|
| **ManifoldDB** | Top-level orchestrator: insert data, build atlas, execute geodesic queries |
| **Atlas** | Collection of charts with transition maps; handles chart location and coordinate transport |
| **Chart** | Local coordinate system (U, φ) on the manifold; provides embed/project/Jacobian |
| **LinearChart** | PCA-based affine patch: φ(x) = origin + B·x |
| **NeuralChart** | ONNX-backed non-linear chart (optional dependency) |
| **MetricTensor** | Riemannian metric g_ij(x) with RBF interpolation or constant mode |
| **MetricStore** | Thread-safe cache + persistent storage for per-chart MetricTensors |
| **GeodesicSolver** | ODE integrator for geodesic equations (RK4, RK45, symplectic, shooting) |
| **TangentSpaceIndex** | R-tree spatial index for fast candidate generation in local coordinates |

---

## Installation

### From PyPI

```bash
pip install manifolddb
```

### From Source (CMake)

Prerequisites: Python 3.10+, C++20 compiler (GCC 11+, Clang 14+, MSVC 2022+), CMake 3.18+, PyTorch, Eigen3, PyBind11.

```bash
# Clone the repository
git clone https://github.com/manifolddb/manifolddb.git
cd manifolddb

# Build the C++ extension
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Install the Python package
cd ..
pip install -e .

# Verify installation
python -c "import manifolddb; print(manifolddb.__version__)"
```

#### Optional: CUDA Support

```bash
cmake .. -DMANIFOLDDB_ENABLE_CUDA=ON
make -j$(nproc)
```

### Dependencies

| Package | Version | Required | Notes |
|---------|---------|----------|-------|
| Python | ≥ 3.9 | Yes | 3.10+ recommended |
| PyTorch | ≥ 1.12 | Yes | For tensor interop |
| NumPy | ≥ 1.21 | Yes | Array bridge to Eigen |
| Eigen3 | ≥ 3.4 | Yes | Linear algebra backend |
| PyBind11 | ≥ 2.10 | Yes | Python/C++ bindings |
| CMake | ≥ 3.18 | Yes (build) | Build system |
| C++ compiler | C++20 | Yes (build) | GCC 11+, Clang 14+ |
| ONNX Runtime | latest | No | NeuralChart support |
| h5py | latest | No | HDF5 metric export |
| scikit-learn | ≥ 1.0 | No | Optional utilities |

---

## Quick Start

### Basic Workflow

The simplest way to use ManifoldDB: insert data, build the atlas, and query by geodesic distance.

```python
import numpy as np
import manifolddb

# Create a database with intrinsic dimension 16
db = manifolddb.ManifoldDB("./my_db", intrinsic_dim=16)

# Insert 10,000 ambient-space points (768-dimensional, e.g., CLIP embeddings)
data = np.random.randn(10000, 768).astype(np.float64)
db.insert(data)

# Build the atlas (PCA-based chart decomposition)
db.build(method="linear")

# Geodesic k-nearest-neighbour query
query = np.random.randn(768).astype(np.float64)
results = db.query_knn(query, k=10)

for r in results:
    print(f"  id={r['id']}  geo_dist={r['distance']:.4f}  "
          f"chart={r['chart_id']}  euc_residual={r['euclidean_residual']:.4f}")
```

### Geodesic k-NN

ManifoldDB's geodesic k-NN differs from Euclidean k-NN: it finds the *k* closest points along the manifold surface rather than through the ambient space.

```python
import numpy as np
import manifolddb

# Generate data on a torus (2D manifold in 3D ambient space)
theta = np.random.uniform(0, 2 * np.pi, 1000)
phi   = np.random.uniform(0, 2 * np.pi, 1000)
R, r  = 2.0, 0.5
x = (R + r * np.cos(theta)) * np.cos(phi)
y = (R + r * np.cos(theta)) * np.sin(phi)
z = r * np.sin(theta)
data = np.column_stack([x, y, z]).astype(np.float64)

db = manifolddb.ManifoldDB("./torus_db", intrinsic_dim=2)
db.insert(data)
db.build()

# Query: find 5 geodesically nearest neighbours
results = db.query_knn(data[0], k=5)
for r in results:
    print(f"Neighbour id={r['id']}, geodesic_dist={r['distance']:.6f}")
```

### Cross-Modal Retrieval

ManifoldDB enables search across different data modalities by building a shared atlas:

```python
import numpy as np
import manifolddb

db = manifolddb.ManifoldDB("./multimodal_db", intrinsic_dim=8)

# Insert text embeddings (modality 0)
text_embeddings = np.random.randn(5000, 128).astype(np.float64)
db.insert(text_embeddings, modality_id=0)

# Insert image embeddings (modality 1)
image_embeddings = np.random.randn(3000, 128).astype(np.float64)
db.insert(image_embeddings, modality_id=1)

# Build a unified atlas covering both modalities
db.build(method="linear")

# Cross-modal query: text query → image results
query_text = text_embeddings[42]
results = db.cross_modal_query(
    query_text,
    source_modality=0,
    target_modality=1,
    k=10,
)

for r in results:
    print(f"  image_id={r['id']}  dist={r['distance']:.4f}")
```

### Schema Evolution

Extend the manifold with new data points — ManifoldDB handles incremental atlas reconstruction:

```python
import numpy as np
import manifolddb

db = manifolddb.ManifoldDB("./evolving_db", intrinsic_dim=4)
db.insert(np.random.randn(500, 32).astype(np.float64))
db.build()

# Later: new data arrives
new_data = np.random.randn(200, 32).astype(np.float64)
db.evolve(new_data)

# The atlas has been rebuilt to incorporate the new points
print(db.stats())
```

### PyTorch Integration

ManifoldDB works natively with PyTorch tensors:

```python
import torch
import manifolddb

db = manifolddb.ManifoldDB("./torch_db", intrinsic_dim=16)

# Insert PyTorch tensors directly (no manual numpy conversion)
embeddings = torch.randn(5000, 768, dtype=torch.float64)
db.insert(embeddings)
db.build()

# Query with a torch tensor
query = torch.randn(768, dtype=torch.float64)
results = db.query_knn(query, k=10)

# Convert results back to torch tensors
from manifolddb import eigen_to_torch
distances = torch.tensor([r['distance'] for r in results])
```

---

## API Reference

### ManifoldDB Class

The top-level class for manifold-aware data management.

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(storage_path, intrinsic_dim, enable_cuda, geodesic_tolerance, max_charts, rbf_bandwidth)` | Create a new ManifoldDB instance |
| `insert` | `(points, modality_id=0)` | Insert data points (numpy array or torch tensor) |
| `build` | `(method='linear')` | Build atlas from inserted data |
| `build_atlas_linear` | `(intrinsic_dim=None)` | Build PCA-based linear charts |
| `query_knn` | `(query, k=10, max_distance=inf)` | K-nearest neighbours by geodesic distance |
| `query_ball` | `(center, radius)` | All points within a geodesic ball |
| `cross_modal_query` | `(query, source_modality, target_modality, k=10)` | Cross-modal geodesic search |
| `geodesic_path` | `(start, end, tolerance=1e-6)` | Compute geodesic path between two points |
| `evolve` | `(new_data)` | Extend manifold with new data points |
| `stats` | `()` | Database statistics (dict) |
| `core` | property | Direct access to C++ engine |
| `atlas` | property | Access the underlying Atlas |
| `metric_store` | property | Access the underlying MetricStore |
| `solver` | property | Access the underlying GeodesicSolver |
| `num_charts` | property | Number of charts in the atlas |
| `total_points` | property | Total points across all modalities |

### Chart Classes

| Class | Description |
|-------|-------------|
| `Chart` | Abstract base: embed, project, jacobian, exponential_map, log_map |
| `LinearChart` | PCA affine patch: φ(x) = origin + B·x |
| `ParametricChart` | User-supplied callback chart |
| `NeuralChart` | ONNX-backed neural network chart (optional) |

### MetricTensor

| Method | Signature | Description |
|--------|-----------|-------------|
| `evaluate` | `(local_coords)` | Evaluate g_ij(x) → d×d SPD matrix |
| `inverse` | `(local_coords)` | Evaluate g^{ij}(x) |
| `christoffel_symbols` | `(local_coords, h=1e-5)` | Γ^k_{ij} via finite differences |
| `sectional_curvature` | `(u, v)` | Sectional curvature K(u, v) |
| `scalar_curvature` | `(local_coords, h=1e-5)` | Scalar curvature S = g^{ij} R_{ij} |
| `update` | `(local_coords, local_metric, weight=1.0)` | Add RBF anchor point |
| `set_constant` | `(metric)` | Set constant metric |
| `set_identity` | `()` | Reset to identity |
| `serialize` / `deserialize` | Binary format | Persistence |

### GeodesicSolver

| Method | Signature | Description |
|--------|-----------|-------------|
| `solve_ivp` | `(start, initial_velocity, t_max, method)` | Solve geodesic IVP |
| `solve_bvp` | `(start, end, method)` | Solve geodesic BVP (shooting) |
| `parallel_transport` | `(path, vector_at_start)` | Levi-Civita parallel transport |
| `geodesic_distance` | `(p, q)` | Geodesic distance d_g(p, q) |
| `batch_geodesic_distance` | `(chart_id, query, candidates)` | Batch distances for k-NN reranking |

### TangentSpaceIndex

| Method | Signature | Description |
|--------|-----------|-------------|
| `build` | `(points)` | Bulk build via STR packing |
| `knn` | `(query_local, k, max_radius)` | K-NN in local coordinates |
| `range_search` | `(query_local, radius)` | Ball query in local coordinates |
| `insert` | `(point)` | Single point insertion |
| `save` / `load` | `(path)` | Binary persistence |

### Atlas

| Method | Signature | Description |
|--------|-----------|-------------|
| `add_chart` | `(chart)` | Register a chart |
| `add_transition` | `(transition)` | Register a transition map |
| `locate_chart` | `(ambient_coords)` | Find best chart for a point |
| `transport` | `(point, target_chart_id)` | Multi-hop coordinate transport |
| `discover_charts_linear` | `(data, dim, num_charts, threshold)` | Auto-discover PCA charts |
| `find_path` | `(from_id, to_id)` | BFS shortest multi-hop path |

### torch_compat Utilities

| Function | Description |
|----------|-------------|
| `ensure_float64(tensor)` | Ensure float64 and CPU |
| `torch_to_eigen(tensor)` | torch.Tensor → contiguous float64 numpy |
| `eigen_to_torch(array)` | numpy → torch.Tensor (cloned) |
| `eigen_to_numpy(array)` | numpy → owned float64 copy |
| `dlpack_export(db, chart_id)` | Export as DLPack (future) |
| `batch_geodesic_distances(query, candidates, ms)` | Batch geodesic distances |

### IO Utilities

| Function | Description |
|----------|-------------|
| `save_manifold(db, path)` | Persist full database state |
| `load_manifold(path)` | Restore database from disk |
| `export_charts_to_json(db, path)` | Export atlas as JSON |
| `import_charts_from_json(path)` | Import atlas from JSON |
| `export_metrics_to_hdf5(db, path)` | Export metrics to HDF5 |
| `import_metrics_from_hdf5(path, db)` | Import metrics from HDF5 |

---

## Mathematical Background

ManifoldDB builds on classical differential geometry. This section provides a concise reference for the key mathematical concepts used throughout the system.

### Riemannian Manifolds

A **Riemannian manifold** (M, g) is a smooth manifold M equipped with a Riemannian metric tensor g. At each point p ∈ M, the metric defines an inner product on the tangent space T_pM:

```
⟨u, v⟩_g = u^i g_{ij}(p) v^j
```

This inner product induces a norm ||v||_g = √(v^i g_{ij} v^j) and a distance function — the **geodesic distance** — which is the infimum of arc lengths of all curves connecting two points:

```
d_g(p, q) = inf ∫₀¹ √(g_{ij}(γ(t)) γ̇^i(t) γ̇^j(t)) dt
```

ManifoldDB assumes data lies on or near a d-dimensional manifold M embedded in D-dimensional Euclidean space R^D (typically D ≫ d). The embedding provides the ambient coordinates y ∈ R^D for each point, while the charts provide local coordinates x ∈ R^d.

### Charts and Atlas

A **chart** (U, φ) is a local coordinate system on the manifold:

```
φ : U ⊂ R^d  →  M ⊂ R^D    (embedding: local → ambient)
φ⁻¹ : M ⊂ R^D →  U ⊂ R^d  (projection: ambient → local)
```

An **atlas** is a collection of charts {(U_α, φ_α)} that covers M. For overlapping charts α and β, the **transition map** converts coordinates between them:

```
ψ_{α→β} = φ_β⁻¹ ∘ φ_α : U_α ∩ U_β → R^d
```

The **Jacobian** (pushforward) of a chart φ at x gives the d×D matrix of partial derivatives:

```
J_ij(x) = ∂y^i / ∂x^j
```

### Metric Tensor

The **induced metric** on a chart is obtained by pulling back the ambient Euclidean metric:

```
g_ij(x) = J(x)^T J(x) = (∂y/∂x)^T (∂y/∂x)
```

This yields a d×d symmetric positive-definite (SPD) matrix at each point. The **Christoffel symbols of the second kind** are derived from the metric:

```
Γ^k_{ij} = ½ g^{kl} (∂g_{li}/∂x^j + ∂g_{lj}/∂x^i - ∂g_{ij}/∂x^l)
```

These encode the Levi-Civita connection — the unique torsion-free, metric-compatible connection on M.

### Geodesics

**Geodesics** are curves that locally minimise arc length. They satisfy the **geodesic equation**:

```
d²x^k/dt² + Γ^k_{ij}(x) (dx^i/dt)(dx^j/dt) = 0
```

This is a second-order ODE system. ManifoldDB solves it using:

- **RK4**: Classical 4th-order Runge-Kutta (fixed step)
- **RK45**: Adaptive Dormand-Prince (variable step, error control)
- **Symplectic**: Störmer-Verlet (energy-preserving)
- **Shooting**: Newton's method for boundary value problems (find geodesic connecting two given points)

### Parallel Transport

**Parallel transport** moves a tangent vector along a curve while preserving the inner product. The Levi-Civita transport equation is:

```
Dv^i/dt = −Γ^i_{jk} v^j (dx^k/dt) = 0
```

ManifoldDB uses parallel transport to move tangent vectors between charts, enabling cross-chart queries and schema evolution.

### Exponential and Logarithmic Maps

The **exponential map** exp_p : T_pM → M maps a tangent vector v at point p to the point reached by following the geodesic in direction v for unit time:

```
exp_p(v) = γ(1)   where γ(0) = p, γ'(0) = v
```

The **logarithmic map** log_p : M → T_pM is the inverse: it finds the tangent vector v such that exp_p(v) = q. ManifoldDB computes these via RK4 integration (exponential) and Newton's method (logarithmic).

---

## Applications

ManifoldDB is designed for any domain where data has intrinsic geometric structure:

| Domain | Use Case |
|--------|----------|
| **Machine Learning** | Geodesic-aware nearest-neighbor search in embedding spaces (CLIP, BERT, SentenceTransformers) |
| **Robotics** | Trajectory retrieval on configuration-space manifolds; motion planning on SE(3) |
| **Scientific Computing** | Molecular conformation search; potential energy surface navigation |
| **Multi-Modal Retrieval** | Cross-modal search (text↔image↔audio) with geometric consistency |
| **Computational Arts** | Style interpolation along geodesic paths; generative model latent space navigation |
| **Bioinformatics** | Protein structure analysis; phylogenetic distance computation |
| **Computer Vision** | Image manifold navigation; face recognition on non-linear face manifolds |
| **Natural Language Processing** | Semantic search in sentence embedding spaces with curved geometry |

---

## Project Structure

```
manifolddb/
├── CMakeLists.txt              # CMake build configuration
├── pyproject.toml              # Python project metadata
├── setup.py                    # Setuptools build script
├── README.md                   # This file
├── LICENSE                     # MIT License
├── CONTRIBUTING.md             # Contribution guidelines
│
├── cpp/
│   ├── include/manifold/
│   │   ├── manifold_types.hpp   # Core types (Scalar, Vector, Matrix, ManifoldPoint)
│   │   ├── chart.hpp            # Abstract Chart base + NeuralChart + ParametricChart
│   │   ├── linear_chart.hpp     # LinearChart: φ(x) = origin + B·x
│   │   ├── atlas.hpp            # Atlas: chart collection + transition maps + BFS transport
│   │   ├── metric_tensor.hpp    # MetricTensor: g_ij(x) with RBF interpolation
│   │   ├── metric_store.hpp     # MetricStore: thread-safe cache + persistence
│   │   ├── geodesic_solver.hpp   # GeodesicSolver: IVP/BVP/transport (RK4/RK45/Symplectic)
│   │   ├── tangent_space_index.hpp # TangentSpaceIndex: R-tree over local coordinates
│   │   └── manifold_db.hpp      # ManifoldDB: top-level API orchestrator
│   └── src/
│       ├── bindings.cpp         # PyBind11 bindings (Python ↔ C++ bridge)
│       ├── manifold_db.cpp      # ManifoldDB implementation
│       ├── atlas.cpp            # Atlas implementation
│       ├── chart.cpp            # Chart implementations
│       ├── linear_chart.cpp     # LinearChart implementation
│       ├── metric_tensor.cpp    # MetricTensor implementation
│       ├── metric_store.cpp     # MetricStore implementation
│       ├── geodesic_solver.cpp  # GeodesicSolver implementation
│       └── tangent_space_index.cpp # TangentSpaceIndex implementation
│
├── python/manifolddb/
│   ├── __init__.py             # High-level Python wrappers (ManifoldDB, ManifoldPoint, etc.)
│   ├── torch_compat.py         # PyTorch ↔ Eigen interop utilities
│   └── io.py                   # Persistence: save/load, JSON, HDF5 export
│
├── tests/
│   ├── conftest.py             # Pytest fixtures
│   ├── test_manifolddb.py      # End-to-end ManifoldDB tests
│   ├── test_manifold_db.py     # Core database tests
│   ├── test_atlas.py           # Atlas and transition map tests
│   ├── test_chart.py           # Chart (linear, parametric) tests
│   ├── test_metric_tensor.py   # Metric tensor and curvature tests
│   ├── test_geodesic.py        # Geodesic path and distance tests
│   └── test_geodesic_solver.py # Solver method tests (RK4, RK45, shooting)
│
├── examples/
│   ├── __init__.py
│   ├── basic_usage.py          # Torus example: insertion, build, geodesic k-NN
│   ├── geodesic_demo.py        # Swiss Roll: geodesic paths, k-NN comparison
│   ├── cross_modal_demo.py     # Cross-modal retrieval (text ↔ image)
│   ├── multimodal_demo.py      # Multi-modal atlas construction
│   └── persistence_demo.py     # Save/load database state
│
└── docs/
    ├── ARCHITECTURE.md         # Detailed architecture document
    └── API.md                  # Full API reference
```

---

## MVP Roadmap

| Phase | Feature | Status | Description |
|-------|---------|--------|-------------|
| 1 | Core Types & Charts | ✅ Implemented | ManifoldPoint, Chart, LinearChart, ParametricChart |
| 2 | Atlas & Transitions | ✅ Implemented | Chart collection, LinearTransitionMap, BFS path finding, PCA discovery |
| 3 | Metric Tensor | ✅ Implemented | Constant + RBF interpolation, Christoffel symbols, curvature |
| 4 | Geodesic Solver | ✅ Implemented | RK4, RK45 (Dormand-Prince), symplectic, Newton shooting BVP |
| 5 | Tangent Space Index | ✅ Implemented | R-tree with STR packing, k-NN, range search, persistence |
| 6 | Top-Level DB | ✅ Implemented | ManifoldDB: insert, build, geodesic k-NN, ball query, cross-modal |
| 7 | Python Bindings | ✅ Implemented | PyBind11 with Eigen + torch::Tensor support |
| 8 | PyTorch Interop | ✅ Implemented | torch_to_eigen, eigen_to_torch, batch_geodesic_distances |
| 9 | Persistence | ✅ Implemented | MetricStore binary, index save/load, JSON/HDF5 export |
| 10 | Schema Evolution | ✅ Implemented | evolve_schema with parallel transport and atlas rebuild |
| 11 | Neural Charts | 🔜 Planned | ONNX Runtime integration for non-linear chart embeddings |
| 12 | CUDA Geodesics | 🔜 Planned | GPU-accelerated RK4 integration kernels |
| 13 | TileDB Backend | 🔜 Planned | Replace file-based storage with TileDB array database |
| 14 | Learned Metrics | 🔜 Planned | Neural network metric tensor fields (end-to-end differentiable) |
| 15 | Distributed Atlas | 🔜 Planned | Sharded atlas across multiple nodes with geodesic routing |
| 16 | WebAssembly | 🔜 Planned | WASM build for browser-based geodesic queries |

---

## Contributing

We welcome contributions to ManifoldDB! Here's how to get started:

### Development Setup

```bash
# Clone and build
git clone https://github.com/manifolddb/manifolddb.git
cd manifolddb
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
cd .. && pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check python/
mypy python/manifolddb/
```

### Contribution Guidelines

1. **Fork** the repository and create a feature branch
2. **Write tests** for any new functionality (aim for >90% coverage)
3. **Document** all public APIs with NumPy-style docstrings
4. **Follow** C++ (clang-format) and Python (ruff) code style
5. **Submit** a pull request with a clear description of changes

### Areas We'd Love Help With

- **CUDA kernels** for GPU-accelerated geodesic integration
- **TileDB** storage backend integration
- **Benchmarks** against FAISS, Annoy, and HNSW on manifold datasets
- **Jupyter notebooks** with interactive examples
- **Documentation** improvements and tutorials
- **Community examples** in robotics, chemistry, and NLP

---

## License

ManifoldDB is released under the **MIT License**. See the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 ManifoldDB Contributors

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

## Acknowledgements

ManifoldDB builds on ideas from several fields:

- **Differential Geometry**: Riemannian geometry, geodesic equations, Levi-Civita connection (do Carmo, Lee)
- **Manifold Learning**: ISOMAP, LLE, UMAP, diffusion maps for non-linear dimensionality reduction
- **Spatial Indexing**: R-tree, R*-tree, STR packing for efficient spatial queries (Guttman, Leutenegger)
- **Numerical ODEs**: Runge-Kutta methods, Dormand-Prince adaptive stepping, Störmer-Verlet symplectic integration
- **PyTorch + Eigen**: PyBind11 for seamless Python/C++ tensor interop

Special thanks to the open-source communities behind Eigen, PyTorch, PyBind11, and NumPy.
