# Changelog

All notable changes to Manifold Database will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-XX-XX

### Added

#### Atlas Layer
- Atlas Manager with automatic chart discovery and overlap detection
- Atlas Builder with 6-stage pipeline (clustering → chart creation → dimensionality estimation → transition maps → quality analysis → registration)
- Chart and TransitionMap classes with full serialisation support
- Adaptive dimensionality estimation (PCA, correlation analysis, nearest-neighbour)

#### Tangent Space Index
- Tangent Space with SVD basis computation, project/lift, log/exp maps, and parallel transport
- Tangent Bundle with KDTree-accelerated nearest-anchor lookup and optimal tangent-space combination
- Tangent Space Index with farthest-point sampling, per-anchor BallTree indices, and multi-anchor search
- Online basis updates with QR re-orthogonalisation

#### Geodesic Query Engine
- Geodesic Solver with RK45 ODE integration and L-BFGS energy minimisation
- Riemannian distance, Wasserstein distance, and Fisher-Rao distance functions
- Exponential and logarithmic map computation with Christoffel symbol curvature correction
- Query DSL (SQL-like) with parser, builder, and engine
- Five query types: k-NN, geodesic ball, range, cross-modal, and temporal
- Explain plan generation with cost estimates

#### Riemannian Metric Store
- Six metric types: Euclidean, Diagonal, Learned (MLP), Fisher-Rao, Wasserstein, and custom
- Metric Tensor Store with per-chart registration and Euclidean fallback
- Online Ricci-flow metric updates
- Sectional, Ricci, and scalar curvature computation

#### Levi-Civita Connection
- Schild's ladder parallel transport with midpoint Christoffel evaluation
- Covariant derivative computation along arbitrary directions
- Cross-chart transport with Jacobian-based pushforward
- Batched transport for high-throughput operations
- Schema Transport for version migration with dimensionality adjustment
- Temporal Transport along time-varying metric paths
- Transport Registry with LRU cache, chain composition, and heatmap cost analysis

#### Storage Backends
- In-memory backend for fast prototyping
- File-based backend (JSON serialisation) for persistence
- SQLite backend for production workloads

#### Interfaces
- REST API (FastAPI) with 15+ endpoints, Pydantic models, middleware (logging, rate limiting, error handling)
- CLI (Typer + Rich) with 20+ commands, progress bars, tables, and coloured output
- Server script with graceful shutdown and startup banner

#### Infrastructure
- Docker support (multi-stage build, docker-compose production and dev stacks)
- CI/CD pipeline (GitHub Actions: lint, test matrix Python 3.10/3.11/3.12, integration, benchmark, Docker)
- Release pipeline (sdist/wheel build, GitHub release, Docker push, PyPI publish)
- Pre-commit hooks (ruff, black, mypy, isort, security checks)
- Makefile with 18 targets
- Example scripts (7 total: quickstart, multi-modal RAG, schema evolution, geodesic analysis, temporal manifold, benchmarks, tutorial walkthrough)
- Documentation (6 docs: architecture overview, atlas, query engine, getting started, advanced queries, API reference)
- MkDocs site with Material theme
