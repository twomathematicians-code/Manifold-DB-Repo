"""
Utils sub-package for the Manifold Database.

Re-exports key utilities for convenient top-level access.
"""

from manifold_db.utils.config import (
    ManifoldConfig,
    AtlasConfig,
    IndexConfig,
    GeodesicConfig,
    MetricConfig,
    StorageConfig,
    ConnectionConfig,
    QueryConfig,
    ServerConfig,
    load_config,
    save_config,
    default_config,
    validate_config,
)

from manifold_db.utils.math_utils import (
    ensure_spd,
    cholesky_decomposition,
    matrix_square_root,
    log_det_spd,
    nearest_spd,
    safe_inverse,
    orthogonalize,
    random_spd_matrix,
    compute_numerical_gradient,
    curvature_bounds,
    volume_element,
    barycentric_projection,
)

from manifold_db.utils.manifold_learning import (
    estimate_intrinsic_dim,
    local_pca,
    diffusion_map,
    spectral_embedding,
    extract_local_patches,
    compute_residual_variance,
    manifold_quality_score,
)

from manifold_db.utils.logging import (
    setup_logging,
    get_logger,
    LogTimer,
    log_execution,
    PerformanceTracker,
)

__all__ = [
    # Config
    "ManifoldConfig",
    "AtlasConfig",
    "IndexConfig",
    "GeodesicConfig",
    "MetricConfig",
    "StorageConfig",
    "ConnectionConfig",
    "QueryConfig",
    "ServerConfig",
    "load_config",
    "save_config",
    "default_config",
    "validate_config",
    # Math
    "ensure_spd",
    "cholesky_decomposition",
    "matrix_square_root",
    "log_det_spd",
    "nearest_spd",
    "safe_inverse",
    "orthogonalize",
    "random_spd_matrix",
    "compute_numerical_gradient",
    "curvature_bounds",
    "volume_element",
    "barycentric_projection",
    # Manifold learning
    "estimate_intrinsic_dim",
    "local_pca",
    "diffusion_map",
    "spectral_embedding",
    "extract_local_patches",
    "compute_residual_variance",
    "manifold_quality_score",
    # Logging
    "setup_logging",
    "get_logger",
    "LogTimer",
    "log_execution",
    "PerformanceTracker",
]
