"""
Utils sub-package for the Manifold Database.

Re-exports key utilities for convenient top-level access.
"""

from manifold_db.utils.config import (
    AtlasConfig,
    ConnectionConfig,
    GeodesicConfig,
    IndexConfig,
    ManifoldConfig,
    MetricConfig,
    QueryConfig,
    ServerConfig,
    StorageConfig,
    default_config,
    load_config,
    save_config,
    validate_config,
)
from manifold_db.utils.logging import (
    PerformanceTracker,
    get_logger,
    log_execution,
    log_timer,
    setup_logging,
)
from manifold_db.utils.manifold_learning import (
    compute_residual_variance,
    diffusion_map,
    estimate_intrinsic_dim,
    extract_local_patches,
    local_pca,
    manifold_quality_score,
    spectral_embedding,
)
from manifold_db.utils.math_utils import (
    barycentric_projection,
    cholesky_decomposition,
    compute_numerical_gradient,
    curvature_bounds,
    ensure_spd,
    log_det_spd,
    matrix_square_root,
    nearest_spd,
    orthogonalize,
    random_spd_matrix,
    safe_inverse,
    volume_element,
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
    "log_timer",
    "log_execution",
    "PerformanceTracker",
]
