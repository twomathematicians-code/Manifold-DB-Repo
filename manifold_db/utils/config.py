"""
Configuration management for manifold database.

Provides a typed configuration hierarchy with sensible defaults, YAML/JSON
serialization, validation, and environment-variable overrides.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# ---------------------------------------------------------------------------
# Sub-configurations
# ---------------------------------------------------------------------------

@dataclass
class AtlasConfig:
    """Parameters governing atlas construction and management."""

    max_charts: int = 100
    min_chart_size: int = 50
    overlap_ratio: float = 0.3
    dim_est_method: str = "pca"


@dataclass
class IndexConfig:
    """Parameters for the tangent-space nearest-neighbor index."""

    n_anchors: int = 256
    leaf_size: int = 32
    metric_type: str = "euclidean"
    cache_size: int = 10_000


@dataclass
class GeodesicConfig:
    """Parameters for geodesic computation."""

    solver: str = "rk4"
    dt: float = 0.01
    max_steps: int = 10_000
    tolerance: float = 1e-8
    gpu_accelerated: bool = False


@dataclass
class MetricConfig:
    """Parameters for metric tensor computation and learning."""

    default_type: str = "euclidean"
    learned_hidden_dim: int = 64
    ricci_flow_dt: float = 0.001


@dataclass
class StorageConfig:
    """Parameters for persistent storage."""

    backend_type: str = "sqlite"
    path: str = "./manifold_db_data"
    cache_size_mb: int = 512
    wal_enabled: bool = True


@dataclass
class ConnectionConfig:
    """Parameters for the Levi-Civita connection and parallel transport."""

    transport_method: str = "schild"
    cache_transports: bool = True
    max_chain_length: int = 20


@dataclass
class QueryConfig:
    """Parameters for query execution."""

    default_k: int = 10
    max_results: int = 1000
    timeout_ms: int = 5000
    batch_size: int = 100


@dataclass
class ServerConfig:
    """Parameters for the network server (if used)."""

    host: str = "127.0.0.1"
    port: int = 8420
    workers: int = 4
    debug: bool = False


# ---------------------------------------------------------------------------
# Top-level configuration
# ---------------------------------------------------------------------------

@dataclass
class ManifoldConfig:
    """Root configuration container for the Manifold Database.

    Attributes
    ----------
    atlas : AtlasConfig
        Atlas construction parameters.
    index : IndexConfig
        Tangent-space index parameters.
    geodesic : GeodesicConfig
        Geodesic solver parameters.
    metric : MetricConfig
        Metric tensor parameters.
    storage : StorageConfig
        Storage backend parameters.
    connection : ConnectionConfig
        Levi-Civita connection parameters.
    query : QueryConfig
        Query engine parameters.
    server : ServerConfig
        Server parameters.
    """

    atlas: AtlasConfig = field(default_factory=AtlasConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    geodesic: GeodesicConfig = field(default_factory=GeodesicConfig)
    metric: MetricConfig = field(default_factory=MetricConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    connection: ConnectionConfig = field(default_factory=ConnectionConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


# ---------------------------------------------------------------------------
# Defaults, loading, saving, validation
# ---------------------------------------------------------------------------

def default_config() -> ManifoldConfig:
    """Return a configuration object populated with sensible defaults."""
    return ManifoldConfig()


def _apply_env_overrides(config: ManifoldConfig) -> ManifoldConfig:
    """Override configuration values from ``MANIFOLD_DB_*`` environment variables.

    Variable names follow the pattern ``MANIFOLD_DB_SECTION_KEY`` (e.g.
    ``MANIFOLD_DB_ATLAS_MAX_CHARTS=200``). Supported types: ``int``, ``float``,
    ``bool`` (``true/false/1/0``), ``str``.
    """
    prefix = "MANIFOLD_DB_"

    # Map of section names to sub-config dataclass instances
    sections: Dict[str, Any] = {
        "atlas": config.atlas,
        "index": config.index,
        "geodesic": config.geodesic,
        "metric": config.metric,
        "storage": config.storage,
        "connection": config.connection,
        "query": config.query,
        "server": config.server,
    }

    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        remainder = env_key[len(prefix):].lower()
        parts = remainder.split("_", 1)
        if len(parts) != 2:
            continue
        section_name, param_name = parts
        if section_name not in sections:
            continue
        section_obj = sections[section_name]
        if not hasattr(section_obj, param_name):
            continue

        # Type-coerce based on the current field type
        current_val = getattr(section_obj, param_name)
        coerced: Any = env_val
        if isinstance(current_val, bool):
            coerced = env_val.lower() in ("true", "1", "yes")
        elif isinstance(current_val, int):
            try:
                coerced = int(env_val)
            except ValueError:
                continue
        elif isinstance(current_val, float):
            try:
                coerced = float(env_val)
            except ValueError:
                continue

        setattr(section_obj, param_name, coerced)

    return config


def load_config(path: str | Path) -> ManifoldConfig:
    """Load configuration from a YAML or JSON file.

    Parameters
    ----------
    path : str or Path
        Path to the configuration file.

    Returns
    -------
    ManifoldConfig
        Parsed and validated configuration, with environment-variable
        overrides applied on top.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    raw_text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        data = yaml.safe_load(raw_text) or {}
    elif path.suffix == ".json":
        data = json.loads(raw_text)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")

    config = _dict_to_config(data)
    config = _apply_env_overrides(config)
    validate_config(config)
    return config


def save_config(config: ManifoldConfig, path: str | Path) -> None:
    """Save configuration to a YAML file.

    Parameters
    ----------
    config : ManifoldConfig
        Configuration to serialize.
    path : str or Path
        Destination path (should have ``.yaml`` or ``.yml`` extension).
    """
    path = Path(path)
    data = _config_to_dict(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False), encoding="utf-8")


def validate_config(config: ManifoldConfig) -> None:
    """Validate configuration values, raising ``ValueError`` on problems.

    Checks ranges, positivity constraints, and enum-like string fields.

    Parameters
    ----------
    config : ManifoldConfig

    Raises
    ------
    ValueError
        If any configuration value is invalid.
    """
    errors: list = []

    # Atlas
    if config.atlas.max_charts < 1:
        errors.append("atlas.max_charts must be >= 1")
    if config.atlas.min_chart_size < 1:
        errors.append("atlas.min_chart_size must be >= 1")
    if not (0.0 <= config.atlas.overlap_ratio < 1.0):
        errors.append("atlas.overlap_ratio must be in [0, 1)")
    if config.atlas.dim_est_method not in ("pca", "mle", "twonn"):
        errors.append("atlas.dim_est_method must be 'pca', 'mle', or 'twonn'")

    # Index
    if config.index.n_anchors < 1:
        errors.append("index.n_anchors must be >= 1")
    if config.index.leaf_size < 1:
        errors.append("index.leaf_size must be >= 1")
    if config.index.cache_size < 0:
        errors.append("index.cache_size must be >= 0")

    # Geodesic
    if config.geodesic.solver not in ("euler", "rk2", "rk4", "verlet"):
        errors.append("geodesic.solver must be 'euler', 'rk2', 'rk4', or 'verlet'")
    if config.geodesic.dt <= 0:
        errors.append("geodesic.dt must be > 0")
    if config.geodesic.max_steps < 1:
        errors.append("geodesic.max_steps must be >= 1")
    if config.geodesic.tolerance <= 0:
        errors.append("geodesic.tolerance must be > 0")

    # Metric
    if config.metric.learned_hidden_dim < 1:
        errors.append("metric.learned_hidden_dim must be >= 1")

    # Storage
    if config.storage.cache_size_mb < 0:
        errors.append("storage.cache_size_mb must be >= 0")

    # Connection
    if config.connection.transport_method not in ("schild", "parallel", "rolling"):
        errors.append("connection.transport_method must be 'schild', 'parallel', or 'rolling'")
    if config.connection.max_chain_length < 1:
        errors.append("connection.max_chain_length must be >= 1")

    # Query
    if config.query.default_k < 1:
        errors.append("query.default_k must be >= 1")
    if config.query.max_results < 1:
        errors.append("query.max_results must be >= 1")
    if config.query.timeout_ms < 1:
        errors.append("query.timeout_ms must be >= 1")
    if config.query.batch_size < 1:
        errors.append("query.batch_size must be >= 1")

    # Server
    if config.server.port < 1 or config.server.port > 65535:
        errors.append("server.port must be in [1, 65535]")
    if config.server.workers < 1:
        errors.append("server.workers must be >= 1")

    if errors:
        raise ValueError("Configuration validation failed:\n  - " + "\n  - ".join(errors))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _config_to_dict(config: ManifoldConfig) -> Dict[str, Any]:
    """Recursively convert a ManifoldConfig to a plain dict."""
    return asdict(config)


def _dict_to_config(data: Dict[str, Any]) -> ManifoldConfig:
    """Build a ManifoldConfig from a nested dict, ignoring unknown keys."""
    def _build_sub(sub_cls: type, sub_data: Optional[Dict[str, Any]]) -> Any:
        if not sub_data or not isinstance(sub_data, dict):
            return sub_cls()
        valid_keys = {f.name for f in sub_cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in sub_data.items() if k in valid_keys}
        return sub_cls(**filtered)

    data = data or {}
    return ManifoldConfig(
        atlas=_build_sub(AtlasConfig, data.get("atlas")),
        index=_build_sub(IndexConfig, data.get("index")),
        geodesic=_build_sub(GeodesicConfig, data.get("geodesic")),
        metric=_build_sub(MetricConfig, data.get("metric")),
        storage=_build_sub(StorageConfig, data.get("storage")),
        connection=_build_sub(ConnectionConfig, data.get("connection")),
        query=_build_sub(QueryConfig, data.get("query")),
        server=_build_sub(ServerConfig, data.get("server")),
    )
