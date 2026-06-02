"""
CLI commands for configuration management.

Provides additional configuration-related commands that extend the
core CLI config group.  These are registered into the main CLI app
by importing this module in ``cli/main.py``.

Commands
--------
    config show              — display current configuration
    config set KEY VALUE     — set a configuration value
    config validate          — validate a configuration file
    config generate-defaults — write a default configuration file
    config list-sections     — list all configuration sections and keys
    config diff              — compare two configuration files
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

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
    _config_to_dict,
    default_config,
    load_config,
)

console = Console()

# ═══════════════════════════════════════════════════════════════
# Configuration introspection helpers
# ═══════════════════════════════════════════════════════════════

# Map section names to their dataclass types
SECTION_TYPES: dict[str, type] = {
    "atlas": AtlasConfig,
    "index": IndexConfig,
    "geodesic": GeodesicConfig,
    "metric": MetricConfig,
    "storage": StorageConfig,
    "connection": ConnectionConfig,
    "query": QueryConfig,
    "server": ServerConfig,
}

# Friendly descriptions for each section
SECTION_DESCRIPTIONS: dict[str, str] = {
    "atlas": "Atlas construction and management",
    "index": "Tangent-space nearest-neighbor index",
    "geodesic": "Geodesic computation and solver",
    "metric": "Metric tensor computation and learning",
    "storage": "Persistent storage backends",
    "connection": "Levi-Civita connection and parallel transport",
    "query": "Query execution parameters",
    "server": "Network server configuration",
}

# Field-level descriptions for common parameters
FIELD_DESCRIPTIONS: dict[str, str] = {
    "max_charts": "Maximum number of charts in the atlas",
    "min_chart_size": "Minimum data points per chart",
    "overlap_ratio": "Overlap fraction between adjacent charts",
    "dim_est_method": "Intrinsic dimension estimation method",
    "n_anchors": "Number of anchor points for tangent index",
    "leaf_size": "Leaf size for ball tree",
    "metric_type": "Default distance metric type",
    "cache_size": "LRU cache capacity",
    "solver": "Geodesic ODE integration method",
    "dt": "Integration time step",
    "max_steps": "Maximum integration steps",
    "tolerance": "Convergence tolerance",
    "gpu_accelerated": "Enable GPU acceleration",
    "default_type": "Default metric tensor type",
    "learned_hidden_dim": "MLP hidden dimension for learned metrics",
    "ricci_flow_dt": "Ricci flow evolution step",
    "backend_type": "Storage backend (memory, file, sqlite)",
    "path": "Base storage path",
    "cache_size_mb": "Cache size in megabytes",
    "wal_enabled": "Enable write-ahead logging",
    "transport_method": "Parallel transport algorithm",
    "cache_transports": "Cache transport computations",
    "max_chain_length": "Maximum transport chain length",
    "default_k": "Default number of nearest neighbors",
    "max_results": "Maximum query result size",
    "timeout_ms": "Query timeout in milliseconds",
    "batch_size": "Batch query size",
    "host": "Server bind address",
    "port": "Server bind port",
    "workers": "Number of uvicorn workers",
    "debug": "Enable debug mode",
}


def _get_field_info(section_name: str) -> list[tuple[str, str, Any, str]]:
    """Return (name, type, default, description) for each field in a section."""
    section_cls = SECTION_TYPES.get(section_name)
    if section_cls is None:
        return []
    info = []
    for f in fields(section_cls):
        desc = FIELD_DESCRIPTIONS.get(f.name, "")
        info.append(
            (
                f.name,
                f.type.__name__ if hasattr(f.type, "__name__") else str(f.type),
                f.default,
                desc,
            )
        )
    return info


def _load_cfg(path: str | None) -> ManifoldConfig:
    """Load configuration, falling back to defaults."""
    if path:
        try:
            return load_config(path)
        except (FileNotFoundError, ValueError) as exc:
            console.print(f"[red]Error loading config from {path}: {exc}[/red]")
            raise typer.Exit(1)
    return default_config()


# ═══════════════════════════════════════════════════════════════
# Commands (usable as standalone helpers or registered into main app)
# ═══════════════════════════════════════════════════════════════


def cmd_config_list_sections(
    config: str | None = None,
) -> None:
    """List all configuration sections with their parameters.

    This function can be used standalone or registered as a Typer command.
    """
    cfg = _load_cfg(config)

    tree = Tree("[bold]Manifold Database Configuration[/bold]")
    for section_name in SECTION_TYPES:
        desc = SECTION_DESCRIPTIONS.get(section_name, "")
        branch = tree.add(f"[cyan]{section_name}[/cyan] — {desc}")
        section_obj = getattr(cfg, section_name)
        for name, type_name, default, field_desc in _get_field_info(section_name):
            current = getattr(section_obj, name)
            label = f"[bold]{name}[/bold]"
            value_str = f"[green]{current!r}[/green]"
            branch.add(f"{label} ({type_name}) = {value_str}  [dim]{field_desc}[/dim]")

    console.print(tree)


def cmd_config_diff(
    file_a: str,
    file_b: str,
) -> None:
    """Compare two configuration files and display differences.

    Parameters
    ----------
    file_a : str
        Path to the first configuration file.
    file_b : str
        Path to the second configuration file.
    """
    cfg_a = _load_cfg(file_a)
    cfg_b = _load_cfg(file_b)
    data_a = _config_to_dict(cfg_a)
    data_b = _config_to_dict(cfg_b)

    # Find differences recursively
    diffs: list[str] = []

    def _compare(path: str, val_a: Any, val_b: Any) -> None:
        if type(val_a) is not type(val_b):
            diffs.append(
                f"{path}: type changed from {type(val_a).__name__} to {type(val_b).__name__}"
            )
        elif isinstance(val_a, dict):
            for key in set(list(val_a.keys()) + list(val_b.keys())):
                _compare(f"{path}.{key}", val_a.get(key), val_b.get(key))
        elif val_a != val_b:
            diffs.append(f"{path}: [red]{val_a!r}[/red] → [green]{val_b!r}[/green]")

    for section_name in SECTION_TYPES:
        _compare(
            section_name, data_a.get(section_name, {}), data_b.get(section_name, {})
        )

    if not diffs:
        console.print("[green]Configurations are identical.[/green]")
        return

    table = Table(title=f"Config Diff: [cyan]{file_a}[/cyan] vs [cyan]{file_b}[/cyan]")
    table.add_column("Path", style="bold")
    table.add_column("Difference")
    for diff in diffs:
        path, _, change = diff.partition(": ")
        table.add_row(path, change)

    console.print(table)
    console.print(f"\n[dim]{len(diffs)} difference(s) found.[/dim]")


def cmd_config_export(
    config: str | None = None,
    format: str = "yaml",
    output: str | None = None,
) -> None:
    """Export configuration in various formats.

    Parameters
    ----------
    config : str or None
        Path to config file. Uses defaults if not provided.
    format : str
        Output format: 'yaml', 'json', or 'env'.
    output : str or None
        Output path. Prints to stdout if not provided.
    """
    cfg = _load_cfg(config)
    data = _config_to_dict(cfg)

    if format == "json":
        content = json.dumps(data, indent=2)
    elif format == "yaml":
        import yaml

        content = yaml.dump(data, default_flow_style=False, sort_keys=False)
    elif format == "env":
        lines = []
        for section_name, section_data in data.items():
            if isinstance(section_data, dict):
                for key, val in section_data.items():
                    env_key = f"MANIFOLD_DB_{section_name.upper()}_{key.upper()}"
                    lines.append(f"{env_key}={val!r}")
        content = "\n".join(lines)
    else:
        console.print(f"[red]Unsupported format: {format}[/red]")
        console.print("Supported formats: yaml, json, env")
        raise typer.Exit(1)

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(content, encoding="utf-8")
        console.print(f"[green]Configuration exported to {output}[/green]")
    else:
        console.print(content)


# ═══════════════════════════════════════════════════════════════
# Registration helper — call from cli/main.py to add these commands
# ═══════════════════════════════════════════════════════════════


def register_extra_config_commands(config_app: typer.Typer) -> None:
    """Register additional config commands into the existing config app group.

    Call this from ``cli/main.py`` after creating the ``config_app``:

        from manifold_db.cli.config_commands import register_extra_config_commands
        register_extra_config_commands(config_app)
    """

    @config_app.command("list-sections")
    def _list_sections(
        config: str | None = typer.Option(
            None,
            "--config",
            "-c",
            help="Path to config file.",
        ),
    ):
        """List all configuration sections and their parameters."""
        cmd_config_list_sections(config)

    @config_app.command("diff")
    def _diff(
        file_a: str = typer.Argument(..., help="First config file."),
        file_b: str = typer.Argument(..., help="Second config file."),
    ):
        """Compare two configuration files and show differences."""
        cmd_config_diff(file_a, file_b)

    @config_app.command("export")
    def _export(
        format: str = typer.Option(
            "yaml",
            "--format",
            "-f",
            help="Output format (yaml, json, env).",
        ),
        output: str | None = typer.Option(
            None,
            "--output",
            "-o",
            help="Output file path.",
        ),
        config: str | None = typer.Option(
            None,
            "--config",
            "-c",
            help="Path to config file.",
        ),
    ):
        """Export configuration in a specified format."""
        cmd_config_export(config, format, output)
