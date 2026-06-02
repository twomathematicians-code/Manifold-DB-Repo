"""
CLI for Manifold Database using Typer + Rich.

Provides a comprehensive command-line interface for all database operations:

    manifold-db insert          — insert data points
    manifold-db query           — execute queries
    manifold-db geodesic-query  — geodesic ball queries
    manifold-db cross-modal      — cross-modal retrieval
    manifold-db atlas            — atlas management
    manifold-db server           — start/stop the REST API server
    manifold-db db               — database init, stats, save, load, reset
    manifold-db config           — configuration management
    manifold-db benchmark       — performance benchmarks
    manifold-db version         — print version

Run ``manifold-db --help`` for usage overview.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import typer
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.tree import Tree

from manifold_db.atlas.atlas_manager import AtlasManager
from manifold_db.query.dsl import ManifoldQuery, MetricType, QueryType
from manifold_db.query.engine import QueryEngine, QueryResult
from manifold_db.storage.backend import StorageManager
from manifold_db.storage.data_store import DataPoint, DataStore
from manifold_db.utils.config import (
    ManifoldConfig,
    _config_to_dict,
    default_config,
    load_config,
    save_config,
    validate_config,
)

__VERSION__ = "0.1.0"

# ── Rich console ───────────────────────────────────────────────
console = Console()

# ── Typer app ───────────────────────────────────────────────────
app = typer.Typer(
    name="manifold-db",
    help="Manifold Database — a Riemannian manifold-structured database with geodesic queries.",
    no_args_is_help=True,
    rich_markup_mode="rich",
    add_completion=True,
)

# ── Sub-apps ───────────────────────────────────────────────────
atlas_app = typer.Typer(
    name="atlas",
    help="Atlas management commands.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
app.add_typer(atlas_app, name="atlas")

server_app = typer.Typer(
    name="server",
    help="Server control commands.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
app.add_typer(server_app, name="server")

db_app = typer.Typer(
    name="db",
    help="Database management commands.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
app.add_typer(db_app, name="db")

config_app = typer.Typer(
    name="config",
    help="Configuration management commands.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
app.add_typer(config_app, name="config")


# ═══════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════


def _parse_vector(vec_str: str) -> np.ndarray:
    """Parse a string representation of a vector into a numpy array."""
    try:
        # Try JSON parse
        vec = json.loads(vec_str)
        if isinstance(vec, (list, tuple)):
            return np.asarray(vec, dtype=np.float64)
    except json.JSONDecodeError:
        pass
    # Try comma/space separated
    try:
        vec = [
            float(x.strip())
            for x in vec_str.strip("[]()").replace(",", " ").split()
            if x.strip()
        ]
        return np.asarray(vec, dtype=np.float64)
    except ValueError:
        raise typer.BadParameter(f"Cannot parse vector from: {vec_str}")


def _parse_metadata(meta_str: str | None) -> dict[str, Any]:
    """Parse a JSON metadata string into a dict."""
    if not meta_str:
        return {}
    try:
        return json.loads(meta_str)
    except json.JSONDecodeError:
        raise typer.BadParameter(f"Cannot parse metadata JSON from: {meta_str}")


def _load_data_file(path: str) -> np.ndarray:
    """Load a data file (.npy, .csv, .json) into a numpy array."""
    p = Path(path)
    if not p.exists():
        raise typer.BadParameter(f"File not found: {path}")

    suffix = p.suffix.lower()
    if suffix == ".npy":
        return np.load(str(p))
    elif suffix == ".csv":
        import csv as csv_mod

        rows = []
        with open(p) as f:
            reader = csv_mod.reader(f)
            for row in reader:
                rows.append([float(x) for x in row])
        return np.array(rows, dtype=np.float64)
    elif suffix == ".json":
        with open(p) as f:
            data = json.load(f)
        if isinstance(data, list):
            return np.array(data, dtype=np.float64)
        raise typer.BadParameter(f"Expected a list of vectors in JSON file: {path}")
    else:
        raise typer.BadParameter(
            f"Unsupported file format: {suffix}. Use .npy, .csv, or .json"
        )


def _init_components(config: ManifoldConfig | None = None) -> tuple:
    """Initialise and return (data_store, atlas_manager, query_engine)."""
    if config is None:
        config = default_config()

    storage_mgr = StorageManager.create(
        backend_type=config.storage.backend_type,
        config={"base_path": config.storage.path},
    )
    data_store = DataStore(backend=storage_mgr)
    atlas_mgr = AtlasManager(name="default_atlas")
    query_engine = QueryEngine(atlas_manager=atlas_mgr)
    return data_store, atlas_mgr, query_engine


def _format_result_table(result: QueryResult, max_rows: int = 20) -> Table:
    """Format a QueryResult as a Rich table."""
    table = Table(
        title=f"Query Results ({len(result)} points, {result.execution_time*1000:.2f}ms)",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Point ID", justify="right")
    table.add_column("Distance", justify="right")
    table.add_column("Metadata", style="dim")

    n = min(len(result), max_rows)
    for i in range(n):
        row = result[i]
        meta_str = json.dumps(row.get("metadata", {})) if row.get("metadata") else ""
        table.add_row(
            str(i),
            str(row["point_id"]),
            f"{row['distance']:.6f}",
            meta_str[:60],
        )

    if len(result) > max_rows:
        table.add_row("...", f"({len(result) - max_rows} more rows)", "", "")

    return table


# ═══════════════════════════════════════════════════════════════
# Version command
# ═══════════════════════════════════════════════════════════════


@app.command()
def version():
    """Print the Manifold Database version and exit."""
    panel = Panel(
        f"[bold]Manifold Database[/bold] v{__VERSION__}\n"
        f"Python {sys.version.split()[0]}",
        title="Version Info",
        border_style="green",
    )
    console.print(panel)


# ═══════════════════════════════════════════════════════════════
# Insert commands
# ═══════════════════════════════════════════════════════════════


@app.command()
def insert(
    data_file: str | None = typer.Option(
        None,
        "--data-file",
        "-f",
        help="Path to .npy/.csv/.json data file with vectors.",
    ),
    vector: str | None = typer.Option(
        None,
        "--vector",
        "-v",
        help="Single vector as JSON array, e.g. '[0.1, 0.2, 0.3]'",
    ),
    modality: str = typer.Option(
        "default",
        "--modality",
        "-m",
        help="Modality tag for the data (text, image, audio, etc.).",
    ),
    metadata: str | None = typer.Option(
        None,
        "--metadata",
        help='JSON metadata string, e.g. \'{"key": "value"}\'',
    ),
    chart_id: str | None = typer.Option(
        None,
        "--chart",
        "-c",
        help="Chart to assign the data to.",
    ),
):
    """Insert data points into the manifold database.

    Use --data-file to insert all vectors from a file, or --vector
    to insert a single point from the command line.
    """
    meta = _parse_metadata(metadata)

    if data_file:
        data = _load_data_file(data_file)
        console.print(f"[cyan]Loading {data.shape[0]} vectors from {data_file}[/cyan]")

        async def _do_batch():
            ds, _, _ = _init_components()
            points = [
                DataPoint(
                    id=str(uuid.uuid4()),
                    vector=data[i],
                    metadata=meta,
                    modality=modality,
                    chart_id=chart_id,
                )
                for i in range(len(data))
            ]
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Inserting...", total=len(points))
                count = await ds.batch_insert(points)
                progress.update(task, completed=len(points))
            await ds.close()
            return count

        count = asyncio.run(_do_batch())
        console.print(f"[green]Inserted {count} data points.[/green]")

    elif vector:
        vec = _parse_vector(vector)

        async def _do_single():
            ds, _, _ = _init_components()
            point_id = str(uuid.uuid4())
            dp = DataPoint(
                id=point_id,
                vector=vec,
                metadata=meta,
                modality=modality,
                chart_id=chart_id,
            )
            await ds.insert(dp)
            await ds.close()
            return point_id

        point_id = asyncio.run(_do_single())
        table = Table(title="Inserted Data Point")
        table.add_column("Field")
        table.add_column("Value")
        table.add_row("ID", point_id)
        table.add_row("Dimension", str(len(vec)))
        table.add_row("Modality", modality)
        if chart_id:
            table.add_row("Chart", chart_id)
        table.add_row("Metadata", json.dumps(meta) if meta else "(none)")
        console.print(table)
        console.print(f"[green]Inserted 1 data point: {point_id}[/green]")
    else:
        console.print("[red]Error: provide --data-file or --vector[/red]")
        raise typer.Exit(1)


# ═══════════════════════════════════════════════════════════════
# Query commands
# ═══════════════════════════════════════════════════════════════


@app.command()
def query(
    query_point: str = typer.Argument(
        ...,
        help="Query vector as JSON array, e.g. '[0.1, 0.2, 0.3]'",
    ),
    modality: str = typer.Option(
        "default",
        "--modality",
        "-m",
        help="Modality to search in.",
    ),
    k: int = typer.Option(
        10,
        "--k",
        "-k",
        help="Number of nearest neighbors to return.",
    ),
    metric: str = typer.Option(
        "geodesic",
        "--metric",
        help="Distance metric (geodesic, euclidean, cosine, wasserstein_riemannian).",
    ),
    chart_id: str | None = typer.Option(
        None,
        "--chart",
        "-c",
        help="Chart to search in.",
    ),
    epsilon: float = typer.Option(
        1.0,
        "--epsilon",
        "-e",
        help="Maximum distance for results.",
    ),
    explain: bool = typer.Option(
        False,
        "--explain",
        help="Show the execution plan instead of results.",
    ),
):
    """Execute a query against the manifold database.

    Finds the k nearest neighbors to the given query point using the
    specified distance metric.
    """
    vec = _parse_vector(query_point)

    try:
        metric_type = MetricType(metric)
    except ValueError:
        console.print(f"[red]Unknown metric: {metric}[/red]")
        console.print(f"Valid options: {[m.value for m in MetricType]}")
        raise typer.Exit(1)

    mq = ManifoldQuery(
        query_type=QueryType.SELECT,
        query_point=vec,
        metric_type=metric_type,
        k=k,
        modality=modality,
        chart_id=chart_id,
        epsilon=epsilon,
    )

    async def _do_query():
        _, _, qe = _init_components()
        if explain:
            plan = await qe.explain(mq)
            return plan, None
        result = await qe.execute(mq)
        return None, result

    plan, result = asyncio.run(_do_query())

    if explain:
        console.print(
            Panel(plan.visualize(), title="Execution Plan", border_style="yellow")
        )
    elif result:
        console.print(_format_result_table(result))


@app.command("geodesic-query")
def geodesic_query(
    center: str = typer.Argument(
        ...,
        help="Center point as JSON array.",
    ),
    epsilon: float = typer.Option(
        0.5,
        "--epsilon",
        "-e",
        help="Radius of the geodesic ball.",
    ),
    metric_type: str = typer.Option(
        "geodesic",
        "--metric-type",
        help="Distance metric type.",
    ),
    modality: str | None = typer.Option(
        None,
        "--modality",
        "-m",
        help="Filter by modality.",
    ),
):
    """Execute a geodesic ball query — find all points within epsilon distance."""
    vec = _parse_vector(center)

    try:
        mt = MetricType(metric_type)
    except ValueError:
        console.print(f"[red]Unknown metric type: {metric_type}[/red]")
        raise typer.Exit(1)

    mq = ManifoldQuery(
        query_type=QueryType.RANGE,
        query_point=vec,
        epsilon=epsilon,
        metric_type=mt,
        modality=modality,
    )

    async def _do():
        _, _, qe = _init_components()
        return await qe.execute(mq)

    result = asyncio.run(_do())
    console.print(_format_result_table(result))


@app.command("cross-modal")
def cross_modal(
    source: str = typer.Option(
        "text",
        "--source",
        "-s",
        help="Source modality.",
    ),
    target: str = typer.Option(
        "image",
        "--target",
        "-t",
        help="Target modality.",
    ),
    query: str = typer.Argument(
        ...,
        help="Query vector in source modality space.",
    ),
    k: int = typer.Option(
        10,
        "--k",
        "-k",
        help="Number of results to return.",
    ),
):
    """Cross-modal retrieval with parallel transport.

    Searches in the target modality using a query from the source modality.
    """
    vec = _parse_vector(query)

    mq = ManifoldQuery(
        query_type=QueryType.CROSS_MODAL,
        query_point=vec,
        modality=source,
        target_modality=target,
        k=k,
    )

    async def _do():
        _, _, qe = _init_components()
        return await qe.execute(mq)

    result = asyncio.run(_do())
    panel = Panel(
        f"Source modality: [cyan]{source}[/cyan]\n"
        f"Target modality: [cyan]{target}[/cyan]\n"
        f"Results: [green]{len(result)}[/green] points",
        title="Cross-Modal Query",
        border_style="magenta",
    )
    console.print(panel)
    console.print(_format_result_table(result))


# ═══════════════════════════════════════════════════════════════
# Atlas commands
# ═══════════════════════════════════════════════════════════════


@atlas_app.command("build")
def atlas_build(
    data_file: str = typer.Argument(
        ...,
        help="Path to .npy/.csv/.json data file.",
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for atlas JSON.",
    ),
    modality: str | None = typer.Option(
        None,
        "--modality",
        "-m",
        help="Modality tag for the atlas.",
    ),
    overlap_ratio: float = typer.Option(
        0.3,
        "--overlap",
        help="Overlap ratio between charts.",
    ),
    min_chart_size: int = typer.Option(
        50,
        "--min-size",
        help="Minimum chart size.",
    ),
    max_charts: int = typer.Option(
        100,
        "--max-charts",
        help="Maximum number of charts.",
    ),
):
    """Build an atlas from a data file."""
    data = _load_data_file(data_file)
    console.print(f"[cyan]Building atlas from {data.shape} data...[/cyan]")

    with console.status("[bold green]Building atlas..."):
        atlas_mgr = AtlasManager(name="built_atlas")
        atlas_mgr.build_atlas(
            data,
            modality=modality,
            overlap_ratio=overlap_ratio,
            min_chart_size=min_chart_size,
            max_charts=max_charts,
        )

    # Print summary
    summary = atlas_mgr.atlas_summary()
    table = Table(title="Atlas Summary")
    table.add_column("Property")
    table.add_column("Value")
    table.add_row("Name", summary["name"])
    table.add_row("Charts", str(summary["n_charts"]))
    table.add_row("Transitions", str(summary["n_transitions"]))
    table.add_row("Dim Range", f"{summary['dim_range'][0]} – {summary['dim_range'][1]}")
    table.add_row(
        "Ambient Dim Range",
        f"{summary['ambient_dim_range'][0]} – {summary['ambient_dim_range'][1]}",
    )
    console.print(table)

    # Save if output specified
    if output:
        atlas_mgr.save(output)
        console.print(f"[green]Atlas saved to {output}[/green]")
    else:
        # Save to current directory
        default_path = "atlas.json"
        atlas_mgr.save(default_path)
        console.print(f"[green]Atlas saved to {default_path}[/green]")


@atlas_app.command("info")
def atlas_info(
    atlas_file: str = typer.Argument(
        "atlas.json",
        help="Path to atlas JSON file.",
    ),
):
    """Display information about an atlas."""
    p = Path(atlas_file)
    if not p.exists():
        console.print(f"[red]Atlas file not found: {atlas_file}[/red]")
        raise typer.Exit(1)

    atlas_mgr = AtlasManager(name="info_atlas")
    atlas_mgr.load(atlas_file)
    summary = atlas_mgr.atlas_summary()

    tree = Tree(f"[bold]Atlas: {summary['name']}[/bold]")
    tree.add(f"Charts: {summary['n_charts']}")
    tree.add(f"Transitions: {summary['n_transitions']}")
    dims_branch = tree.add("Dimensions")
    dims_branch.add(f"Intrinsic: {summary['dim_range'][0]} – {summary['dim_range'][1]}")
    dims_branch.add(
        f"Ambient: {summary['ambient_dim_range'][0]} – {summary['ambient_dim_range'][1]}"
    )

    charts_branch = tree.add("Charts")
    for chart_info in summary.get("charts", []):
        cid = chart_info.get("chart_id", "?")
        cname = chart_info.get("name", "?")
        cdim = chart_info.get("dim", "?")
        charts_branch.add(f"[cyan]{cname}[/cyan] (id={cid}, dim={cdim})")

    console.print(tree)


@atlas_app.command("list-charts")
def atlas_list_charts(
    atlas_file: str = typer.Argument(
        "atlas.json",
        help="Path to atlas JSON file.",
    ),
):
    """List all charts in an atlas."""
    p = Path(atlas_file)
    if not p.exists():
        console.print(f"[red]Atlas file not found: {atlas_file}[/red]")
        raise typer.Exit(1)

    atlas_mgr = AtlasManager(name="list_atlas")
    atlas_mgr.load(atlas_file)
    charts = atlas_mgr.get_all_charts()

    table = Table(title=f"Charts in Atlas ({len(charts)} total)")
    table.add_column("Chart ID", style="cyan")
    table.add_column("Name")
    table.add_column("Dim", justify="right")
    table.add_column("Ambient Dim", justify="right")
    table.add_column("Anchors", justify="right")
    table.add_column("Has Bounds", justify="center")

    for chart in charts:
        summary = chart.summary()
        table.add_row(
            chart.chart_id[:12] + "...",
            chart.name,
            str(summary.get("dim", "?")),
            str(summary.get("ambient_dim", "?")),
            str(summary.get("n_anchor_points", 0)),
            "[green]Yes[/green]" if summary.get("has_bounds") else "[dim]No[/dim]",
        )

    console.print(table)


# ═══════════════════════════════════════════════════════════════
# Server commands
# ═══════════════════════════════════════════════════════════════


@server_app.command("start")
def server_start(
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        help="Bind host.",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        "-p",
        help="Bind port.",
    ),
    workers: int = typer.Option(
        1,
        "--workers",
        "-w",
        help="Number of uvicorn workers.",
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        help="Enable auto-reload for development.",
    ),
    config_file: str | None = typer.Option(
        None,
        "--config",
        help="Path to configuration YAML file.",
    ),
):
    """Start the Manifold DB REST API server."""
    # Set config env var if provided
    if config_file:
        os.environ["MANIFOLD_CONFIG"] = config_file

    console.print(
        Panel(
            f"[bold]Manifold Database API Server[/bold]\n"
            f"Version: {__VERSION__}\n"
            f"Host: [cyan]{host}[/cyan]\n"
            f"Port: [cyan]{port}[/cyan]\n"
            f"Workers: [cyan]{workers}[/cyan]\n"
            f"Docs: [cyan]http://{host}:{port}/docs[/cyan]",
            title="Starting Server",
            border_style="green",
        )
    )

    try:
        import uvicorn
    except ImportError:
        console.print("[red]uvicorn is required to start the server.[/red]")
        console.print("Install it with: [cyan]pip install uvicorn[standard][/cyan]")
        raise typer.Exit(1)

    uvicorn.run(
        "manifold_db.api.server:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level="info",
    )


@server_app.command("stop")
def server_stop(
    port: int = typer.Option(
        8000,
        "--port",
        "-p",
        help="Port the server is running on.",
    ),
):
    """Stop a running Manifold DB server."""
    console.print(f"[yellow]Attempting to stop server on port {port}...[/yellow]")
    try:
        # Find and kill the uvicorn process
        result = subprocess.run(
            ["lsof", "-t", "-i", f":{port}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split("\n")
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                    console.print(f"[green]Sent SIGTERM to PID {pid}[/green]")
                except ProcessLookupError:
                    console.print(f"[dim]PID {pid} not found (already stopped)[/dim]")
                except PermissionError:
                    console.print(f"[red]Permission denied to kill PID {pid}[/red]")
            console.print("[green]Server stop signal sent.[/green]")
        else:
            console.print("[yellow]No process found on that port.[/yellow]")
    except FileNotFoundError:
        console.print("[yellow]lsof not found. Use 'kill' manually.[/yellow]")
    except subprocess.TimeoutExpired:
        console.print("[red]Timeout while looking up port.[/red]")


# ═══════════════════════════════════════════════════════════════
# Database commands
# ═══════════════════════════════════════════════════════════════


@db_app.command("init")
def db_init(
    config: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config YAML file.",
    ),
):
    """Initialise a new manifold database."""
    console.print("[cyan]Initialising manifold database...[/cyan]")

    cfg = default_config()
    if config:
        try:
            cfg = load_config(config)
            console.print(f"[green]Loaded configuration from {config}[/green]")
        except (FileNotFoundError, ValueError) as exc:
            console.print(f"[red]Failed to load config: {exc}[/red]")
            raise typer.Exit(1)

    # Validate config
    try:
        validate_config(cfg)
        console.print("[green]Configuration validated.[/green]")
    except ValueError as exc:
        console.print(f"[red]Configuration validation failed: {exc}[/red]")
        raise typer.Exit(1)

    # Create storage directory
    storage_path = Path(cfg.storage.path)
    storage_path.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]Storage directory ready: {storage_path}[/green]")

    # Create default config file if none exists
    default_config_path = storage_path / "config.yaml"
    if not default_config_path.exists():
        save_config(cfg, str(default_config_path))
        console.print(f"[green]Default config written to {default_config_path}[/green]")

    console.print("[bold green]Database initialised successfully.[/bold green]")


@db_app.command("stats")
def db_stats(
    config: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config YAML file.",
    ),
):
    """Display database statistics."""
    console.print("[cyan]Gathering database statistics...[/cyan]")

    cfg = default_config()
    if config:
        try:
            cfg = load_config(config)
        except (FileNotFoundError, ValueError):
            pass

    async def _do():
        ds, am, _ = _init_components(cfg)
        ds_stats = await ds.stats()
        am_stats = am.atlas_summary()
        await ds.close()
        return ds_stats, am_stats

    ds_stats, am_stats = asyncio.run(_do())

    table = Table(title="Database Statistics")
    table.add_column("Property", style="bold")
    table.add_column("Value")
    table.add_row("Version", __VERSION__)
    table.add_row("Total Points", str(ds_stats.get("total_points", 0)))
    table.add_row(
        "Modalities", ", ".join(ds_stats.get("modalities_list", [])) or "(none)"
    )
    table.add_row("Charts", str(am_stats.get("n_charts", 0)))
    table.add_row("Transitions", str(am_stats.get("n_transitions", 0)))
    table.add_row("Storage Backend", cfg.storage.backend_type)
    table.add_row("Storage Path", cfg.storage.path)

    # Modality distribution
    mod_dist = ds_stats.get("modalities", {})
    if mod_dist:
        table.add_row("", "")
        table.add_row("[bold]Modality Distribution[/bold]", "")
        for mod, count in mod_dist.items():
            table.add_row(f"  {mod}", str(count))

    console.print(table)


@db_app.command("save")
def db_save(
    path: str = typer.Argument(
        "./manifold_data",
        help="Directory to save data to.",
    ),
    config: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config YAML file.",
    ),
):
    """Save the database to disk."""
    cfg = default_config()
    if config:
        try:
            cfg = load_config(config)
        except (FileNotFoundError, ValueError):
            pass

    async def _do():
        ds, am, _ = _init_components(cfg)
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save atlas
        atlas_file = save_path / "atlas.json"
        am.save(str(atlas_file))

        # Export data
        data_file = save_path / "data.json"
        await ds.export(format="json", path=str(data_file))

        # Save config
        config_file = save_path / "config.yaml"
        save_config(cfg, str(config_file))

        await ds.close()

    with console.status("[bold green]Saving database..."):
        asyncio.run(_do())
    console.print(f"[bold green]Database saved to {path}[/bold green]")


@db_app.command("load")
def db_load(
    path: str = typer.Argument(
        "./manifold_data",
        help="Directory to load data from.",
    ),
    config: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config YAML file.",
    ),
):
    """Load the database from disk."""
    load_path = Path(path)
    if not load_path.exists():
        console.print(f"[red]Path not found: {path}[/red]")
        raise typer.Exit(1)

    cfg = default_config()
    config_file = load_path / "config.yaml"
    if config_file.exists():
        try:
            cfg = load_config(str(config_file))
            console.print(f"[green]Loaded config from {config_file}[/green]")
        except ValueError:
            pass
    elif config:
        try:
            cfg = load_config(config)
        except (FileNotFoundError, ValueError):
            pass

    async def _do():
        ds, am, _ = _init_components(cfg)
        atlas_file = load_path / "atlas.json"
        data_file = load_path / "data.json"

        if atlas_file.exists():
            am.load(str(atlas_file))
            console.print(f"[green]Loaded atlas from {atlas_file}[/green]")

        if data_file.exists():
            count = await ds.import_data(str(data_file), format="json")
            console.print(f"[green]Loaded {count} data points[/green]")
        else:
            console.print(f"[yellow]No data file found at {data_file}[/yellow]")

        await ds.close()

    with console.status("[bold green]Loading database..."):
        asyncio.run(_do())
    console.print(f"[bold green]Database loaded from {path}[/bold green]")


@db_app.command("reset")
def db_reset(
    confirm: bool = typer.Option(
        False,
        "--confirm",
        help="Skip confirmation prompt.",
    ),
    path: str = typer.Option(
        "./manifold_data",
        "--path",
        help="Data directory to reset.",
    ),
):
    """Reset the database by deleting all stored data."""
    if not confirm:
        if not typer.confirm(
            "[bold red]This will permanently delete all database data![/bold red] Continue?",
            default=False,
        ):
            console.print("[yellow]Cancelled.[/yellow]")
            raise typer.Exit(0)

    storage_path = Path(path)
    if storage_path.exists():
        import shutil

        shutil.rmtree(storage_path)
        console.print(f"[green]Removed: {storage_path}[/green]")
    else:
        console.print(f"[yellow]Path does not exist: {storage_path}[/yellow]")

    console.print("[bold green]Database reset complete.[/bold green]")


# ═══════════════════════════════════════════════════════════════
# Config commands (inline — extended in config_commands.py)
# ═══════════════════════════════════════════════════════════════


@config_app.command("show")
def config_show(
    config: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config file. Uses defaults if not provided.",
    ),
):
    """Display the current configuration."""
    cfg = default_config()
    if config:
        try:
            cfg = load_config(config)
        except (FileNotFoundError, ValueError) as exc:
            console.print(f"[red]Error loading config: {exc}[/red]")
            raise typer.Exit(1)

    data = _config_to_dict(cfg)
    console.print(JSON(json.dumps(data, indent=2)))


@config_app.command("validate")
def config_validate(
    config: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config file to validate.",
    ),
):
    """Validate a configuration file."""
    cfg = default_config()
    if config:
        try:
            cfg = load_config(config)
        except (FileNotFoundError, ValueError) as exc:
            console.print(f"[red]Error loading config: {exc}[/red]")
            raise typer.Exit(1)

    try:
        validate_config(cfg)
        console.print("[bold green]Configuration is valid.[/bold green]")
    except ValueError as exc:
        console.print("[bold red]Configuration validation failed:[/bold red]")
        console.print(str(exc))
        raise typer.Exit(1)


@config_app.command("set")
def config_set(
    key: str = typer.Argument(
        ...,
        help="Config key in section.key format, e.g. 'server.port'.",
    ),
    value: str = typer.Argument(
        ...,
        help="Value to set.",
    ),
    config_file: str | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Config file to modify. Creates default if not provided.",
    ),
):
    """Set a configuration value."""
    cfg_path = Path(config_file) if config_file else Path("config.yaml")
    if cfg_path.exists():
        try:
            cfg = load_config(str(cfg_path))
        except (FileNotFoundError, ValueError) as exc:
            console.print(f"[red]Error loading config: {exc}[/red]")
            raise typer.Exit(1)
    else:
        cfg = default_config()

    # Parse section.key
    parts = key.lower().split(".", 1)
    if len(parts) != 2:
        console.print(f"[red]Key must be in 'section.key' format, got: {key}[/red]")
        raise typer.Exit(1)

    section_name, param_name = parts
    sections = {
        "atlas": cfg.atlas,
        "index": cfg.index,
        "geodesic": cfg.geodesic,
        "metric": cfg.metric,
        "storage": cfg.storage,
        "connection": cfg.connection,
        "query": cfg.query,
        "server": cfg.server,
    }

    if section_name not in sections:
        console.print(
            f"[red]Unknown section: {section_name}. Valid: {list(sections.keys())}[/red]"
        )
        raise typer.Exit(1)

    section_obj = sections[section_name]
    if not hasattr(section_obj, param_name):
        console.print(f"[red]Unknown parameter: {section_name}.{param_name}[/red]")
        raise typer.Exit(1)

    # Type coerce
    current_val = getattr(section_obj, param_name)
    if isinstance(current_val, bool):
        coerced = value.lower() in ("true", "1", "yes")
    elif isinstance(current_val, int):
        try:
            coerced = int(value)
        except ValueError:
            console.print(f"[red]Cannot convert '{value}' to int[/red]")
            raise typer.Exit(1)
    elif isinstance(current_val, float):
        try:
            coerced = float(value)
        except ValueError:
            console.print(f"[red]Cannot convert '{value}' to float[/red]")
            raise typer.Exit(1)
    else:
        coerced = value

    setattr(section_obj, param_name, coerced)

    # Validate after change
    try:
        validate_config(cfg)
    except ValueError as exc:
        console.print(f"[red]Validation failed after change: {exc}[/red]")
        raise typer.Exit(1)

    save_config(cfg, str(cfg_path))
    console.print(f"[green]Set {section_name}.{param_name} = {coerced}[/green]")
    console.print(f"[green]Saved to {cfg_path}[/green]")


@config_app.command("generate-defaults")
def config_generate(
    output: str = typer.Argument(
        "config.yaml",
        help="Output path for default config.",
    ),
):
    """Generate a default configuration file."""
    cfg = default_config()
    save_config(cfg, output)
    console.print(f"[green]Default configuration written to {output}[/green]")


# ═══════════════════════════════════════════════════════════════
# Benchmark command
# ═══════════════════════════════════════════════════════════════


@app.command()
def benchmark(
    data_size: int = typer.Option(
        10000,
        "--data-size",
        "-n",
        help="Number of data points to generate.",
    ),
    queries: int = typer.Option(
        100,
        "--queries",
        "-q",
        help="Number of queries to execute.",
    ),
    dimension: int = typer.Option(
        128,
        "--dimension",
        "-d",
        help="Dimensionality of embeddings.",
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for benchmark results (JSON).",
    ),
):
    """Run a performance benchmark on the manifold database."""
    console.print(
        Panel(
            f"Data size: [cyan]{data_size}[/cyan] points\n"
            f"Dimension: [cyan]{dimension}[/cyan]\n"
            f"Queries: [cyan]{queries}[/cyan]",
            title="Benchmark Configuration",
            border_style="yellow",
        )
    )

    # Generate random data
    console.print("[cyan]Generating random data...[/cyan]")
    rng = np.random.default_rng(42)
    data = rng.standard_normal((data_size, dimension)).astype(np.float64)

    # Create database
    ds, am, qe = _init_components()

    async def _do_benchmark():
        # Insert data
        console.print("[cyan]Inserting data points...[/cyan]")
        points = [
            DataPoint(id=str(uuid.uuid4()), vector=data[i], modality="benchmark")
            for i in range(data_size)
        ]
        t0 = time.perf_counter()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Inserting...", total=data_size)
            # Batch insert in chunks
            chunk_size = 1000
            for i in range(0, data_size, chunk_size):
                chunk = points[i : i + chunk_size]
                await ds.batch_insert(chunk)
                progress.update(task, completed=min(i + chunk_size, data_size))
        insert_time = time.perf_counter() - t0

        # Build index
        console.print("[cyan]Building query engine index...[/cyan]")
        # Query engine uses stubs, so we benchmark via data_store search
        am_stats = am.atlas_summary()

        # Run queries
        console.print("[cyan]Running benchmark queries...[/cyan]")
        query_points = rng.standard_normal((queries, dimension)).astype(np.float64)
        query_times = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Querying...", total=queries)
            for i in range(queries):
                t0 = time.perf_counter()
                await ds.search(
                    query_vector=query_points[i],
                    k=10,
                    metric="euclidean",
                )
                elapsed = time.perf_counter() - t0
                query_times.append(elapsed)
                progress.update(task, completed=i + 1)

        await ds.close()

        # Compile results
        query_times_arr = np.array(query_times)
        results_data = {
            "version": __VERSION__,
            "data_size": data_size,
            "dimension": dimension,
            "num_queries": queries,
            "insert_time_s": round(insert_time, 4),
            "insert_throughput": round(data_size / insert_time, 2),
            "query_mean_ms": round(float(query_times_arr.mean() * 1000), 4),
            "query_median_ms": round(float(np.median(query_times_arr) * 1000), 4),
            "query_p95_ms": round(float(np.percentile(query_times_arr, 95) * 1000), 4),
            "query_p99_ms": round(float(np.percentile(query_times_arr, 99) * 1000), 4),
            "query_min_ms": round(float(query_times_arr.min() * 1000), 4),
            "query_max_ms": round(float(query_times_arr.max() * 1000), 4),
            "queries_per_second": round(queries / float(query_times_arr.sum()), 2),
            "n_charts": am_stats.get("n_charts", 0),
        }

        return results_data

    with console.status("[bold green]Running benchmark..."):
        results = asyncio.run(_do_benchmark())

    # Display results
    table = Table(title="Benchmark Results")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Data Size", f"{results['data_size']:,}")
    table.add_row("Dimension", str(results["dimension"]))
    table.add_row("Num Queries", f"{results['num_queries']:,}")
    table.add_row("", "")
    table.add_row(
        "[bold]Insert Throughput[/bold]", f"{results['insert_throughput']:,.1f} pts/s"
    )
    table.add_row("[bold]Insert Time[/bold]", f"{results['insert_time_s']:.4f}s")
    table.add_row("", "")
    table.add_row("[bold]Query Mean[/bold]", f"{results['query_mean_ms']:.4f}ms")
    table.add_row("[bold]Query Median[/bold]", f"{results['query_median_ms']:.4f}ms")
    table.add_row("[bold]Query P95[/bold]", f"{results['query_p95_ms']:.4f}ms")
    table.add_row("[bold]Query P99[/bold]", f"{results['query_p99_ms']:.4f}ms")
    table.add_row("[bold]QPS[/bold]", f"{results['queries_per_second']:,.1f}")

    console.print(table)

    # Save results
    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"[green]Results saved to {output}[/green]")


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════


def main():
    """Entry point for ``python -m manifold_db.cli``."""
    app()


if __name__ == "__main__":
    main()
