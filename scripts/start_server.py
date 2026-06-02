#!/usr/bin/env python3
"""
Standalone script to start the Manifold DB REST API server.

This script can be invoked directly:

    python scripts/start_server.py
    python scripts/start_server.py --host 0.0.0.0 --port 8000 --workers 4
    python scripts/start_server.py --config /path/to/config.yaml

It handles:
    - Configuration loading from YAML/JSON or defaults
    - Environment variable overrides (MANIFOLD_DB_* and MANIFOLD_CONFIG)
    - Uvicorn server startup with configurable host, port, and workers
    - Graceful shutdown on SIGINT / SIGTERM
    - Logging setup
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

# ── Ensure project root is on sys.path ─────────────────────────
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="start_server",
        description="Start the Manifold Database REST API server.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind to. (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port to listen on. (default: 8000)",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of uvicorn worker processes. (default: 1)",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to a YAML configuration file.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable auto-reload for development.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Log level. (default: info)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override the data storage directory.",
    )
    return parser.parse_args()


def setup_logging(level_name: str) -> None:
    """Configure Python logging with a consistent format."""
    log_level = getattr(logging, level_name.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    # Quiet noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def load_configuration(config_path: Optional[str], data_dir: Optional[str]) -> None:
    """Load and apply configuration from file.

    Sets the MANIFOLD_CONFIG environment variable so that the FastAPI
    lifespan handler picks it up during startup.

    Parameters
    ----------
    config_path : str or None
        Path to the YAML/JSON configuration file.
    data_dir : str or None
        Override for the data storage directory.
    """
    if config_path:
        p = Path(config_path)
        if not p.exists():
            logging.error("Configuration file not found: %s", config_path)
            sys.exit(1)
        os.environ["MANIFOLD_CONFIG"] = str(p.resolve())
        logging.info("Configuration file set: %s", p.resolve())

    if data_dir:
        os.environ["MANIFOLD_DB_STORAGE_PATH"] = data_dir
        logging.info("Data directory override: %s", data_dir)


def print_banner(host: str, port: int, workers: int) -> None:
    """Print a startup banner to stderr."""
    banner = f"""
╔══════════════════════════════════════════════════════════════╗
║           Manifold Database REST API Server                ║
║                                                              ║
║  Host:    {host:<50s} ║
║  Port:    {port:<50d} ║
║  Workers: {workers:<50d} ║
║  Docs:    http://{host}:{port}/docs{' ' * (38 - len(host) - len(str(port)))}║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner, file=sys.stderr)


def main() -> None:
    """Entry point: parse args, configure, and start uvicorn."""
    args = parse_args()

    # ── Logging ─────────────────────────────────────────────────
    setup_logging(args.log_level)
    logger = logging.getLogger("start_server")

    # ── Configuration ──────────────────────────────────────────
    load_configuration(args.config, args.data_dir)

    # ── Banner ─────────────────────────────────────────────────
    print_banner(args.host, args.port, args.workers)

    # ── Check for uvicorn ──────────────────────────────────────
    try:
        import uvicorn
    except ImportError:
        logger.error(
            "uvicorn is required. Install it with: pip install uvicorn[standard]"
        )
        sys.exit(1)

    # ── Determine app module ────────────────────────────────────
    app_module = "manifold_db.api.server:app"

    # ── Build uvicorn config ──────────────────────────────────
    uvicorn_config: dict = {
        "app": app_module,
        "host": args.host,
        "port": args.port,
        "workers": args.workers,
        "log_level": args.log_level,
        "access_log": args.log_level == "debug",
    }

    if args.reload:
        uvicorn_config["reload"] = True
        uvicorn_config["reload_dirs"] = [str(project_root)]
        logger.info("Auto-reload enabled (development mode)")

    logger.info(
        "Starting Manifold DB server: %s://%s:%s (workers=%d)",
        "http", args.host, args.port, args.workers,
    )

    # ── Handle graceful shutdown ───────────────────────────────
    shutdown_requested = False

    def _signal_handler(signum: int, frame: object) -> None:
        nonlocal shutdown_requested
        if shutdown_requested:
            logger.warning("Force shutdown requested (second signal)")
            sys.exit(1)
        shutdown_requested = True
        logger.info(
            "Shutdown signal received (%s). Press Ctrl+C again to force exit.",
            signal.Signals(signum).name,
        )

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # ── Start server ───────────────────────────────────────────
    try:
        uvicorn.run(**uvicorn_config)
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Manifold DB server stopped.")


if __name__ == "__main__":
    main()
