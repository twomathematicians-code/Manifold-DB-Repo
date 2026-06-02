"""
manifold_db.cli — Command-line interface for the Manifold Database.

Provides a rich, user-friendly CLI built on Typer + Rich for all database
operations including data insertion, querying, atlas management, server
control, configuration, and benchmarking.

Public API:
    app       — the root Typer application (invoke via ``manifold-db``)
    main      — synchronous entry point for ``python -m manifold_db.cli``
"""

from manifold_db.cli.main import app, main

__all__ = [
    "app",
    "main",
]
