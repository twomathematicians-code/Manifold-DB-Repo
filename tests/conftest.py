"""
conftest.py - Shared pytest fixtures and configuration for ManifoldDB tests
=============================================================================

This module provides:
- A session-level skip if the C++ extension ``_manifolddb_core`` is not available.
- Common fixtures (random data generators, temporary database paths, etc.).

All tests in the ``tests/`` package share these fixtures automatically.
"""

from __future__ import annotations

import os
import tempfile
import shutil
from typing import Sequence

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Check that the C++ extension is importable
# ---------------------------------------------------------------------------

def _core_available() -> bool:
    """Return ``True`` if ``_manifolddb_core`` can be imported."""
    try:
        import manifolddb._manifolddb_core  # noqa: F401
        return True
    except Exception:
        pass
    try:
        import _manifolddb_core  # noqa: F401
        return True
    except Exception:
        pass
    return False


CORE_AVAILABLE = _core_available()
CORE_SKIP_REASON = (
    "C++ extension _manifolddb_core is not available. "
    "Build ManifoldDB from source first: pip install -e ."
)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_core: skip test if _manifolddb_core is unavailable"
    )


# ---------------------------------------------------------------------------
# Session-level auto-skip
# ---------------------------------------------------------------------------

def pytest_runtest_setup(item):
    """Skip every test if the core extension is missing."""
    # Allow tests that explicitly opt out with @pytest.mark.skipif.
    if "requires_core" in item.keywords:
        if not CORE_AVAILABLE:
            pytest.skip(CORE_SKIP_REASON)
    # For all other tests, also skip if core is unavailable
    elif not CORE_AVAILABLE:
        pytest.skip(CORE_SKIP_REASON)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def core():
    """Return the ``_manifolddb_core`` module."""
    try:
        import manifolddb._manifolddb_core as mod
        return mod
    except ImportError:
        import _manifolddb_core as mod
        return mod


@pytest.fixture
def temp_db_path(tmp_path):
    """Return a temporary directory path for ManifoldDB storage.

    The directory is automatically cleaned up by ``tmp_path`` (pytest builtin).
    """
    p = tmp_path / "manifolddb_test_data"
    p.mkdir(exist_ok=True)
    return str(p)


@pytest.fixture
def rng():
    """Return a seeded NumPy random generator for reproducible tests."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def random_points_3d(rng) -> np.ndarray:
    """Generate 100 random points in R^3 (shape ``(100, 3)``)."""
    return rng.standard_normal((100, 3))


@pytest.fixture
def random_points_10d(rng) -> np.ndarray:
    """Generate 200 random points in R^10 (shape ``(200, 10)``)."""
    return rng.standard_normal((200, 10))


@pytest.fixture
def identity_basis_3d() -> np.ndarray:
    """3x3 identity matrix, usable as an orthonormal basis for a 3-D LinearChart."""
    return np.eye(3, dtype=np.float64)


@pytest.fixture
def identity_basis_3d_in_5d() -> np.ndarray:
    """5x3 matrix embedding R^3 into R^5 (first 3 columns of I_5)."""
    return np.eye(5, 3, dtype=np.float64)


@pytest.fixture
def simple_chart(core, identity_basis_3d):
    """Create a simple 3-D LinearChart with identity basis and zero origin."""
    origin = np.zeros(3, dtype=np.float64)
    return core.LinearChart(id=0, basis=identity_basis_3d, origin=origin)


@pytest.fixture
def metric_store(core, temp_db_path):
    """Create a MetricStore backed by the temp database path."""
    return core.MetricStore(temp_db_path + "/metrics")


@pytest.fixture
def solver(core, metric_store):
    """Create a GeodesicSolver with default configuration."""
    config = core.SolverConfig()
    config.bvp_tolerance = 1e-4
    config.max_bvp_iterations = 30
    config.tolerance = 1e-6
    return core.GeodesicSolver(metric_store=metric_store, config=config)


@pytest.fixture
def atlas(core):
    """Create an empty Atlas."""
    return core.Atlas()
