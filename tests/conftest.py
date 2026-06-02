"""
Shared fixtures for all manifold_db tests.
Provides synthetic datasets, chart objects, and commonly used components.
"""

import os
import tempfile

import numpy as np
import pytest


# ---------- Reproducibility ----------
@pytest.fixture(autouse=True)
def _set_random_seed():
    np.random.seed(42)


# ---------- Synthetic datasets ----------

@pytest.fixture
def random_data():
    """1000 points in 50-D ambient space."""
    return np.random.randn(1000, 50).astype(np.float64)


@pytest.fixture
def swiss_roll_data():
    """Classic Swiss-roll: 1000 points in 3-D."""
    n = 1000
    t = np.linspace(1.5 * np.pi, 4.5 * np.pi, n)
    x = t * np.sin(t) + np.random.randn(n) * 0.1
    y = t * np.cos(t) + np.random.randn(n) * 0.1
    z = np.random.randn(n) * 0.5
    return np.column_stack([x, y, z]).astype(np.float64)


@pytest.fixture
def low_dim_data():
    """500 points in 10-D that roughly live on a 3-D manifold (linear + noise)."""
    n = 500
    basis = np.random.randn(10, 3)
    latent = np.random.randn(n, 3)
    noise = np.random.randn(n, 10) * 0.05
    return (latent @ basis.T + noise).astype(np.float64)


@pytest.fixture
def multi_modal_data():
    """Two modalities with an overlap region."""
    # Text-like embeddings: 768-D (simulated)
    text_data = np.random.randn(300, 50).astype(np.float64)
    # Image-like embeddings: 512-D (simulated) - mapped to same dim for simplicity
    image_data = np.random.randn(300, 50).astype(np.float64) + 2.0
    # Overlap region: 50 points that blend both
    overlap_data = np.random.randn(50, 50).astype(np.float64) + 1.0
    modality_labels = (
        ["text"] * 300 + ["image"] * 300 + ["overlap"] * 50
    )
    combined = np.vstack([text_data, image_data, overlap_data])
    return combined, modality_labels


# ---------- Chart fixture ----------

@pytest.fixture
def simple_chart():
    """A basic Chart wrapping an identity projection in 3-D."""
    from manifold_db.atlas import Chart

    chart = Chart(
        name="test_chart",
        dim=3,
        ambient_dim=3,
    )
    return chart


@pytest.fixture
def chart_with_data(random_data):
    """Chart with embedded data from random_data (first 100 points)."""
    from manifold_db.atlas import Chart

    data = random_data[:100]
    chart = Chart(name="data_chart", dim=50, ambient_dim=50)
    # Manually set embedded data snapshot so bounds/anchors work
    chart._embedded_data = data.copy()
    chart._data_count = len(data)
    return chart


# ---------- Tangent-space fixture ----------

@pytest.fixture
def tangent_space():
    """TangentSpace built from a small cluster of points."""
    from manifold_db.tangent_index import TangentSpace

    center = np.zeros(10)
    neighbors = np.random.randn(50, 10) * 0.5
    return TangentSpace(base_point=center, data=neighbors)


# ---------- Storage fixtures ----------

@pytest.fixture
def temp_storage_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_data_points():
    """A small list of DataPoint-like dicts."""
    pts = []
    for i in range(20):
        pts.append({
            "id": f"pt_{i}",
            "vector": np.random.randn(10).astype(np.float64),
            "metadata": {"source": "test"},
            "modality": "default",
        })
    return pts
