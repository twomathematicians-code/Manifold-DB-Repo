"""
Integration test fixtures — provide real database instances.
"""

import numpy as np
import pytest

from manifold_db.atlas import AtlasBuilder, AtlasManager
from manifold_db.metric import EuclideanMetric, MetricTensorStore
from manifold_db.tangent_index import TangentSpaceIndex


@pytest.fixture
def small_dataset():
    """100 points in 10-D."""
    np.random.seed(0)
    return np.random.randn(100, 10)


@pytest.fixture
def built_atlas(small_dataset):
    """An atlas built from small_dataset."""
    builder = AtlasBuilder(min_chart_size=30, n_neighbors=10)
    atlas = AtlasManager(name="integration")
    builder.build(small_dataset, atlas)
    return atlas


@pytest.fixture
def metric_store(built_atlas):
    """MetricTensorStore with Euclidean metrics for all charts."""
    store = MetricTensorStore()
    for chart in built_atlas.get_all_charts():
        store.register_metric(chart.chart_id, EuclideanMetric(dim=10))
    return store


@pytest.fixture
def tangent_index(small_dataset):
    """TangentSpaceIndex built on small_dataset."""
    idx = TangentSpaceIndex(intrinsic_dim=5, leaf_size=20)
    ids = [f"pt_{i}" for i in range(len(small_dataset))]
    idx.build_from_data(ids, small_dataset, n_anchors=10)
    return idx
