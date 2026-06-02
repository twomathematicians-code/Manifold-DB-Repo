"""
Integration tests — end-to-end pipelines for Manifold Database.
"""

import numpy as np
import pytest

from manifold_db.atlas import AtlasBuilder, AtlasManager, Chart
from manifold_db.geodesic import GeodesicSolver, RiemannianDistance
from manifold_db.metric import EuclideanMetric, MetricTensorStore
from manifold_db.tangent_index import TangentSpaceIndex


def _euclidean_metric(x):
    return np.eye(len(x))


class TestEndToEnd:
    """Full pipeline: build atlas → build index → query."""

    def test_insert_and_search(self):
        """Build an atlas and index, then search for nearest neighbors."""
        np.random.seed(0)
        data = np.random.randn(300, 20)

        # 1. Build atlas
        builder = AtlasBuilder(min_chart_size=30, n_neighbors=10)
        atlas = AtlasManager(name="e2e_test")
        builder.build(data, atlas)
        assert len(atlas) >= 1

        # 2. Build tangent index
        idx = TangentSpaceIndex(intrinsic_dim=5, leaf_size=20)
        ids = [f"pt_{i}" for i in range(len(data))]
        idx.build_from_data(ids, data, n_anchors=10)
        assert idx.size == 300

        # 3. Search
        results = idx.search(data[0], k=5)
        assert len(results) == 5
        # First result should be the query point itself (or very close)
        top_id, top_dist = results[0]
        assert top_id == "pt_0"
        assert top_dist < 1.0

    def test_multi_chart_query(self, multi_modal_data):
        """Build multi-modal atlas and execute cross-modal-like queries."""
        data, labels = multi_modal_data
        builder = AtlasBuilder(min_chart_size=30, n_neighbors=10)
        atlas = AtlasManager(name="multi_modal")
        builder.build(data, atlas, modality_labels=labels)
        assert len(atlas) >= 1

        # Build index
        idx = TangentSpaceIndex(intrinsic_dim=5, leaf_size=20)
        ids = [f"pt_{i}" for i in range(len(data))]
        idx.build_from_data(ids, data, n_anchors=10)

        # Query from text region
        results = idx.search(data[0], k=5)
        assert len(results) == 5

    def test_geodesic_solver_with_atlas(self):
        """Build atlas, register metrics, and run geodesic computations."""
        np.random.seed(1)
        data = np.random.randn(200, 10)

        # Build atlas
        builder = AtlasBuilder(min_chart_size=50, n_neighbors=10)
        atlas = AtlasManager(name="geo_test")
        builder.build(data, atlas)

        # Register metrics
        store = MetricTensorStore()
        for chart in atlas.get_all_charts():
            store.register_metric(chart.chart_id, EuclideanMetric(dim=10))

        # Solve geodesic
        solver = GeodesicSolver(metric_fn=_euclidean_metric, dim=10)
        p0 = data[0]
        v0 = data[1] - data[0]
        v0 = v0 / (np.linalg.norm(v0) + 1e-8) * 0.5
        result = solver.solve_geodesic(p0, v0, t_span=(0.0, 1.0), method="rk4", dt=0.05)
        assert result.success
        assert result.trajectory.shape[0] > 1

    def test_persistence_roundtrip(self, tmp_path):
        """Save and load a complete atlas."""
        np.random.seed(2)
        data = np.random.randn(150, 10)

        # Build
        builder = AtlasBuilder(min_chart_size=50, n_neighbors=10)
        atlas = AtlasManager(name="persist_test")
        builder.build(data, atlas)

        # Save
        path = str(tmp_path / "atlas.json")
        atlas.save(path)

        # Load
        atlas2 = AtlasManager()
        atlas2.load(path)
        assert len(atlas2) == len(atlas)
        assert atlas2.get_all_charts()[0].name == atlas.get_all_charts()[0].name

    def test_distance_computation_pipeline(self):
        """Compute various distance metrics on data."""
        np.random.seed(3)
        data = np.random.randn(100, 5)

        from manifold_db.geodesic import DistanceComputer

        dc = DistanceComputer(metric_tensor_fn=_euclidean_metric)

        p, q = data[0], data[1]
        d_euc = dc.compute(p, q, metric_type="euclidean")
        assert d_euc > 0

        # Batch
        batch_a = data[:5]
        batch_b = data[5:10]
        dists = dc.batch_compute(batch_a, batch_b, metric_type="euclidean")
        assert dists.shape == (5,)
        assert np.all(dists >= 0)
