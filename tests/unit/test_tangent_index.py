"""
Unit tests for manifold_db.tangent_index — TangentSpace, TangentBundle, TangentSpaceIndex.
"""

import numpy as np
import pytest

from manifold_db.tangent_index import TangentBundle, TangentSpace, TangentSpaceIndex


# ================================================================
# TangentSpace
# ================================================================


class TestTangentSpace:
    def test_creation_with_data(self):
        center = np.zeros(10)
        neighbors = np.random.randn(50, 10)
        ts = TangentSpace(base_point=center, data=neighbors, intrinsic_dim=5)
        assert ts.dimension <= 10

    def test_creation_without_data(self):
        ts = TangentSpace(base_point=np.zeros(5))
        assert ts.dimension == 5  # fallback to ambient dim

    def test_project_lift_roundtrip(self):
        center = np.zeros(10)
        neighbors = np.random.randn(50, 10)
        ts = TangentSpace(base_point=center, data=neighbors, intrinsic_dim=5)
        data = np.random.randn(20, 10)
        projected = ts.project(data)
        lifted = ts.lift(projected)
        # Lifting should recover approximate original shape
        assert lifted.shape == data.shape

    def test_log_exp_roundtrip(self):
        center = np.zeros(5)
        neighbors = np.random.randn(30, 5)
        ts = TangentSpace(base_point=center, data=neighbors)
        target = np.random.randn(5)
        log_vec = ts.log_map(target)
        exp_pt = ts.exp_map(log_vec)
        assert log_vec.shape == (ts.dimension,)
        assert exp_pt.shape == center.shape

    def test_compute_metric(self):
        ts = TangentSpace(base_point=np.zeros(5), data=np.random.randn(30, 5))
        a = np.random.randn(ts.dimension)
        b = np.random.randn(ts.dimension)
        val = ts.compute_metric(a, b)
        assert isinstance(val, float)

    def test_compute_distance(self):
        ts = TangentSpace(base_point=np.zeros(5), data=np.random.randn(30, 5))
        a = np.random.randn(ts.dimension)
        b = np.random.randn(ts.dimension)
        dist = ts.compute_distance(a, b)
        assert dist >= 0

    def test_parallel_transport(self):
        ts_a = TangentSpace(base_point=np.zeros(5), data=np.random.randn(30, 5))
        ts_b = TangentSpace(base_point=np.ones(5), data=np.random.randn(30, 5) + 1)
        vec = np.random.randn(ts_a.dimension)
        transported = ts_a.parallel_transport(vec, ts_b)
        assert transported.shape == vec.shape

    def test_update_basis(self):
        ts = TangentSpace(base_point=np.zeros(5), data=np.random.randn(30, 5))
        new_batch = np.random.randn(10, 5)
        ts.update_basis(new_batch)  # Should not raise

    def test_serialization_roundtrip(self):
        ts = TangentSpace(base_point=np.zeros(5), data=np.random.randn(30, 5))
        d = ts.to_dict()
        ts2 = TangentSpace.from_dict(d)
        assert ts2.dimension == ts.dimension


# ================================================================
# TangentBundle
# ================================================================


class TestTangentBundle:
    def test_reindex(self, random_data):
        tb = TangentBundle(intrinsic_dim=5, metric_eps=0.2)
        tb.reindex(random_data[:200], n_anchors=10)
        assert tb.n_spaces >= 1

    def test_nearest_anchor(self, random_data):
        tb = TangentBundle(intrinsic_dim=5, metric_eps=0.2)
        tb.reindex(random_data[:200], n_anchors=10)
        query = random_data[0]
        nearest = tb.nearest_anchor(query)
        assert len(nearest) >= 1
        anchor_id, dist = nearest[0]
        assert isinstance(anchor_id, str)
        assert dist >= 0

    def test_get_optimal_tangent_space(self, random_data):
        tb = TangentBundle(intrinsic_dim=5, metric_eps=0.2)
        tb.reindex(random_data[:200], n_anchors=10)
        query = random_data[0]
        ts, coords = tb.get_optimal_tangent_space(query, k_neighbors=3)
        assert isinstance(ts, TangentSpace)
        assert coords.ndim == 1

    def test_coverage_analysis(self, random_data):
        tb = TangentBundle(intrinsic_dim=5, metric_eps=0.5)
        tb.reindex(random_data[:200], n_anchors=10)
        result = tb.coverage_analysis(random_data[:200])
        assert "coverage_ratio" in result

    def test_serialization_roundtrip(self, random_data):
        tb = TangentBundle(intrinsic_dim=5, metric_eps=0.2)
        tb.reindex(random_data[:200], n_anchors=5)
        d = tb.serialize()
        tb2 = TangentBundle.deserialize(d)
        assert tb2.n_spaces == tb.n_spaces


# ================================================================
# TangentSpaceIndex
# ================================================================


class TestTangentSpaceIndex:
    def test_build_from_data(self, random_data):
        idx = TangentSpaceIndex(intrinsic_dim=5, leaf_size=20)
        ids = [f"pt_{i}" for i in range(200)]
        result = idx.build_from_data(ids, random_data[:200], n_anchors=10)
        assert idx.size == 200
        assert "n_anchors" in result

    def test_search(self, random_data):
        idx = TangentSpaceIndex(intrinsic_dim=5, leaf_size=20)
        ids = [f"pt_{i}" for i in range(200)]
        idx.build_from_data(ids, random_data[:200], n_anchors=10)
        results = idx.search(random_data[0], k=5)
        assert len(results) == 5
        for pid, dist in results:
            assert isinstance(pid, str)
            assert dist >= 0

    def test_insert_and_search(self, random_data):
        idx = TangentSpaceIndex(intrinsic_dim=5, leaf_size=20)
        ids = [f"pt_{i}" for i in range(100)]
        idx.build_from_data(ids, random_data[:100], n_anchors=5)
        # Insert a new point
        new_id = "pt_new"
        new_vec = random_data[150]
        idx.insert(new_id, new_vec)
        assert idx.size == 101

    def test_delete(self, random_data):
        idx = TangentSpaceIndex(intrinsic_dim=5, leaf_size=20)
        ids = [f"pt_{i}" for i in range(100)]
        idx.build_from_data(ids, random_data[:100], n_anchors=5)
        deleted = idx.delete("pt_0")
        assert deleted is True
        assert idx.size == 99

    def test_update(self, random_data):
        idx = TangentSpaceIndex(intrinsic_dim=5, leaf_size=20)
        ids = [f"pt_{i}" for i in range(100)]
        idx.build_from_data(ids, random_data[:100], n_anchors=5)
        updated = idx.update("pt_0", np.random.randn(50))
        assert updated is True

    def test_stats(self, random_data):
        idx = TangentSpaceIndex(intrinsic_dim=5, leaf_size=20)
        ids = [f"pt_{i}" for i in range(100)]
        idx.build_from_data(ids, random_data[:100], n_anchors=5)
        stats = idx.stats
        assert "size" in stats

    def test_serialization_roundtrip(self, random_data):
        idx = TangentSpaceIndex(intrinsic_dim=5, leaf_size=20)
        ids = [f"pt_{i}" for i in range(50)]
        idx.build_from_data(ids, random_data[:50], n_anchors=3)
        d = idx.to_dict()
        idx2 = TangentSpaceIndex.from_dict(d)
        assert idx2.size == idx.size
