"""
test_metric_tensor.py - Unit tests for MetricTensor and MetricStore
====================================================================

Tests cover:
- Identity and constant metrics
- Inverse metric computation
- Christoffel symbols for constant metrics (should be zero)
- Anchor-based metric interpolation
- Serialisation / deserialization round-trips
"""

from __future__ import annotations

import numpy as np
import pytest


class TestMetricTensor:
    """Tests for the ``MetricTensor`` class."""

    def test_identity_metric(self, core):
        """A freshly constructed MetricTensor should evaluate to the identity
        matrix (set_identity is the default).
        """
        dim = 3
        mt = core.MetricTensor(chart_id=0, dim=dim)

        local = np.zeros(dim, dtype=np.float64)
        g = mt.evaluate(local)

        np.testing.assert_allclose(
            g, np.eye(dim, dtype=np.float64), atol=1e-12,
            err_msg="Default metric should be identity"
        )

        assert mt.is_constant() is True
        assert mt.dim() == dim
        assert mt.chart_id() == 0
        assert mt.num_anchors() == 0

    def test_constant_metric(self, core):
        """Set a constant metric and verify it evaluates correctly at
        multiple points.
        """
        dim = 2
        mt = core.MetricTensor(chart_id=1, dim=dim)

        # Set a non-identity constant metric
        g_const = np.array([
            [2.0, 0.5],
            [0.5, 3.0],
        ], dtype=np.float64)
        mt.set_constant(g_const)

        assert mt.is_constant() is True

        # Evaluate at several points; all should return g_const
        test_points = [
            np.zeros(dim, dtype=np.float64),
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([-5.0, 100.0], dtype=np.float64),
        ]

        for pt in test_points:
            g = mt.evaluate(pt)
            np.testing.assert_allclose(
                g, g_const, atol=1e-12,
                err_msg=f"Constant metric should be the same at {pt}"
            )

    def test_inverse_metric(self, core):
        """Check that the inverse of the metric is correct.

        For a known metric g, verify g @ g^{-1} ≈ I.
        """
        dim = 3
        mt = core.MetricTensor(chart_id=0, dim=dim)

        # Non-trivial metric
        g_const = np.array([
            [4.0, 1.0, 0.0],
            [1.0, 3.0, 0.5],
            [0.0, 0.5, 2.0],
        ], dtype=np.float64)
        mt.set_constant(g_const)

        local = np.zeros(dim, dtype=np.float64)
        g_inv = mt.inverse(local)

        # Verify g @ g^{-1} = I
        product = g_const @ g_inv
        np.testing.assert_allclose(
            product, np.eye(dim, dtype=np.float64), atol=1e-10,
            err_msg="g @ g_inv should equal identity"
        )

        # Also verify g^{-1} @ g = I
        product_rev = g_inv @ g_const
        np.testing.assert_allclose(
            product_rev, np.eye(dim, dtype=np.float64), atol=1e-10,
        )

    def test_christoffel_zero_for_constant(self, core):
        """For a constant metric tensor, all Christoffel symbols should be
        zero (derivatives of a constant are zero).
        """
        dim = 3
        mt = core.MetricTensor(chart_id=0, dim=dim)

        g_const = np.array([
            [2.0, 0.3, -0.1],
            [0.3, 5.0, 0.0],
            [-0.1, 0.0, 1.0],
        ], dtype=np.float64)
        mt.set_constant(g_const)

        local = np.array([0.5, -1.0, 2.0], dtype=np.float64)
        Gamma = mt.christoffel_symbols(local, h=1e-5)

        assert Gamma.shape == (dim, dim, dim)
        np.testing.assert_allclose(
            np.asarray(Gamma), 0.0, atol=1e-8,
            err_msg="Christoffel symbols should be zero for a constant metric"
        )

    def test_update_anchor(self, core):
        """Add an anchor point to the metric tensor and verify evaluation.

        After adding an anchor at a specific location with a custom metric,
        evaluating at that location should return (approximately) the
        anchor's metric.
        """
        dim = 2
        mt = core.MetricTensor(chart_id=0, dim=dim)

        anchor_coords = np.array([1.0, 0.0], dtype=np.float64)
        anchor_metric = np.array([
            [3.0, 0.0],
            [0.0, 3.0],
        ], dtype=np.float64)

        mt.update(anchor_coords, anchor_metric, weight=1.0)

        assert mt.is_constant() is False
        assert mt.num_anchors() == 1

        # Evaluate at the anchor location – should be close to anchor_metric
        g = mt.evaluate(anchor_coords)
        np.testing.assert_allclose(
            g, anchor_metric, atol=1e-10,
            err_msg="Metric at anchor should match the anchor metric"
        )

        # Evaluate far from the anchor – may differ due to RBF weighting,
        # but with only one anchor the interpolation should still match closely
        far = np.array([100.0, 100.0], dtype=np.float64)
        g_far = mt.evaluate(far)
        # With one anchor, RBF weight decays but the metric may not be identity
        assert g_far.shape == (dim, dim)

    def test_update_multiple_anchors(self, core):
        """Add multiple anchors and verify interpolation blends them."""
        dim = 2
        mt = core.MetricTensor(chart_id=0, dim=dim)

        # Anchor 1: 2*I at origin
        mt.update(np.zeros(dim), 2.0 * np.eye(dim), weight=1.0)
        # Anchor 2: 4*I at (1, 1)
        mt.update(np.ones(dim), 4.0 * np.eye(dim), weight=1.0)

        assert mt.num_anchors() == 2

        # At origin, should be close to 2*I
        g0 = mt.evaluate(np.zeros(dim))
        np.testing.assert_allclose(g0, 2.0 * np.eye(dim), atol=0.5)

        # At (1,1), should be close to 4*I
        g1 = mt.evaluate(np.ones(dim))
        np.testing.assert_allclose(g1, 4.0 * np.eye(dim), atol=0.5)

    def test_serialize_deserialize(self, core):
        """Round-trip: serialize a MetricTensor, deserialize it, and verify
        the result matches.
        """
        dim = 3
        mt = core.MetricTensor(chart_id=5, dim=dim)

        g_const = np.array([
            [1.5, 0.2, 0.0],
            [0.2, 2.0, -0.3],
            [0.0, -0.3, 0.8],
        ], dtype=np.float64)
        mt.set_constant(g_const)

        # Add an anchor
        mt.update(
            np.array([0.5, 0.0, -0.5], dtype=np.float64),
            np.eye(dim, dtype=np.float64),
            weight=1.0,
        )

        # Serialize
        data = mt.serialize()
        assert len(data) > 0

        # Deserialize into a fresh MetricTensor
        mt2 = core.MetricTensor(chart_id=99, dim=dim)  # wrong values initially
        mt2.deserialize(data)

        # Verify metadata
        assert mt2.chart_id() == 5
        assert mt2.dim() == 3

        # Verify the constant metric portion matches
        # After deserialization, the metric should evaluate consistently
        test_pt = np.zeros(dim, dtype=np.float64)
        g_orig = mt.evaluate(test_pt)
        g_loaded = mt2.evaluate(test_pt)

        np.testing.assert_allclose(
            g_orig, g_loaded, atol=1e-12,
            err_msg="Deserialized metric should evaluate identically"
        )


class TestMetricStore:
    """Tests for the ``MetricStore`` class."""

    def test_create_and_get_metric(self, core, temp_db_path):
        """Create a metric via MetricStore and retrieve it."""
        store = core.MetricStore(temp_db_path + "/metrics_test")

        metric = store.create_metric(chart_id=0, dim=3)
        assert metric is not None
        assert metric.chart_id() == 0
        assert metric.dim() == 3
        assert metric.is_constant() is True

        # Retrieve the same metric
        cached = store.get_metric(0)
        assert cached is not None

    def test_commit_and_retrieve(self, core, temp_db_path):
        """Commit a modified metric and retrieve it."""
        store = core.MetricStore(temp_db_path + "/metrics_commit")

        metric = store.create_metric(chart_id=1, dim=2)
        g_custom = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float64)
        metric.set_constant(g_custom)

        store.commit(1, metric)

        # Retrieve and verify
        retrieved = store.get_metric(1)
        assert retrieved is not None
        g = retrieved.evaluate(np.zeros(2, dtype=np.float64))
        np.testing.assert_allclose(g, g_custom, atol=1e-12)

    def test_has_chart(self, core, temp_db_path):
        """Test the ``has_chart`` method."""
        store = core.MetricStore(temp_db_path + "/metrics_has")

        assert store.has_chart(0) is False
        store.create_metric(chart_id=0, dim=3)
        assert store.has_chart(0) is True

    def test_num_charts(self, core, temp_db_path):
        """Test the ``num_charts`` counter."""
        store = core.MetricStore(temp_db_path + "/metrics_num")

        assert store.num_charts() == 0
        store.create_metric(chart_id=0, dim=3)
        store.create_metric(chart_id=1, dim=5)
        assert store.num_charts() == 2

    def test_batch_evaluate(self, core, temp_db_path):
        """Test batch evaluation at multiple points."""
        store = core.MetricStore(temp_db_path + "/metrics_batch")

        metric = store.create_metric(chart_id=0, dim=2)
        metric.set_identity()
        store.commit(0, metric)

        points = [
            np.zeros(2, dtype=np.float64),
            np.array([1.0, 2.0], dtype=np.float64),
        ]

        results = store.batch_evaluate(0, points)
        assert len(results) == 2
        for g in results:
            np.testing.assert_allclose(g, np.eye(2, dtype=np.float64), atol=1e-12)
