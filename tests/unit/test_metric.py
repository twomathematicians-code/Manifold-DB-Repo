"""
Unit tests for manifold_db.metric — MetricTensor, EuclideanMetric, MetricTensorStore.
"""

import numpy as np
import pytest

from manifold_db.metric import DiagonalMetric, EuclideanMetric, MetricTensorStore


class TestEuclideanMetric:
    def test_evaluate_returns_identity(self):
        m = EuclideanMetric(dim=3)
        x = np.random.randn(3)
        g = m.evaluate(x)
        np.testing.assert_allclose(g, np.eye(3))

    def test_inverse_returns_identity(self):
        m = EuclideanMetric(dim=3)
        g_inv = m.inverse(np.zeros(3))
        np.testing.assert_allclose(g_inv, np.eye(3))

    def test_christoffel_symbols_zero(self):
        m = EuclideanMetric(dim=3)
        Gamma = m.christoffel_symbols(np.zeros(3))
        np.testing.assert_allclose(Gamma, 0.0, atol=1e-12)

    def test_determinant_one(self):
        m = EuclideanMetric(dim=3)
        det = m.determinant(np.zeros(3))
        assert abs(det - 1.0) < 1e-10

    def test_log_det_zero(self):
        m = EuclideanMetric(dim=3)
        ld = m.log_det(np.zeros(3))
        assert abs(ld) < 1e-10

    def test_ricci_curvature_zero(self):
        m = EuclideanMetric(dim=3)
        ricci = m.ricci_curvature(np.zeros(3))
        np.testing.assert_allclose(ricci, 0.0, atol=1e-8)

    def test_scalar_curvature_zero(self):
        m = EuclideanMetric(dim=3)
        sc = m.scalar_curvature(np.zeros(3))
        assert abs(sc) < 1e-8

    def test_serialization_roundtrip(self):
        m = EuclideanMetric(dim=5)
        d = m.to_dict()
        m2 = EuclideanMetric.from_dict(d)
        assert m2.evaluate(np.zeros(5)).shape == (5, 5)


class TestDiagonalMetric:
    def test_evaluate_returns_diagonal(self):
        weights = np.array([1.0, 2.0, 3.0])
        m = DiagonalMetric(weights=weights)
        g = m.evaluate(np.zeros(3))
        np.testing.assert_allclose(np.diag(g), weights)

    def test_determinant(self):
        weights = np.array([2.0, 3.0, 4.0])
        m = DiagonalMetric(weights=weights)
        det = m.determinant(np.zeros(3))
        assert abs(det - 24.0) < 1e-10

    def test_inverse(self):
        weights = np.array([1.0, 2.0, 4.0])
        m = DiagonalMetric(weights=weights)
        g_inv = m.inverse(np.zeros(3))
        np.testing.assert_allclose(np.diag(g_inv), 1.0 / weights)

    def test_serialization_roundtrip(self):
        weights = np.array([1.0, 2.0, 3.0])
        m = DiagonalMetric(weights=weights)
        d = m.to_dict()
        m2 = DiagonalMetric.from_dict(d)
        np.testing.assert_allclose(m2.evaluate(np.zeros(3)), m.evaluate(np.zeros(3)))


class TestMetricTensorStore:
    def test_register_and_get(self):
        store = MetricTensorStore()
        m = EuclideanMetric(dim=5)
        store.register_metric("chart_0", m)
        retrieved = store.get_metric("chart_0")
        assert isinstance(retrieved, EuclideanMetric)

    def test_get_missing_raises(self):
        store = MetricTensorStore()
        with pytest.raises(KeyError):
            store.get_metric("nonexistent")

    def test_list_charts(self):
        store = MetricTensorStore()
        store.register_metric("a", EuclideanMetric(dim=3))
        store.register_metric("b", DiagonalMetric(dim=3))
        assert set(store.list_charts()) == {"a", "b"}

    def test_serialize_deserialize(self):
        store = MetricTensorStore()
        store.register_metric("a", EuclideanMetric(dim=3))
        d = store.serialize()
        store2 = MetricTensorStore()
        store2.deserialize(d)
        assert "a" in store2.list_charts()
