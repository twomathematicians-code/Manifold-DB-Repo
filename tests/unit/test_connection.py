"""
Unit tests for manifold_db.connection — LeviCivitaConnection, TransportRegistry, SchemaTransport.
"""

import numpy as np
import pytest

from manifold_db.connection import (
    LeviCivitaConnection,
    SchemaTransport,
    TemporalTransport,
    TransportRegistry,
)


def _euclidean_metric(x):
    return np.eye(len(x))


class TestLeviCivitaConnection:
    def test_parallel_transport_euclidean(self):
        conn = LeviCivitaConnection()
        path = np.linspace(0, 1, 20).reshape(-1, 1) * np.array([[1.0, 0.0, 0.0]])
        # Actually need 3D points
        path_3d = np.zeros((20, 3))
        path_3d[:, 0] = np.linspace(0, 1, 20)
        vector = np.array([0.0, 1.0, 0.0])
        transported = conn.parallel_transport(vector, path_3d, _euclidean_metric)
        # In Euclidean space, parallel transport preserves the vector
        assert transported.shape == vector.shape

    def test_connection_coefficients_euclidean(self):
        conn = LeviCivitaConnection()
        Gamma = conn.connection_coefficients(np.zeros(3), _euclidean_metric)
        np.testing.assert_allclose(Gamma, 0.0, atol=1e-10)

    def test_parallel_transport_along_geodesic(self):
        conn = LeviCivitaConnection()
        start = np.zeros(3)
        end = np.array([1.0, 0.0, 0.0])
        vector = np.array([0.0, 1.0, 0.0])
        result = conn.parallel_transport_along_geodesic(
            vector, start, end, _euclidean_metric, n_steps=20
        )
        assert result.shape == vector.shape

    def test_covariant_derivative(self):
        conn = LeviCivitaConnection()

        def const_field(x):
            return np.array([0.0, 1.0, 0.0])

        result = conn.covariant_derivative(const_field, np.array([1.0, 0.0, 0.0]),
                                             np.zeros(3), _euclidean_metric)
        # Constant field in Euclidean space → zero derivative
        np.testing.assert_allclose(result, 0.0, atol=0.1)

    def test_transport_across_charts(self):
        conn = LeviCivitaConnection()

        def identity_map(x):
            return x

        result = conn.transport_across_charts(
            np.array([1.0, 0.0]),
            "chart_a", "chart_b",
            identity_map,
            metric_fn_source=_euclidean_metric,
            metric_fn_target=_euclidean_metric,
        )
        assert result.shape == (2,)

    def test_batch_transport(self):
        conn = LeviCivitaConnection()
        vectors = np.random.randn(3, 3)
        path = np.zeros((10, 3))
        path[:, 0] = np.linspace(0, 1, 10)
        paths = [path] * 3
        result = conn.transport_batch(vectors, paths, _euclidean_metric)
        assert result.shape == vectors.shape


class TestSchemaTransport:
    def test_register_and_transport(self):
        conn = LeviCivitaConnection()
        st = SchemaTransport(connection=conn)
        st.register_schema("v1", np.zeros(5), _euclidean_metric)
        st.register_schema("v2", np.ones(5), _euclidean_metric)
        coords = np.random.randn(5)
        result = st.transport_query(coords, "v1", "v2", n_steps=10)
        assert result.shape == coords.shape


class TestTemporalTransport:
    def test_transport_in_time(self):
        tt = TemporalTransport()
        vector = np.array([1.0, 0.0])
        result = tt.transport_in_time(vector, 0.0, 1.0, n_steps=10)
        assert result.shape == vector.shape


class TestTransportRegistry:
    def test_register_and_get(self):
        reg = TransportRegistry()

        def simple_transport(v):
            return v * 2

        reg.register_transport("a", "b", simple_transport)
        fn = reg.get_transport("a", "b")
        assert fn is not None
        np.testing.assert_allclose(fn(np.array([1.0])), np.array([2.0]))

    def test_has_transport(self):
        reg = TransportRegistry()
        reg.register_transport("a", "b", lambda v: v)
        assert reg.has_transport("a", "b")
        assert not reg.has_transport("a", "c")

    def test_list_transports(self):
        reg = TransportRegistry()
        reg.register_transport("a", "b", lambda v: v)
        reg.register_transport("b", "c", lambda v: v)
        pairs = reg.list_transports()
        assert len(pairs) == 2

    def test_compute_chain(self):
        reg = TransportRegistry()
        reg.register_transport("a", "b", lambda v: v + 1)
        reg.register_transport("b", "c", lambda v: v * 2)
        chain = [("a", "b"), ("b", "c")]
        result = reg.compute_chain(chain, np.array([1.0]))
        np.testing.assert_allclose(result, np.array([4.0]))  # (1+1)*2

    def test_invalidate(self):
        reg = TransportRegistry()
        reg.register_transport("a", "b", lambda v: v)
        reg.register_transport("b", "c", lambda v: v)
        count = reg.invalidate("b")
        assert count == 2
        assert not reg.has_transport("a", "b")

    def test_serialization(self):
        reg = TransportRegistry()
        reg.register_transport("a", "b", lambda v: v)
        d = reg.serialize()
        reg2 = TransportRegistry()
        reg2.deserialize(d)
        assert reg2.has_transport("a", "b")
