"""
Unit tests for manifold_db.atlas — Chart, TransitionMap, AtlasManager, AtlasBuilder.
"""

import numpy as np
import pytest

from manifold_db.atlas import (
    AffineTransition,
    AtlasBuilder,
    AtlasManager,
    Chart,
    LinearTransition,
)


# ================================================================
# Chart
# ================================================================


class TestChart:
    def test_creation_defaults(self):
        c = Chart(name="c0", dim=3, ambient_dim=5)
        assert c.name == "c0"
        assert c.dim == 3
        assert c.ambient_dim == 5

    def test_embed_identity_fallback(self, simple_chart):
        data = np.random.randn(4, 3)
        embedded = simple_chart.embed(data)
        np.testing.assert_allclose(embedded, data)

    def test_inverse_identity_fallback(self, simple_chart):
        coords = np.random.randn(4, 3)
        lifted = simple_chart.inverse(coords)
        np.testing.assert_allclose(lifted, coords)

    def test_embed_inverse_roundtrip(self):
        """Custom embedding_fn + inverse_fn should round-trip."""
        shift = np.array([1.0, 2.0, 3.0])

        def embed(x):
            return x + shift

        def inverse(y):
            return y - shift

        c = Chart(name="shifted", dim=3, ambient_dim=3, embedding_fn=embed, inverse_fn=inverse)
        original = np.random.randn(5, 3)
        np.testing.assert_allclose(c.inverse(c.embed(original)), original, atol=1e-12)

    def test_contains(self, simple_chart):
        coords = np.array([[0.0, 0.0, 0.0]])
        # Without explicit bounds and no data, contains may return True/False
        result = simple_chart.contains(coords)
        assert isinstance(result, np.ndarray)

    def test_bounds_lazy(self, simple_chart):
        # Bounds should be None when no data has been embedded
        assert simple_chart.bounds is None

    def test_serialization_roundtrip(self, simple_chart):
        d = simple_chart.to_dict()
        c2 = Chart.from_dict(d)
        assert c2.name == simple_chart.name
        assert c2.dim == simple_chart.dim

    def test_summary(self, simple_chart):
        s = simple_chart.summary()
        assert isinstance(s, dict)
        assert "name" in s

    def test_repr(self, simple_chart):
        r = repr(simple_chart)
        assert "Chart" in r


# ================================================================
# Transition Maps
# ================================================================


class TestLinearTransition:
    def test_forward(self):
        M = np.eye(3)
        lt = LinearTransition(source_chart_id="a", target_chart_id="b",
                              overlap_region=np.array([[-5, -5, -5], [5, 5, 5]]),
                              matrix=M)
        coords = np.random.randn(4, 3)
        np.testing.assert_allclose(lt.forward(coords), coords)

    def test_inverse(self):
        M = np.diag([1.0, 2.0, 3.0])
        lt = LinearTransition(source_chart_id="a", target_chart_id="b",
                              overlap_region=np.array([[-5, -5, -5], [5, 5, 5]]),
                              matrix=M)
        x = np.random.randn(4, 3)
        np.testing.assert_allclose(lt.inverse(lt.forward(x)), x, atol=1e-12)

    def test_jacobian(self):
        M = np.random.randn(3, 3)
        lt = LinearTransition(source_chart_id="a", target_chart_id="b",
                              overlap_region=np.array([[-5, -5, -5], [5, 5, 5]]),
                              matrix=M)
        J = lt.jacobian(np.zeros(3))
        np.testing.assert_allclose(J, M)


class TestAffineTransition:
    def test_roundtrip(self):
        M = np.diag([1.0, 2.0, 3.0])
        b = np.array([1.0, 0.5, -1.0])
        at = AffineTransition(source_chart_id="a", target_chart_id="b",
                              overlap_region=np.array([[-10, -10, -10], [10, 10, 10]]),
                              matrix=M, bias=b)
        x = np.random.randn(5, 3)
        np.testing.assert_allclose(at.inverse(at.forward(x)), x, atol=1e-12)

    def test_bias_applied(self):
        b = np.array([1.0, 2.0, 3.0])
        at = AffineTransition(source_chart_id="a", target_chart_id="b",
                              overlap_region=np.array([[-5, -5, -5], [5, 5, 5]]),
                              matrix=np.eye(3), bias=b)
        zero = np.zeros((1, 3))
        np.testing.assert_allclose(at.forward(zero), b.reshape(1, 3))


# ================================================================
# AtlasManager
# ================================================================


class TestAtlasManager:
    def test_add_and_get_chart(self, simple_chart):
        am = AtlasManager(name="test")
        am.add_chart(simple_chart)
        assert len(am) == 1
        retrieved = am.get_chart(simple_chart.chart_id)
        assert retrieved.name == "test_chart"

    def test_remove_chart(self, simple_chart):
        am = AtlasManager()
        am.add_chart(simple_chart)
        am.remove_chart(simple_chart.chart_id)
        assert len(am) == 0
        with pytest.raises(KeyError):
            am.get_chart(simple_chart.chart_id)

    def test_get_all_charts(self, simple_chart):
        am = AtlasManager()
        am.add_chart(simple_chart)
        charts = am.get_all_charts()
        assert len(charts) == 1

    def test_add_transition_map(self, simple_chart):
        am = AtlasManager()
        c2 = Chart(name="c2", dim=3, ambient_dim=3)
        am.add_chart(simple_chart)
        am.add_chart(c2)
        overlap = np.array([[-5, -5, -5], [5, 5, 5]])
        tmap = LinearTransition(
            source_chart_id=simple_chart.chart_id,
            target_chart_id=c2.chart_id,
            overlap_region=overlap,
            matrix=np.eye(3),
        )
        am.add_transition_map(tmap)
        retrieved = am.get_transition(simple_chart.chart_id, c2.chart_id)
        assert isinstance(retrieved, LinearTransition)

    def test_get_transition_missing(self, simple_chart):
        am = AtlasManager()
        am.add_chart(simple_chart)
        with pytest.raises(KeyError):
            am.get_transition(simple_chart.chart_id, "nonexistent")

    def test_atlas_summary(self, simple_chart):
        am = AtlasManager()
        am.add_chart(simple_chart)
        s = am.atlas_summary()
        assert s["n_charts"] == 1

    def test_serialize_deserialize(self, simple_chart):
        am = AtlasManager(name="serialize_test")
        am.add_chart(simple_chart)
        d = am.serialize()
        am2 = AtlasManager()
        am2.deserialize(d)
        assert len(am2) == 1
        assert am2.get_chart(simple_chart.chart_id).name == "test_chart"

    def test_save_load(self, simple_chart, tmp_path):
        am = AtlasManager(name="persist_test")
        am.add_chart(simple_chart)
        filepath = str(tmp_path / "atlas.json")
        am.save(filepath)
        am2 = AtlasManager()
        am2.load(filepath)
        assert len(am2) == 1


# ================================================================
# AtlasBuilder
# ================================================================


class TestAtlasBuilder:
    def test_build_atlas_creates_charts(self, random_data):
        ab = AtlasBuilder(min_chart_size=50, n_neighbors=15)
        atlas = AtlasManager()
        ab.build(random_data, atlas)
        assert len(atlas) >= 1

    def test_build_multi_modal(self, multi_modal_data):
        data, labels = multi_modal_data
        ab = AtlasBuilder(min_chart_size=30, n_neighbors=10)
        atlas = AtlasManager()
        ab.build(data, atlas, modality_labels=labels)
        assert len(atlas) >= 1

    def test_analyze_quality(self, random_data):
        ab = AtlasBuilder(min_chart_size=50, n_neighbors=15)
        atlas = AtlasManager()
        ab.build(random_data, atlas)
        quality = ab.analyze_quality(atlas, random_data)
        assert "coverage" in quality
