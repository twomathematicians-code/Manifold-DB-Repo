"""
test_manifold_db.py - Integration tests for the ManifoldDB database
======================================================================

Tests cover the full workflow:
- Insert data and build atlas
- kNN queries
- Geodesic ball queries
- Statistics retrieval
- Multi-modal data insertion and cross-modal queries
"""

from __future__ import annotations

import numpy as np
import pytest


class TestManifoldDBBasic:
    """Basic integration tests for ManifoldDB using the high-level Python API."""

    @pytest.fixture
    def db(self, temp_db_path):
        """Create a ManifoldDB instance for testing."""
        from manifolddb import ManifoldDB

        return ManifoldDB(
            storage_path=temp_db_path,
            intrinsic_dim=3,
            enable_cuda=False,
            geodesic_tolerance=1e-6,
        )

    def test_insert_and_build(self, db, rng):
        """Insert data points and build the atlas. Verify that charts are
        created and points are assigned.
        """
        n_points = 100
        ambient_dim = 10
        data = rng.standard_normal((n_points, ambient_dim))

        db.insert(data, modality=0)
        db.build()

        stats = db.stats()
        assert stats["num_charts"] >= 1, (
            "At least one chart should be discovered from the data"
        )
        assert stats["total_points"] == n_points

    def test_knn_query(self, db, rng):
        """Insert data, build, and query kNN. Verify that results are returned."""
        n_points = 100
        ambient_dim = 10
        data = rng.standard_normal((n_points, ambient_dim))

        db.insert(data, modality=0)
        db.build()

        query = data[0]
        results = db.query_knn(query, k=5)

        assert len(results) > 0, "kNN query should return at least one result"
        assert len(results) <= 5

        # Each result should have the expected keys
        for r in results:
            assert "id" in r
            assert "distance" in r
            assert "chart_id" in r
            assert r["distance"] >= 0.0

        # The nearest neighbour should be the query point itself (or very close)
        if len(results) > 0:
            assert results[0]["distance"] < 1.0, (
                "Nearest neighbour should be very close to the query"
            )

    def test_ball_query(self, db, rng):
        """Query a geodesic ball around a center point."""
        n_points = 200
        ambient_dim = 8
        data = rng.standard_normal((n_points, ambient_dim))

        db.insert(data, modality=0)
        db.build()

        center = data[n_points // 2]
        radius = 5.0

        results = db.query_ball(center, radius=radius)

        assert isinstance(results, list)

        # Each result should have point data
        for r in results:
            assert "id" in r
            assert "chart_id" in r
            assert "local_coords" in r

    def test_stats(self, db, rng):
        """Check that stats are populated correctly after building."""
        n_points = 150
        ambient_dim = 10
        data = rng.standard_normal((n_points, ambient_dim))

        db.insert(data, modality=0)
        db.build()

        stats = db.stats()

        assert "num_charts" in stats
        assert "total_points" in stats
        assert "index_size" in stats
        assert stats["total_points"] == n_points
        assert stats["num_charts"] >= 1


class TestManifoldDBMultimodal:
    """Integration tests for multi-modal data in ManifoldDB."""

    @pytest.fixture
    def db(self, temp_db_path):
        """Create a ManifoldDB instance for multi-modal testing."""
        from manifolddb import ManifoldDB

        return ManifoldDB(
            storage_path=temp_db_path,
            intrinsic_dim=4,
            enable_cuda=False,
            geodesic_tolerance=1e-6,
        )

    @staticmethod
    def _generate_cluster_data(rng, n_per_cluster=50, dim=16, n_clusters=3,
                                separation=5.0, seed_offset=0):
        """Generate clustered data simulating multi-modal embeddings."""
        rng_local = np.random.default_rng(seed=42 + seed_offset)

        all_data = []
        all_labels = []

        for i in range(n_clusters):
            angle = 2 * np.pi * i / n_clusters
            centroid = np.zeros(dim, dtype=np.float64)
            centroid[0] = separation * np.cos(angle)
            centroid[1] = separation * np.sin(angle)

            cluster = rng_local.normal(
                loc=centroid, scale=0.5, size=(n_per_cluster, dim)
            )
            all_data.append(cluster)
            all_labels.extend([i] * n_per_cluster)

        data = np.vstack(all_data)
        return data, all_labels

    def test_multimodal_insert(self, db, rng):
        """Insert two modalities and verify they are both stored."""
        dim = 16
        text_data, _ = self._generate_cluster_data(rng, dim=dim, seed_offset=0)
        image_data, _ = self._generate_cluster_data(rng, dim=dim, seed_offset=100)

        db.insert(text_data, modality=0)
        db.insert(image_data, modality=1)
        db.build()

        stats = db.stats()
        expected_total = text_data.shape[0] + image_data.shape[0]
        assert stats["total_points"] == expected_total

    def test_multimodal_cross_modal_query(self, db, rng):
        """Insert two modalities, build, and perform a cross-modal query."""
        dim = 16
        n_per_cluster = 40
        text_data, text_labels = self._generate_cluster_data(
            rng, n_per_cluster=n_per_cluster, dim=dim, seed_offset=0
        )
        image_data, image_labels = self._generate_cluster_data(
            rng, n_per_cluster=n_per_cluster, dim=dim, seed_offset=100
        )

        db.insert(text_data, modality=0)
        db.insert(image_data, modality=1)
        db.build()

        # Query: use a text point to find similar images
        query = text_data[0]
        k = 5
        results = db.cross_modal_query(
            query, source=0, target=1, k=k
        )

        assert isinstance(results, list)
        # May return fewer than k if no target points share the chart
        assert len(results) <= k

        # Verify result structure
        for r in results:
            assert "id" in r
            assert "distance" in r
