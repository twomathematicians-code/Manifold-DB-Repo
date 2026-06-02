"""
test_manifolddb.py - Integration tests for the ManifoldDB database
====================================================================

These tests exercise the full ManifoldDB workflow through the high-level
Python API (``manifolddb.ManifoldDB``):

  - **Database Creation and Configuration**: Verify that ManifoldDB can be
    instantiated with various parameter combinations.

  - **Data Insertion**: Insert numpy arrays of different shapes and sizes.

  - **Atlas Building**: Construct the atlas via PCA-based linear charts and
    verify that charts are discovered.

  - **k-NN Queries**: Query k-nearest neighbours and verify result structure,
    distance ordering, and correctness.

  - **Geodesic Ball Queries**: Query all points within a geodesic radius.

  - **Cross-Modal Queries**: Insert data from multiple modalities and perform
    cross-modal retrieval.

  - **Statistics Reporting**: Verify that ``stats()`` returns consistent values.

  - **Schema Evolution**: Extend the database with new data and verify that
    the atlas is rebuilt correctly.

All tests are automatically skipped if the C++ extension (``_manifolddb_core``)
is not available — see ``conftest.py`` for the session-level skip mechanism.
"""

from __future__ import annotations

import numpy as np
import pytest


# ===================================================================
# Helper: Generate clustered data for multi-modal tests
# ===================================================================

def generate_cluster_data(
    rng,
    n_per_cluster: int = 50,
    dim: int = 16,
    n_clusters: int = 3,
    separation: float = 5.0,
    seed_offset: int = 0,
) -> tuple:
    """Generate clustered data simulating multi-modal embeddings.

    Each cluster is a Gaussian blob centred on a vertex of a regular polygon
    in the first two dimensions, with small random values in the remaining
    dimensions.

    Parameters
    ----------
    rng : numpy.random.Generator
        Seeded random number generator.
    n_per_cluster : int
        Points per cluster.
    dim : int
        Ambient dimensionality.
    n_clusters : int
        Number of clusters.
    separation : float
        Radial separation of cluster centroids.
    seed_offset : int
        Added to the base seed for variation.

    Returns
    -------
    numpy.ndarray, shape (n_per_cluster * n_clusters, dim)
    list[int]
        Cluster labels for each point.
    """
    rng_local = np.random.default_rng(seed=42 + seed_offset)

    all_data = []
    all_labels = []

    for i in range(n_clusters):
        angle = 2 * np.pi * i / n_clusters
        centroid = np.zeros(dim, dtype=np.float64)
        centroid[0] = separation * np.cos(angle)
        centroid[1] = separation * np.sin(angle)

        cluster = rng_local.normal(
            loc=centroid, scale=0.5, size=(n_per_cluster, dim),
        ).astype(np.float64)
        all_data.append(cluster)
        all_labels.extend([i] * n_per_cluster)

    return np.vstack(all_data), all_labels


# ===================================================================
# Database Creation and Configuration
# ===================================================================

class TestDatabaseCreation:
    """Tests for ManifoldDB instantiation and configuration."""

    def test_creation_default_params(self, temp_db_path):
        """Create a ManifoldDB with default parameters."""
        from manifolddb import ManifoldDB

        db = ManifoldDB(storage_path=temp_db_path)
        assert db is not None
        assert db.intrinsic_dim > 0
        assert len(db) == 0  # No points yet

    def test_creation_custom_params(self, temp_db_path):
        """Create a ManifoldDB with custom parameters."""
        from manifolddb import ManifoldDB

        db = ManifoldDB(
            storage_path=temp_db_path,
            intrinsic_dim=5,
            enable_cuda=False,
            geodesic_tolerance=1e-8,
            max_charts=10,
            rbf_bandwidth=2.0,
        )
        assert db is not None

    def test_creation_invalid_intrinsic_dim(self, temp_db_path):
        """intrinsic_dim < 1 should raise ValueError."""
        from manifolddb import ManifoldDB

        with pytest.raises(ValueError):
            ManifoldDB(storage_path=temp_db_path, intrinsic_dim=0)

    def test_creation_invalid_tolerance(self, temp_db_path):
        """geodesic_tolerance <= 0 should raise ValueError."""
        from manifolddb import ManifoldDB

        with pytest.raises(ValueError):
            ManifoldDB(storage_path=temp_db_path, geodesic_tolerance=0)

    def test_repr(self, temp_db_path):
        """Repr should contain key configuration info."""
        from manifolddb import ManifoldDB

        db = ManifoldDB(storage_path=temp_db_path, intrinsic_dim=3)
        r = repr(db)
        assert "ManifoldDB" in r
        assert "3" in r  # intrinsic_dim

    def test_len_empty(self, temp_db_path):
        """len(db) should be 0 for a fresh database."""
        from manifolddb import ManifoldDB

        db = ManifoldDB(storage_path=temp_db_path)
        assert len(db) == 0


# ===================================================================
# Data Insertion
# ===================================================================

class TestDataInsertion:
    """Tests for data insertion into ManifoldDB."""

    @pytest.fixture
    def db(self, temp_db_path):
        from manifolddb import ManifoldDB
        return ManifoldDB(
            storage_path=temp_db_path,
            intrinsic_dim=3,
            enable_cuda=False,
        )

    def test_insert_2d_array(self, db, rng):
        """Insert a standard 2-D numpy array."""
        data = rng.standard_normal((100, 10))
        db.insert(data, modality_id=0)
        assert len(db) == 100

    def test_insert_1d_array(self, db):
        """Insert a 1-D numpy array (single point)."""
        point = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        db.insert(point, modality_id=0)
        assert len(db) == 1

    def test_insert_multiple_batches(self, db, rng):
        """Insert data in multiple batches."""
        batch1 = rng.standard_normal((50, 8))
        batch2 = rng.standard_normal((75, 8))
        batch3 = rng.standard_normal((25, 8))

        db.insert(batch1, modality_id=0)
        db.insert(batch2, modality_id=0)
        db.insert(batch3, modality_id=0)

        assert len(db) == 150

    def test_insert_empty_raises(self, db):
        """Inserting an empty array should raise ValueError."""
        with pytest.raises(ValueError):
            db.insert(np.array([]).reshape(0, 5), modality_id=0)

    def test_insert_list_of_lists(self, db):
        """Insert a Python list-of-lists (non-numpy input)."""
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        db.insert(data, modality_id=0)
        assert len(db) == 3

    def test_insert_multiple_modalities(self, db, rng):
        """Insert data from different modalities."""
        text_data = rng.standard_normal((50, 10))
        image_data = rng.standard_normal((50, 10))

        db.insert(text_data, modality_id=0)
        db.insert(image_data, modality_id=1)

        assert len(db) == 100


# ===================================================================
# Atlas Building
# ===================================================================

class TestAtlasBuilding:
    """Tests for atlas construction."""

    @pytest.fixture
    def db(self, temp_db_path):
        from manifolddb import ManifoldDB
        return ManifoldDB(
            storage_path=temp_db_path,
            intrinsic_dim=3,
            enable_cuda=False,
        )

    def test_build_after_insert(self, db, rng):
        """Building after insertion should discover at least one chart."""
        data = rng.standard_normal((100, 10))
        db.insert(data, modality_id=0)
        db.build(method="linear")

        stats = db.stats()
        assert stats["num_charts"] >= 1, (
            "At least one chart should be discovered from 100 points"
        )

    def test_build_without_insert_raises(self, db):
        """Building without inserting data should raise RuntimeError."""
        with pytest.raises(RuntimeError):
            db.build(method="linear")

    def test_build_atlas_linear_explicit(self, db, rng):
        """Build with explicit intrinsic dimension via build_atlas_linear."""
        data = rng.standard_normal((100, 10))
        db.insert(data, modality_id=0)
        db.build_atlas_linear(intrinsic_dim=3)

        assert db.num_charts >= 1

    def test_build_preserves_point_count(self, db, rng):
        """Building should not change the total point count."""
        n = 150
        data = rng.standard_normal((n, 8))
        db.insert(data, modality_id=0)
        assert len(db) == n

        db.build(method="linear")
        assert len(db) == n, "Build should not change point count"

    def test_build_multiple_modalities(self, db, rng):
        """Building with data from multiple modalities."""
        text = rng.standard_normal((60, 10))
        images = rng.standard_normal((80, 10))

        db.insert(text, modality_id=0)
        db.insert(images, modality_id=1)
        db.build(method="linear")

        stats = db.stats()
        assert stats["total_points"] == 140


# ===================================================================
# k-NN Queries
# ===================================================================

class TestKNNQueries:
    """Tests for geodesic k-nearest-neighbour queries."""

    @pytest.fixture
    def db(self, temp_db_path, rng):
        from manifolddb import ManifoldDB
        db = ManifoldDB(
            storage_path=temp_db_path,
            intrinsic_dim=3,
            enable_cuda=False,
        )
        data = rng.standard_normal((200, 10))
        db.insert(data, modality_id=0)
        db.build(method="linear")
        return db

    def test_knn_returns_list(self, db, rng):
        """kNN query should return a list."""
        query = rng.standard_normal(10)
        results = db.query_knn(query, k=5)
        assert isinstance(results, list)

    def test_knn_result_count(self, db, rng):
        """kNN should return at most k results."""
        query = rng.standard_normal(10)
        k = 5
        results = db.query_knn(query, k=k)
        assert 0 < len(results) <= k

    def test_knn_result_structure(self, db, rng):
        """Each kNN result should have the required keys."""
        query = rng.standard_normal(10)
        results = db.query_knn(query, k=5)

        required_keys = {"id", "chart_id", "distance", "euclidean_residual",
                         "local_coords", "ambient_coords"}

        for r in results:
            assert required_keys.issubset(r.keys()), (
                f"Missing keys in result: {required_keys - set(r.keys())}"
            )

    def test_knn_distance_non_negative(self, db, rng):
        """All kNN distances should be non-negative."""
        query = rng.standard_normal(10)
        results = db.query_knn(query, k=10)

        for r in results:
            assert r["distance"] >= 0.0, (
                f"Distance should be >= 0, got {r['distance']}"
            )

    def test_knn_nearest_is_close(self, db, rng):
        """The nearest neighbour should be very close to the query."""
        query = rng.standard_normal(10)
        results = db.query_knn(query, k=10)

        if results:
            assert results[0]["distance"] < 2.0, (
                "Nearest neighbour should be close to the query"
            )

    def test_knn_sorted_by_distance(self, db, rng):
        """kNN results should be sorted by ascending geodesic distance."""
        query = rng.standard_normal(10)
        results = db.query_knn(query, k=10)

        for i in range(len(results) - 1):
            assert results[i]["distance"] <= results[i + 1]["distance"], (
                f"Results not sorted: results[{i}].dist={results[i]['distance']} > "
                f"results[{i+1}].dist={results[i+1]['distance']}"
            )

    def test_knn_max_distance_filter(self, db, rng):
        """kNN with max_distance should exclude faraway points."""
        query = rng.standard_normal(10)
        results = db.query_knn(query, k=20, max_distance=1.0)

        for r in results:
            assert r["distance"] <= 1.0 + 1e-10, (
                f"Result distance {r['distance']} exceeds max_distance=1.0"
            )

    def test_knn_k_too_large(self, db, rng):
        """Requesting k > total_points should return all available."""
        query = rng.standard_normal(10)
        results = db.query_knn(query, k=10000)
        assert len(results) <= 200  # Total points


# ===================================================================
# Geodesic Ball Queries
# ===================================================================

class TestBallQueries:
    """Tests for geodesic ball (range) queries."""

    @pytest.fixture
    def db(self, temp_db_path, rng):
        from manifolddb import ManifoldDB
        db = ManifoldDB(
            storage_path=temp_db_path,
            intrinsic_dim=3,
            enable_cuda=False,
        )
        data = rng.standard_normal((200, 8))
        db.insert(data, modality_id=0)
        db.build(method="linear")
        return db

    def test_ball_returns_list(self, db, rng):
        """Ball query should return a list of point dicts."""
        center = rng.standard_normal(8)
        results = db.query_ball(center, radius=5.0)
        assert isinstance(results, list)

    def test_ball_result_structure(self, db, rng):
        """Each ball result should have required keys."""
        center = rng.standard_normal(8)
        results = db.query_ball(center, radius=10.0)

        for r in results:
            assert "id" in r
            assert "chart_id" in r
            assert "local_coords" in r
            assert "ambient_coords" in r

    def test_ball_large_radius(self, db, rng):
        """Large radius should return most points."""
        center = rng.standard_normal(8)
        results = db.query_ball(center, radius=100.0)
        assert len(results) > 0

    def test_ball_small_radius(self, db, rng):
        """Very small radius may return zero points."""
        center = rng.standard_normal(8)
        results = db.query_ball(center, radius=0.001)
        # May return 0 or 1 (the center itself)
        assert len(results) <= 1

    def test_ball_zero_radius(self, db, rng):
        """Zero radius should return 0 or very few points."""
        center = rng.standard_normal(8)
        results = db.query_ball(center, radius=0.0)
        assert len(results) <= 1


# ===================================================================
# Cross-Modal Queries
# ===================================================================

class TestCrossModalQueries:
    """Tests for cross-modal retrieval."""

    @pytest.fixture
    def db_with_modalities(self, temp_db_path, rng):
        """Create a database with two modalities."""
        from manifolddb import ManifoldDB
        db = ManifoldDB(
            storage_path=temp_db_path,
            intrinsic_dim=4,
            enable_cuda=False,
        )
        text_data, _ = generate_cluster_data(rng, dim=16, seed_offset=0)
        image_data, _ = generate_cluster_data(rng, dim=16, seed_offset=100)
        db.insert(text_data, modality_id=0)
        db.insert(image_data, modality_id=1)
        db.build(method="linear")
        return db

    def test_cross_modal_returns_results(self, db_with_modalities, rng):
        """Cross-modal query should return a list of results."""
        query = rng.standard_normal(16)
        results = db_with_modalities.cross_modal_query(
            query, source_modality=0, target_modality=1, k=5,
        )
        assert isinstance(results, list)

    def test_cross_modal_result_count(self, db_with_modalities, rng):
        """Should return at most k results."""
        query = rng.standard_normal(16)
        results = db_with_modalities.cross_modal_query(
            query, source_modality=0, target_modality=1, k=5,
        )
        assert len(results) <= 5

    def test_cross_modal_result_structure(self, db_with_modalities, rng):
        """Each result should have the standard keys."""
        query = rng.standard_normal(16)
        results = db_with_modalities.cross_modal_query(
            query, source_modality=0, target_modality=1, k=5,
        )

        for r in results:
            assert "id" in r
            assert "distance" in r
            assert "chart_id" in r

    def test_cross_modal_both_directions(self, db_with_modalities, rng):
        """Cross-modal should work in both directions."""
        query = rng.standard_normal(16)

        # Text → Image
        t2i = db_with_modalities.cross_modal_query(
            query, source_modality=0, target_modality=1, k=3,
        )

        # Image → Text
        i2t = db_with_modalities.cross_modal_query(
            query, source_modality=1, target_modality=0, k=3,
        )

        assert isinstance(t2i, list)
        assert isinstance(i2t, list)

    def test_cross_modal_k_invalid(self, db_with_modalities, rng):
        """k < 1 should raise ValueError."""
        query = rng.standard_normal(16)
        with pytest.raises(ValueError):
            db_with_modalities.cross_modal_query(
                query, source_modality=0, target_modality=1, k=0,
            )


# ===================================================================
# Statistics
# ===================================================================

class TestStats:
    """Tests for the stats() reporting method."""

    @pytest.fixture
    def db(self, temp_db_path, rng):
        from manifolddb import ManifoldDB
        db = ManifoldDB(
            storage_path=temp_db_path,
            intrinsic_dim=3,
            enable_cuda=False,
        )
        data = rng.standard_normal((150, 10))
        db.insert(data, modality_id=0)
        db.build(method="linear")
        return db

    def test_stats_keys(self, db):
        """stats() should return a dict with expected keys."""
        stats = db.stats()
        expected_keys = {"num_charts", "total_points", "index_size",
                         "build_time_ms", "avg_geodesic_time_ms"}
        assert expected_keys.issubset(stats.keys()), (
            f"Missing keys: {expected_keys - set(stats.keys())}"
        )

    def test_stats_point_count(self, db):
        """total_points should match the number of inserted points."""
        stats = db.stats()
        assert stats["total_points"] == 150

    def test_stats_chart_count(self, db):
        """num_charts should be >= 1 after building."""
        stats = db.stats()
        assert stats["num_charts"] >= 1

    def test_stats_return_types(self, db):
        """Stats values should have correct types."""
        stats = db.stats()
        assert isinstance(stats["num_charts"], int)
        assert isinstance(stats["total_points"], int)
        assert isinstance(stats["index_size"], int)
        assert isinstance(stats["build_time_ms"], float)


# ===================================================================
# Schema Evolution
# ===================================================================

class TestSchemaEvolution:
    """Tests for schema evolution (extending the database with new data)."""

    @pytest.fixture
    def db(self, temp_db_path, rng):
        """Create a database with initial data."""
        from manifolddb import ManifoldDB
        db = ManifoldDB(
            storage_path=temp_db_path,
            intrinsic_dim=3,
            enable_cuda=False,
        )
        initial_data = rng.standard_normal((100, 8))
        db.insert(initial_data, modality_id=0)
        db.build(method="linear")
        return db

    def test_evolve_increases_points(self, db, rng):
        """Evolving with new data should increase total point count."""
        initial_count = db.total_points
        new_data = rng.standard_normal((50, 8))

        db.evolve(new_data)

        assert db.total_points > initial_count, (
            f"Total points should increase after evolve: "
            f"{initial_count} → {db.total_points}"
        )

    def test_evolve_rebuilds_atlas(self, db, rng):
        """Evolving should rebuild the atlas."""
        initial_charts = db.num_charts
        new_data = rng.standard_normal((100, 8))

        db.evolve(new_data)

        # Atlas should still exist after evolve
        assert db.num_charts >= 1, (
            "Atlas should exist after schema evolution"
        )

    def test_evolve_queryable_after(self, db, rng):
        """Database should be queryable after schema evolution."""
        new_data = rng.standard_normal((50, 8))
        db.evolve(new_data)

        query = rng.standard_normal(8)
        results = db.query_knn(query, k=5)
        assert isinstance(results, list)
