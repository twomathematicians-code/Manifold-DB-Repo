"""
Performance benchmarks for Manifold Database components.
"""

import time

import numpy as np
import pytest


def _euclidean_metric(x):
    return np.eye(len(x))


@pytest.mark.slow
class TestBenchmarkInsert:
    def test_insert_1000_points(self):
        from manifold_db.tangent_index import TangentSpaceIndex

        np.random.seed(0)
        data = np.random.randn(1000, 50)
        ids = [f"pt_{i}" for i in range(1000)]

        idx = TangentSpaceIndex(intrinsic_dim=10, leaf_size=40)
        t0 = time.perf_counter()
        idx.build_from_data(ids, data, n_anchors=30)
        elapsed = time.perf_counter() - t0
        print(f"\nInsert 1000 points: {elapsed:.3f}s")
        assert idx.size == 1000


@pytest.mark.slow
class TestBenchmarkSearch:
    def test_search_k10_1000_points(self):
        from manifold_db.tangent_index import TangentSpaceIndex

        np.random.seed(0)
        data = np.random.randn(1000, 50)
        ids = [f"pt_{i}" for i in range(1000)]

        idx = TangentSpaceIndex(intrinsic_dim=10, leaf_size=40)
        idx.build_from_data(ids, data, n_anchors=30)

        # Warm up
        idx.search(data[0], k=10)

        queries = data[:100]
        t0 = time.perf_counter()
        for q in queries:
            idx.search(q, k=10)
        elapsed = time.perf_counter() - t0
        avg_ms = (elapsed / 100) * 1000
        print(f"\nSearch k=10 (avg over 100 queries): {avg_ms:.2f}ms")


@pytest.mark.slow
class TestBenchmarkGeodesic:
    def test_geodesic_distance_100_pairs(self):
        from manifold_db.geodesic import GeodesicSolver

        np.random.seed(0)
        solver = GeodesicSolver(metric_fn=_euclidean_metric, dim=10)
        points = np.random.randn(100, 10)

        t0 = time.perf_counter()
        for i in range(100):
            solver.geodesic_distance(points[i], points[(i + 1) % 100], n_paths=5)
        elapsed = time.perf_counter() - t0
        avg_ms = (elapsed / 100) * 1000
        print(f"\nGeodesic distance (avg over 100 pairs): {avg_ms:.2f}ms")


@pytest.mark.slow
class TestBenchmarkAtlasBuilding:
    def test_atlas_building_5000_points(self):
        from manifold_db.atlas import AtlasBuilder, AtlasManager

        np.random.seed(0)
        data = np.random.randn(5000, 30)

        builder = AtlasBuilder(min_chart_size=100, n_neighbors=15)
        atlas = AtlasManager(name="bench")
        t0 = time.perf_counter()
        builder.build(data, atlas)
        elapsed = time.perf_counter() - t0
        print(f"\nAtlas building 5000 points: {elapsed:.3f}s ({len(atlas)} charts)")
        assert len(atlas) >= 1
