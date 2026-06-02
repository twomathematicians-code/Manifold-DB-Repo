"""
Unit tests for manifold_db.geodesic — solver, distances, exponential maps.
"""

import numpy as np
import pytest

from manifold_db.geodesic import (
    DistanceComputer,
    ExponentialMap,
    FisherRaoDistance,
    GeodesicSolver,
    RetractionMap,
    RiemannianDistance,
    WassersteinDistance,
    compute_christoffel_symbols,
)


def _euclidean_metric(x):
    """Identity metric for Euclidean space."""
    dim = len(x)
    return np.eye(dim)


class TestChristoffelSymbols:
    def test_zero_for_euclidean(self):
        n = 3
        g = np.eye(n)
        # Derivatives of identity metric are all zero
        dg = np.zeros((n, n, n))
        Gamma = compute_christoffel_symbols(g, dg)
        np.testing.assert_allclose(Gamma, 0.0, atol=1e-15)


class TestGeodesicSolver:
    def test_euclidean_straight_line(self):
        solver = GeodesicSolver(metric_fn=_euclidean_metric, dim=3)
        p0 = np.zeros(3)
        v0 = np.array([1.0, 0.0, 0.0])
        result = solver.solve_geodesic(p0, v0, t_span=(0.0, 1.0), method="rk4", dt=0.01)
        assert result.success
        # Final point should be approximately (1, 0, 0)
        np.testing.assert_allclose(result.trajectory[-1], [1.0, 0.0, 0.0], atol=0.01)

    def test_geodesic_distance_euclidean(self):
        solver = GeodesicSolver(metric_fn=_euclidean_metric, dim=3)
        p = np.zeros(3)
        q = np.array([3.0, 4.0, 0.0])
        dist = solver.geodesic_distance(p, q, n_paths=5)
        assert abs(dist - 5.0) < 1.0  # Should approximate Euclidean distance

    def test_geodesic_ball(self):
        solver = GeodesicSolver(metric_fn=_euclidean_metric, dim=3)
        center = np.zeros(3)
        pts = solver.geodesic_ball(center, radius=1.0, n_samples=100)
        assert pts.shape[0] == 100
        assert pts.shape[1] == 3
        # Points should be roughly within radius
        dists = np.linalg.norm(pts, axis=1)
        assert np.mean(dists) < 2.0


class TestRiemannianDistance:
    def test_tangent_approximation(self):
        rd = RiemannianDistance(metric_tensor_fn=_euclidean_metric)
        p = np.zeros(3)
        q = np.array([1.0, 0.0, 0.0])
        d = rd.tangent_approx_distance(p, q)
        assert abs(d - 1.0) < 0.5


class TestWassersteinDistance:
    def test_sinkhorn_same_distribution(self):
        wd = WassersteinDistance(reg=0.1, max_iter=200)
        mu = np.array([0.5, 0.5])
        nu = np.array([0.5, 0.5])
        cost = np.array([[0.0, 1.0], [1.0, 0.0]])
        dist = wd.sinkhorn_distance(mu, nu, cost)
        assert dist >= 0

    def test_sinkhorn_different_distributions(self):
        wd = WassersteinDistance(reg=0.1, max_iter=200)
        mu = np.array([1.0, 0.0])
        nu = np.array([0.0, 1.0])
        cost = np.array([[0.0, 1.0], [1.0, 0.0]])
        dist = wd.sinkhorn_distance(mu, nu, cost)
        assert dist > 0.1


class TestFisherRaoDistance:
    def test_identical_distributions(self):
        fr = FisherRaoDistance()
        p = np.array([0.5, 0.5])
        q = np.array([0.5, 0.5])
        dist = fr.fisher_rao_distance(p, q)
        assert dist < 1e-10

    def test_orthogonal_distributions(self):
        fr = FisherRaoDistance()
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        dist = fr.fisher_rao_distance(p, q)
        assert dist > 0


class TestDistanceComputer:
    def test_euclidean_compute(self):
        dc = DistanceComputer(metric_tensor_fn=_euclidean_metric)
        p = np.zeros(3)
        q = np.array([1.0, 0.0, 0.0])
        d = dc.compute(p, q, metric_type="euclidean")
        assert abs(d - 1.0) < 0.5

    def test_batch_compute(self):
        dc = DistanceComputer(metric_tensor_fn=_euclidean_metric)
        points_a = np.zeros((5, 3))
        points_b = np.ones((5, 3))
        dists = dc.batch_compute(points_a, points_b, metric_type="euclidean")
        assert dists.shape == (5,)
        assert np.all(dists >= 0)


class TestExponentialMap:
    def test_exp_map_euclidean(self):
        em = ExponentialMap(metric_fn=_euclidean_metric)
        base = np.zeros(3)
        tangent = np.array([1.0, 0.0, 0.0])
        result = em.exp_map(base, tangent, steps=50)
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0], atol=0.1)

    def test_log_exp_roundtrip(self):
        em = ExponentialMap(metric_fn=_euclidean_metric)
        base = np.zeros(3)
        target = np.array([0.5, 0.3, 0.1])
        log_vec = em.log_map(base, target, max_iter=20)
        exp_pt = em.exp_map(base, log_vec, steps=50)
        np.testing.assert_allclose(exp_pt, target, atol=0.3)


class TestRetractionMap:
    def test_retract_euclidean(self):
        rm = RetractionMap(metric_fn=_euclidean_metric)
        base = np.zeros(3)
        vec = np.array([1.0, 2.0, 3.0])
        result = rm.retract(base, vec)
        # First-order approximation should be close to base + vec
        np.testing.assert_allclose(result, vec, atol=0.5)
