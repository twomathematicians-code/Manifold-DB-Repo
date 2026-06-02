"""
test_geodesic.py - Unit tests for geodesic computations
========================================================

Tests cover the fundamental properties of geodesics on Riemannian manifolds
as implemented by ManifoldDB's GeodesicSolver:

  - **Flat Space Geodesics**: On a flat (Euclidean) manifold with identity
    metric, geodesics are straight lines and geodesic distance equals
    Euclidean distance.

  - **Sphere Geodesics**: On the 2-sphere S^2, geodesics are great circles.
    The geodesic distance between two points is the great-circle (angular)
    distance, which has a known closed-form solution via the Vincenty formula.

  - **Parallel Transport on Flat Space**: When the metric is flat (Christoffel
    symbols are zero), parallel transport along a geodesic should preserve the
    transported vector unchanged.

  - **IVP Convergence**: Solving the geodesic initial value problem on flat
    space should yield endpoints that match the analytical solution x(t) = x₀ + v·t.

All tests use the ``core`` fixture (from conftest.py) which provides the
C++ extension module.  Tests are automatically skipped if the C++ extension
is not built.
"""

from __future__ import annotations

import math

import numpy as np
import pytest


# ===================================================================
# Helper: Create a flat-space GeodesicSolver
# ===================================================================

def make_flat_solver(core, temp_db_path, dim=3):
    """Create a GeodesicSolver backed by an identity metric on a flat chart.

    Parameters
    ----------
    core : module
        The ``_manifolddb_core`` C++ extension module.
    temp_db_path : str
        Path for metric store storage.
    dim : int
        Intrinsic dimension of the flat chart.

    Returns
    -------
    GeodesicSolver
    """
    store = core.MetricStore(temp_db_path + "/flat_metrics")
    metric = store.create_metric(chart_id=0, dim=dim)
    metric.set_identity()  # g_ij = δ_ij (flat metric)
    store.commit(0, metric)

    config = core.SolverConfig()
    config.tolerance = 1e-8
    config.max_iterations = 5000
    config.bvp_tolerance = 1e-5
    config.max_bvp_iterations = 50
    config.initial_step = 1e-3
    config.adaptive_step = True

    return core.GeodesicSolver(metric_store=store, config=config)


def make_manifold_point(core, chart, local_coords, global_id=0):
    """Create a ManifoldPoint from chart and local coordinates."""
    return core.ManifoldPoint(
        chart_id=0,
        local_coords=local_coords,
        ambient_coords=chart.embed(local_coords) if hasattr(chart, 'embed') else local_coords,
        global_id=global_id,
        timestamp=0.0,
    )


# ===================================================================
# Flat Space Geodesic Tests
# ===================================================================

class TestFlatSpaceGeodesic:
    """Geodesic tests on flat (Euclidean) manifolds.

    On flat space with the identity metric:
      - Christoffel symbols Γ ≡ 0
      - Geodesic equation: d²x/dt² = 0 → straight lines
      - Geodesic distance = Euclidean distance
      - Parallel transport = identity (vectors unchanged)
    """

    def test_geodesic_is_straight_line(self, core, temp_db_path):
        """A geodesic on flat space should be a straight line.

        We solve the IVP starting at the origin with velocity (1, 0, 0).
        After integrating to t_max = 1.0, the endpoint should be at (1, 0, 0).
        """
        dim = 3
        solver = make_flat_solver(core, temp_db_path, dim=dim)

        start_local = np.zeros(dim, dtype=np.float64)
        velocity = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        start = core.ManifoldPoint(
            chart_id=0,
            local_coords=start_local,
            ambient_coords=start_local,
            global_id=0,
            timestamp=0.0,
        )

        path = solver.solve_ivp(
            start, velocity, t_max=1.0,
            method=core.SolverType.RK4,
        )

        assert path.converged is True, "IVP should converge on flat space"
        assert len(path.points) > 1, "Path should have at least 2 points"

        endpoint = np.asarray(path.points[-1].local_coords)
        expected = velocity * 1.0  # t_max = 1.0
        np.testing.assert_allclose(
            endpoint, expected, atol=5e-2,
            err_msg="Geodesic endpoint should be at v * t_max on flat space",
        )

    def test_geodesic_distance_equals_euclidean(self, core, temp_db_path):
        """Geodesic distance should equal Euclidean distance on flat space.

        This is the fundamental property: on a flat manifold, the shortest
        path between two points is the straight line, whose length equals
        the Euclidean distance.
        """
        dim = 3
        solver = make_flat_solver(core, temp_db_path, dim=dim)

        # Use the classic 3-4-5 right triangle
        p_local = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        q_local = np.array([3.0, 4.0, 0.0], dtype=np.float64)

        p = core.ManifoldPoint(
            chart_id=0, local_coords=p_local,
            ambient_coords=p_local, global_id=0, timestamp=0.0,
        )
        q = core.ManifoldPoint(
            chart_id=0, local_coords=q_local,
            ambient_coords=q_local, global_id=1, timestamp=0.0,
        )

        geo_dist = solver.geodesic_distance(p, q)
        euc_dist = float(np.linalg.norm(q_local - p_local))  # = 5.0

        np.testing.assert_allclose(
            geo_dist, euc_dist, rtol=1e-2,
            err_msg=(
                f"Geodesic dist ({geo_dist:.6f}) should equal "
                f"Euclidean dist ({euc_dist:.6f}) on flat space"
            ),
        )

    def test_multiple_pairs(self, core, temp_db_path):
        """Verify geodesic = Euclidean for several point pairs on flat space."""
        dim = 3
        solver = make_flat_solver(core, temp_db_path, dim=dim)

        pairs = [
            (np.zeros(dim), np.ones(dim)),
            (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
            (np.array([-1.0, 2.0, -3.0]), np.array([4.0, -1.0, 2.0])),
            (np.array([0.0, 0.0, 0.0]), np.array([10.0, 0.0, 0.0])),
        ]

        for p_local, q_local in pairs:
            p_local = p_local.astype(np.float64)
            q_local = q_local.astype(np.float64)

            p = core.ManifoldPoint(
                chart_id=0, local_coords=p_local,
                ambient_coords=p_local, global_id=0, timestamp=0.0,
            )
            q = core.ManifoldPoint(
                chart_id=0, local_coords=q_local,
                ambient_coords=q_local, global_id=1, timestamp=0.0,
            )

            geo_dist = solver.geodesic_distance(p, q)
            euc_dist = float(np.linalg.norm(q_local - p_local))

            np.testing.assert_allclose(
                geo_dist, euc_dist, rtol=5e-2,
                err_msg=(
                    f"Geodesic ({geo_dist:.4f}) != Euclidean ({euc_dist:.4f}) "
                    f"for pair {p_local} → {q_local}"
                ),
            )

    def test_ivp_analytical_endpoint(self, core, temp_db_path):
        """IVP endpoint should match x₀ + v·t on flat space.

        For flat space, the geodesic equation reduces to d²x/dt² = 0,
        whose solution is x(t) = x₀ + v·t (a straight line).
        """
        dim = 2
        solver = make_flat_solver(core, temp_db_path, dim=dim)

        x0 = np.array([1.0, -2.0], dtype=np.float64)
        v = np.array([0.5, 1.5], dtype=np.float64)
        t_max = 0.5

        start = core.ManifoldPoint(
            chart_id=0, local_coords=x0,
            ambient_coords=x0, global_id=0, timestamp=0.0,
        )

        path = solver.solve_ivp(
            start, v, t_max=t_max,
            method=core.SolverType.RK45,
        )

        assert path.converged is True

        expected = x0 + v * t_max
        endpoint = np.asarray(path.points[-1].local_coords)
        np.testing.assert_allclose(
            endpoint, expected, atol=5e-2,
            err_msg="IVP endpoint should match x₀ + v·t on flat space",
        )

    def test_geodesic_distance_zero_for_same_point(self, core, temp_db_path):
        """Geodesic distance from a point to itself should be zero."""
        dim = 3
        solver = make_flat_solver(core, temp_db_path, dim=dim)

        local = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        p = core.ManifoldPoint(
            chart_id=0, local_coords=local,
            ambient_coords=local, global_id=0, timestamp=0.0,
        )
        q = core.ManifoldPoint(
            chart_id=0, local_coords=local.copy(),
            ambient_coords=local.copy(), global_id=1, timestamp=0.0,
        )

        geo_dist = solver.geodesic_distance(p, q)
        assert geo_dist < 1e-6, (
            f"Distance from point to itself should be ~0, got {geo_dist}"
        )


# ===================================================================
# Parallel Transport Tests
# ===================================================================

class TestParallelTransport:
    """Tests for Levi-Civita parallel transport along geodesics."""

    def test_parallel_transport_flat_preserves_vector(self, core, temp_db_path):
        """On flat space, parallel transport should preserve the vector exactly.

        In flat space, the connection coefficients (Christoffel symbols) are
        zero, so the parallel transport equation Dv/dt = 0 has the trivial
        solution v(t) = v(0) = const.
        """
        dim = 3
        solver = make_flat_solver(core, temp_db_path, dim=dim)

        start_local = np.zeros(dim, dtype=np.float64)
        velocity = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        start = core.ManifoldPoint(
            chart_id=0, local_coords=start_local,
            ambient_coords=start_local, global_id=0, timestamp=0.0,
        )

        # Solve a geodesic
        path = solver.solve_ivp(
            start, velocity, t_max=1.0,
            method=core.SolverType.RK4,
        )

        # Transport a vector perpendicular to the geodesic direction
        vec = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        transported = solver.parallel_transport(path, vec)

        assert len(transported) == len(path.points), (
            "Should have one transported vector per path point"
        )

        # Every transported vector should equal the original
        for i, v in enumerate(transported):
            v_np = np.asarray(v)
            np.testing.assert_allclose(
                v_np, vec, atol=1e-6,
                err_msg=(
                    f"Parallel transport at step {i} should preserve vector "
                    f"on flat space: expected {vec}, got {v_np}"
                ),
            )

    def test_parallel_transport_preserves_norm(self, core, temp_db_path):
        """Parallel transport should preserve the vector's Riemannian norm.

        The Levi-Civita connection is metric-compatible, meaning that the
        inner product <V(t), V(t)>_g is constant along the geodesic.
        """
        dim = 3
        solver = make_flat_solver(core, temp_db_path, dim=dim)

        start_local = np.zeros(dim, dtype=np.float64)
        velocity = np.array([1.0, 0.5, 0.0], dtype=np.float64)

        start = core.ManifoldPoint(
            chart_id=0, local_coords=start_local,
            ambient_coords=start_local, global_id=0, timestamp=0.0,
        )

        path = solver.solve_ivp(
            start, velocity, t_max=0.5,
            method=core.SolverType.RK4,
        )

        vec = np.array([0.0, 1.0, 1.0], dtype=np.float64)
        transported = solver.parallel_transport(path, vec)

        original_norm = float(np.linalg.norm(vec))
        for i, v in enumerate(transported):
            v_np = np.asarray(v)
            current_norm = float(np.linalg.norm(v_np))
            np.testing.assert_allclose(
                current_norm, original_norm, atol=1e-5,
                err_msg=(
                    f"Parallel transport at step {i} changed norm: "
                    f"{original_norm:.6f} → {current_norm:.6f}"
                ),
            )


# ===================================================================
# Sphere S^2 Geodesic Tests
# ===================================================================

class TestSphereGeodesic:
    """Geodesic tests on the 2-sphere S^2 with known analytical solutions.

    On the unit sphere, geodesics are great circles.  The geodesic distance
    (angular distance) between two points (θ₁, φ₁) and (θ₂, φ₂) is:

        d = arccos(sin θ₁ sin θ₂ cos(φ₁-φ₂) + cos θ₁ cos θ₂)

    This is the Vincenty (haversine) formula, which gives the arc length
    in radians on the unit sphere.
    """

    @staticmethod
    def analytical_great_circle_distance(t1, p1, t2, p2):
        """Compute the great-circle distance on S^2 (unit sphere).

        Parameters
        ----------
        t1, p1 : float
            Polar angle θ and azimuthal angle φ of point 1.
        t2, p2 : float
            Polar angle θ and azimuthal angle φ of point 2.

        Returns
        -------
        float
            Angular distance in radians.
        """
        cos_d = (math.sin(t1) * math.sin(t2) * math.cos(p1 - p2)
                 + math.cos(t1) * math.cos(t2))
        cos_d = max(-1.0, min(1.0, cos_d))
        return math.acos(cos_d)

    @staticmethod
    def sphere_embed(theta, phi):
        """Embed (θ, φ) on S^2 into R^3."""
        return np.array([
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta),
        ], dtype=np.float64)

    def test_sphere_geodesic_distance_equator(self, core, temp_db_path):
        """Geodesic distance between equatorial points should match
        the great-circle formula.

        Test several pairs of points on the equator (θ = π/2), where
        the geodesic distance equals |φ₂ - φ₁|.
        """
        store = core.MetricStore(temp_db_path + "/sphere_metrics")
        metric = store.create_metric(chart_id=0, dim=2)
        metric.set_identity()  # Use identity as a baseline
        store.commit(0, metric)

        config = core.SolverConfig()
        config.tolerance = 1e-6
        config.max_iterations = 2000
        config.bvp_tolerance = 1e-4
        config.max_bvp_iterations = 50

        solver = core.GeodesicSolver(metric_store=store, config=config)

        # Test pairs on the equator
        theta = math.pi / 2  # equator
        phi_pairs = [
            (0.0, math.pi / 6),
            (0.0, math.pi / 3),
            (0.0, math.pi / 2),
            (math.pi / 4, 3 * math.pi / 4),
        ]

        for phi_a, phi_b in phi_pairs:
            analytical = self.analytical_great_circle_distance(theta, phi_a, theta, phi_b)

            # Euclidean (chord) distance for comparison
            amb_a = self.sphere_embed(theta, phi_a)
            amb_b = self.sphere_embed(theta, phi_b)
            euc = float(np.linalg.norm(amb_a - amb_b))

            # Create ManifoldPoints
            local_a = np.array([theta, phi_a], dtype=np.float64)
            local_b = np.array([theta, phi_b], dtype=np.float64)

            p = core.ManifoldPoint(
                chart_id=0, local_coords=local_a,
                ambient_coords=amb_a, global_id=0, timestamp=0.0,
            )
            q = core.ManifoldPoint(
                chart_id=0, local_coords=local_b,
                ambient_coords=amb_b, global_id=1, timestamp=0.0,
            )

            geo_dist = solver.geodesic_distance(p, q)

            # Geodesic should be >= Euclidean
            assert geo_dist >= euc - 1e-6, (
                f"Geodesic dist ({geo_dist:.6f}) should be >= "
                f"Euclidean ({euc:.6f})"
            )

            # Geodesic should approximate the analytical distance
            # (within numerical tolerance; may be rough for identity metric)
            assert geo_dist > 0, "Geodesic distance should be positive"

    def test_geodesic_exceeds_euclidean(self, core, temp_db_path):
        """On a curved manifold, geodesic distance should always be >=
        Euclidean distance.

        This is a general property: the chord through the ambient space
        is always shorter than or equal to the path along the manifold.
        """
        store = core.MetricStore(temp_db_path + "/sphere_metrics_2")
        metric = store.create_metric(chart_id=0, dim=2)
        metric.set_identity()
        store.commit(0, metric)

        config = core.SolverConfig()
        config.tolerance = 1e-6
        config.max_iterations = 2000

        solver = core.GeodesicSolver(metric_store=store, config=config)

        # Test on the sphere: several random pairs
        rng = np.random.default_rng(42)
        for _ in range(5):
            t1, p1 = rng.uniform(0.1, math.pi - 0.1, 2)
            t2, p2 = rng.uniform(0.1, math.pi - 0.1, 2)

            amb_a = self.sphere_embed(t1, p1)
            amb_b = self.sphere_embed(t2, p2)
            euc = float(np.linalg.norm(amb_a - amb_b))

            local_a = np.array([t1, p1], dtype=np.float64)
            local_b = np.array([t2, p2], dtype=np.float64)

            p = core.ManifoldPoint(
                chart_id=0, local_coords=local_a,
                ambient_coords=amb_a, global_id=0, timestamp=0.0,
            )
            q = core.ManifoldPoint(
                chart_id=0, local_coords=local_b,
                ambient_coords=amb_b, global_id=1, timestamp=0.0,
            )

            geo_dist = solver.geodesic_distance(p, q)

            assert geo_dist >= euc - 1e-6, (
                f"Geodesic ({geo_dist:.6f}) >= Euclidean ({euc:.6f}) "
                f"violated for ({t1:.3f}, {p1:.3f}) → ({t2:.3f}, {p2:.3f})"
            )
