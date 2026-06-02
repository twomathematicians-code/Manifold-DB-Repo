"""
test_geodesic_solver.py - Unit tests for the GeodesicSolver
============================================================

Tests cover:
- Geodesic computation on flat space (should be a straight line)
- Geodesic distance on flat space (should equal Euclidean distance)
- Parallel transport on flat space (should preserve vectors exactly)
- IVP convergence verification
"""

from __future__ import annotations

import numpy as np
import pytest


class TestGeodesicSolverFlatSpace:
    """Tests for the GeodesicSolver on flat (Euclidean) manifolds.

    In flat space with an identity metric, geodesics are straight lines,
    geodesic distance equals Euclidean distance, and parallel transport
    is trivial (vectors are preserved unchanged).
    """

    def _make_flat_solver(self, core, temp_db_path, dim=3):
        """Create a GeodesicSolver with an identity metric on a flat chart."""
        store = core.MetricStore(temp_db_path + "/flat_metrics")
        metric = store.create_metric(chart_id=0, dim=dim)
        metric.set_identity()
        store.commit(0, metric)

        config = core.SolverConfig()
        config.tolerance = 1e-8
        config.max_iterations = 5000
        config.bvp_tolerance = 1e-5
        config.max_bvp_iterations = 50
        config.initial_step = 1e-3
        config.adaptive_step = True

        solver = core.GeodesicSolver(metric_store=store, config=config)
        return solver

    def test_geodesic_flat_space(self, core, temp_db_path):
        """On a flat manifold, a geodesic should be a straight line.

        Solve the IVP starting at the origin with velocity (1,0,...,0).
        The endpoint should lie along the initial velocity direction.
        """
        dim = 3
        solver = self._make_flat_solver(core, temp_db_path, dim=dim)

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

        assert path.converged is True
        assert len(path.points) > 1

        # The endpoint should be near (1, 0, 0) (velocity * t_max)
        endpoint = np.asarray(path.points[-1].local_coords)
        expected = velocity * 1.0  # t_max = 1.0
        np.testing.assert_allclose(
            endpoint, expected, atol=5e-2,
            err_msg="On flat space, geodesic should be a straight line"
        )

    def test_geodesic_distance_flat(self, core, temp_db_path):
        """On a flat manifold with identity metric, geodesic distance should
        equal the Euclidean distance between two points.
        """
        dim = 3
        solver = self._make_flat_solver(core, temp_db_path, dim=dim)

        p_local = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        q_local = np.array([3.0, 4.0, 0.0], dtype=np.float64)

        p = core.ManifoldPoint(
            chart_id=0,
            local_coords=p_local,
            ambient_coords=p_local,
            global_id=0,
            timestamp=0.0,
        )
        q = core.ManifoldPoint(
            chart_id=0,
            local_coords=q_local,
            ambient_coords=q_local,
            global_id=1,
            timestamp=0.0,
        )

        geo_dist = solver.geodesic_distance(p, q)
        euc_dist = np.linalg.norm(q_local - p_local)

        np.testing.assert_allclose(
            geo_dist, euc_dist, rtol=1e-2,
            err_msg="Geodesic distance should equal Euclidean distance in flat space"
        )

    def test_parallel_transport_flat(self, core, temp_db_path):
        """On flat space, parallel transport should preserve vectors exactly
        (no change in the transported vector).
        """
        dim = 3
        solver = self._make_flat_solver(core, temp_db_path, dim=dim)

        start_local = np.zeros(dim, dtype=np.float64)
        velocity = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        start = core.ManifoldPoint(
            chart_id=0,
            local_coords=start_local,
            ambient_coords=start_local,
            global_id=0,
            timestamp=0.0,
        )

        # Solve a geodesic
        path = solver.solve_ivp(
            start, velocity, t_max=1.0,
            method=core.SolverType.RK4,
        )

        # Transport a vector along the geodesic
        vec = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        transported = solver.parallel_transport(path, vec)

        assert len(transported) == len(path.points)

        # In flat space, the transported vector should be preserved at every step
        for i, v in enumerate(transported):
            v_np = np.asarray(v)
            np.testing.assert_allclose(
                v_np, vec, atol=1e-6,
                err_msg=f"Parallel transport at step {i} should preserve the vector"
            )

    def test_ivp_convergence(self, core, temp_db_path):
        """Solve an IVP on flat space and verify the endpoint is close to
        the analytically expected position.

        For flat space: x(t) = x0 + v*t.
        """
        dim = 2
        solver = self._make_flat_solver(core, temp_db_path, dim=dim)

        x0 = np.array([1.0, -2.0], dtype=np.float64)
        v = np.array([0.5, 1.5], dtype=np.float64)
        t_max = 0.5

        start = core.ManifoldPoint(
            chart_id=0,
            local_coords=x0,
            ambient_coords=x0,
            global_id=0,
            timestamp=0.0,
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
            err_msg="IVP endpoint should match analytical solution on flat space"
        )

    def test_geodesic_distance_multiple_pairs(self, core, temp_db_path):
        """Check geodesic distances for several point pairs on flat space."""
        dim = 3
        solver = self._make_flat_solver(core, temp_db_path, dim=dim)

        pairs = [
            (np.zeros(dim), np.ones(dim)),
            (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
            (np.array([-1.0, 2.0, -3.0]), np.array([4.0, -1.0, 2.0])),
        ]

        for p_local, q_local in pairs:
            p = core.ManifoldPoint(
                chart_id=0,
                local_coords=p_local.astype(np.float64),
                ambient_coords=p_local.astype(np.float64),
                global_id=0,
            )
            q = core.ManifoldPoint(
                chart_id=0,
                local_coords=q_local.astype(np.float64),
                ambient_coords=q_local.astype(np.float64),
                global_id=1,
            )

            geo_dist = solver.geodesic_distance(p, q)
            euc_dist = float(np.linalg.norm(q_local - p_local))

            np.testing.assert_allclose(
                geo_dist, euc_dist, rtol=5e-2,
                err_msg=f"Geodesic distance failed for pair {p_local} -> {q_local}"
            )
