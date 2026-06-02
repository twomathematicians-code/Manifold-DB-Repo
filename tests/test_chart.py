"""
test_chart.py - Unit tests for Chart classes (LinearChart, ParametricChart)
===========================================================================

Tests cover the core geometric operations of ManifoldDB chart objects:

  - **Embed / Project Roundtrip**: For a LinearChart with orthonormal basis,
    embed(x) maps local coords to ambient coords and project(y) maps them
    back.  The roundtrip embed → project should recover the original local
    coordinates exactly.

  - **Jacobian Consistency**: For a LinearChart, the Jacobian is constant
    (equal to the basis matrix) regardless of the evaluation point.

  - **Metric Tensor**: The induced metric g_ij = J^T J should be identity
    for an orthonormal basis, and should match the expected analytical form
    for parametric charts (e.g., sphere S^2).

  - **Exponential and Logarithmic Maps**: On flat space (identity metric),
    exp_p(v) = p + v and log_p(q) = q - p.  The two maps are inverses.

  - **Various Dimensions**: Tests with d = D (square basis), d < D
    (rectangular basis), and different dimensionalities.

All tests require the C++ extension.  The conftest.py session-level skip
mechanism handles the case where ``_manifolddb_core`` is not available.
"""

from __future__ import annotations

import math

import numpy as np
import pytest


# ===================================================================
# LinearChart Tests
# ===================================================================

class TestLinearChartEmbedProject:
    """Tests for LinearChart embed/project round-trip fidelity."""

    def test_square_basis_roundtrip(self, core, rng):
        """embed(x) then project(ambient) should recover x for d = D.

        With a square orthonormal basis B (B^T B = I), the LinearChart
        implements:
            embed(x)    = origin + B @ x
            project(y)  = B^T @ (y - origin)

        So project(embed(x)) = B^T B x = x  (exact for orthonormal B).
        """
        dim = 5
        basis = np.eye(dim, dtype=np.float64)
        origin = rng.standard_normal(dim).astype(np.float64)

        chart = core.LinearChart(id=0, basis=basis, origin=origin)

        for _ in range(10):
            local = rng.standard_normal(dim).astype(np.float64)
            ambient = chart.embed(local)
            projected = chart.project(ambient)

            np.testing.assert_allclose(
                projected, local, atol=1e-12,
                err_msg="Roundtrip embed→project failed for square orthonormal basis",
            )

    def test_rectangular_basis_roundtrip(self, core, rng):
        """Roundtrip should also work when d < D (non-square basis).

        Use a 7×3 orthonormal basis: embed R^3 into R^7.  Since B^T B = I_3,
        the roundtrip is still exact.
        """
        d = 3   # intrinsic dimension
        D = 7   # ambient dimension

        # Construct orthonormal columns via QR decomposition of a random matrix
        random_mat = rng.standard_normal((D, d)).astype(np.float64)
        Q, _ = np.linalg.qr(random_mat)
        basis = Q.astype(np.float64)

        origin = np.zeros(D, dtype=np.float64)
        chart = core.LinearChart(id=1, basis=basis, origin=origin)

        for _ in range(10):
            local = rng.standard_normal(d).astype(np.float64)
            ambient = chart.embed(local)
            projected = chart.project(ambient)

            np.testing.assert_allclose(
                projected, local, atol=1e-11,
                err_msg="Roundtrip embed→project failed for rectangular basis",
            )

    def test_embed_output_dimension(self, core):
        """embed() should return a vector of ambient dimension D."""
        D, d = 8, 3
        basis = np.zeros((D, d), dtype=np.float64)
        basis[:d, :d] = np.eye(d)
        origin = np.zeros(D, dtype=np.float64)

        chart = core.LinearChart(id=0, basis=basis, origin=origin)
        local = np.ones(d, dtype=np.float64)
        ambient = chart.embed(local)

        assert ambient.shape == (D,), (
            f"embed output should have shape ({D},), got {ambient.shape}"
        )

    def test_project_output_dimension(self, core):
        """project() should return a vector of intrinsic dimension d."""
        D, d = 6, 2
        basis = np.zeros((D, d), dtype=np.float64)
        basis[:d, :d] = np.eye(d)
        origin = np.zeros(D, dtype=np.float64)

        chart = core.LinearChart(id=0, basis=basis, origin=origin)
        ambient = np.ones(D, dtype=np.float64)
        local = chart.project(ambient)

        assert local.shape == (d,), (
            f"project output should have shape ({d},), got {local.shape}"
        )


class TestLinearChartJacobian:
    """Tests verifying that the Jacobian of a LinearChart is constant."""

    def test_jacobian_equals_basis(self, core):
        """The Jacobian J(x) of a LinearChart should equal the basis matrix B
        at every point x, because the embedding is affine (linear + shift).

        For an affine map φ(x) = origin + Bx, the derivative dφ/dx = B
        is constant everywhere.
        """
        D, d = 5, 3
        basis = np.zeros((D, d), dtype=np.float64)
        basis[:d, :d] = np.eye(d)
        basis[3, 0] = 0.5   # non-trivial entry
        basis[4, 1] = -0.3

        origin = np.array([1.0, -2.0, 0.0, 3.0, 0.0], dtype=np.float64)
        chart = core.LinearChart(id=2, basis=basis, origin=origin)

        # Evaluate at diverse points
        test_points = [
            np.zeros(d, dtype=np.float64),
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([100.0, -50.0, 25.0], dtype=np.float64),
            rng_standard_normal(d),
        ]

        for pt in test_points:
            J = chart.jacobian(pt)
            np.testing.assert_allclose(
                J, basis, atol=1e-12,
                err_msg=f"Jacobian should equal basis at point {pt}",
            )

    def test_jacobian_shape(self, core):
        """Jacobian should be D × d."""
        D, d = 10, 4
        basis = np.zeros((D, d), dtype=np.float64)
        basis[:d, :d] = np.eye(d)

        chart = core.LinearChart(id=0, basis=basis, origin=np.zeros(D, dtype=np.float64))
        J = chart.jacobian(np.zeros(d, dtype=np.float64))

        assert J.shape == (D, d), (
            f"Jacobian should be ({D}×{d}), got {J.shape}"
        )


def rng_standard_normal(dim, seed=42):
    """Helper to generate a random normal vector (for test points)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).astype(np.float64)


class TestLinearChartMetric:
    """Tests for the induced metric tensor g_ij = J^T J on LinearCharts."""

    def test_orthonormal_basis_identity_metric(self, core):
        """For an orthonormal basis, g = B^T B = I."""
        dim = 4
        basis = np.eye(dim, dtype=np.float64)
        origin = np.zeros(dim, dtype=np.float64)
        chart = core.LinearChart(id=0, basis=basis, origin=origin)

        local = np.array([0.5, -0.3, 1.2, 0.0], dtype=np.float64)
        g = chart.compute_local_metric(local)

        np.testing.assert_allclose(
            g, np.eye(dim, dtype=np.float64), atol=1e-12,
            err_msg="Orthonormal basis should yield identity metric",
        )

    def test_scaled_basis_metric(self, core):
        """For a basis with scaled columns, g = B^T B should reflect the scaling.

        If B = [[2, 0], [0, 3]], then g = [[4, 0], [0, 9]].
        """
        basis = np.diag([2.0, 3.0]).astype(np.float64)  # 2×2
        origin = np.zeros(2, dtype=np.float64)
        chart = core.LinearChart(id=0, basis=basis, origin=origin)

        local = np.zeros(2, dtype=np.float64)
        g = chart.compute_local_metric(local)

        expected = np.diag([4.0, 9.0]).astype(np.float64)
        np.testing.assert_allclose(g, expected, atol=1e-12)


# ===================================================================
# Exponential Map and Logarithmic Map
# ===================================================================

class TestExponentialLogMaps:
    """Tests for the exponential and logarithmic maps on LinearCharts.

    On flat space with the identity metric, these maps are trivial:
        exp_p(v) = p + v   (geodesics are straight lines)
        log_p(q) = q - p   (tangent vector is the displacement)

    The two maps are inverses: exp_p(log_p(q)) = q.
    """

    def test_exponential_map_flat_space(self, core):
        """exp_p(v) should return a point whose local coords are p + v
        when the metric is flat (identity).
        """
        dim = 3
        basis = np.eye(dim, dtype=np.float64)
        origin = np.zeros(dim, dtype=np.float64)
        chart = core.LinearChart(id=0, basis=basis, origin=origin)

        # Base point at the origin
        base_local = np.zeros(dim, dtype=np.float64)
        base = core.ManifoldPoint(
            chart_id=0,
            local_coords=base_local,
            ambient_coords=chart.embed(base_local),
            global_id=0,
            timestamp=0.0,
        )

        # Tangent vector
        tangent = np.array([1.0, 2.0, -1.0], dtype=np.float64)
        result = chart.exponential_map(base, tangent, step_size=1e-3, max_steps=1000)

        expected_local = base_local + tangent
        np.testing.assert_allclose(
            np.asarray(result.local_coords), expected_local, atol=1e-3,
            err_msg="Exponential map in flat space should be base + tangent_vec",
        )

    def test_log_map_flat_space(self, core):
        """log_p(q) should return q - p on flat space.

        The logarithmic map computes the tangent vector v such that
        exp_p(v) = q.  In flat space, v = q - p.
        """
        dim = 3
        basis = np.eye(dim, dtype=np.float64)
        origin = np.zeros(dim, dtype=np.float64)
        chart = core.LinearChart(id=0, basis=basis, origin=origin)

        base_local = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        target_local = np.array([3.0, -1.0, 2.0], dtype=np.float64)

        base = core.ManifoldPoint(
            chart_id=0,
            local_coords=base_local,
            ambient_coords=chart.embed(base_local),
            global_id=0,
            timestamp=0.0,
        )
        target = core.ManifoldPoint(
            chart_id=0,
            local_coords=target_local,
            ambient_coords=chart.embed(target_local),
            global_id=1,
            timestamp=0.0,
        )

        log_vec = chart.log_map(base, target, tolerance=1e-6, max_iterations=100)

        expected = target_local - base_local
        np.testing.assert_allclose(
            np.asarray(log_vec), expected, atol=1e-3,
            err_msg="Log map in flat space should return target - base",
        )

    def test_exp_log_inverse(self, core):
        """exp_p(log_p(q)) should recover q (the maps are inverses).

        This is a fundamental property of the exponential/logarithmic map
        pair on Riemannian manifolds.
        """
        dim = 2
        basis = np.eye(dim, dtype=np.float64)
        origin = np.zeros(dim, dtype=np.float64)
        chart = core.LinearChart(id=0, basis=basis, origin=origin)

        base_local = np.array([0.5, -0.2], dtype=np.float64)
        target_local = np.array([2.0, 1.5], dtype=np.float64)

        base = core.ManifoldPoint(
            chart_id=0,
            local_coords=base_local,
            ambient_coords=chart.embed(base_local),
            global_id=0,
            timestamp=0.0,
        )
        target = core.ManifoldPoint(
            chart_id=0,
            local_coords=target_local,
            ambient_coords=chart.embed(target_local),
            global_id=1,
            timestamp=0.0,
        )

        # log_p(q) gives the tangent vector
        log_vec = chart.log_map(base, target, tolerance=1e-6, max_iterations=100)

        # exp_p(log_p(q)) should give q back
        recovered = chart.exponential_map(
            base, np.asarray(log_vec), step_size=1e-3, max_steps=1000,
        )

        np.testing.assert_allclose(
            np.asarray(recovered.local_coords), target_local, atol=5e-2,
            err_msg="exp_p(log_p(q)) should recover q",
        )

    def test_exp_log_higher_dimension(self, core):
        """Test exp/log inverse property in higher dimensions (d=5, D=8)."""
        d, D = 5, 8

        # Construct orthonormal basis via QR
        rng = np.random.default_rng(42)
        random_mat = rng.standard_normal((D, d)).astype(np.float64)
        Q, _ = np.linalg.qr(random_mat)
        basis = Q.astype(np.float64)

        origin = np.zeros(D, dtype=np.float64)
        chart = core.LinearChart(id=0, basis=basis, origin=origin)

        base_local = np.zeros(d, dtype=np.float64)
        target_local = rng.standard_normal(d).astype(np.float64) * 2.0

        base = core.ManifoldPoint(
            chart_id=0, local_coords=base_local,
            ambient_coords=chart.embed(base_local),
            global_id=0, timestamp=0.0,
        )
        target = core.ManifoldPoint(
            chart_id=0, local_coords=target_local,
            ambient_coords=chart.embed(target_local),
            global_id=1, timestamp=0.0,
        )

        log_vec = chart.log_map(base, target, tolerance=1e-6, max_iterations=100)
        recovered = chart.exponential_map(
            base, np.asarray(log_vec), step_size=1e-3, max_steps=1500,
        )

        np.testing.assert_allclose(
            np.asarray(recovered.local_coords), target_local, atol=1e-1,
            err_msg="exp/log inverse should hold in higher dimensions",
        )


# ===================================================================
# Christoffel Symbols on Flat Space
# ===================================================================

class TestChristoffelFlatSpace:
    """Christoffel symbols should vanish on flat space (constant metric)."""

    def test_christoffel_zero_flat(self, core):
        """Both kinds of Christoffel symbols should be zero in flat space.

        For a LinearChart with identity metric, the metric g_ij is constant,
        so all partial derivatives ∂g_ij/∂x^k = 0, which means all
        Christoffel symbols are zero.
        """
        dim = 3
        basis = np.eye(dim, dtype=np.float64)
        origin = np.zeros(dim, dtype=np.float64)
        chart = core.LinearChart(id=0, basis=basis, origin=origin)

        local = np.array([0.5, -0.3, 1.0], dtype=np.float64)

        # Second kind Γ^k_{ij}
        Gamma2 = chart.christoffel_second_kind(local, h=1e-5)
        assert Gamma2.shape == (dim, dim, dim)
        np.testing.assert_allclose(
            np.asarray(Gamma2), 0.0, atol=1e-8,
            err_msg="Christoffel symbols (2nd kind) should be zero in flat space",
        )

        # First kind Γ_{ijk}
        Gamma1 = chart.christoffel_first_kind(local, h=1e-5)
        assert Gamma1.shape == (dim, dim, dim)
        np.testing.assert_allclose(
            np.asarray(Gamma1), 0.0, atol=1e-8,
            err_msg="Christoffel symbols (1st kind) should be zero in flat space",
        )


# ===================================================================
# ParametricChart Tests (Sphere S^2)
# ===================================================================

class TestParametricChartSphere:
    """Tests for a ParametricChart representing the 2-sphere S^2."""

    @staticmethod
    def _sphere_embed(local_coords):
        theta, phi = float(local_coords[0]), float(local_coords[1])
        return np.array([
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta),
        ], dtype=np.float64)

    @staticmethod
    def _sphere_project(ambient_coords):
        x, y, z = (float(ambient_coords[0]),
                    float(ambient_coords[1]),
                    float(ambient_coords[2]))
        theta = math.acos(max(-1.0, min(1.0, z)))
        phi = math.atan2(y, x)
        return np.array([theta, phi], dtype=np.float64)

    @staticmethod
    def _sphere_jacobian(local_coords):
        theta = float(local_coords[0])
        phi = float(local_coords[1])
        st, ct = math.sin(theta), math.cos(theta)
        sp, cp = math.sin(phi), math.cos(phi)
        return np.array([
            [ct * cp, -st * sp],
            [ct * sp,  st * cp],
            [-st,     0.0],
        ], dtype=np.float64)

    def _make_sphere_chart(self, core):
        """Create a ParametricChart for S^2."""
        return core.ParametricChart(
            id=10,
            intrinsic_dim=2,
            ambient_dim=3,
            embed_fn=self._sphere_embed,
            project_fn=self._sphere_project,
            jacobian_fn=self._sphere_jacobian,
        )

    def test_sphere_embed_project_roundtrip(self, core):
        """Embed → project should recover the original (θ, φ) coordinates."""
        chart = self._make_sphere_chart(core)

        test_points = [
            np.array([math.pi / 2, 0.0], dtype=np.float64),
            np.array([math.pi / 4, math.pi / 4], dtype=np.float64),
            np.array([math.pi / 3, math.pi], dtype=np.float64),
            np.array([math.pi / 6, 3 * math.pi / 2], dtype=np.float64),
        ]

        for local in test_points:
            ambient = chart.embed(local)
            projected = chart.project(ambient)
            np.testing.assert_allclose(
                projected, local, atol=1e-10,
                err_msg=f"Sphere roundtrip failed for {local}",
            )

    def test_sphere_metric_at_equator(self, core):
        """At the equator (θ = π/2, φ = 0), the sphere metric should be
        the identity matrix g = [[1, 0], [0, 1]].

        This is because sin(θ) = sin(π/2) = 1, and the sphere metric is:
            g = [[1, 0], [0, sin²(θ)]]
        """
        chart = self._make_sphere_chart(core)
        local = np.array([math.pi / 2, 0.0], dtype=np.float64)
        g = chart.compute_local_metric(local)

        np.testing.assert_allclose(
            g, np.eye(2, dtype=np.float64), atol=1e-10,
        )

    def test_sphere_metric_at_pole_offset(self, core):
        """At θ = π/4: g_{22} = sin²(π/4) = 0.5.

        The sphere metric is:
            g = [[1, 0], [0, sin²(θ)]]
        """
        chart = self._make_sphere_chart(core)
        local = np.array([math.pi / 4, 0.0], dtype=np.float64)
        g = chart.compute_local_metric(local)

        np.testing.assert_allclose(g[0, 0], 1.0, atol=1e-10)
        np.testing.assert_allclose(g[1, 1], 0.5, atol=1e-6)
        np.testing.assert_allclose(g[0, 1], 0.0, atol=1e-10)
        np.testing.assert_allclose(g[1, 0], 0.0, atol=1e-10)
