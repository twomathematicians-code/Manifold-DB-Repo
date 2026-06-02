"""
Geodesic Solver - solves the geodesic equation on Riemannian manifolds.

The geodesic equation:
    d²xⁱ/dt² + Γⁱ_jk dxʲ/dt dxᵏ/dt = 0

where Γ are the Christoffel symbols of the Levi-Civita connection.

Supports multiple integration methods (Euler, RK4, RK45 adaptive) and
GPU-accelerated batch solving via PyTorch.
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class IntegrationMethod(Enum):
    """Available ODE integration methods for geodesic solving."""

    EULER = "euler"
    RK4 = "rk4"
    RK45 = "rk45"


@dataclass
class GeodesicResult:
    """Result container for a single geodesic solve."""

    trajectory: np.ndarray  # (n_steps, dim) positions
    velocities: np.ndarray  # (n_steps, dim) velocities
    times: np.ndarray  # (n_steps,) parameter values
    success: bool = True
    message: str = ""


# ---------------------------------------------------------------------------
# Christoffel symbols
# ---------------------------------------------------------------------------


def compute_christoffel_symbols(
    metric_tensor: np.ndarray,
    derivatives: np.ndarray,
) -> np.ndarray:
    """
    Compute Christoffel symbols of the second kind from the metric tensor
    and its partial derivatives.

    Formula:
        Γⁱ_jk = ½ g^{il} (∂g_{lj}/∂x^k + ∂g_{lk}/∂x^j − ∂g_{jk}/∂x^l)

    Parameters
    ----------
    metric_tensor : np.ndarray, shape (n, n)
        The Riemannian metric tensor g_{ij} at a point.
    derivatives : np.ndarray, shape (n, n, n)
        Partial derivatives ∂g_{ij}/∂x^k where the last axis indexes the
        differentiation variable.

    Returns
    -------
    np.ndarray, shape (n, n, n)
        Christoffel symbols Γⁱ_{jk}.  The first index is the upper index.
    """
    n = metric_tensor.shape[0]
    # Inverse metric g^{ij}
    try:
        g_inv = np.linalg.inv(metric_tensor)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Metric tensor is singular; cannot compute inverse.") from exc

    gamma = np.zeros((n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                s = 0.0
                for idx_l in range(n):
                    # ∂g_{lj}/∂x^k  +  ∂g_{lk}/∂x^j  −  ∂g_{jk}/∂x^l
                    term = (
                        derivatives[idx_l, j, k]
                        + derivatives[idx_l, k, j]
                        - derivatives[j, k, idx_l]
                    )
                    s += g_inv[i, idx_l] * term
                gamma[i, j, k] = 0.5 * s
    return gamma


# ---------------------------------------------------------------------------
# Geodesic Solver
# ---------------------------------------------------------------------------


class GeodesicSolver:
    """
    Solves the geodesic equation on a Riemannian manifold.

    Parameters
    ----------
    metric_fn : callable
        Function x → g_{ij}(x) returning an (n, n) metric tensor.
    christoffel_fn : callable or None
        Function x → Γⁱ_{jk}(x) returning an (n, n, n) array.
        If *None* and ``metric_fn`` is provided, Christoffel symbols are
        approximated numerically from the metric via finite differences.
    dim : int or None
        Manifold dimension.  Inferred on first call if *None*.
    """

    def __init__(
        self,
        metric_fn: Callable[[np.ndarray], np.ndarray],
        christoffel_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        dim: int | None = None,
    ) -> None:
        self.metric_fn = metric_fn
        self.christoffel_fn = christoffel_fn
        self.dim = dim
        self._eps = 1e-7  # finite-difference step

    # ----- internal helpers ------------------------------------------------

    def _get_christoffel(self, x: np.ndarray) -> np.ndarray:
        """Return Christoffel symbols at *x*, computing from metric if needed."""
        if self.christoffel_fn is not None:
            return self.christoffel_fn(x)
        # Numerical differentiation of metric_fn
        g0 = self.metric_fn(x)
        n = g0.shape[0]
        if self.dim is None:
            self.dim = n
        dg = np.zeros((n, n, n))
        for k in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[k] += self._eps
            x_minus[k] -= self._eps
            dg[:, :, k] = (self.metric_fn(x_plus) - self.metric_fn(x_minus)) / (
                2.0 * self._eps
            )
        return compute_christoffel_symbols(g0, dg)

    def _geodesic_rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """Right-hand side of the first-order geodesic ODE system.

        State vector y = [x⁰, …, x^{n-1}, v⁰, …, v^{n-1}] where v = dx/dt.
        Returns dy/dt = [v, -Γⁱ_{jk} v^j v^k].
        """
        n = self.dim or len(y) // 2
        pos = y[:n]
        vel = y[n:]
        gamma = self._get_christoffel(pos)
        acc = np.zeros(n)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    acc[i] -= gamma[i, j, k] * vel[j] * vel[k]
        return np.concatenate([vel, acc])

    # ----- integration methods ---------------------------------------------

    def solve_geodesic(
        self,
        initial_point: np.ndarray,
        initial_velocity: np.ndarray,
        t_span: tuple[float, float] = (0.0, 1.0),
        method: str = "rk45",
        dt: float = 0.01,
        **kwargs: Any,
    ) -> GeodesicResult:
        """
        Solve the geodesic equation from an initial point with initial velocity.

        Parameters
        ----------
        initial_point : np.ndarray, shape (n,)
            Starting point on the manifold.
        initial_velocity : np.ndarray, shape (n,)
            Initial tangent vector (velocity).
        t_span : (float, float)
            Integration parameter interval (t_start, t_end).
        method : str
            Integration method: ``'euler'``, ``'rk4'``, or ``'rk45'``.
        dt : float
            Time step (for Euler/RK4) or initial step size hint (for RK45).
        **kwargs
            Extra arguments forwarded to ``scipy.integrate.solve_ivp``
            (for RK45) or ignored otherwise.

        Returns
        -------
        GeodesicResult
            Container with trajectory, velocities, and metadata.
        """
        n = len(initial_point)
        if self.dim is None:
            self.dim = n
        y0 = np.concatenate([initial_point, initial_velocity])
        t_start, t_end = t_span
        method_enum = IntegrationMethod(method.lower())

        if method_enum == IntegrationMethod.EULER:
            return self._solve_euler(y0, t_start, t_end, dt, n)
        elif method_enum == IntegrationMethod.RK4:
            return self._solve_rk4(y0, t_start, t_end, dt, n)
        else:
            return self._solve_rk45(y0, t_span, dt, n, **kwargs)

    def _solve_euler(
        self,
        y0: np.ndarray,
        t0: float,
        t1: float,
        dt: float,
        n: int,
    ) -> GeodesicResult:
        steps = max(1, int(np.ceil(abs(t1 - t0) / dt)))
        dt_actual = (t1 - t0) / steps
        trajectory = np.zeros((steps + 1, n))
        velocities = np.zeros((steps + 1, n))
        trajectory[0] = y0[:n]
        velocities[0] = y0[n:]
        y = y0.copy()
        for i in range(1, steps + 1):
            dy = self._geodesic_rhs(t0 + (i - 1) * dt_actual, y)
            y = y + dt_actual * dy
            trajectory[i] = y[:n]
            velocities[i] = y[n:]
        times = np.linspace(t0, t1, steps + 1)
        return GeodesicResult(trajectory, velocities, times, True)

    def _solve_rk4(
        self,
        y0: np.ndarray,
        t0: float,
        t1: float,
        dt: float,
        n: int,
    ) -> GeodesicResult:
        steps = max(1, int(np.ceil(abs(t1 - t0) / dt)))
        dt_actual = (t1 - t0) / steps
        trajectory = np.zeros((steps + 1, n))
        velocities = np.zeros((steps + 1, n))
        trajectory[0] = y0[:n]
        velocities[0] = y0[n:]
        y = y0.copy()
        for i in range(1, steps + 1):
            t = t0 + (i - 1) * dt_actual
            k1 = self._geodesic_rhs(t, y)
            k2 = self._geodesic_rhs(t + 0.5 * dt_actual, y + 0.5 * dt_actual * k1)
            k3 = self._geodesic_rhs(t + 0.5 * dt_actual, y + 0.5 * dt_actual * k2)
            k4 = self._geodesic_rhs(t + dt_actual, y + dt_actual * k3)
            y = y + (dt_actual / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            trajectory[i] = y[:n]
            velocities[i] = y[n:]
        times = np.linspace(t0, t1, steps + 1)
        return GeodesicResult(trajectory, velocities, times, True)

    def _solve_rk45(
        self,
        y0: np.ndarray,
        t_span: tuple[float, float],
        dt: float,
        n: int,
        **kwargs: Any,
    ) -> GeodesicResult:
        t_eval = np.linspace(
            t_span[0], t_span[1], max(2, int(abs(t_span[1] - t_span[0]) / dt) + 1)
        )
        sol = solve_ivp(
            self._geodesic_rhs,
            t_span,
            y0,
            method="RK45",
            t_eval=t_eval,
            rtol=1e-8,
            atol=1e-10,
            max_step=dt,
            **kwargs,
        )
        if not sol.success:
            logger.warning("RK45 integration failed: %s", sol.message)
        traj = sol.y[:n].T
        vel = sol.y[n:].T
        return GeodesicResult(traj, vel, sol.t, sol.success, sol.message)

    # ----- geodesic distance via shooting ----------------------------------

    def geodesic_distance(
        self,
        point_a: np.ndarray,
        point_b: np.ndarray,
        n_paths: int = 10,
    ) -> float:
        """
        Approximate geodesic distance using the shooting method.

        Tries several initial velocities aimed at *point_b*, integrates the
        geodesic, and returns the length of the best (shortest) path.

        Parameters
        ----------
        point_a, point_b : np.ndarray, shape (n,)
        n_paths : int
            Number of random initial directions to try.

        Returns
        -------
        float
            Approximate geodesic distance.
        """
        diff = point_b - point_a
        base_vel = diff.copy()
        n = len(point_a)
        best_len = np.inf

        for _ in range(n_paths):
            if _ == 0:
                vel = base_vel.copy()
            else:
                perturbation = np.random.randn(n) * np.linalg.norm(diff) * 0.3
                vel = base_vel + perturbation

            # Estimate affine parameter so we reach the target roughly
            metric_at_a = self.metric_fn(point_a)
            speed_sq = vel @ metric_at_a @ vel
            t_end = np.linalg.norm(diff) / max(np.sqrt(speed_sq), 1e-12) * 1.5

            result = self.solve_geodesic(
                point_a, vel, (0.0, t_end), method="rk45", dt=0.02
            )
            if not result.success:
                continue

            # Compute Riemannian arc length along the trajectory
            length = self._path_length(result)
            if length < best_len:
                best_len = length

        if best_len == np.inf:
            # Fallback: Euclidean distance in the metric
            metric_mid = self.metric_fn(0.5 * (point_a + point_b))
            delta = point_b - point_a
            best_len = float(np.sqrt(delta @ metric_mid @ delta))
        return best_len

    def _path_length(self, result: GeodesicResult) -> float:
        """Compute Riemannian arc length of a geodesic result."""
        length = 0.0
        for i in range(1, len(result.trajectory)):
            x_prev = result.trajectory[i - 1]
            x_curr = result.trajectory[i]
            dt = result.times[i] - result.times[i - 1]
            if dt == 0:
                continue
            vel = (x_curr - x_prev) / dt
            mid = 0.5 * (x_prev + x_curr)
            g = self.metric_fn(mid)
            ds_sq = vel @ g @ vel * dt * dt
            if ds_sq > 0:
                length += np.sqrt(ds_sq)
        return length

    # ----- geodesic ball ---------------------------------------------------

    def geodesic_ball(
        self,
        center: np.ndarray,
        radius: float,
        n_samples: int = 1000,
        t_max: float = 2.0,
    ) -> np.ndarray:
        """
        Sample points inside a geodesic ball of given radius.

        Uses uniform random directions and distances (volume-corrected for
        approximate uniform sampling).

        Parameters
        ----------
        center : np.ndarray, shape (n,)
        radius : float
            Geodesic radius.
        n_samples : int
        t_max : float
            Maximum affine parameter for integration.

        Returns
        -------
        np.ndarray, shape (n_samples, n)
            Sampled points.
        """
        n = len(center)
        metric_c = self.metric_fn(center)
        g_inv = np.linalg.inv(metric_c)
        points = []
        count = 0
        max_attempts = n_samples * 10
        attempts = 0

        while count < n_samples and attempts < max_attempts:
            attempts += 1
            # Random tangent vector (uniform on the sphere in the metric)
            raw = np.random.randn(n)
            norm_metric = float(np.sqrt(raw @ g_inv @ raw))
            if norm_metric < 1e-14:
                continue
            tangent = raw / norm_metric
            # Sample radius with volume correction: r ∝ radius * u^{1/n}
            r = radius * (np.random.random() ** (1.0 / n))

            result = self.solve_geodesic(
                center, tangent * r, (0.0, t_max), method="rk45", dt=0.05
            )
            if not result.success:
                continue

            # Walk along trajectory to find the point at the right arc-length distance
            accumulated = 0.0
            for i in range(1, len(result.trajectory)):
                ds = np.linalg.norm(result.trajectory[i] - result.trajectory[i - 1])
                accumulated += ds
                if accumulated >= r:
                    points.append(result.trajectory[i])
                    count += 1
                    break

        if count == 0:
            return np.zeros((0, n))
        return np.array(points)

    # ----- shortest path via graph approximation ---------------------------

    def shortest_path(
        self,
        point_a: np.ndarray,
        point_b: np.ndarray,
        graph_approx: Any | None = None,
        n_intermediate: int = 50,
    ) -> np.ndarray:
        """
        Find approximate shortest path using a graph approximation followed
        by geodesic refinement.

        If no graph approximation is provided, a simple straight-line
        interpolation in the embedding space is used as a starting guess,
        then refined via energy minimization.

        Parameters
        ----------
        point_a, point_b : np.ndarray, shape (n,)
        graph_approx : object or None
            An object with a ``shortest_path(a, b)`` method returning a list
            of waypoints.  If *None*, uses linear interpolation.
        n_intermediate : int
            Number of intermediate waypoints for the refined path.

        Returns
        -------
        np.ndarray, shape (n_intermediate + 2, n)
            The refined path from *point_a* to *point_b*.
        """
        if graph_approx is not None and hasattr(graph_approx, "shortest_path"):
            raw_path = graph_approx.shortest_path(point_a, point_b)
            if raw_path is not None and len(raw_path) > 1:
                init_path = np.array(raw_path)
            else:
                init_path = self._linear_interpolation(point_a, point_b, n_intermediate)
        else:
            init_path = self._linear_interpolation(point_a, point_b, n_intermediate)

        return self._refine_path(init_path)

    def _linear_interpolation(
        self,
        a: np.ndarray,
        b: np.ndarray,
        n: int,
    ) -> np.ndarray:
        ts = np.linspace(0, 1, n + 2)
        return (
            a[np.newaxis, :] * (1 - ts[:, np.newaxis])
            + b[np.newaxis, :] * ts[:, np.newaxis]
        )

    def _refine_path(
        self, path: np.ndarray, iterations: int = 100, lr: float = 0.01
    ) -> np.ndarray:
        """
        Refine a path by gradient descent on the total Riemannian energy.

        Energy = Σᵢ g_{ij}(xᵢ) (xᵢ₊₁ - xᵢ)(xᵢ₊₁ - xᵢ).
        """
        waypoints = path.copy().astype(np.float64)
        endpoints_fixed = True
        for _ in range(iterations):
            grad = np.zeros_like(waypoints)
            total_energy = 0.0
            for i in range(len(waypoints) - 1):
                diff = waypoints[i + 1] - waypoints[i]
                g = self.metric_fn(waypoints[i])
                energy_seg = float(diff @ g @ diff)
                total_energy += energy_seg
                dE_dx_i = -2.0 * (g @ diff)
                dE_dx_ip1 = 2.0 * (g @ diff)
                grad[i] += dE_dx_i
                grad[i + 1] += dE_dx_ip1

            if endpoints_fixed:
                grad[0] = 0.0
                grad[-1] = 0.0

            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-12:
                break
            waypoints -= lr * grad
        return waypoints

    # ----- GPU batch solve -------------------------------------------------

    def solve_geodesic_batch(
        self,
        initial_points: np.ndarray,
        initial_velocities: np.ndarray,
        t_span: tuple[float, float] = (0.0, 1.0),
        dt: float = 0.01,
        method: str = "rk4",
    ) -> np.ndarray:
        """
        Batch geodesic solving accelerated with PyTorch when available.

        Parameters
        ----------
        initial_points : np.ndarray, shape (B, n)
        initial_velocities : np.ndarray, shape (B, n)
        t_span, dt, method : see ``solve_geodesic``.

        Returns
        -------
        np.ndarray, shape (B, S, n)
            Trajectory for each initial condition.  *S* = number of time steps.
        """
        _has_torch = importlib.util.find_spec("torch") is not None

        if _has_torch and method.lower() == "rk4":
            return self._solve_batch_torch(
                initial_points, initial_velocities, t_span, dt
            )
        else:
            return self._solve_batch_numpy(
                initial_points, initial_velocities, t_span, dt, method
            )

    def _solve_batch_torch(
        self,
        points: np.ndarray,
        velocities: np.ndarray,
        t_span: tuple[float, float],
        dt: float,
    ) -> np.ndarray:
        """RK4 batch integration using PyTorch."""
        import torch

        pts = torch.tensor(points, dtype=torch.float64, device=self._get_torch_device())
        vels = torch.tensor(
            velocities, dtype=torch.float64, device=self._get_torch_device()
        )
        B, n = pts.shape

        steps = max(1, int(np.ceil(abs(t_span[1] - t_span[0]) / dt)))
        dt_t = torch.tensor(dt, dtype=torch.float64, device=self._get_torch_device())
        trajectories = torch.zeros(steps + 1, B, n, device=self._get_torch_device())
        trajectories[0] = pts.clone()

        pos = pts.clone()
        vel = vels.clone()
        for i in range(1, steps + 1):
            # Batch Christoffel computation
            gamma = self._batch_christoffel_torch(pos)  # (B, n, n, n)
            acc = torch.zeros_like(pos)
            for ii in range(n):
                for jj in range(n):
                    for kk in range(n):
                        acc[:, ii] -= gamma[:, ii, jj, kk] * vel[:, jj] * vel[:, kk]
            # RK4
            k1_v = acc
            k1_x = vel

            k2_x = vel + 0.5 * dt_t * k1_v
            k2_v = self._batch_accel_torch(
                pos + 0.5 * dt_t * k1_x, vel + 0.5 * dt_t * k1_v
            )

            k3_x = vel + 0.5 * dt_t * k2_v
            k3_v = self._batch_accel_torch(
                pos + 0.5 * dt_t * k2_x, vel + 0.5 * dt_t * k2_v
            )

            k4_x = vel + dt_t * k3_v
            k4_v = self._batch_accel_torch(pos + dt_t * k3_x, vel + dt_t * k3_v)

            pos = pos + (dt_t / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
            vel = vel + (dt_t / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
            trajectories[i] = pos.clone()

        return trajectories.permute(1, 0, 2).cpu().numpy()

    def _batch_christoffel_torch(self, pos: Any) -> Any:
        """Compute Christoffel symbols for a batch of points using PyTorch."""
        import torch

        B, n = pos.shape
        gamma_batch = torch.zeros(
            B, n, n, n, dtype=torch.float64, device=self._get_torch_device()
        )
        for idx in range(B):
            x_np = pos[idx].detach().cpu().numpy()
            g = self._get_christoffel(x_np)
            gamma_batch[idx] = torch.tensor(
                g, dtype=torch.float64, device=self._get_torch_device()
            )
        return gamma_batch

    def _batch_accel_torch(self, pos: Any, vel: Any) -> Any:
        """Batch acceleration computation for RK4 intermediate stages."""
        import torch

        gamma = self._batch_christoffel_torch(pos)
        B, n = pos.shape
        acc = torch.zeros_like(pos)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    acc[:, i] -= gamma[:, i, j, k] * vel[:, j] * vel[:, k]
        return acc

    def _get_torch_device(self) -> Any:
        try:
            import torch

            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            return "cpu"

    def _solve_batch_numpy(
        self,
        points: np.ndarray,
        velocities: np.ndarray,
        t_span: tuple[float, float],
        dt: float,
        method: str,
    ) -> np.ndarray:
        """Fallback: solve each geodesic individually with numpy."""
        B, n = points.shape
        steps = max(1, int(np.ceil(abs(t_span[1] - t_span[0]) / dt)))
        trajectories = np.zeros((B, steps + 1, n))
        for b in range(B):
            result = self.solve_geodesic(points[b], velocities[b], t_span, method, dt)
            actual_steps = min(len(result.trajectory), steps + 1)
            trajectories[b, :actual_steps] = result.trajectory[:actual_steps]
        return trajectories
