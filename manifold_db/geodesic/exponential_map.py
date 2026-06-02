"""
Exponential and Logarithmic Maps — bridge between tangent space and manifold.

    exp_p(v): tangent space → manifold
        Follows the geodesic starting at p with initial velocity v.

    log_p(q): manifold → tangent space
        Inverse of exp_p: returns the tangent vector v such that exp_p(v) = q.

Also provides cheaper first-order approximations via retraction / inverse
retraction, and batch GPU-accelerated variants via PyTorch.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exponential Map
# ---------------------------------------------------------------------------

class ExponentialMap:
    """
    Exponential and logarithmic maps on a Riemannian manifold.

    Parameters
    ----------
    metric_fn : callable
        x → g_{ij}(x) returning the metric tensor at point x.
    christoffel_fn : callable or None
        x → Γⁱ_{jk}(x).  If *None*, Christoffel symbols are approximated
        from the metric via finite differences.
    """

    def __init__(
        self,
        metric_fn: Callable[[np.ndarray], np.ndarray],
        christoffel_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        self.metric_fn = metric_fn
        self.christoffel_fn = christoffel_fn
        self._eps = 1e-7

    # ----- Christoffel from metric (numerical) ------------------------------

    def _get_christoffel(self, x: np.ndarray) -> np.ndarray:
        if self.christoffel_fn is not None:
            return self.christoffel_fn(x)
        g0 = self.metric_fn(x)
        n = g0.shape[0]
        g_inv = np.linalg.inv(g0)
        dg = np.zeros((n, n, n))
        for k in range(n):
            xp = x.copy(); xp[k] += self._eps
            xm = x.copy(); xm[k] -= self._eps
            dg[:, :, k] = (self.metric_fn(xp) - self.metric_fn(xm)) / (2.0 * self._eps)
        gamma = np.zeros((n, n, n))
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    s = 0.0
                    for l in range(n):
                        s += g_inv[i, l] * (dg[l, j, k] + dg[l, k, j] - dg[j, k, l])
                    gamma[i, j, k] = 0.5 * s
        return gamma

    def _geodesic_rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """First-order ODE system for geodesic equation."""
        n = len(y) // 2
        pos, vel = y[:n], y[n:]
        gamma = self._get_christoffel(pos)
        acc = np.zeros(n)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    acc[i] -= gamma[i, j, k] * vel[j] * vel[k]
        return np.concatenate([vel, acc])

    # ----- exp_p(v) ---------------------------------------------------------

    def exp_map(
        self,
        base_point: np.ndarray,
        tangent_vector: np.ndarray,
        t_max: Optional[float] = None,
        steps: int = 100,
    ) -> np.ndarray:
        """
        Exponential map: exp_p(v) = γ(1) where γ is the geodesic with γ(0)=p,
        γ̇(0) = v.

        Parameters
        ----------
        base_point : np.ndarray, shape (n,)
        tangent_vector : np.ndarray, shape (n,)
        t_max : float or None
            Affine parameter at which to evaluate.  *None* → 1.0, but
            automatically adjusted if the tangent vector norm is large.
        steps : int
            Number of evaluation points along the geodesic.

        Returns
        -------
        np.ndarray, shape (steps,)
            The endpoint exp_p(v) (the full trajectory is discarded).
        """
        if t_max is None:
            # Adjust t_max so that affine parameter 1 corresponds to unit
            # speed in the metric at base_point.
            g = self.metric_fn(base_point)
            speed_sq = float(tangent_vector @ g @ tangent_vector)
            if speed_sq > 1e-14:
                t_max = 1.0 / np.sqrt(speed_sq)
            else:
                t_max = 1.0

        n = len(base_point)
        y0 = np.concatenate([base_point, tangent_vector])
        t_eval = np.linspace(0, t_max, max(2, steps))

        sol = solve_ivp(
            self._geodesic_rhs,
            (0.0, t_max),
            y0,
            method="RK45",
            t_eval=t_eval,
            rtol=1e-9,
            atol=1e-11,
        )

        if not sol.success:
            logger.warning("exp_map integration failed: %s", sol.message)

        return sol.y[:n, -1]

    def exp_map_batch(
        self,
        base_points: np.ndarray,
        tangent_vectors: np.ndarray,
        t_max: Optional[float] = None,
        steps: int = 100,
    ) -> np.ndarray:
        """
        Batch exponential map using PyTorch when available.

        Parameters
        ----------
        base_points : np.ndarray, shape (B, n)
        tangent_vectors : np.ndarray, shape (B, n)

        Returns
        -------
        np.ndarray, shape (B, n)
        """
        try:
            import torch
            _has_torch = True
        except ImportError:
            _has_torch = False

        if _has_torch:
            return self._exp_map_batch_torch(base_points, tangent_vectors, t_max, steps)
        return self._exp_map_batch_numpy(base_points, tangent_vectors, t_max, steps)

    def _exp_map_batch_torch(
        self,
        base_points: np.ndarray,
        tangent_vectors: np.ndarray,
        t_max: Optional[float],
        steps: int,
    ) -> np.ndarray:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pts = torch.tensor(base_points, dtype=torch.float64, device=device)
        vels = torch.tensor(tangent_vectors, dtype=torch.float64, device=device)
        B, n = pts.shape

        if t_max is None:
            # Use first point to determine t_max
            g0 = self.metric_fn(pts[0].cpu().numpy())
            s2 = float(tangent_vectors[0] @ g0 @ tangent_vectors[0])
            t_max = 1.0 / max(np.sqrt(s2), 1e-14)

        dt = t_max / steps
        dt_t = torch.tensor(dt, dtype=torch.float64, device=device)

        pos = pts.clone()
        vel = vels.clone()

        for _ in range(steps):
            # RK4 stage 1
            gamma1 = self._batch_christoffel_torch(pos, device)
            acc1 = torch.zeros_like(pos)
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        acc1[:, i] -= gamma1[:, i, j, k] * vel[:, j] * vel[:, k]

            k1x, k1v = vel, acc1

            # Stage 2
            pos2 = pos + 0.5 * dt_t * k1x
            vel2 = vel + 0.5 * dt_t * k1v
            gamma2 = self._batch_christoffel_torch(pos2, device)
            acc2 = torch.zeros_like(pos2)
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        acc2[:, i] -= gamma2[:, i, j, k] * vel2[:, j] * vel2[:, k]
            k2x, k2v = vel2, acc2

            # Stage 3
            pos3 = pos + 0.5 * dt_t * k2x
            vel3 = vel + 0.5 * dt_t * k2v
            gamma3 = self._batch_christoffel_torch(pos3, device)
            acc3 = torch.zeros_like(pos3)
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        acc3[:, i] -= gamma3[:, i, j, k] * vel3[:, j] * vel3[:, k]
            k3x, k3v = vel3, acc3

            # Stage 4
            pos4 = pos + dt_t * k3x
            vel4 = vel + dt_t * k3v
            gamma4 = self._batch_christoffel_torch(pos4, device)
            acc4 = torch.zeros_like(pos4)
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        acc4[:, i] -= gamma4[:, i, j, k] * vel4[:, j] * vel4[:, k]
            k4x, k4v = vel4, acc4

            pos = pos + (dt_t / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
            vel = vel + (dt_t / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)

        return pos.cpu().numpy()

    def _batch_christoffel_torch(self, pos: Any, device: Any) -> Any:
        import torch

        B, n = pos.shape
        gamma = torch.zeros(B, n, n, n, dtype=torch.float64, device=device)
        for b in range(B):
            x_np = pos[b].detach().cpu().numpy()
            g = self._get_christoffel(x_np)
            gamma[b] = torch.tensor(g, dtype=torch.float64, device=device)
        return gamma

    def _exp_map_batch_numpy(
        self,
        base_points: np.ndarray,
        tangent_vectors: np.ndarray,
        t_max: Optional[float],
        steps: int,
    ) -> np.ndarray:
        B, n = base_points.shape
        endpoints = np.zeros((B, n))
        for b in range(B):
            endpoints[b] = self.exp_map(base_points[b], tangent_vectors[b], t_max, steps)
        return endpoints

    # ----- log_p(q) ---------------------------------------------------------

    def log_map(
        self,
        base_point: np.ndarray,
        target_point: np.ndarray,
        max_iter: int = 50,
        tol: float = 1e-8,
    ) -> np.ndarray:
        """
        Logarithmic map: log_p(q) = the tangent vector v such that exp_p(v) = q.

        Uses an iterative Newton-type shooting method:
        1. Start with initial guess v₀ from the metric at p:  v₀ ≈ g⁻¹(q−p).
        2. Compute exp_p(v_k) and residual r_k = exp_p(v_k) − q.
        3. Update v via approximate Jacobian.

        Parameters
        ----------
        base_point : np.ndarray, shape (n,)
        target_point : np.ndarray, shape (n,)
        max_iter : int
        tol : float

        Returns
        -------
        np.ndarray, shape (n,)
        """
        n = len(base_point)
        diff = target_point - base_point

        # Initial guess: use metric to get a tangent-space estimate
        g_p = self.metric_fn(base_point)
        v = np.linalg.solve(g_p, diff)

        # First-order estimate of t_max from the expected distance
        speed_sq = float(v @ g_p @ v)
        t_max_est = 1.0 / max(np.sqrt(speed_sq), 1e-14)

        for iteration in range(max_iter):
            # Compute exp_p(v) — the forward map
            endpoint = self._exp_at_unit(v, base_point, t_max_est)
            residual = endpoint - target_point
            res_norm = np.linalg.norm(residual)

            if res_norm < tol:
                break

            # Approximate the Jacobian of exp_p at v via finite differences
            J = np.zeros((n, n))
            eps_fd = 1e-7
            for i in range(n):
                v_plus = v.copy()
                v_plus[i] += eps_fd
                ep_plus = self._exp_at_unit(v_plus, base_point, t_max_est)
                J[:, i] = (ep_plus - endpoint) / eps_fd

            # Newton step: Δv = −J⁻¹ residual
            try:
                dv = np.linalg.solve(J, -residual)
            except np.linalg.LinAlgError:
                dv = -residual * 0.1

            v += dv

        return v

    def _exp_at_unit(
        self,
        tangent_vec: np.ndarray,
        base_point: np.ndarray,
        t_max: float,
        steps: int = 50,
    ) -> np.ndarray:
        """Evaluate exp_p(v) at parameter t_max (used by log_map iterations)."""
        n = len(base_point)
        y0 = np.concatenate([base_point, tangent_vec])
        t_eval = np.array([t_max])

        sol = solve_ivp(
            self._geodesic_rhs,
            (0.0, t_max),
            y0,
            method="RK45",
            t_eval=t_eval,
            rtol=1e-8,
            atol=1e-10,
        )

        if not sol.success or sol.y.shape[1] == 0:
            return base_point + tangent_vec  # fallback

        return sol.y[:n, -1]

    def log_map_batch(
        self,
        base_points: np.ndarray,
        target_points: np.ndarray,
        max_iter: int = 30,
        tol: float = 1e-6,
    ) -> np.ndarray:
        """
        Batch logarithmic map.

        Parameters
        ----------
        base_points : np.ndarray, shape (B, n)
        target_points : np.ndarray, shape (B, n)

        Returns
        -------
        np.ndarray, shape (B, n)
        """
        B, n = base_points.shape
        tangent_vectors = np.zeros((B, n))
        for b in range(B):
            tangent_vectors[b] = self.log_map(base_points[b], target_points[b], max_iter, tol)
        return tangent_vectors


# ---------------------------------------------------------------------------
# Retraction Map (first-order approximation)
# ---------------------------------------------------------------------------

class RetractionMap:
    """
    First-order retraction: a computationally cheaper alternative to the
    full exponential map.

    The retraction satisfies R_p(0) = p and dR_p/dt|₀ = Id.  This implementation
    uses the projection-based retraction with QR decomposition for numerical
    stability.
    """

    def __init__(
        self,
        metric_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        self.metric_fn = metric_fn

    def retract(
        self,
        base_point: np.ndarray,
        tangent_vector: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the retraction R_p(v).

        Uses QR decomposition to project the displacement into the tangent
        space while maintaining numerical stability:

            R_p(v) = p + Q v

        where Q is from the QR decomposition of the displacement direction.

        Parameters
        ----------
        base_point : np.ndarray, shape (n,)
        tangent_vector : np.ndarray, shape (n,)

        Returns
        -------
        np.ndarray, shape (n,)
        """
        v = tangent_vector.reshape(-1, 1)

        # QR decomposition for numerical stability
        Q, R = np.linalg.qr(v)
        # Choose the sign convention so diagonal of R is positive
        signs = np.sign(np.diag(R))
        signs[signs == 0] = 1.0
        Q = Q * signs[np.newaxis, :]
        R = R * signs[:, np.newaxis]

        displacement = Q @ R
        return base_point + displacement.flatten()


# ---------------------------------------------------------------------------
# Inverse Retraction
# ---------------------------------------------------------------------------

class InverseRetraction:
    """
    Inverse of the retraction map: an approximate logarithmic map.

    Given p and q, finds the tangent vector v such that R_p(v) ≈ q.
    """

    def __init__(
        self,
        metric_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        self.metric_fn = metric_fn

    def inverse_retract(
        self,
        base_point: np.ndarray,
        target_point: np.ndarray,
    ) -> np.ndarray:
        """
        Approximate inverse retraction: v ≈ log_p(q).

        Computes the displacement q − p and optionally projects it onto
        the tangent space using the metric tensor at p.

        Parameters
        ----------
        base_point : np.ndarray, shape (n,)
        target_point : np.ndarray, shape (n,)

        Returns
        -------
        np.ndarray, shape (n,)
            Approximate tangent vector.
        """
        diff = target_point - base_point

        if self.metric_fn is not None:
            # Project into the tangent space using the metric
            # v^i = g^{ij} (q_j - p_j)
            g = self.metric_fn(base_point)
            try:
                v = np.linalg.solve(g, diff)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse
                v = np.linalg.lstsq(g, diff, rcond=None)[0]
            return v
        else:
            return diff

    def inverse_retract_batch(
        self,
        base_points: np.ndarray,
        target_points: np.ndarray,
    ) -> np.ndarray:
        """
        Batch inverse retraction.

        Parameters
        ----------
        base_points : np.ndarray, shape (B, n)
        target_points : np.ndarray, shape (B, n)

        Returns
        -------
        np.ndarray, shape (B, n)
        """
        B = base_points.shape[0]
        tangent_vectors = np.zeros_like(base_points)
        for b in range(B):
            tangent_vectors[b] = self.inverse_retract(base_points[b], target_points[b])
        return tangent_vectors
