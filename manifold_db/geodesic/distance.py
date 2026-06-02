"""
Distance computation module for Riemannian manifolds.

Provides various distance measures beyond Euclidean, including:
  - Geodesic (Riemannian) distance
  - Tangent-space approximation
  - Curvature-corrected (second-order) distance
  - Wasserstein (optimal transport) distance
  - Fisher-Rao information distance

All computations use NumPy/SciPy with vectorised operations for speed.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Riemannian Distance
# ---------------------------------------------------------------------------

class RiemannianDistance:
    """
    Compute distances on a Riemannian manifold.

    Parameters
    ----------
    metric_tensor_fn : callable
        Function x → g_{ij}(x) returning the metric tensor at point x.
    christoffel_fn : callable or None
        Function x → Γⁱ_{jk}(x).  Used for curvature corrections.
    tangent_space : np.ndarray or None
        Pre-computed tangent space basis at a reference point.
    """

    def __init__(
        self,
        metric_tensor_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        christoffel_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        tangent_space: Optional[np.ndarray] = None,
    ) -> None:
        self.metric_tensor_fn = metric_tensor_fn
        self.christoffel_fn = christoffel_fn
        self.tangent_space = tangent_space

    def geodesic_distance(
        self,
        p: np.ndarray,
        q: np.ndarray,
    ) -> float:
        """
        True geodesic distance via energy minimisation on the manifold.

        Minimises ∫₀¹ g_{ij}(γ(t)) γ̇ⁱ(t) γ̇ʲ(t) dt discretised over
        a piecewise-linear path with waypoints.

        Parameters
        ----------
        p, q : np.ndarray, shape (n,)
            Points on the manifold.

        Returns
        -------
        float
            Approximate geodesic distance.
        """
        if self.metric_tensor_fn is None:
            raise RuntimeError("metric_tensor_fn is required for geodesic_distance")

        n = len(p)
        n_waypoints = 20

        def _energy(waypoints_flat: np.ndarray) -> float:
            waypoints = waypoints_flat.reshape(n_waypoints, n)
            path = np.vstack([p, waypoints, q])
            energy = 0.0
            for i in range(len(path) - 1):
                diff = path[i + 1] - path[i]
                mid = 0.5 * (path[i] + path[i + 1])
                g = self.metric_tensor_fn(mid)
                energy += float(diff @ g @ diff)
            return energy

        def _energy_grad(waypoints_flat: np.ndarray) -> np.ndarray:
            waypoints = waypoints_flat.reshape(n_waypoints, n)
            path = np.vstack([p, waypoints, q])
            grad = np.zeros_like(waypoints)
            for i in range(len(path) - 1):
                diff = path[i + 1] - path[i]
                mid = 0.5 * (path[i] + path[i + 1])
                g = self.metric_tensor_fn(mid)
                g_diff = g @ diff
                idx = i - 1  # index into waypoints (path[0]=p, path[-1]=q)
                if 0 <= idx < n_waypoints:
                    grad[idx] -= 2.0 * g_diff
                if 1 <= idx + 1 < n_waypoints + 1:
                    wp_idx = idx + 1
                    if wp_idx < n_waypoints:
                        grad[wp_idx] += 2.0 * g_diff
            return grad.ravel()

        init = np.zeros(n_waypoints * n)
        for i in range(n_waypoints):
            t = (i + 1) / (n_waypoints + 1)
            init[i * n:(i + 1) * n] = p + t * (q - p)

        res = minimize(_energy, init, jac=_energy_grad, method="L-BFGS-B",
                       options={"maxiter": 200, "ftol": 1e-12})
        return float(np.sqrt(res.fun))

    def tangent_approx_distance(
        self,
        p: np.ndarray,
        q: np.ndarray,
        tangent_space: Optional[np.ndarray] = None,
    ) -> float:
        """
        Approximate distance using the tangent-space representation.

        Parameters
        ----------
        p, q : np.ndarray, shape (n,)
        tangent_space : np.ndarray, shape (n, n) or None
            Orthonormal basis for the tangent space.  Identity is used
            when *None*.

        Returns
        -------
        float
            Tangent-space approximate distance.
        """
        ts = tangent_space if tangent_space is not None else self.tangent_space
        if ts is None:
            ts = np.eye(len(p))

        diff = q - p
        if self.metric_tensor_fn is not None:
            g_mid = self.metric_tensor_fn(0.5 * (p + q))
            return float(np.sqrt(diff @ g_mid @ diff))
        else:
            t_coords = ts.T @ diff
            return float(np.linalg.norm(t_coords))

    def curvature_corrected_distance(
        self,
        p: np.ndarray,
        q: np.ndarray,
    ) -> float:
        """
        Second-order curvature correction to the geodesic distance.

        d(p,q) ≈ d₀(p,q) · (1 − (1/12) R_{ijkl} vⁱ vʲ vᵏ vˡ / |v|⁴)

        where d₀ is the first-order distance and R is the Riemann curvature
        tensor evaluated at the midpoint.

        Parameters
        ----------
        p, q : np.ndarray, shape (n,)

        Returns
        -------
        float
            Curvature-corrected distance.
        """
        if self.metric_tensor_fn is None or self.christoffel_fn is None:
            logger.warning("Need metric_fn and christoffel_fn for curvature correction; "
                           "falling back to tangent approximation.")
            return self.tangent_approx_distance(p, q)

        mid = 0.5 * (p + q)
        g_mid = self.metric_tensor_fn(mid)
        diff = q - p
        d0 = float(np.sqrt(diff @ g_mid @ diff))

        # Approximate Riemann tensor via numerical differentiation of Christoffel symbols
        R = self._approximate_riemann_tensor(mid)

        # Corrective term
        g_inv = np.linalg.inv(g_mid)
        # Convert diff to contravariant velocity
        v_cov = g_mid @ diff  # covariant components of displacement
        v_contra = g_inv @ v_cov
        v_norm = float(np.linalg.norm(v_contra))
        if v_norm < 1e-14:
            return d0

        # ∑ R_{ijkl} vⁱ vʲ vᵏ vˡ
        n = len(p)
        correction_sum = 0.0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        correction_sum += R[i, j, k, l] * v_contra[i] * v_contra[j] * v_contra[k] * v_contra[l]

        correction = 1.0 - (1.0 / 12.0) * correction_sum / (v_norm ** 4)
        correction = max(correction, 0.5)  # clamp to avoid negative distances
        return d0 * correction

    def _approximate_riemann_tensor(self, x: np.ndarray) -> np.ndarray:
        """
        Approximate the Riemann curvature tensor numerically:

            Rⁱ_{jkl} = ∂Γⁱ_{jk}/∂x^l − ∂Γⁱ_{jl}/∂x^k + Γⁱ_{lk} Γˡ_{jm} − Γⁱ_{lm} Γˡ_{jk}

        Returns the fully covariant form R_{ijkl} = g_{im} R^m_{jkl}.
        """
        n = len(x)
        eps = 1e-5
        R = np.zeros((n, n, n, n))

        gamma_center = self.christoffel_fn(x)
        g = self.metric_tensor_fn(x)
        g_inv = np.linalg.inv(g)

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        x_plus_l = x.copy(); x_plus_l[l] += eps
                        x_minus_l = x.copy(); x_minus_l[l] -= eps
                        x_plus_k = x.copy(); x_plus_k[k] += eps
                        x_minus_k = x.copy(); x_minus_k[k] -= eps

                        dg_jk_l = (self.christoffel_fn(x_plus_l)[i, j, k]
                                   - self.christoffel_fn(x_minus_l)[i, j, k]) / (2 * eps)
                        dg_jl_k = (self.christoffel_fn(x_plus_k)[i, j, l]
                                   - self.christoffel_fn(x_minus_k)[i, j, l]) / (2 * eps)

                        R_upper = dg_jk_l - dg_jl_k
                        for m in range(n):
                            R_upper += (gamma_center[i, m, k] * gamma_center[m, j, l]
                                        - gamma_center[i, m, l] * gamma_center[m, j, k])

                        # Lower the first index: R_{ijkl} = g_{im} R^m_{jkl}
                        for m in range(n):
                            R[i, j, k, l] += g[i, m] * R_upper / n  # average for stability
        return R


# ---------------------------------------------------------------------------
# Wasserstein Distance (Entropy-Regularised Optimal Transport)
# ---------------------------------------------------------------------------

class WassersteinDistance:
    """
    Wasserstein distance between probability distributions on the manifold.

    Uses the Sinkhorn algorithm for entropy-regularised optimal transport.
    """

    def __init__(
        self,
        reg: float = 0.1,
        max_iter: int = 500,
        tol: float = 1e-8,
    ) -> None:
        """
        Parameters
        ----------
        reg : float
            Entropy regularisation parameter (ε > 0).
        max_iter : int
            Maximum Sinkhorn iterations.
        tol : float
            Convergence tolerance on dual variable change.
        """
        self.reg = reg
        self.max_iter = max_iter
        self.tol = tol

    def sinkhorn_distance(
        self,
        mu: np.ndarray,
        nu: np.ndarray,
        cost_matrix: np.ndarray,
        reg: Optional[float] = None,
    ) -> float:
        """
        Entropy-regularised optimal transport distance (Sinkhorn).

        Parameters
        ----------
        mu : np.ndarray, shape (m,)
            Source distribution (must sum to 1).
        nu : np.ndarray, shape (n,)
            Target distribution (must sum to 1).
        cost_matrix : np.ndarray, shape (m, n)
            Transport cost C_{ij} between source i and target j.
        reg : float or None
            Regularisation strength.  Uses ``self.reg`` if *None*.

        Returns
        -------
        float
            Sinkhorn distance W_ε(μ, ν).
        """
        eps = reg if reg is not None else self.reg
        m, n = cost_matrix.shape

        # K = exp(-C / eps)
        K = np.exp(-cost_matrix / eps)
        K_tilde = K / (mu[:, np.newaxis] * nu[np.newaxis, :]).max()  # numerical stability

        # Dual variables
        u = np.ones(m)
        v = np.ones(n)

        for iteration in range(self.max_iter):
            u_prev = u.copy()
            v_prev = v.copy()

            # Bregman projections
            Kv = K_tilde @ v
            Kv = np.maximum(Kv, 1e-300)  # avoid log(0)
            u = mu / Kv

            Ktu = K_tilde.T @ u
            Ktu = np.maximum(Ktu, 1e-300)
            v = nu / Ktu

            # Check convergence
            u_change = np.max(np.abs(u - u_prev))
            v_change = np.max(np.abs(v - v_prev))
            if u_change < self.tol and v_change < self.tol:
                break

        # Transport plan
        transport = np.diag(u) @ K_tilde @ np.diag(v)
        transport = transport / transport.sum() * 1.0  # normalise

        # Sinkhorn distance = <transport, cost>
        distance = float(np.sum(transport * cost_matrix))
        logger.debug("Sinkhorn converged in %d iterations, distance=%.6f", iteration + 1, distance)
        return distance

    def batch_sinkhorn_distance(
        self,
        mu_batch: np.ndarray,
        nu_batch: np.ndarray,
        cost_matrices: np.ndarray,
    ) -> np.ndarray:
        """
        Batch version of Sinkhorn distance.

        Parameters
        ----------
        mu_batch : np.ndarray, shape (B, m)
        nu_batch : np.ndarray, shape (B, n)
        cost_matrices : np.ndarray, shape (B, m, n)

        Returns
        -------
        np.ndarray, shape (B,)
        """
        B = len(mu_batch)
        distances = np.zeros(B)
        for b in range(B):
            distances[b] = self.sinkhorn_distance(mu_batch[b], nu_batch[b], cost_matrices[b])
        return distances


# ---------------------------------------------------------------------------
# Fisher-Rao Distance
# ---------------------------------------------------------------------------

class FisherRaoDistance:
    """
    Fisher-Rao information metric distance for probability distributions.

    Uses the square-root representation (Hellinger) for efficient computation:

        d_{FR}(p, q) = 2 arccos(∑_x √(p(x)) √(q(x)))
    """

    def fisher_rao_distance(
        self,
        p_dist: np.ndarray,
        q_dist: np.ndarray,
    ) -> float:
        """
        Compute the Fisher-Rao distance between two discrete distributions.

        Parameters
        ----------
        p_dist, q_dist : np.ndarray, shape (n,)
            Probability distributions (non-negative, sum to 1).

        Returns
        -------
        float
            Fisher-Rao distance ∈ [0, π].
        """
        p = np.asarray(p_dist, dtype=np.float64)
        q = np.asarray(q_dist, dtype=np.float64)

        # Normalise
        p = p / p.sum()
        q = q / q.sum()

        # Hellinger affinity: ∑ √(p_i q_i)
        sqrt_p = np.sqrt(np.maximum(p, 0.0))
        sqrt_q = np.sqrt(np.maximum(q, 0.0))
        affinity = float(np.sum(sqrt_p * sqrt_q))
        affinity = np.clip(affinity, -1.0, 1.0)

        return 2.0 * np.arccos(affinity)

    def fisher_rao_metric_tensor(
        self,
        distribution: np.ndarray,
        param_indices: Optional[tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Compute the Fisher information matrix for a parametric family.

        For a discrete distribution parameterised by θ, the Fisher metric is:
            I_{ij} = ∑_x p(x|θ) ∂log p(x|θ)/∂θⁱ ∂log p(x|θ)/∂θʲ

        Here we compute it numerically via finite differences of log-probs.

        Parameters
        ----------
        distribution : np.ndarray, shape (n,)
            Current distribution values.
        param_indices : tuple of two indices or None
            If provided, only compute the 2×2 sub-block for those params.

        Returns
        -------
        np.ndarray
            Fisher information metric tensor.
        """
        n = len(distribution)
        eps = 1e-6
        log_p = np.log(np.maximum(distribution, 1e-300))

        fisher = np.zeros((n, n))
        for i in range(n):
            p_plus = distribution.copy()
            p_minus = distribution.copy()
            p_plus[i] += eps
            p_minus[i] -= eps
            p_plus = np.maximum(p_plus, 1e-300)
            p_minus = np.maximum(p_minus, 1e-300)

            dlog_i_plus = np.log(p_plus) - log_p
            dlog_i_minus = log_p - np.log(p_minus)
            dlog_i = 0.5 * (dlog_i_plus + dlog_i_minus)

            for j in range(i, n):
                p_plus_j = distribution.copy()
                p_minus_j = distribution.copy()
                p_plus_j[j] += eps
                p_minus_j[j] -= eps
                p_plus_j = np.maximum(p_plus_j, 1e-300)
                p_minus_j = np.maximum(p_minus_j, 1e-300)

                dlog_j_plus = np.log(p_plus_j) - log_p
                dlog_j_minus = log_p - np.log(p_minus_j)
                dlog_j = 0.5 * (dlog_j_plus + dlog_j_minus)

                fisher[i, j] = float(np.sum(distribution * dlog_i * dlog_j))
                fisher[j, i] = fisher[i, j]

        return fisher


# ---------------------------------------------------------------------------
# Distance Computer (unified interface)
# ---------------------------------------------------------------------------

class DistanceComputer:
    """
    Unified interface for computing various distances on a Riemannian manifold.

    Parameters
    ----------
    metric_tensor_fn : callable or None
        x → g_{ij}(x).  Required for geodesic and curvature-corrected distances.
    christoffel_fn : callable or None
        x → Γⁱ_{jk}(x).
    """

    def __init__(
        self,
        metric_tensor_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        christoffel_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        self.metric_tensor_fn = metric_tensor_fn
        self.christoffel_fn = christoffel_fn
        self._riemannian = RiemannianDistance(metric_tensor_fn, christoffel_fn)
        self._wasserstein = WassersteinDistance()

    def compute(
        self,
        p: Any,
        q: Any,
        metric_type: str = "geodesic",
        **kwargs: Any,
    ) -> float:
        """
        Compute a distance between two points / distributions.

        Parameters
        ----------
        p, q : np.ndarray
            Points on the manifold or probability distributions.
        metric_type : str
            One of: ``'geodesic'``, ``'tangent'``, ``'curvature'``,
            ``'wasserstein'``, ``'fisher_rao'``.
        **kwargs
            Additional arguments forwarded to the specific distance function.

        Returns
        -------
        float
        """
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)

        if metric_type == "geodesic":
            return self._riemannian.geodesic_distance(p, q)
        elif metric_type == "tangent":
            return self._riemannian.tangent_approx_distance(p, q, kwargs.get("tangent_space"))
        elif metric_type == "curvature":
            return self._riemannian.curvature_corrected_distance(p, q)
        elif metric_type == "wasserstein":
            cost = kwargs.get("cost_matrix")
            if cost is None:
                raise ValueError("cost_matrix required for wasserstein distance")
            return self._wasserstein.sinkhorn_distance(p, q, cost, kwargs.get("reg"))
        elif metric_type == "fisher_rao":
            return FisherRaoDistance().fisher_rao_distance(p, q)
        else:
            raise ValueError(f"Unknown metric_type: {metric_type}")

    def batch_compute(
        self,
        points_a: np.ndarray,
        points_b: np.ndarray,
        metric_type: str = "geodesic",
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Compute distances between corresponding pairs of points.

        Parameters
        ----------
        points_a : np.ndarray, shape (B, n) or (B,)
        points_b : np.ndarray, shape (B, n) or (B,)
        metric_type : str

        Returns
        -------
        np.ndarray, shape (B,)
            Distances for each pair.
        """
        points_a = np.asarray(points_a, dtype=np.float64)
        points_b = np.asarray(points_b, dtype=np.float64)

        if points_a.ndim == 1:
            points_a = points_a[np.newaxis, :]
        if points_b.ndim == 1:
            points_b = points_b[np.newaxis, :]

        B = points_a.shape[0]
        distances = np.zeros(B)

        # Use vectorised shortcuts where possible
        if metric_type == "tangent" and self.metric_tensor_fn is not None:
            for b in range(B):
                mid = 0.5 * (points_a[b] + points_b[b])
                diff = points_b[b] - points_a[b]
                g = self.metric_tensor_fn(mid)
                distances[b] = float(np.sqrt(diff @ g @ diff))
            return distances

        if metric_type == "fisher_rao":
            fr = FisherRaoDistance()
            for b in range(B):
                distances[b] = fr.fisher_rao_distance(points_a[b], points_b[b])
            return distances

        # General case
        for b in range(B):
            distances[b] = self.compute(points_a[b], points_b[b], metric_type, **kwargs)
        return distances
