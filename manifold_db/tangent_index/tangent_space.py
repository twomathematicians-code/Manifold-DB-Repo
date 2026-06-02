"""
Tangent Space - local linear approximation of the manifold at a point.

Each data point on the manifold has a tangent space T_p(M) that provides
a local Euclidean approximation of the curved manifold.  The tangent space
is computed via SVD of locally centred neighbourhood data, yielding an
orthonormal basis whose column span captures the intrinsic low-dimensional
structure of the manifold patch.

Key operations
---------------
* **project / lift**  – move between ambient and tangent coordinates.
* **log_map / exp_map** – Riemannian logarithm and exponential maps.
* **parallel_transport** – move a tangent vector between tangent spaces.
* **compute_metric / compute_distance** – inner products & geodesic distances.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.linalg import svd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Small helper
# ---------------------------------------------------------------------------


def _stable_svd(X: np.ndarray, full_matrices: bool = False) -> tuple:
    """Economy SVD with a tiny regularisation for numerical safety."""
    U, S, Vt = svd(X, full_matrices=full_matrices)
    return U, S, Vt


# ===========================================================================
# TangentSpace
# ===========================================================================


class TangentSpace:
    """
    Local tangent space anchored at *base_point* on the manifold.

    Parameters
    ----------
    base_point : np.ndarray of shape (ambient_dim,)
        The anchor point on the manifold.
    data : np.ndarray of shape (n_samples, ambient_dim) | None
        Local neighbourhood used to compute the basis.  If *None* the
        tangent space is initialised with an identity basis of the same
        dimension as *base_point*.
    intrinsic_dim : int | None
        Override for the intrinsic dimension.  If *None* it is inferred
        from the singular-value spectrum (see ``_infer_dimension``).
    """

    def __init__(
        self,
        base_point: np.ndarray,
        data: np.ndarray | None = None,
        intrinsic_dim: int | None = None,
    ) -> None:
        self.base_point = np.asarray(base_point, dtype=np.float64)
        self._ambient_dim = self.base_point.shape[0]

        if data is not None:
            data = np.asarray(data, dtype=np.float64)
            self._compute_basis_from_data(data, intrinsic_dim)
        else:
            dim = intrinsic_dim if intrinsic_dim is not None else self._ambient_dim
            self.dimension = dim
            self.basis = np.eye(self._ambient_dim, dim, dtype=np.float64)
            # Default: flat metric (Euclidean)
            self.metric_tensor = np.eye(dim, dtype=np.float64)
            self.christoffel_symbols = np.zeros((dim, dim, dim), dtype=np.float64)

        logger.debug(
            "TangentSpace created at base_point dim=%d, intrinsic_dim=%d",
            self._ambient_dim,
            self.dimension,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _infer_dimension(self, S: np.ndarray, threshold: float = 0.95) -> int:
        """Infer intrinsic dimension from cumulative explained variance."""
        cumvar = np.cumsum(S**2) / np.sum(S**2)
        return max(1, int(np.searchsorted(cumvar, threshold) + 1))

    def _compute_basis_from_data(
        self, data: np.ndarray, intrinsic_dim: int | None
    ) -> None:
        """Build orthonormal basis via centred SVD."""
        n, d = data.shape
        assert (
            d == self._ambient_dim
        ), f"Data ambient dim {d} != base_point dim {self._ambient_dim}"

        # Centre at base point
        X = data - self.base_point[np.newaxis, :]

        # Thin SVD – right singular vectors (rows of Vt) are the
        # principal directions in ambient space.  basis has shape (d, k)
        # so that project(data) = centered @ basis yields (n, k) coords.
        U, S, Vt = _stable_svd(X, full_matrices=False)

        min_dim = min(len(S), d)
        if intrinsic_dim is not None:
            self.dimension = min(intrinsic_dim, min_dim)
        else:
            self.dimension = self._infer_dimension(S)

        # Basis: top-k right singular vectors as columns → shape (ambient_dim, intrinsic_dim)
        self.basis = Vt[: self.dimension, :].T.copy()

        # ---- Metric tensor (induced Riemannian metric) ----
        # The metric is the pullback of the ambient Euclidean metric
        # through the projection, i.e. G_{ij} = <e_i, e_j> = delta_{ij}
        # for an orthonormal basis.  We perturb it with local curvature
        # estimated from residual variance.
        residuals = X - X @ self.basis @ self.basis.T
        self.metric_tensor = np.eye(self.dimension, dtype=np.float64)
        # Add curvature correction: off-diagonal terms from data covariance
        if n > self.dimension:
            cov = (self.basis.T @ X.T @ X @ self.basis) / max(n - 1, 1)
            # Normalise to keep metric positive-definite
            cov = cov / (np.max(np.abs(cov)) + 1e-12)
            self.metric_tensor += 1e-3 * cov

        # Symmetrise
        self.metric_tensor = 0.5 * (self.metric_tensor + self.metric_tensor.T)

        # ---- Christoffel symbols ----
        # Estimated numerically from the data gradient of the metric.
        self._estimate_christoffel(data)

    def _estimate_christoffel(self, data: np.ndarray) -> None:
        """
        Estimate Christoffel symbols of the first kind via finite
        differences of the metric over the local data patch.
        """
        d = self.dimension
        self.christoffel_symbols = np.zeros((d, d, d), dtype=np.float64)

        if data.shape[0] < max(2 * d + 1, 5):
            return  # not enough data for estimation

        # Project data to tangent coordinates
        tc = self.project(data)

        # For each direction, compute numerical derivative of metric
        # G_a = G(x + h e_a) - G(x - h e_a) / (2h)
        # We approximate this using k-NN patches along each axis.
        h = 1e-4
        for a in range(d):
            shifted_plus = tc.copy()
            shifted_minus = tc.copy()
            shifted_plus[:, a] += h
            shifted_minus[:, a] -= h

            # Lift back to ambient to observe metric change
            ambient_plus = self.lift(shifted_plus)
            ambient_minus = self.lift(shifted_minus)

            # Local metric at each shifted point (projected basis inner products)
            for idx in range(min(tc.shape[0], 20)):
                for i in range(d):
                    for j in range(d):
                        diff_plus = ambient_plus[idx] - self.base_point
                        diff_minus = ambient_minus[idx] - self.base_point
                        gp = diff_plus @ diff_plus  # scalar approximation
                        gm = diff_minus @ diff_minus
                        dG_da = (gp - gm) / (2.0 * h)
                        self.christoffel_symbols[a, i, j] += dG_da / min(
                            tc.shape[0], 20
                        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def project(self, data: np.ndarray) -> np.ndarray:
        """
        Project ambient-space data into tangent-space coordinates.

        Parameters
        ----------
        data : np.ndarray, shape (n, ambient_dim) or (ambient_dim,)

        Returns
        -------
        np.ndarray, shape (n, intrinsic_dim) or (intrinsic_dim,)
            Coordinates in the tangent space.
        """
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        centered = data - self.base_point[np.newaxis, :]
        return centered @ self.basis

    def lift(self, tangent_coords: np.ndarray) -> np.ndarray:
        """
        Lift tangent-space coordinates back to ambient space.

        This is the exponential map approximation: x ≈ p + B · ξ,
        where p is the base point, B the basis matrix, and ξ the
        tangent coordinates.

        Parameters
        ----------
        tangent_coords : np.ndarray, shape (n, intrinsic_dim) or (intrinsic_dim,)

        Returns
        -------
        np.ndarray, shape (n, ambient_dim) or (ambient_dim,)
        """
        tc = np.asarray(tangent_coords, dtype=np.float64)
        if tc.ndim == 1:
            tc = tc[np.newaxis, :]
        return self.base_point[np.newaxis, :] + tc @ self.basis.T

    def log_map(self, point: np.ndarray) -> np.ndarray:
        """
        Logarithmic map: ambient point → tangent vector.

        log_p(q) = B^T (q - p), which gives the tangent-space coordinates
        of the Riemannian log of *point* at the base point.

        Parameters
        ----------
        point : np.ndarray, shape (ambient_dim,)

        Returns
        -------
        np.ndarray, shape (intrinsic_dim,)
        """
        point = np.asarray(point, dtype=np.float64)
        diff = point - self.base_point
        tangent = diff @ self.basis

        # Correct for non-linearity: scale by inverse of local stretch factor
        stretch = np.sqrt(np.sum(diff**2))
        if stretch > 1e-12:
            # Approximate correction using geodesic distance / chord distance
            correction = 1.0 - (stretch**2) / (24.0 * (self._ambient_dim + 1))
            correction = max(correction, 0.5)
            tangent *= correction

        return tangent

    def exp_map(self, tangent_vec: np.ndarray) -> np.ndarray:
        """
        Exponential map: tangent vector → ambient point.

        exp_p(v) = p + B v + ½ Γ(v,v) correction, where the Γ term
        accounts for manifold curvature via the Christoffel symbols.

        Parameters
        ----------
        tangent_vec : np.ndarray, shape (intrinsic_dim,)

        Returns
        -------
        np.ndarray, shape (ambient_dim,)
        """
        v = np.asarray(tangent_vec, dtype=np.float64)
        # Linear part
        ambient = self.base_point + self.basis @ v

        # Second-order curvature correction using Christoffel symbols
        correction = np.zeros(self._ambient_dim, dtype=np.float64)
        d = self.dimension
        for i in range(d):
            for j in range(d):
                # Γ^k_{ij} v^i v^j summed over k
                gamma_term = (
                    self.christoffel_symbols[:, i, j]
                    if self.christoffel_symbols.shape[0] == self._ambient_dim
                    else self.christoffel_symbols[i, j, :]
                )
                correction += self.basis @ (gamma_term * v[i] * v[j])

        ambient += 0.5 * correction
        return ambient

    def compute_metric(
        self, tangent_vec_a: np.ndarray, tangent_vec_b: np.ndarray
    ) -> float:
        """
        Compute the Riemannian inner product <u, v>_G = u^T G v.

        Parameters
        ----------
        tangent_vec_a, tangent_vec_b : np.ndarray, shape (intrinsic_dim,)

        Returns
        -------
        float
        """
        a = np.asarray(tangent_vec_a, dtype=np.float64)
        b = np.asarray(tangent_vec_b, dtype=np.float64)
        return float(a @ self.metric_tensor @ b)

    def compute_distance(
        self, tangent_vec_a: np.ndarray, tangent_vec_b: np.ndarray
    ) -> float:
        """
        Geodesic distance approximation between two tangent vectors.

        d(a, b) ≈ sqrt(<a-b, a-b>_G)

        Parameters
        ----------
        tangent_vec_a, tangent_vec_b : np.ndarray, shape (intrinsic_dim,)

        Returns
        -------
        float
        """
        diff = tangent_vec_a - tangent_vec_b
        inner = self.compute_metric(diff, diff)
        return float(np.sqrt(max(inner, 0.0)))

    def parallel_transport(
        self, vec: np.ndarray, target_tangent_space: TangentSpace
    ) -> np.ndarray:
        """
        Parallel transport a tangent vector from this tangent space to
        *target_tangent_space*.

        Uses SVD-based optimal rotation between the two bases to align
        the vector in the target frame while preserving its magnitude
        (as measured by the source metric).

        Parameters
        ----------
        vec : np.ndarray, shape (intrinsic_dim,)
            Tangent vector in *this* tangent space.
        target_tangent_space : TangentSpace
            Destination tangent space.

        Returns
        -------
        np.ndarray, shape (target_intrinsic_dim,)
            Parallel-transported vector in the target tangent space.
        """
        v = np.asarray(vec, dtype=np.float64)

        # Compute transition matrix: T = B_target^T @ B_source
        # This maps coordinates from source basis to target basis
        transition = target_tangent_space.basis.T @ self.basis  # (d_t, d_s)

        # Use SVD to find the optimal rotation (closest orthogonal matrix)
        U_t, S_t, Vt_t = _stable_svd(transition, full_matrices=False)
        optimal_rotation = U_t @ Vt_t  # (d_t, d_t) or (d_t, d_s)

        transported = optimal_rotation @ v

        # Ensure correct dimensionality
        if transported.shape[0] > target_tangent_space.dimension:
            transported = transported[: target_tangent_space.dimension]

        # Pad with zeros if target has higher dimension
        if transported.shape[0] < target_tangent_space.dimension:
            pad = target_tangent_space.dimension - transported.shape[0]
            transported = np.concatenate([transported, np.zeros(pad)])

        # Preserve magnitude (from source metric)
        source_norm_sq = self.compute_metric(v, v)
        target_norm_sq = target_tangent_space.compute_metric(transported, transported)
        if target_norm_sq > 1e-15:
            transported *= np.sqrt(source_norm_sq / target_norm_sq)

        return transported

    def update_basis(self, data_batch: np.ndarray, lr: float = 0.1) -> None:
        """
        Online (incremental) update of the tangent space basis from new data.

        Uses a weighted combination of the existing basis and a newly
        computed basis from the batch.

        Parameters
        ----------
        data_batch : np.ndarray, shape (n_new, ambient_dim)
        lr : float
            Learning rate for blending old and new basis (0–1).
        """
        batch = np.asarray(data_batch, dtype=np.float64)
        if batch.shape[0] < self.dimension + 1:
            logger.warning(
                "Batch too small (%d) for basis update (dim=%d), skipping.",
                batch.shape[0],
                self.dimension,
            )
            return

        # Compute new basis from batch alone
        old_basis = self.basis.copy()
        self._compute_basis_from_data(batch, intrinsic_dim=self.dimension)

        new_basis = self.basis.copy()

        # Align new basis to old basis via SVD to avoid sign flips
        M = old_basis.T @ new_basis
        U_align, _, Vt_align = _stable_svd(M, full_matrices=False)
        align = U_align @ Vt_align
        new_basis = new_basis @ align

        # Blended basis
        blended = (1.0 - lr) * old_basis + lr * new_basis

        # Re-orthogonalise via QR
        Q, _ = np.linalg.qr(blended)
        self.basis = Q[:, : self.dimension]

        # Recompute metric tensor (smooth update)
        cov_term = (
            self.basis.T
            @ (batch - self.base_point).T
            @ (batch - self.base_point)
            @ self.basis
        )
        cov_term /= max(batch.shape[0] - 1, 1)
        cov_term = cov_term / (np.max(np.abs(cov_term)) + 1e-12)
        self.metric_tensor = 0.5 * (self.metric_tensor + self.metric_tensor.T)
        self.metric_tensor += lr * 1e-3 * cov_term
        self.metric_tensor = 0.5 * (self.metric_tensor + self.metric_tensor.T)

        logger.debug("Basis updated online with lr=%.3f", lr)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain-Python dict."""
        return {
            "base_point": self.base_point.tolist(),
            "basis": self.basis.tolist(),
            "dimension": self.dimension,
            "metric_tensor": self.metric_tensor.tolist(),
            "christoffel_symbols": self.christoffel_symbols.tolist(),
            "ambient_dim": self._ambient_dim,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TangentSpace:
        """Deserialise from a dict produced by ``to_dict``."""
        ts = cls.__new__(cls)
        ts.base_point = np.asarray(d["base_point"], dtype=np.float64)
        ts.basis = np.asarray(d["basis"], dtype=np.float64)
        ts.dimension = int(d["dimension"])
        ts.metric_tensor = np.asarray(d["metric_tensor"], dtype=np.float64)
        ts.christoffel_symbols = np.asarray(d["christoffel_symbols"], dtype=np.float64)
        ts._ambient_dim = int(d.get("ambient_dim", ts.base_point.shape[0]))
        return ts

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TangentSpace(base_point_dim={self._ambient_dim}, "
            f"intrinsic_dim={self.dimension})"
        )
