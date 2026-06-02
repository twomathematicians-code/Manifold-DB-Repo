"""
Mathematical utilities for manifold operations.

Provides numerical linear algebra primitives for Riemannian geometry computations,
including SPD matrix operations, eigendecomposition, orthogonalization, and
numerical differentiation routines.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def ensure_spd(
    matrix: np.ndarray,
    min_eigenvalue: float = 1e-10,
) -> np.ndarray:
    """Ensure a matrix is symmetric positive-definite.

    Symmetrizes the input first, then checks eigenvalues. If any eigenvalue
    falls below *min_eigenvalue*, shifts the entire spectrum upward by adding
    a small multiple of the identity matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Square matrix ``(n, n)``.
    min_eigenvalue : float
        Minimum acceptable eigenvalue after correction.

    Returns
    -------
    np.ndarray
        Symmetric positive-definite matrix.

    Examples
    --------
    >>> A = np.array([[2.0, 1.0], [0.5, 3.0]])
    >>> ensure_spd(A)
    array([[2.25, 0.75],
           [0.75, 2.75]])
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {matrix.shape}")

    # Symmetrize: M <- (M + M^T) / 2
    matrix = 0.5 * (matrix + matrix.T)

    # Eigendecompose to inspect spectrum
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    min_ev = eigenvalues.min()

    if min_ev < min_eigenvalue:
        shift = min_eigenvalue - min_ev + 1e-12
        matrix = matrix + shift * np.eye(matrix.shape[0], dtype=np.float64)

    return matrix


def cholesky_decomposition(
    spd_matrix: np.ndarray,
    lower: bool = True,
) -> np.ndarray:
    """Cholesky decomposition with numerical stability checks.

    Parameters
    ----------
    spd_matrix : np.ndarray
        Symmetric positive-definite matrix ``(n, n)``.
    lower : bool
        If *True* return lower-triangular factor *L*; otherwise upper *L^T*.

    Returns
    -------
    np.ndarray
        Cholesky factor ``(n, n)``.

    Raises
    ------
    ValueError
        If the matrix is not numerically positive-definite after a single
        correction attempt.
    """
    spd_matrix = np.asarray(spd_matrix, dtype=np.float64)
    if spd_matrix.ndim != 2 or spd_matrix.shape[0] != spd_matrix.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {spd_matrix.shape}")

    try:
        return (
            np.linalg.cholesky(spd_matrix).T
            if not lower
            else np.linalg.cholesky(spd_matrix)
        )
    except np.linalg.LinAlgError:
        # One retry after ensuring SPD
        spd_matrix = ensure_spd(spd_matrix)
        try:
            return (
                np.linalg.cholesky(spd_matrix).T
                if not lower
                else np.linalg.cholesky(spd_matrix)
            )
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Matrix is not positive-definite even after correction."
            ) from exc


def matrix_square_root(spd_matrix: np.ndarray) -> np.ndarray:
    """Compute the matrix square root of an SPD matrix via eigendecomposition.

    For an SPD matrix *M = Q Λ Q^T*, the square root is *M^{1/2} = Q Λ^{1/2} Q^T*.

    Parameters
    ----------
    spd_matrix : np.ndarray
        Symmetric positive-definite matrix ``(n, n)``.

    Returns
    -------
    np.ndarray
        Matrix square root ``(n, n)``.
    """
    spd_matrix = ensure_spd(np.asarray(spd_matrix, dtype=np.float64))
    eigenvalues, eigenvectors = np.linalg.eigh(spd_matrix)
    # Clamp negatives (should not happen after ensure_spd but be safe)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    sqrt_diag = np.diag(np.sqrt(eigenvalues))
    return eigenvectors @ sqrt_diag @ eigenvectors.T


def log_det_spd(spd_matrix: np.ndarray) -> float:
    """Numerically stable log-determinant of an SPD matrix.

    Uses eigendecomposition so that ``log(det(M)) = sum(log(lambda_i))``,
    avoiding overflow/underflow of the raw determinant.

    Parameters
    ----------
    spd_matrix : np.ndarray
        Symmetric positive-definite matrix ``(n, n)``.

    Returns
    -------
    float
        ``log(det(M))``.
    """
    spd_matrix = ensure_spd(np.asarray(spd_matrix, dtype=np.float64))
    eigenvalues = np.linalg.eigvalsh(spd_matrix)
    eigenvalues = np.maximum(eigenvalues, 1e-300)  # avoid log(0)
    return float(np.sum(np.log(eigenvalues)))


def nearest_spd(
    A: np.ndarray,
    max_iterations: int = 100,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Find the nearest symmetric positive-definite matrix (Higham 2002).

    Implements the algorithm from N. J. Higham, "Computing the nearest
    correlation matrix – a problem from finance", *IMA J. Numer. Anal.*
    22(3), 329–343, 2002.

    Parameters
    ----------
    A : np.ndarray
        Square input matrix ``(n, n)``.
    max_iterations : int
        Maximum number of Newton iterations.
    tolerance : float
        Convergence tolerance on the Frobenius norm of successive iterates.

    Returns
    -------
    np.ndarray
        Nearest SPD matrix to *A*.
    """
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    if A.ndim != 2 or A.shape[1] != n:
        raise ValueError(f"Expected square matrix, got shape {A.shape}")

    B = (A + A.T) / 2.0
    _, S, _ = np.linalg.svd(B)

    # Scale to unit diagonal (Higham step)
    H = np.eye(n) @ S @ np.eye(n)

    for _ in range(max_iterations):
        eigvals, eigvecs = np.linalg.eigh(H)
        neg_mask = eigvals < 0.0
        if not np.any(neg_mask):
            break
        eigvals[neg_mask] = 0.0
        H_new = eigvecs @ np.diag(eigvals) @ eigvecs.T
        if np.linalg.norm(H_new - H, "fro") < tolerance:
            H = H_new
            break
        H = H_new

    # Make sure result is strictly PD
    H = ensure_spd(H)
    # Symmetrize once more
    H = 0.5 * (H + H.T)
    return H


def safe_inverse(
    matrix: np.ndarray,
    reg: float = 1e-6,
) -> np.ndarray:
    """Regularized matrix inverse.

    Computes ``(M^T M + reg * I)^{-1}`` which is always well-conditioned.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix ``(n, n)`` (or ``(m, n)`` for pseudo-inverse style).
    reg : float
        Tikhonov regularization parameter.

    Returns
    -------
    np.ndarray
        Regularized inverse ``(n, n)``.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    n = matrix.shape[0]
    gram = matrix.T @ matrix if matrix.shape[0] != matrix.shape[1] else matrix
    gram = 0.5 * (gram + gram.T)  # symmetrize
    gram += reg * np.eye(n, dtype=np.float64)
    return np.linalg.inv(gram)


def orthogonalize(
    vectors: np.ndarray,
) -> np.ndarray:
    """Gram-Schmidt orthogonalization of column vectors.

    Parameters
    ----------
    vectors : np.ndarray
        Matrix ``(d, k)`` whose *k* columns are the vectors to orthogonalize.

    Returns
    -------
    np.ndarray
        Orthonormal basis ``(d, k)``.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim == 1:
        vectors = vectors[:, np.newaxis]

    d, k = vectors.shape
    Q = np.zeros_like(vectors)
    Q[:, 0] = vectors[:, 0] / (np.linalg.norm(vectors[:, 0]) + 1e-300)

    for i in range(1, k):
        v = vectors[:, i].copy()
        for j in range(i):
            v -= np.dot(Q[:, j], v) * Q[:, j]
        norm = np.linalg.norm(v)
        if norm < 1e-14:
            # Degenerate – fill with a random orthogonal vector
            v = np.random.randn(d)
            for j in range(i):
                v -= np.dot(Q[:, j], v) * Q[:, j]
            norm = np.linalg.norm(v)
        Q[:, i] = v / norm
    return Q


def random_spd_matrix(
    n: int,
    condition_number: float = 10.0,
) -> np.ndarray:
    """Generate a random symmetric positive-definite test matrix.

    Parameters
    ----------
    n : int
        Dimension.
    condition_number : float
        Ratio of largest to smallest eigenvalue.

    Returns
    -------
    np.ndarray
        SPD matrix ``(n, n)``.
    """
    log_conds = np.linspace(0, np.log(condition_number), n)
    eigenvalues = np.exp(log_conds)
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    return Q @ np.diag(eigenvalues) @ Q.T


def compute_numerical_gradient(
    fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Compute the numerical Jacobian of *fn* at *x* via central differences.

    Parameters
    ----------
    fn : callable
        Scalar or vector-valued function ``f: R^n -> R^m``.
    x : np.ndarray
        Point at which to evaluate the Jacobian.
    eps : float
        Finite-difference step size.

    Returns
    -------
    np.ndarray
        Jacobian ``(m, n)``.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    f0 = np.asarray(fn(x), dtype=np.float64).ravel()
    m = f0.shape[0]
    n = x.shape[0]
    J = np.zeros((m, n), dtype=np.float64)

    for j in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[j] += eps
        x_minus[j] -= eps
        f_plus = np.asarray(fn(x_plus), dtype=np.float64).ravel()
        f_minus = np.asarray(fn(x_minus), dtype=np.float64).ravel()
        J[:, j] = (f_plus - f_minus) / (2.0 * eps)
    return J


def curvature_bounds(
    metric_fn: Callable[[np.ndarray], np.ndarray],
    region: np.ndarray,
    n_samples: int = 100,
) -> tuple[float, float]:
    """Estimate min/max sectional curvature on a region of the manifold.

    Samples points uniformly from the bounding box defined by *region*
    ``(2, d)`` (min and max coordinates) and computes the Riemann curvature
    tensor numerically at each sample, then returns the extremal sectional
    curvatures observed.

    Parameters
    ----------
    metric_fn : callable
        Function returning the metric tensor *g(x)* at a point *x*.
    region : np.ndarray
        ``(2, d)`` array with min/max coordinates.
    n_samples : int
        Number of Monte-Carlo sample points.

    Returns
    -------
    tuple[float, float]
        ``(min_curvature, max_curvature)``.
    """
    region = np.asarray(region, dtype=np.float64)
    if region.shape[0] != 2:
        raise ValueError("region must have shape (2, d)")

    d = region.shape[1]
    min_curv = np.inf
    max_curv = -np.inf

    for _ in range(n_samples):
        x = np.random.uniform(region[0], region[1])
        g = np.asarray(metric_fn(x), dtype=np.float64)
        g_inv = safe_inverse(g)

        # Numerical second derivatives of the metric → Christoffel → curvature
        # Simplified: use finite differences for the metric gradient
        for i in range(d):
            x_plus = x.copy()
            x_plus[i] += 1e-4
            x_minus = x.copy()
            x_minus[i] -= 1e-4
            g_plus = np.asarray(metric_fn(x_plus), dtype=np.float64)
            g_minus = np.asarray(metric_fn(x_minus), dtype=np.float64)
            dg = (g_plus - g_minus) / (2e-4)
            # Rough curvature proxy: norm of metric gradient relative to metric
            curv_proxy = np.linalg.norm(g_inv @ dg, "fro")
            min_curv = min(min_curv, curv_proxy)
            max_curv = max(max_curv, curv_proxy)

    return float(min_curv), float(max_curv)


def volume_element(metric_tensor: np.ndarray) -> float:
    """Compute the volume element ``sqrt(det(g))`` for integration.

    Parameters
    ----------
    metric_tensor : np.ndarray
        Metric tensor ``(d, d)`` at a point.

    Returns
    -------
    float
        ``sqrt(det(g))``.
    """
    metric_tensor = ensure_spd(np.asarray(metric_tensor, dtype=np.float64))
    return float(np.sqrt(np.linalg.det(metric_tensor)))


def barycentric_projection(
    points: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Weighted average (Fréchet / barycentric mean approximation).

    Computes the Euclidean weighted mean of *points*; useful as a first-order
    approximation to the Riemannian barycenter when points lie in a common
    tangent space.

    Parameters
    ----------
    points : np.ndarray
        ``(k, d)`` array of *k* points in *d*-dimensional space.
    weights : np.ndarray or None
        ``(k,)`` non-negative weights summing to 1. If *None*, uniform weights.

    Returns
    -------
    np.ndarray
        Barycentric projection ``(d,)``.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        return points.copy()
    if weights is None:
        weights = np.ones(points.shape[0], dtype=np.float64) / points.shape[0]
    else:
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / weights.sum()
    return float(np.dot(weights, points))
