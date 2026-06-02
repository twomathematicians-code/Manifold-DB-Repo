"""
Manifold learning utilities.

Provides dimensionality estimation, local PCA, diffusion maps, spectral
embedding, and quality assessment for manifold-structured data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from scipy.spatial import KDTree

# ---------------------------------------------------------------------------
# Intrinsic dimension estimation
# ---------------------------------------------------------------------------


def estimate_intrinsic_dim(
    data: np.ndarray,
    method: str = "pca",
    threshold: float = 0.95,
) -> int:
    """Estimate the intrinsic dimensionality of a dataset.

    Parameters
    ----------
    data : np.ndarray
        ``(n_samples, n_features)`` data matrix.
    method : str
        One of ``'pca'``, ``'mle'``, or ``'twonn'``.
    threshold : float
        For PCA: cumulative explained-variance threshold.

    Returns
    -------
    int
        Estimated intrinsic dimension.
    """
    data = np.asarray(data, dtype=np.float64)
    n, d = data.shape
    if n < 2:
        return min(d, 1)

    method = method.lower().strip()
    if method == "pca":
        return _pca_dim(data, threshold)
    elif method == "mle":
        return _mle_dim(data)
    elif method == "twonn":
        return _twonn_dim(data)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'pca', 'mle', or 'twonn'.")


def _pca_dim(data: np.ndarray, threshold: float = 0.95) -> int:
    """PCA-based intrinsic dimension: count eigenvalues explaining threshold variance."""
    centered = data - data.mean(axis=0)
    cov = (centered.T @ centered) / (len(data) - 1)
    eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
    total_var = eigenvalues.sum()
    if total_var == 0:
        return 1
    cumvar = np.cumsum(eigenvalues) / total_var
    return int(np.searchsorted(cumvar, threshold) + 1)


def _mle_dim(data: np.ndarray, k: int = 30) -> int:
    """Maximum likelihood estimator (Levina & Bickel, 2004)."""
    n = data.shape[0]
    k = min(k, n - 1)
    tree = KDTree(data)
    distances, _ = tree.query(data, k=k + 1)  # +1 because self is included
    # Exclude the zero self-distance
    distances = distances[:, 1:]  # (n, k)

    log_k = np.log(k)
    # For each sample, compute the sum of log-distances
    log_dists = np.log(distances)
    # MLE formula per sample
    mle_per_sample = -(log_k - 1.0 / k * log_dists.sum(axis=1))
    # MLE per sample gives the local intrinsic dimension; take average
    d_hat = float(np.mean(mle_per_sample))
    return max(1, int(np.round(d_hat)))


def _twonn_dim(data: np.ndarray) -> int:
    """Two-nearest-neighbors estimator (Fausto-Ferrari et al.)."""
    tree = KDTree(data)
    distances, _ = tree.query(data, k=3)  # 0=self, 1=NN1, 2=NN2
    r1 = distances[:, 1]
    r2 = distances[:, 2]

    # Avoid zeros
    mask = (r1 > 0) & (r2 > 0)
    r1, r2 = r1[mask], r2[mask]

    if len(r1) < 2:
        return 1

    mu = r2 / r1
    # Fit power-law: estimate d from the empirical CDF
    # d is estimated as -1 / E[log(mu)]
    log_mu = np.log(mu)
    d_hat = -1.0 / (np.mean(log_mu) + 1e-300)
    return max(1, min(int(np.round(d_hat)), data.shape[1]))


# ---------------------------------------------------------------------------
# Local PCA
# ---------------------------------------------------------------------------


def local_pca(
    data: np.ndarray,
    k_neighbors: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-point local PCA for tangent-space estimation.

    Parameters
    ----------
    data : np.ndarray
        ``(n_samples, n_features)``.
    k_neighbors : int
        Number of nearest neighbors for each local PCA.

    Returns
    -------
    tuple
        *eigenvalues* ``(n, d)``, *eigenvectors* ``(n, d, d)``,
        *local_means* ``(n, d)``.
    """
    data = np.asarray(data, dtype=np.float64)
    n, d = data.shape
    k_neighbors = min(k_neighbors, n - 1)

    tree = KDTree(data)
    _, indices = tree.query(data, k=k_neighbors + 1)
    indices = indices[:, 1:]  # drop self

    eigenvalues = np.zeros((n, d), dtype=np.float64)
    eigenvectors = np.zeros((n, d, d), dtype=np.float64)
    local_means = np.zeros((n, d), dtype=np.float64)

    for i in range(n):
        neighbors = data[indices[i]]
        mean = neighbors.mean(axis=0)
        centered = neighbors - mean
        cov = centered.T @ centered / max(len(neighbors) - 1, 1)
        try:
            evals, evecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            evals = np.ones(d)
            evecs = np.eye(d)

        idx = np.argsort(evals)[::-1]
        eigenvalues[i] = evals[idx]
        eigenvectors[i] = evecs[:, idx]
        local_means[i] = mean

    return eigenvalues, eigenvectors, local_means


# ---------------------------------------------------------------------------
# Diffusion maps
# ---------------------------------------------------------------------------


def diffusion_map(
    data: np.ndarray,
    n_components: int = 2,
    alpha: float = 0.5,
    n_neighbors: int = 15,
) -> np.ndarray:
    """Coifman-Lafon diffusion maps for nonlinear embedding.

    Parameters
    ----------
    data : np.ndarray
        ``(n_samples, n_features)``.
    n_components : int
        Number of diffusion coordinates to return.
    alpha : float
        Anisotropy parameter in ``[0, 1]``.
    n_neighbors : int
        Neighbors for Gaussian kernel bandwidth.

    Returns
    -------
    np.ndarray
        Embedding ``(n_samples, n_components)``.
    """
    data = np.asarray(data, dtype=np.float64)
    n = data.shape[0]
    n_components = min(n_components, n - 2)
    n_neighbors = min(n_neighbors, n - 1)

    tree = KDTree(data)
    distances, _ = tree.query(data, k=n_neighbors + 1)
    distances = distances[:, 1:]  # drop self

    # Local bandwidth = distance to k-th neighbor
    bandwidths = distances[:, -1:]  # (n, 1)
    bandwidths = np.maximum(bandwidths, 1e-10)

    # Gaussian kernel with anisotropic normalization
    K = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j_idx in range(n_neighbors):
            j = _neighbor_index(tree, data[i], j_idx)
            d_ij = distances[i, j_idx]
            K[i, j] = np.exp(-(d_ij**2) / (bandwidths[i, 0] * bandwidths[j, 0]))
            K[j, i] = K[i, j]

    # Anisotropic normalization q_i = sum_j K_ij
    q = K.sum(axis=1, keepdims=True) ** alpha
    q = np.maximum(q, 1e-300)
    K_aniso = K / (q * q.T)

    # Row-normalize to get transition matrix P
    d_row = K_aniso.sum(axis=1)
    d_row = np.maximum(d_row, 1e-300)
    P = K_aniso / d_row[:, None]

    # Eigen-decomposition of P (use symmetric version)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d_row))
    L_sym = D_inv_sqrt @ P @ np.diag(np.sqrt(d_row))
    # This is equivalent to D^{-1/2} K D^{-1/2}
    L_sym = (L_sym + L_sym.T) / 2.0

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L_sym)
    except np.linalg.LinAlgError:
        # Fallback: use scipy sparse eigsh
        L_sparse = csr_matrix(L_sym)
        eigenvalues, eigenvectors = eigsh(L_sparse, k=n_components + 1, which="LM")

    # Skip trivial eigenvalue (=1) and take next n_components
    idx = np.argsort(-eigenvalues)
    idx = idx[1 : n_components + 1]  # skip the first (lambda=1)
    embedding = eigenvectors[:, idx]

    # Normalize by eigenvalues
    for col_i in range(n_components):
        ev = eigenvalues[idx[col_i]]
        if ev > 0:
            embedding[:, col_i] *= ev

    return embedding


def _neighbor_index(
    tree: KDTree,
    point: np.ndarray,
    k: int,
) -> int:
    """Get the actual index of the k-th neighbor from a KDTree query result."""
    _, indices = tree.query(point.reshape(1, -1), k=k + 2)
    return int(indices[0, k + 1])


# ---------------------------------------------------------------------------
# Spectral embedding (Laplacian eigenmaps)
# ---------------------------------------------------------------------------


def spectral_embedding(
    data: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 10,
) -> np.ndarray:
    """Laplacian eigenmaps for nonlinear dimensionality reduction.

    Parameters
    ----------
    data : np.ndarray
        ``(n_samples, n_features)``.
    n_components : int
        Embedding dimension.
    n_neighbors : int
        Neighborhood size.

    Returns
    -------
    np.ndarray
        Embedding ``(n_samples, n_components)``.
    """
    data = np.asarray(data, dtype=np.float64)
    n = data.shape[0]
    n_components = min(n_components, n - 2)
    n_neighbors = min(n_neighbors, n - 1)

    tree = KDTree(data)
    distances, indices = tree.query(data, k=n_neighbors + 1)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    # Build adjacency with Gaussian kernel
    rows, cols, vals = [], [], []
    sigma = np.median(distances)
    if sigma == 0:
        sigma = 1.0

    for i in range(n):
        for j_off in range(n_neighbors):
            j = indices[i, j_off]
            w = np.exp(-distances[i, j_off] ** 2 / (2 * sigma**2))
            if w > 1e-12:
                rows.extend([i, j])
                cols.extend([j, i])
                vals.extend([w, w])

    W = csr_matrix((vals, (rows, cols)), shape=(n, n))
    D_inv_sqrt = diags(
        1.0 / np.sqrt(np.maximum(np.array(W.sum(axis=1)).ravel(), 1e-300))
    )
    L_norm = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt

    # Symmetrize for numerical safety
    L_dense = 0.5 * (L_norm.toarray() + L_norm.toarray().T)
    L_sparse = csr_matrix(L_dense)

    try:
        eigenvalues, eigenvectors = eigsh(L_sparse, k=n_components + 1, which="SM")
    except Exception:
        # Fallback: dense eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(L_dense)

    idx = np.argsort(eigenvalues)
    # Skip the first trivial eigenvalue (≈0)
    idx = idx[1 : n_components + 1]
    return eigenvectors[:, idx]


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------


def extract_local_patches(
    data: np.ndarray,
    patch_size: int = 100,
    overlap: float = 0.5,
) -> list:
    """Extract overlapping local patches using a sliding window.

    Parameters
    ----------
    data : np.ndarray
        ``(n_samples, n_features)``.
    patch_size : int
        Number of consecutive samples per patch.
    overlap : float
        Fractional overlap between patches in ``[0, 1)``.

    Returns
    -------
    list[np.ndarray]
        List of ``(patch_size, n_features)`` arrays.
    """
    data = np.asarray(data, dtype=np.float64)
    n = data.shape[0]
    step = max(1, int(patch_size * (1 - overlap)))
    patches: list = []
    start = 0
    while start + patch_size <= n:
        patches.append(data[start : start + patch_size])
        start += step
    return patches


# ---------------------------------------------------------------------------
# Residual variance
# ---------------------------------------------------------------------------


def compute_residual_variance(
    data: np.ndarray,
    intrinsic_dim: int,
    n_neighbors: int = 20,
) -> float:
    """Measure how well local PCA reconstructs data.

    Computes the average fraction of variance *not* captured by the top
    *intrinsic_dim* principal components in local neighborhoods.

    Parameters
    ----------
    data : np.ndarray
        ``(n_samples, n_features)``.
    intrinsic_dim : int
        Number of dimensions to keep.
    n_neighbors : int
        Neighborhood size.

    Returns
    -------
    float
        Mean residual variance in ``[0, 1]``.
    """
    data = np.asarray(data, dtype=np.float64)
    n, d = data.shape
    n_neighbors = min(n_neighbors, n - 1)
    intrinsic_dim = min(intrinsic_dim, d)

    tree = KDTree(data)
    _, indices = tree.query(data, k=n_neighbors + 1)
    indices = indices[:, 1:]

    residuals: list = []
    for i in range(n):
        neighbors = data[indices[i]]
        mean = neighbors.mean(axis=0)
        centered = neighbors - mean
        evals, evecs = np.linalg.eigh(centered.T @ centered)
        idx = np.argsort(evals)[::-1]
        total_var = evals[idx].sum()
        if total_var < 1e-300:
            residuals.append(0.0)
        else:
            captured = evals[idx[:intrinsic_dim]].sum()
            residuals.append(1.0 - captured / total_var)

    return float(np.mean(residuals))


# ---------------------------------------------------------------------------
# Manifold quality score
# ---------------------------------------------------------------------------


def manifold_quality_score(
    data: np.ndarray,
    atlas: dict[str, Any],
) -> dict[str, float]:
    """Comprehensive quality metric for a manifold atlas.

    Evaluates coverage, reconstruction error, dimension consistency, and
    overlap quality.

    Parameters
    ----------
    data : np.ndarray
        ``(n_samples, n_features)``.
    atlas : dict
        Atlas metadata containing ``'charts'`` key with a list of chart dicts
        that may contain ``'center'``, ``'indices'``, ``'dimension'``, etc.

    Returns
    -------
    dict[str, float]
        Keys: ``coverage``, ``reconstruction_error``, ``dim_consistency``,
        ``overlap_quality``, ``overall``.
    """
    data = np.asarray(data, dtype=np.float64)
    n = data.shape[0]
    charts = atlas.get("charts", [])

    if not charts:
        return {
            "coverage": 0.0,
            "reconstruction_error": 1.0,
            "dim_consistency": 0.0,
            "overlap_quality": 0.0,
            "overall": 0.0,
        }

    # Coverage: fraction of points assigned to at least one chart
    covered = set()
    chart_indices_list: list = []
    for chart in charts:
        idx = chart.get("indices", [])
        covered.update(idx)
        chart_indices_list.append(set(idx))
    coverage = len(covered) / n if n > 0 else 0.0

    # Dimension consistency: std of chart dimensions (lower = better)
    dims = [c.get("dimension", 0) for c in charts if c.get("dimension")]
    dim_consistency = 1.0 - (np.std(dims) / (np.mean(dims) + 1e-300)) if dims else 0.0
    dim_consistency = max(0.0, min(1.0, dim_consistency))

    # Overlap quality: average Jaccard index between chart pairs (sampled)
    overlap_scores: list = []
    nc = len(chart_indices_list)
    max_pairs = min(nc * (nc - 1) // 2, 200)
    pair_count = 0
    for i in range(nc):
        for j in range(i + 1, nc):
            if pair_count >= max_pairs:
                break
            s_i, s_j = chart_indices_list[i], chart_indices_list[j]
            intersection = len(s_i & s_j)
            union = len(s_i | s_j)
            if union > 0:
                overlap_scores.append(intersection / union)
            pair_count += 1
    overlap_quality = float(np.mean(overlap_scores)) if overlap_scores else 0.0

    # Reconstruction error: residual variance of local PCA at atlas charts
    dims_list = [c.get("dimension", 2) for c in charts if c.get("dimension")]
    avg_dim = int(np.mean(dims_list)) if dims_list else 2
    reconstruction_error = compute_residual_variance(data, avg_dim)

    overall = (
        0.25 * coverage
        + 0.25 * (1.0 - reconstruction_error)
        + 0.2 * dim_consistency
        + 0.3 * overlap_quality
    )

    return {
        "coverage": float(coverage),
        "reconstruction_error": float(reconstruction_error),
        "dim_consistency": float(dim_consistency),
        "overlap_quality": float(overlap_quality),
        "overall": float(overall),
    }
