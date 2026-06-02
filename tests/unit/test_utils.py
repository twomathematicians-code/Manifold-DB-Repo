"""
Unit tests for manifold_db.utils — math utilities, manifold learning, config.
"""

import numpy as np
import pytest

from manifold_db.utils import (
    default_config,
    estimate_intrinsic_dim,
    load_config,
    nearest_spd,
    save_config,
    validate_config,
)
from manifold_db.utils.math_utils import (
    cholesky_decomposition,
    ensure_spd,
    log_det_spd,
    matrix_square_root,
    orthogonalize,
    random_spd_matrix,
    safe_inverse,
)


class TestMathUtils:
    def test_ensure_spd_already_spd(self):
        M = np.eye(3)
        result = ensure_spd(M)
        np.testing.assert_allclose(result, M)

    def test_ensure_spd_near_singular(self):
        M = np.array([[1.0, 0.99], [0.99, 1.0]])
        result = ensure_spd(M)
        # Should still be close to original but positive definite
        eigvals = np.linalg.eigvalsh(result)
        assert np.all(eigvals > 0)

    def test_nearest_spd(self):
        A = np.random.randn(5, 5)
        B = nearest_spd(A)
        eigvals = np.linalg.eigvalsh(B)
        assert np.all(eigvals > -1e-10)
        np.testing.assert_allclose(B, B.T, atol=1e-10)  # symmetric

    def test_cholesky_decomposition(self):
        M = random_spd_matrix(5)
        L = cholesky_decomposition(M)
        assert L.shape == (5, 5)
        reconstructed = L @ L.T
        np.testing.assert_allclose(reconstructed, M, atol=1e-8)

    def test_matrix_square_root(self):
        M = random_spd_matrix(4)
        Msqrt = matrix_square_root(M)
        np.testing.assert_allclose(Msqrt @ Msqrt, M, atol=1e-8)

    def test_log_det_spd(self):
        M = random_spd_matrix(4)
        ld = log_det_spd(M)
        expected = np.linalg.slogdet(M)[1]
        assert abs(ld - expected) < 1e-8

    def test_safe_inverse(self):
        M = np.eye(4) * 0.01  # well-conditioned
        M_inv = safe_inverse(M)
        np.testing.assert_allclose(M @ M_inv, np.eye(4), atol=1e-8)

    def test_orthogonalize(self):
        v = np.random.randn(5, 5)
        Q = orthogonalize(v)
        # Q^T Q should be close to identity
        np.testing.assert_allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-8)

    def test_random_spd_matrix(self):
        M = random_spd_matrix(5, condition_number=10.0)
        eigvals = np.linalg.eigvalsh(M)
        assert np.all(eigvals > 0)
        cond = eigvals[-1] / eigvals[0]
        assert 5.0 < cond < 20.0  # rough check


class TestManifoldLearning:
    def test_estimate_intrinsic_dim_pca(self):
        # Generate data that lies on a 3-D subspace of 10-D
        basis = np.random.randn(10, 3)
        latent = np.random.randn(500, 3)
        data = latent @ basis.T
        dim = estimate_intrinsic_dim(data, method="pca", threshold=0.95)
        assert dim <= 5  # Should detect ~3

    def test_estimate_intrinsic_dim_mle(self):
        data = np.random.randn(500, 10)
        dim = estimate_intrinsic_dim(data, method="mle")
        assert 1 <= dim <= 10


class TestConfig:
    def test_default_config(self):
        cfg = default_config()
        assert cfg.atlas.max_charts > 0
        assert cfg.query.default_k == 10

    def test_validate_config(self):
        cfg = default_config()
        validate_config(cfg)  # Should not raise

    def test_save_load_roundtrip(self, tmp_path):
        cfg = default_config()
        path = str(tmp_path / "config.yaml")
        save_config(cfg, path)
        loaded = load_config(path)
        assert loaded.atlas.max_charts == cfg.atlas.max_charts
        assert loaded.server.port == cfg.server.port
