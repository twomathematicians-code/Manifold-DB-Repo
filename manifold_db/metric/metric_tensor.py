"""
Riemannian Metric Tensor — defines distances and angles on the manifold.

The metric tensor g_ij(x) is a symmetric positive-definite matrix field
that varies from point to point on the manifold.  This module provides:
  - Abstract ``MetricTensor`` base class with curvature primitives
  - ``EuclideanMetric``   (flat fallback)
  - ``DiagonalMetric``    (simplest non-trivial case)
  - ``LearnedMetric``     (torch MLP parameterised via Cholesky factors)
  - ``FisherRaoMetric``   (information-geometric metric on distributions)
  - ``WassersteinMetric`` (optimal-transport induced metric)
  - ``MetricTensorStore``  (per-chart registry with Ricci-flow smoothing)
"""

from __future__ import annotations

import abc
import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.linalg import sqrtm

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:
    import types

    # Create stub module so that class definitions don't fail at import time.
    # The _require_torch() guard inside __init__ prevents runtime use without torch.
    torch = types.ModuleType("torch")  # type: ignore[assignment]
    nn = types.ModuleType("torch.nn")  # type: ignore[assignment]
    nn.Module = object  # type: ignore[attr-defined]
    nn.Linear = object  # type: ignore[attr-defined]
    nn.Softplus = object  # type: ignore[attr-defined]
    nn.Sequential = object  # type: ignore[attr-defined]
    _HAS_TORCH = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Abstract base
# ---------------------------------------------------------------------------


class MetricTensor(abc.ABC):
    """Abstract base class for a Riemannian metric tensor field.

    Every subclass must implement :meth:`evaluate` which returns the
    symmetric positive-definite matrix ``g_ij`` at a given point.
    """

    @abc.abstractmethod
    def evaluate(self, point: np.ndarray) -> np.ndarray:
        """Return g_ij(x) as an (n, n) numpy array."""

    # ---- derived quantities (default implementations via finite diff.) ----

    def inverse(self, point: np.ndarray) -> np.ndarray:
        """Return g^{ij}(x), the inverse metric tensor."""
        g = self.evaluate(point)
        return np.linalg.inv(g)

    def determinant(self, point: np.ndarray) -> float:
        """Return det(g_ij(x))."""
        return float(np.linalg.det(self.evaluate(point)))

    def log_det(self, point: np.ndarray) -> float:
        """Return log det(g_ij(x)), numerically stable for volume forms."""
        return float(np.linalg.slogdet(self.evaluate(point))[1])

    def christoffel_symbols(self, point: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        r"""Compute Christoffel symbols of the first kind numerically.

        .. math::
            \Gamma^i_{jk} = \frac{1}{2} g^{il}
                \left( \partial_j g_{lk} + \partial_k g_{lj} - \partial_l g_{jk} \right)

        Returns an (n, n, n) array where ``result[i, j, k]`` = Γⁱ_jk.
        """
        g = self.evaluate(point)
        n = g.shape[0]
        g_inv = np.linalg.inv(g)
        Gamma = np.zeros((n, n, n), dtype=np.float64)

        for idx_l in range(n):
            for j in range(n):
                dp = np.zeros(n)
                dm = np.zeros(n)
                dp[j] = eps
                dm[j] = eps
                dg_lk = (
                    self.evaluate(point + dp)[idx_l, :]
                    - self.evaluate(point - dp)[idx_l, :]
                ) / (2.0 * eps)
                for k in range(n):
                    dp2 = np.zeros(n)
                    dm2 = np.zeros(n)
                    dp2[k] = eps
                    dm2[k] = eps
                    dg_lj = (
                        self.evaluate(point + dp2)[idx_l, j]
                        - self.evaluate(point - dp2)[idx_l, j]
                    ) / (2.0 * eps)
                    dl_gjk = (
                        self.evaluate(point + dp)[j, k]
                        - self.evaluate(point - dp)[j, k]
                    ) / (2.0 * eps)
                    Gamma[:, j, k] += g_inv[:, idx_l] * (dg_lk[k] + dg_lj - dl_gjk)

        Gamma *= 0.5
        return Gamma

    def sectional_curvature(
        self, point: np.ndarray, u: np.ndarray, v: np.ndarray
    ) -> float:
        r"""Sectional curvature K(u, v) at *point*.

        .. math::
            K(u,v) = \frac{R(u,v,v,u)}
            {\langle u,u \rangle \langle v,v \rangle - \langle u,v \rangle^2}

        Uses the full Riemann tensor approximated from the Ricci tensor
        (constant sectional curvature assumption for numerical tractability).
        """
        g = self.evaluate(point)
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        norm_sq_u = u @ g @ u
        norm_sq_v = v @ g @ v
        inner_uv = u @ g @ v
        denom = norm_sq_u * norm_sq_v - inner_uv**2
        if abs(denom) < 1e-14:
            return 0.0

        # Riem(u,v,v,u) ≈ Ricci approximation via scalar curvature / n(n-1)
        n = g.shape[0]
        R_scalar = self.scalar_curvature(point)
        R_const = R_scalar / (n * (n - 1)) if n > 1 else 0.0
        Riem = R_const * (norm_sq_u * norm_sq_v - inner_uv**2)
        return float(Riem / denom)

    def ricci_curvature(self, point: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        r"""Ricci curvature tensor R_ij(x).

        Computed from the Riemann tensor via contraction:
        R_ij = R^k_{ikj}.  We evaluate the Riemann tensor numerically.
        """
        n = self.evaluate(point).shape[0]
        Ric = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            for j in range(n):
                val = 0.0
                for k in range(n):
                    # R^k_{ikj} via finite-difference of Christoffel symbols
                    def christoffel_component(
                        p: np.ndarray, idx_k: int, idx_i: int, idx_j: int
                    ) -> float:
                        G = self.christoffel_symbols(p, eps=eps * 0.5)
                        return G[idx_k, idx_i, idx_j]

                    for idx_l in range(n):
                        dG = (
                            christoffel_component(
                                point + eps * _e(idx_l, n), k, i, j
                            )
                            - christoffel_component(
                                point - eps * _e(idx_l, n), k, i, j
                            )
                        ) / (2.0 * eps)
                        dG2 = (
                            christoffel_component(point + eps * _e(j, n), k, i, idx_l)
                            - christoffel_component(point - eps * _e(j, n), k, i, idx_l)
                        ) / (2.0 * eps)
                        dG3 = (
                            christoffel_component(point + eps * _e(i, n), k, idx_l, j)
                            - christoffel_component(point - eps * _e(i, n), k, idx_l, j)
                        ) / (2.0 * eps)
                        val += dG - dG2 + dG3
                Ric[i, j] = val
        return Ric

    def scalar_curvature(self, point: np.ndarray) -> float:
        r"""Scalar curvature R = g^{ij} R_ij."""
        g_inv = self.inverse(point)
        Ric = self.ricci_curvature(point)
        return float(np.einsum("ij,ij->", g_inv, Ric))

    # ---- serialisation -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.__class__.__name__}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricTensor:
        """Deserialise a metric from a dictionary."""
        kind = data.pop("type", None)
        registry: dict[str, type] = {
            "EuclideanMetric": EuclideanMetric,
            "DiagonalMetric": DiagonalMetric,
            "LearnedMetric": LearnedMetric,
            "FisherRaoMetric": FisherRaoMetric,
            "WassersteinMetric": WassersteinMetric,
        }
        if kind is None or kind not in registry:
            raise ValueError(f"Unknown metric type: {kind}")
        return registry[kind].from_dict(data)


def _e(i: int, n: int) -> np.ndarray:
    """Return the i-th standard basis vector in R^n."""
    v = np.zeros(n, dtype=np.float64)
    v[i] = 1.0
    return v


# ---------------------------------------------------------------------------
#  Euclidean metric
# ---------------------------------------------------------------------------


class EuclideanMetric(MetricTensor):
    """Flat Euclidean metric g_ij = δ_ij."""

    def __init__(self, dim: int = 3) -> None:
        self.dim = dim

    def evaluate(self, point: np.ndarray) -> np.ndarray:
        return np.eye(self.dim, dtype=np.float64)

    def inverse(self, point: np.ndarray) -> np.ndarray:
        return np.eye(self.dim, dtype=np.float64)

    def christoffel_symbols(self, point: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        return np.zeros((self.dim, self.dim, self.dim), dtype=np.float64)

    def ricci_curvature(self, point: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        return np.zeros((self.dim, self.dim), dtype=np.float64)

    def scalar_curvature(self, point: np.ndarray) -> float:
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"type": "EuclideanMetric", "dim": self.dim}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EuclideanMetric:
        return cls(dim=data.get("dim", 3))


# ---------------------------------------------------------------------------
#  Diagonal metric
# ---------------------------------------------------------------------------


class DiagonalMetric(MetricTensor):
    """Diagonal metric g_ij = diag(w_0, …, w_{n-1}) with optional spatial variation."""

    def __init__(
        self,
        weights: np.ndarray | None = None,
        dim: int = 3,
        variation_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        if weights is not None:
            self.weights = np.asarray(weights, dtype=np.float64)
        else:
            self.weights = np.ones(dim, dtype=np.float64)
        self.dim = len(self.weights)
        self.variation_fn = variation_fn  # point → weight modifier array

    def evaluate(self, point: np.ndarray) -> np.ndarray:
        w = self.weights.copy()
        if self.variation_fn is not None:
            w = w * np.asarray(self.variation_fn(point), dtype=np.float64)
        w = np.maximum(w, 1e-12)  # enforce positivity
        return np.diag(w)

    def inverse(self, point: np.ndarray) -> np.ndarray:
        w = 1.0 / np.maximum(self.weights, 1e-12)
        if self.variation_fn is not None:
            v = np.maximum(
                np.asarray(self.variation_fn(point), dtype=np.float64), 1e-12
            )
            w = w / v
        return np.diag(w)

    def determinant(self, point: np.ndarray) -> float:
        g = self.evaluate(point)
        return float(np.prod(np.diag(g)))

    def log_det(self, point: np.ndarray) -> float:
        return float(np.sum(np.log(np.maximum(np.diag(self.evaluate(point)), 1e-12))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "DiagonalMetric",
            "weights": self.weights.tolist(),
            "dim": self.dim,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiagonalMetric:
        weights = np.array(data["weights"]) if "weights" in data else None
        return cls(weights=weights, dim=data.get("dim", 3))


# ---------------------------------------------------------------------------
#  Learned (neural) metric
# ---------------------------------------------------------------------------


def _require_torch() -> None:
    """Raise ImportError if torch is not available."""
    if not _HAS_TORCH:
        raise ImportError(
            "torch is required for LearnedMetric but is not installed.  "
            "Install it with: pip install torch"
        )


class MetricMLP(nn.Module):
    """Small MLP that maps manifold coordinates to Cholesky factor L of g.

    g = L L^T  ensures positive-definiteness by construction.
    """

    def __init__(self, dim: int, hidden: int = 64, n_layers: int = 3) -> None:
        _require_torch()
        super().__init__()
        self.dim = dim
        self.n_components = dim * (dim + 1) // 2  # lower-triangular entries
        layers: list[nn.Module] = []
        in_feat = dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(in_feat, hidden))
            layers.append(nn.Softplus())
            in_feat = hidden
        layers.append(nn.Linear(in_feat, self.n_components))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, dim) → L_flat: (batch, n_components)."""
        return self.net(x)


class LearnedMetric(MetricTensor):
    """Neural-network metric tensor parameterised via Cholesky decomposition.

    g(x) = L(x) L(x)^T  where L(x) is the lower-triangular Cholesky factor
    output by a small MLP.  This guarantees positive-definiteness everywhere.
    """

    def __init__(
        self, dim: int, hidden: int = 64, n_layers: int = 3, lr: float = 1e-3
    ) -> None:
        _require_torch()
        self.dim = dim
        self.device = torch.device("cpu")
        self.mlp = MetricMLP(dim, hidden, n_layers).to(self.device)
        self.lr = lr
        self._optimizer = torch.optim.Adam(self.mlp.parameters(), lr=self.lr)

    def _point_to_tensor(self, point: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(
            point, dtype=torch.float64, device=self.device
        ).unsqueeze(0)

    def _flat_to_matrix(self, flat: torch.Tensor) -> torch.Tensor:
        """Convert flat lower-triangular vector to Cholesky matrix."""
        batch = flat.shape[0]
        L = torch.zeros(batch, self.dim, self.dim, dtype=flat.dtype, device=self.device)
        idx = 0
        for i in range(self.dim):
            for j in range(i + 1):
                L[:, i, j] = flat[:, idx]
                idx += 1
        # Exponentiate diagonal to ensure positivity
        diag_idx = torch.arange(self.dim, device=self.device)
        L[:, diag_idx, diag_idx] = torch.exp(L[:, diag_idx, diag_idx])
        return L

    def evaluate(self, point: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = self._point_to_tensor(point).float()
            flat = self.mlp(x)
            L = self._flat_to_matrix(flat)
            g = L @ L.transpose(-1, -2)
        return g.squeeze(0).double().numpy()

    def evaluate_batch(self, points: np.ndarray) -> np.ndarray:
        """Evaluate metric at multiple points.  points: (N, dim)."""
        with torch.no_grad():
            x = torch.as_tensor(points, dtype=torch.float32, device=self.device)
            flat = self.mlp(x)
            L = self._flat_to_matrix(flat)
            g = L @ L.transpose(-1, -2)
        return g.double().numpy()

    def train(
        self,
        samples: np.ndarray,
        tangent_pairs: np.ndarray | None = None,
        n_epochs: int = 100,
        batch_size: int = 32,
    ) -> dict[str, list[float]]:
        """Learn metric from data.

        Parameters
        ----------
        samples : (N, dim)
            Points on the manifold.
        tangent_pairs : (M, 2, dim) or None
            Pairs of nearby points used to learn local inner products.
        n_epochs : int
        batch_size : int

        Returns
        -------
        dict with 'loss' history.
        """
        x = torch.as_tensor(samples, dtype=torch.float32, device=self.device)
        history: list[float] = []
        n = len(samples)

        for epoch in range(n_epochs):
            perm = torch.randperm(n, device=self.device)
            epoch_loss = 0.0
            count = 0
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                xb = x[idx]
                flat = self.mlp(xb)
                L = self._flat_to_matrix(flat)
                g = L @ L.transpose(-1, -2)

                # Reconstruction loss: encourage g ≈ I near sample mean
                loss = torch.mean(
                    (g - torch.eye(self.dim, device=self.device).unsqueeze(0)) ** 2
                )

                # Tangent pair loss: inner products should be consistent
                if tangent_pairs is not None:
                    pairs = torch.as_tensor(
                        tangent_pairs, dtype=torch.float32, device=self.device
                    )
                    for k in range(min(len(pairs), batch_size)):
                        p1, p2 = pairs[k, 0], pairs[k, 1]
                        diff = (p1 - p2).unsqueeze(0)
                        flat1 = self.mlp(p1.unsqueeze(0))
                        L1 = self._flat_to_matrix(flat1)
                        g1 = (L1 @ L1.transpose(-1, -2)).squeeze(0)
                        inner = diff @ g1 @ diff.T
                        loss = loss + 0.1 * torch.relu(-inner + 1e-3)

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                epoch_loss += loss.item()
                count += 1

            avg_loss = epoch_loss / max(count, 1)
            history.append(avg_loss)
            if epoch % 20 == 0:
                logger.info(
                    "LearnedMetric epoch %d/%d  loss=%.6f", epoch, n_epochs, avg_loss
                )

        return {"loss": history}

    def to_dict(self) -> dict[str, Any]:
        state = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in {
                "dim": self.dim,
                "hidden": self.mlp.net[0].out_features if len(self.mlp.net) > 0 else 64,
                "n_layers": sum(1 for m in self.mlp.net if isinstance(m, nn.Linear)),
                "lr": self.lr,
            }.items()
        }
        state["state_dict"] = {
            k: v.cpu().numpy().tolist() for k, v in self.mlp.state_dict().items()
        }
        return {"type": "LearnedMetric", **state}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearnedMetric:
        metric = cls(
            dim=data["dim"],
            hidden=data.get("hidden", 64),
            n_layers=data.get("n_layers", 3),
            lr=data.get("lr", 1e-3),
        )
        state_dict = {}
        for k, v in data["state_dict"].items():
            state_dict[k] = torch.as_tensor(v)
        metric.mlp.load_state_dict(state_dict)
        return metric


# ---------------------------------------------------------------------------
#  Fisher-Rao metric
# ---------------------------------------------------------------------------


class FisherRaoMetric(MetricTensor):
    r"""Fisher-Rao metric on the space of probability distributions.

    For an exponential-family distribution with natural parameter θ,
    the Fisher information matrix I(θ) is the metric tensor:

    .. math:: g_{ij}(θ) = \mathbb{E}\left[
        \frac{\partial \log p(x;θ)}{\partial θ^i}
        \frac{\partial \log p(x;θ)}{\partial θ^j}
    \right]
    """

    def __init__(
        self,
        dim: int = 3,
        fisher_matrix: np.ndarray | None = None,
        samples_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        dim : int
            Dimension of the parameter space.
        fisher_matrix : (dim, dim) or None
            Constant Fisher information matrix.  If None, an identity matrix
            scaled by ``dim`` is used.
        samples_fn : callable or None
            Function point → (N, dim) gradient samples for numerical Fisher
            estimation.  If provided, overrides ``fisher_matrix``.
        """
        self.dim = dim
        if fisher_matrix is not None:
            self.fisher_matrix = np.asarray(fisher_matrix, dtype=np.float64)
        else:
            self.fisher_matrix = np.eye(dim, dtype=np.float64) * dim
        self.samples_fn = samples_fn

    def evaluate(self, point: np.ndarray) -> np.ndarray:
        if self.samples_fn is not None:
            return self._estimate_fisher(point)
        return self.fisher_matrix.copy()

    def _estimate_fisher(self, point: np.ndarray, n_samples: int = 500) -> np.ndarray:
        """Numerically estimate Fisher information from gradient samples."""
        grads = self.samples_fn(point)  # (N, dim)
        if len(grads) == 0:
            return np.eye(self.dim, dtype=np.float64)
        return (grads.T @ grads) / len(grads)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "FisherRaoMetric",
            "dim": self.dim,
            "fisher_matrix": self.fisher_matrix.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FisherRaoMetric:
        fm = np.array(data["fisher_matrix"]) if "fisher_matrix" in data else None
        return cls(dim=data.get("dim", 3), fisher_matrix=fm)


# ---------------------------------------------------------------------------
#  Wasserstein metric
# ---------------------------------------------------------------------------


class WassersteinMetric(MetricTensor):
    r"""Wasserstein (optimal transport) metric.

    On the space of probability measures with finite second moment,
    the tangent-space inner product at a measure μ is given by the
    Benamou–Brenier formula.  We approximate this using Sinkhorn
    divergences for numerical tractability.

    For discrete distributions with ``n_bins`` bins, the metric is
    an ``(n_bins, n_bins)`` matrix computed from the optimal transport
    plan.
    """

    def __init__(self, n_bins: int = 10, cost_matrix: np.ndarray | None = None) -> None:
        """
        Parameters
        ----------
        n_bins : int
            Number of histogram bins / atoms.
        cost_matrix : (n_bins, n_bins) or None
            Ground-metric cost matrix.  Defaults to |i - j|^2.
        """
        self.n_bins = n_bins
        if cost_matrix is not None:
            self.cost = np.asarray(cost_matrix, dtype=np.float64)
        else:
            idx = np.arange(n_bins, dtype=np.float64)
            self.cost = (idx[:, None] - idx[None, :]) ** 2

    def _sinkhorn(
        self,
        mu: np.ndarray,
        nu: np.ndarray,
        reg: float = 1e-2,
        max_iter: int = 100,
    ) -> tuple[np.ndarray, float]:
        """Sinkhorn algorithm for entropy-regularised optimal transport.

        Returns the transport plan T and Wasserstein distance W_2.
        """
        K = np.exp(-self.cost / reg)
        u = np.ones(self.n_bins) / self.n_bins
        v = np.ones(self.n_bins) / self.n_bins

        for _ in range(max_iter):
            u_prev = u.copy()
            u = mu / (K @ v + 1e-16)
            v = nu / (K.T @ u + 1e-16)
            if np.max(np.abs(u - u_prev)) < 1e-8:
                break

        T = np.diag(u) @ K @ np.diag(v)
        W2 = np.sum(T * self.cost)
        return T, W2

    def evaluate(self, point: np.ndarray) -> np.ndarray:
        """Evaluate the Wasserstein metric at a discrete distribution.

        ``point`` should be a probability vector of length ``n_bins``.
        The metric is the inverse of the transport Hessian, approximated
        here via the cost-weighted outer product of the optimal plan.
        """
        mu = np.abs(point[: self.n_bins])
        total = mu.sum()
        if total < 1e-12:
            mu = np.ones(self.n_bins) / self.n_bins
        else:
            mu = mu / total

        # Metric tensor from cost matrix: g = C^{-1/2} approx
        C = self.cost.copy()
        C += np.eye(self.n_bins) * 1e-6
        try:
            C_inv_sqrt = np.real(sqrtm(np.linalg.inv(C)))
            g = C_inv_sqrt @ np.diag(mu) @ C_inv_sqrt
            g = (g + g.T) / 2.0
            # Ensure positive-definiteness
            eigvals = np.linalg.eigvalsh(g)
            if eigvals.min() < 1e-10:
                g += np.eye(self.n_bins) * (1e-10 - eigvals.min())
        except np.linalg.LinAlgError:
            g = np.eye(self.n_bins, dtype=np.float64)
        return g

    def wasserstein_distance(
        self, mu: np.ndarray, nu: np.ndarray, reg: float = 1e-2
    ) -> float:
        """Compute W_2 distance between two discrete distributions."""
        mu_n = mu[: self.n_bins].copy()
        nu_n = nu[: self.n_bins].copy()
        mu_n /= mu_n.sum() + 1e-16
        nu_n /= nu_n.sum() + 1e-16
        _, W2 = self._sinkhorn(mu_n, nu_n, reg=reg)
        return float(np.sqrt(max(W2, 0.0)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "WassersteinMetric",
            "n_bins": self.n_bins,
            "cost_matrix": self.cost.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WassersteinMetric:
        cm = np.array(data["cost_matrix"]) if "cost_matrix" in data else None
        return cls(n_bins=data.get("n_bins", 10), cost_matrix=cm)


# ---------------------------------------------------------------------------
#  Metric Tensor Store
# ---------------------------------------------------------------------------


class MetricTensorStore:
    """Manages per-chart metric tensors with Ricci-flow smoothing.

    The store maintains a mapping from chart identifiers to metric tensors
    and provides methods for online updates, Ricci-flow-based smoothing,
    and covariant transformation across charts.
    """

    def __init__(self) -> None:
        self._metrics: dict[str, MetricTensor] = {}
        self._chart_data: dict[str, list[np.ndarray]] = {}  # accumulated points

    def register_metric(self, chart_id: str, metric: MetricTensor) -> None:
        """Register a metric tensor for a given chart."""
        if chart_id in self._metrics:
            logger.warning("Overwriting existing metric for chart %s", chart_id)
        self._metrics[chart_id] = metric
        self._chart_data.setdefault(chart_id, [])
        logger.info(
            "Registered metric %s for chart %s", type(metric).__name__, chart_id
        )

    def get_metric(self, chart_id: str) -> MetricTensor:
        """Retrieve the metric tensor for a chart; fallback to Euclidean."""
        if chart_id not in self._metrics:
            logger.warning(
                "No metric for chart %s, falling back to Euclidean", chart_id
            )
            return EuclideanMetric()
        return self._metrics[chart_id]

    def update_online(self, chart_id: str, new_data: np.ndarray) -> None:
        """Accumulate new point data and trigger a Ricci-flow update.

        Parameters
        ----------
        chart_id : str
            Chart to update.
        new_data : (N, dim) or (dim,)
            New data points observed in this chart.
        """
        pts = np.atleast_2d(new_data)
        self._chart_data.setdefault(chart_id, []).extend(pts.tolist())
        # Trigger a small Ricci-flow smoothing step
        self.ricci_flow_step(chart_id, dt=0.01)

    def ricci_flow_step(self, chart_id: str, dt: float = 0.01) -> None:
        r"""Perform one step of (normalised) Ricci flow.

        .. math::
            \frac{\partial g_{ij}}{\partial t} = -2 R_{ij} + \frac{2}{n} R \, g_{ij}

        This smooths the metric tensor toward constant curvature.
        """
        metric = self._metrics.get(chart_id)
        if metric is None or isinstance(metric, EuclideanMetric):
            logger.debug(
                "Skipping Ricci flow for chart %s (no non-trivial metric)", chart_id
            )
            return

        data = self._chart_data.get(chart_id, [])
        if not data:
            return

        # Compute average Ricci correction over stored points
        points = np.array(data[-min(len(data), 50) :])  # use last 50 points
        n = metric.evaluate(points[0]).shape[0]

        ricci_avg = np.zeros((n, n), dtype=np.float64)
        for pt in points:
            ricci_avg += metric.ricci_curvature(pt)
        ricci_avg /= len(points)
        R_scalar = np.trace(metric.inverse(points[0]) @ ricci_avg)

        # Build a DiagonalMetric that approximates the Ricci-flow correction
        current = metric.evaluate(points[-1])
        correction = -2.0 * ricci_avg * dt + (2.0 / n) * R_scalar * current * dt
        new_g = current + correction

        # Ensure symmetry and positive-definiteness
        new_g = (new_g + new_g.T) / 2.0
        try:
            eigvals, eigvecs = np.linalg.eigh(new_g)
            eigvals = np.maximum(eigvals, 1e-8)
            new_g = eigvecs @ np.diag(eigvals) @ eigvecs.T
        except np.linalg.LinAlgError:
            logger.warning(
                "Ricci flow step failed for chart %s, keeping old metric", chart_id
            )
            return

        # Replace with a DiagonalMetric using the eigenvalues
        self._metrics[chart_id] = DiagonalMetric(weights=np.diag(new_g))
        logger.debug("Ricci flow step applied to chart %s, dt=%.4f", chart_id, dt)

    def transform_metric(
        self,
        source_chart: str,
        target_chart: str,
        transition_map: Callable[[np.ndarray], np.ndarray],
        reference_point: np.ndarray | None = None,
    ) -> np.ndarray:
        r"""Covariantly transform a metric between charts.

        Given a transition map φ: source → target, the metric transforms as:

        .. math:: g'_{ij}(y) = \frac{\partial x^k}{\partial y^i}
            \frac{\partial x^l}{\partial y^j} g_{kl}(x)

        We approximate the Jacobian numerically.
        """
        metric = self.get_metric(source_chart)
        dim = metric.evaluate(
            np.zeros(max(metric.dim if hasattr(metric, "dim") else 3, 1))
        ).shape[0]

        if reference_point is None:
            reference_point = np.zeros(dim, dtype=np.float64)
        eps = 1e-5

        # Jacobian of transition_map: J^k_i = ∂φ^k/∂x^i
        J = np.zeros((dim, dim), dtype=np.float64)
        for i in range(dim):
            dp = np.zeros(dim)
            dm = np.zeros(dim)
            dp[i] = eps
            dm[i] = eps
            J[:, i] = (
                transition_map(reference_point + dp)
                - transition_map(reference_point - dm)
            ) / (2.0 * eps)

        # Inverse Jacobian for pullback: (∂x/∂y) = J^{-1}
        try:
            J_inv = np.linalg.inv(J)
        except np.linalg.LinAlgError:
            logger.warning(
                "Transition map Jacobian is singular between %s → %s",
                source_chart,
                target_chart,
            )
            return metric.evaluate(reference_point)

        g_source = metric.evaluate(reference_point)
        g_target = J_inv.T @ g_source @ J_inv

        # Symmetrise and ensure PD
        g_target = (g_target + g_target.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(g_target)
        eigvals = np.maximum(eigvals, 1e-8)
        g_target = eigvecs @ np.diag(eigvals) @ eigvecs.T

        logger.debug(
            "Transformed metric from chart %s to %s", source_chart, target_chart
        )
        return g_target

    def list_charts(self) -> list[str]:
        return list(self._metrics.keys())

    def serialize(self) -> dict[str, Any]:
        return {
            "metrics": {cid: m.to_dict() for cid, m in self._metrics.items()},
        }

    def deserialize(self, data: dict[str, Any]) -> None:
        for cid, md in data.get("metrics", {}).items():
            self._metrics[cid] = MetricTensor.from_dict(md)
            self._chart_data.setdefault(cid, [])
        logger.info("Deserialized %d chart metrics", len(self._metrics))
