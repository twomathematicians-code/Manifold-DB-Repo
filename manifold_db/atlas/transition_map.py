"""
Transition maps - smooth invertible functions between overlapping charts.

Implements diffeomorphisms that map coordinates between chart overlaps.
Three concrete strategies are provided:

* **LinearTransition** – a simple linear (matrix + bias) map.
* **AffineTransition** – extends linear with full affine parameters.
* **NeuralTransition** – a small invertible MLP (coupling-layer style) built
  in PyTorch.

All transition maps expose a uniform interface: ``forward``, ``inverse``,
and ``jacobian``.
"""

from __future__ import annotations

import abc
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np

logger = logging.getLogger(__name__)


# ======================================================================
# Base
# ======================================================================
class TransitionMap(abc.ABC):
    """Abstract base class for coordinate transition maps.

    A transition map ψ: V ⊂ R^d → R^d is a diffeomorphism between the
    overlap region of two charts.  Subclasses must implement :meth:`forward`,
    :meth:`inverse`, and :meth:`jacobian`.
    """

    def __init__(
        self,
        source_chart_id: str,
        target_chart_id: str,
        dim: int,
        overlap_region: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        transition_id: Optional[str] = None,
    ) -> None:
        self.source_chart_id = source_chart_id
        self.target_chart_id = target_chart_id
        self.dim = dim
        self.transition_id = transition_id or str(uuid.uuid4())
        self._overlap_region = overlap_region
        logger.info(
            "TransitionMap %s: %s -> %s (dim=%d)",
            self.transition_id, source_chart_id, target_chart_id, dim,
        )

    @property
    def overlap_region(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """ ``(min_coords, max_coords)`` defining the overlap domain in source coords. """
        return self._overlap_region

    @overlap_region.setter
    def overlap_region(self, value: Optional[Tuple[np.ndarray, np.ndarray]]) -> None:
        self._overlap_region = value

    # ---- abstract interface --------------------------------------------------
    @abc.abstractmethod
    def forward(self, coords: np.ndarray) -> np.ndarray:
        """Map coordinates from the *source* chart to the *target* chart.

        Parameters
        ----------
        coords : ndarray of shape (N, dim) or (dim,)
            Coordinates in the source chart.

        Returns
        -------
        ndarray of shape (N, dim)
            Corresponding coordinates in the target chart.

        Raises
        ------
        ValueError
            If any coordinate falls outside the overlap region (when bounds
            are set).
        """

    @abc.abstractmethod
    def inverse(self, coords: np.ndarray) -> np.ndarray:
        """Map coordinates from the *target* chart back to the *source* chart."""

    @abc.abstractmethod
    def jacobian(self, coords: np.ndarray) -> np.ndarray:
        """Compute the Jacobian matrix at each point.

        Parameters
        ----------
        coords : ndarray of shape (N, dim)

        Returns
        -------
        ndarray of shape (N, dim, dim)
            Jacobian ``∂output_i / ∂input_j`` at each point.
        """

    # ---- overlap validation --------------------------------------------------
    def _check_overlap(self, coords: np.ndarray) -> None:
        """Raise ``ValueError`` if any row is outside the overlap region."""
        if self._overlap_region is None:
            return
        mn, mx = self._overlap_region
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)
        outside = np.any((coords < mn) | (coords > mx), axis=1)
        if np.any(outside):
            idx = int(np.argmax(outside))
            logger.warning(
                "TransitionMap %s: %d/%d points outside overlap region "
                "(first violator idx=%d).",
                self.transition_id, int(outside.sum()), len(coords), idx,
            )
            raise ValueError(
                f"Coordinate at index {idx} is outside the overlap region. "
                f"Overlap bounds: min={mn.tolist()}, max={mx.tolist()}. "
                f"Violator: {coords[idx].tolist()}"
            )

    # ---- serialization -------------------------------------------------------
    @abc.abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""

    @classmethod
    @abc.abstractmethod
    def from_dict(cls: Type["TransitionMap"], d: Dict[str, Any]) -> "TransitionMap":
        """Deserialize from a dictionary."""

    # ---- convenience ---------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"id={self.transition_id!r}, "
            f"{self.source_chart_id!r} -> {self.target_chart_id!r}, "
            f"dim={self.dim})"
        )


# ======================================================================
# Linear Transition
# ======================================================================
class LinearTransition(TransitionMap):
    """Linear transition: ``y = M @ x``.

    Parameters
    ----------
    matrix : ndarray of shape (dim, dim)
        Square linear transformation matrix.
    """

    def __init__(
        self,
        source_chart_id: str,
        target_chart_id: str,
        dim: int,
        matrix: Optional[np.ndarray] = None,
        overlap_region: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        transition_id: Optional[str] = None,
    ) -> None:
        super().__init__(source_chart_id, target_chart_id, dim, overlap_region, transition_id)
        if matrix is None:
            self.matrix = np.eye(dim, dtype=np.float64)
        else:
            self.matrix = np.asarray(matrix, dtype=np.float64)
            if self.matrix.shape != (dim, dim):
                raise ValueError(
                    f"Matrix must be ({dim}, {dim}), got {self.matrix.shape}"
                )

    def forward(self, coords: np.ndarray) -> np.ndarray:
        self._check_overlap(coords)
        c = np.asarray(coords, dtype=np.float64)
        if c.ndim == 1:
            c = c.reshape(1, -1)
        result = (self.matrix @ c.T).T
        if coords.ndim == 1:
            return result[0]
        return result

    def inverse(self, coords: np.ndarray) -> np.ndarray:
        c = np.asarray(coords, dtype=np.float64)
        if c.ndim == 1:
            c = c.reshape(1, -1)
        inv_mat = np.linalg.inv(self.matrix)
        result = (inv_mat @ c.T).T
        if coords.ndim == 1:
            return result[0]
        return result

    def jacobian(self, coords: np.ndarray) -> np.ndarray:
        c = np.asarray(coords, dtype=np.float64)
        if c.ndim == 1:
            c = c.reshape(1, -1)
        N = c.shape[0]
        # Constant Jacobian for linear map
        return np.broadcast_to(self.matrix[np.newaxis, :, :], (N, self.dim, self.dim)).copy()

    def to_dict(self) -> Dict[str, Any]:
        ov = self._overlap_region
        return {
            "type": "LinearTransition",
            "transition_id": self.transition_id,
            "source_chart_id": self.source_chart_id,
            "target_chart_id": self.target_chart_id,
            "dim": self.dim,
            "matrix": self.matrix.tolist(),
            "overlap_region": [ov[0].tolist(), ov[1].tolist()] if ov is not None else None,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LinearTransition":
        ov_raw = d.get("overlap_region")
        ov = (
            (np.asarray(ov_raw[0]), np.asarray(ov_raw[1]))
            if ov_raw is not None
            else None
        )
        return cls(
            source_chart_id=d["source_chart_id"],
            target_chart_id=d["target_chart_id"],
            dim=d["dim"],
            matrix=np.asarray(d["matrix"]),
            overlap_region=ov,
            transition_id=d.get("transition_id"),
        )


# ======================================================================
# Affine Transition
# ======================================================================
class AffineTransition(TransitionMap):
    """Affine transition: ``y = M @ x + b``.

    Parameters
    ----------
    matrix : ndarray of shape (dim, dim)
    bias : ndarray of shape (dim,)
    """

    def __init__(
        self,
        source_chart_id: str,
        target_chart_id: str,
        dim: int,
        matrix: Optional[np.ndarray] = None,
        bias: Optional[np.ndarray] = None,
        overlap_region: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        transition_id: Optional[str] = None,
    ) -> None:
        super().__init__(source_chart_id, target_chart_id, dim, overlap_region, transition_id)
        self.matrix = np.eye(dim, dtype=np.float64) if matrix is None else np.asarray(matrix, dtype=np.float64)
        self.bias = np.zeros(dim, dtype=np.float64) if bias is None else np.asarray(bias, dtype=np.float64)
        if self.matrix.shape != (dim, dim):
            raise ValueError(f"Matrix must be ({dim}, {dim}), got {self.matrix.shape}")
        if self.bias.shape != (dim,):
            raise ValueError(f"Bias must be ({dim},), got {self.bias.shape}")

    def forward(self, coords: np.ndarray) -> np.ndarray:
        self._check_overlap(coords)
        c = np.asarray(coords, dtype=np.float64)
        if c.ndim == 1:
            c = c.reshape(1, -1)
        result = (self.matrix @ c.T).T + self.bias
        if coords.ndim == 1:
            return result[0]
        return result

    def inverse(self, coords: np.ndarray) -> np.ndarray:
        c = np.asarray(coords, dtype=np.float64)
        if c.ndim == 1:
            c = c.reshape(1, -1)
        inv_mat = np.linalg.inv(self.matrix)
        result = (inv_mat @ (c - self.bias).T).T
        if coords.ndim == 1:
            return result[0]
        return result

    def jacobian(self, coords: np.ndarray) -> np.ndarray:
        c = np.asarray(coords, dtype=np.float64)
        if c.ndim == 1:
            c = c.reshape(1, -1)
        N = c.shape[0]
        return np.broadcast_to(self.matrix[np.newaxis, :, :], (N, self.dim, self.dim)).copy()

    def to_dict(self) -> Dict[str, Any]:
        ov = self._overlap_region
        return {
            "type": "AffineTransition",
            "transition_id": self.transition_id,
            "source_chart_id": self.source_chart_id,
            "target_chart_id": self.target_chart_id,
            "dim": self.dim,
            "matrix": self.matrix.tolist(),
            "bias": self.bias.tolist(),
            "overlap_region": [ov[0].tolist(), ov[1].tolist()] if ov is not None else None,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AffineTransition":
        ov_raw = d.get("overlap_region")
        ov = (
            (np.asarray(ov_raw[0]), np.asarray(ov_raw[1]))
            if ov_raw is not None
            else None
        )
        return cls(
            source_chart_id=d["source_chart_id"],
            target_chart_id=d["target_chart_id"],
            dim=d["dim"],
            matrix=np.asarray(d["matrix"]),
            bias=np.asarray(d["bias"]),
            overlap_region=ov,
            transition_id=d.get("transition_id"),
        )


# ======================================================================
# Neural Transition (invertible MLP via coupling layers)
# ======================================================================
class _InvertibleMLP(torch.nn.Module):
    """
    Small invertible network using affine coupling layers.

    Splits input in half.  One half passes through identity; the other is
    transformed conditioned on the first half via an affine transformation
    (scale & shift predicted by tiny MLPs).

    Architecture
    -------------
    For input dim *d*:
    * ``scale_net``: Linear(d//2, hidden) → ReLU → Linear(hidden, d//2)
    * ``shift_net``: Linear(d//2, hidden) → ReLU → Linear(hidden, d//2)
    * 4 coupling layers total, alternating halves.

    This guarantees invertibility by construction.
    """

    def __init__(self, dim: int, hidden: int = 64, n_layers: int = 4) -> None:
        super().__init__()
        half = max(dim // 2, 1)
        self.dim = dim
        self.half = half
        self.n_layers = n_layers

        self.scale_nets = torch.nn.ModuleList()
        self.shift_nets = torch.nn.ModuleList()
        for _ in range(n_layers):
            self.scale_nets.append(torch.nn.Sequential(
                torch.nn.Linear(half, hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden, half),
            ))
            self.shift_nets.append(torch.nn.Sequential(
                torch.nn.Linear(half, hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden, half),
            ))

    def _coupling_forward(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Single coupling layer forward pass."""
        x1, x2 = x[:, : self.half], x[:, self.half :]
        if layer_idx % 2 == 1:
            x1, x2 = x2, x1
        s = self.scale_nets[layer_idx](x1)
        t = self.shift_nets[layer_idx](x1)
        y2 = x2 * torch.exp(s) + t
        if layer_idx % 2 == 1:
            return torch.cat([y2, x1], dim=1)
        return torch.cat([x1, y2], dim=1)

    def _coupling_inverse(self, y: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Single coupling layer inverse pass."""
        y1, y2 = y[:, : self.half], y[:, self.half :]
        if layer_idx % 2 == 1:
            y1, y2 = y2, y1
        s = self.scale_nets[layer_idx](y1)
        t = self.shift_nets[layer_idx](y1)
        x2 = (y2 - t) * torch.exp(-s)
        if layer_idx % 2 == 1:
            return torch.cat([x2, y1], dim=1)
        return torch.cat([y1, x2], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for i in range(self.n_layers):
            out = self._coupling_forward(out, i)
        return out

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        out = y
        for i in reversed(range(self.n_layers)):
            out = self._coupling_inverse(out, i)
        return out


class NeuralTransition(TransitionMap):
    """Neural-network transition map using an invertible coupling-layer MLP.

    The network is trained via MSE loss on paired data (source coords,
    target coords).  After training the forward pass maps source → target
    and the inverse maps target → source.

    Parameters
    ----------
    dim : int
        Coordinate dimensionality.
    hidden : int
        Width of internal MLP layers (default 64).
    n_coupling_layers : int
        Number of affine coupling layers (default 4).
    """

    def __init__(
        self,
        source_chart_id: str,
        target_chart_id: str,
        dim: int,
        hidden: int = 64,
        n_coupling_layers: int = 4,
        overlap_region: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        transition_id: Optional[str] = None,
    ) -> None:
        super().__init__(source_chart_id, target_chart_id, dim, overlap_region, transition_id)
        self.hidden = hidden
        self.n_coupling_layers = n_coupling_layers
        self._model = _InvertibleMLP(dim, hidden=hidden, n_layers=n_coupling_layers)
        self._trained = False
        logger.info(
            "NeuralTransition %s created: dim=%d hidden=%d layers=%d",
            self.transition_id, dim, hidden, n_coupling_layers,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(
        self,
        source_coords: np.ndarray,
        target_coords: np.ndarray,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 256,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Train the invertible MLP on paired coordinate data.

        Parameters
        ----------
        source_coords : ndarray of shape (N, dim)
            Coordinates in the source chart.
        target_coords : ndarray of shape (N, dim)
            Corresponding coordinates in the target chart.
        epochs : int
            Training epochs.
        lr : float
            Learning rate for Adam.
        batch_size : int
            Mini-batch size.
        verbose : bool
            If True, log per-epoch loss.

        Returns
        -------
        dict
            ``{"final_loss": float, "epochs": int}``
        """
        src = torch.as_tensor(source_coords, dtype=torch.float32)
        tgt = torch.as_tensor(target_coords, dtype=torch.float32)
        N = src.shape[0]
        if N < 2:
            raise ValueError("Need at least 2 data points to train.")
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        history: List[float] = []
        for epoch in range(epochs):
            perm = torch.randperm(N)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, N, batch_size):
                idx = perm[start : start + batch_size]
                batch_src = src[idx]
                batch_tgt = tgt[idx]
                optimizer.zero_grad()
                pred = self._model(batch_src)
                loss = torch.nn.functional.mse_loss(pred, batch_tgt)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            avg_loss = epoch_loss / max(n_batches, 1)
            history.append(avg_loss)
            if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
                logger.info(
                    "NeuralTransition %s epoch %d/%d  loss=%.6f",
                    self.transition_id, epoch, epochs, avg_loss,
                )
        self._trained = True
        self._model.eval()
        result = {"final_loss": history[-1], "epochs": epochs}
        logger.info("NeuralTransition %s training complete: %s", self.transition_id, result)
        return result

    # ------------------------------------------------------------------
    # Forward / Inverse / Jacobian
    # ------------------------------------------------------------------
    def forward(self, coords: np.ndarray) -> np.ndarray:
        if not self._trained:
            logger.warning(
                "NeuralTransition %s used before training; results may be meaningless.",
                self.transition_id,
            )
        self._check_overlap(coords)
        c = np.asarray(coords, dtype=np.float64)
        squeeze = c.ndim == 1
        if squeeze:
            c = c.reshape(1, -1)
        with torch.no_grad():
            x = torch.as_tensor(c, dtype=torch.float32)
            out = self._model(x).numpy().astype(np.float64)
        if coords.ndim == 1:
            return out[0]
        return out

    def inverse(self, coords: np.ndarray) -> np.ndarray:
        if not self._trained:
            logger.warning(
                "NeuralTransition %s inverse used before training.",
                self.transition_id,
            )
        c = np.asarray(coords, dtype=np.float64)
        squeeze = c.ndim == 1
        if squeeze:
            c = c.reshape(1, -1)
        with torch.no_grad():
            y = torch.as_tensor(c, dtype=torch.float32)
            out = self._model.inverse(y).numpy().astype(np.float64)
        if coords.ndim == 1:
            return out[0]
        return out

    def jacobian(self, coords: np.ndarray) -> np.ndarray:
        """Compute Jacobian via finite differences (requires _trained)."""
        if not self._trained:
            logger.warning(
                "NeuralTransition %s jacobian before training.",
                self.transition_id,
            )
        c = np.asarray(coords, dtype=np.float64)
        if c.ndim == 1:
            c = c.reshape(1, -1)
        N, d = c.shape
        J = np.zeros((N, d, d), dtype=np.float64)
        eps = 1e-5
        for i in range(d):
            c_plus = c.copy()
            c_minus = c.copy()
            c_plus[:, i] += eps
            c_minus[:, i] -= eps
            with torch.no_grad():
                fp = self._model(
                    torch.as_tensor(c_plus, dtype=torch.float32)
                ).numpy()
                fm = self._model(
                    torch.as_tensor(c_minus, dtype=torch.float32)
                ).numpy()
            J[:, :, i] = (fp - fm) / (2.0 * eps)
        return J

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        ov = self._overlap_region
        state = {
            "model_state": {
                k: v.tolist() for k, v in self._model.state_dict().items()
            }
        }
        return {
            "type": "NeuralTransition",
            "transition_id": self.transition_id,
            "source_chart_id": self.source_chart_id,
            "target_chart_id": self.target_chart_id,
            "dim": self.dim,
            "hidden": self.hidden,
            "n_coupling_layers": self.n_coupling_layers,
            "trained": self._trained,
            "state": state,
            "overlap_region": [ov[0].tolist(), ov[1].tolist()] if ov is not None else None,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NeuralTransition":
        ov_raw = d.get("overlap_region")
        ov = (
            (np.asarray(ov_raw[0]), np.asarray(ov_raw[1]))
            if ov_raw is not None
            else None
        )
        obj = cls(
            source_chart_id=d["source_chart_id"],
            target_chart_id=d["target_chart_id"],
            dim=d["dim"],
            hidden=d.get("hidden", 64),
            n_coupling_layers=d.get("n_coupling_layers", 4),
            overlap_region=ov,
            transition_id=d.get("transition_id"),
        )
        # Restore weights
        state_raw = d.get("state", {}).get("model_state", {})
        if state_raw:
            state = {
                k: torch.as_tensor(v, dtype=torch.float32)
                for k, v in state_raw.items()
            }
            obj._model.load_state_dict(state)
            obj._trained = d.get("trained", False)
        obj._model.eval()
        return obj


# ======================================================================
# Factory / registry
# ======================================================================
_TRANSITION_REGISTRY: Dict[str, Type[TransitionMap]] = {
    "LinearTransition": LinearTransition,
    "AffineTransition": AffineTransition,
    "NeuralTransition": NeuralTransition,
}


def create_transition_map(d: Dict[str, Any]) -> TransitionMap:
    """Instantiate a :class:`TransitionMap` from a serialised dict.

    Looks up ``d["type"]`` in the registry.

    Raises
    ------
    ValueError
        If the type string is not recognised.
    """
    t = d.get("type")
    if t not in _TRANSITION_REGISTRY:
        raise ValueError(
            f"Unknown transition type '{t}'. Available: {list(_TRANSITION_REGISTRY)}"
        )
    return _TRANSITION_REGISTRY[t].from_dict(d)
