"""
Chart module - local coordinate patches on the manifold.

Each chart represents a region of the manifold with its own coordinate system.
A chart (U, φ) consists of an open set U ⊂ M (the domain) and a homeomorphism
φ: U → R^d mapping the patch into d-dimensional Euclidean space.

Internally uses numpy arrays for all coordinate operations.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import InitVar, dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Chart:
    """
    A local coordinate patch on the manifold.

    Represents an open subset of the manifold together with a coordinate map
    that identifies points in that subset with points in R^d.  The *intrinsic*
    dimension ``dim`` is the dimension of the coordinate image; ``ambient_dim``
    is the dimension of the embedding space the data originally lives in.

    Parameters
    ----------
    name : str
        Human-readable identifier for the chart.
    dim : int
        Intrinsic dimension of the chart (coordinate dimensionality).
    ambient_dim : int
        Dimensionality of the ambient (embedding) space.
    embedding_fn : callable or None
        Function that projects ambient-space data into chart coordinates.
        Signature: ``(data: ndarray(N, ambient_dim)) -> ndarray(N, dim)``.
    inverse_fn : callable or None
        Function that lifts chart coordinates back to ambient space.
        Signature: ``(coords: ndarray(N, dim)) -> ndarray(N, ambient_dim)``.
    bounds : tuple of ndarray or None
        Optional ``(min_coords, max_coords)`` each of shape ``(dim,)`` that
        defines the axis-aligned bounding box of the chart domain.
    anchor_points : ndarray or None
        Representative points in chart coordinates used for tangent-space
        computations.  Shape ``(n_anchors, dim)``.
    chart_id : str
        Unique identifier.  Auto-generated when *None*.
    metadata : dict
        Arbitrary user-level metadata attached to the chart.

    Examples
    --------
    >>> import numpy as np
    >>> dim, ambient = 2, 10
    >>> chart = Chart(name="patch_0", dim=dim, ambient_dim=ambient)
    >>> data = np.random.randn(5, ambient)
    >>> coords = chart.embed(data)          # project into chart coords
    >>> recovered = chart.inverse(coords)   # lift back to ambient
    """

    name: str
    dim: int
    ambient_dim: int
    embedding_fn: Callable[[np.ndarray], np.ndarray] | None = None
    inverse_fn: Callable[[np.ndarray], np.ndarray] | None = None
    bounds: InitVar[tuple[np.ndarray, np.ndarray] | None] = None
    anchor_points: InitVar[np.ndarray | None] = None
    chart_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Internal bookkeeping
    # ------------------------------------------------------------------
    _min_coords: np.ndarray | None = field(default=None, init=False, repr=False)
    _max_coords: np.ndarray | None = field(default=None, init=False, repr=False)
    _data_snapshot: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(
        self,
        bounds: tuple[np.ndarray, np.ndarray] | None,
        anchor_points: np.ndarray | None,
    ) -> None:
        """Validate parameters and set up internal state."""
        if self.dim <= 0:
            raise ValueError(f"Intrinsic dimension must be > 0, got {self.dim}")
        if self.ambient_dim <= 0:
            raise ValueError(f"Ambient dimension must be > 0, got {self.ambient_dim}")
        if self.embedding_fn is None:
            logger.debug(
                "Chart '%s' (%s): no embedding_fn provided; "
                "falling back to identity projection (first %d dims).",
                self.name,
                self.chart_id,
                self.dim,
            )
            self._set_default_embedding()
        if self.inverse_fn is None:
            logger.debug(
                "Chart '%s' (%s): no inverse_fn provided; "
                "falling back to zero-padded lift.",
                self.name,
                self.chart_id,
            )
            self._set_default_inverse()
        # Validate explicitly set bounds
        if bounds is not None:
            mn, mx = bounds
            mn = np.asarray(mn, dtype=np.float64)
            mx = np.asarray(mx, dtype=np.float64)
            if mn.shape != (self.dim,):
                raise ValueError(
                    f"min_coords shape must be ({self.dim},), got {mn.shape}"
                )
            if mx.shape != (self.dim,):
                raise ValueError(
                    f"max_coords shape must be ({self.dim},), got {mx.shape}"
                )
            self._min_coords = mn
            self._max_coords = mx
        if anchor_points is not None:
            self.anchor_points = anchor_points
        logger.info(
            "Chart created: name='%s' id='%s' dim=%d ambient_dim=%d",
            self.name,
            self.chart_id,
            self.dim,
            self.ambient_dim,
        )

    # ------------------------------------------------------------------
    # Default embedding / inverse
    # ------------------------------------------------------------------
    def _set_default_embedding(self) -> None:
        """Default embedding: project ambient data onto first `dim` columns."""
        dim = self.dim

        def _embed(data: np.ndarray) -> np.ndarray:
            d = np.asarray(data, dtype=np.float64)
            if d.ndim == 1:
                d = d.reshape(1, -1)
            if d.shape[1] < dim:
                raise ValueError(
                    f"Data has {d.shape[1]} features but chart dim is {dim}"
                )
            return d[:, :dim].copy()

        self.embedding_fn = _embed

    def _set_default_inverse(self) -> None:
        """Default inverse: zero-pad chart coords back to ambient_dim."""
        ambient_dim = self.ambient_dim

        def _inverse(coords: np.ndarray) -> np.ndarray:
            c = np.asarray(coords, dtype=np.float64)
            if c.ndim == 1:
                c = c.reshape(1, -1)
            out = np.zeros((c.shape[0], ambient_dim), dtype=np.float64)
            dim = min(c.shape[1], ambient_dim)
            out[:, :dim] = c[:, :dim]
            return out

        self.inverse_fn = _inverse

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Axis-aligned bounding box ``(min_coords, max_coords)`` of the chart domain.

        If no explicit bounds were supplied and data has been embedded, the
        bounds are computed from the embedded data snapshot.
        """
        if self._min_coords is not None and self._max_coords is not None:
            return (self._min_coords, self._max_coords)
        if self._data_snapshot is not None:
            self._min_coords = self._data_snapshot.min(axis=0)
            self._max_coords = self._data_snapshot.max(axis=0)
            return (self._min_coords, self._max_coords)
        return None

    @bounds.setter
    def bounds(self, value: tuple[np.ndarray, np.ndarray] | None) -> None:
        if value is None:
            self._min_coords = None
            self._max_coords = None
            return
        mn, mx = value
        self._min_coords = np.asarray(mn, dtype=np.float64)
        self._max_coords = np.asarray(mx, dtype=np.float64)

    @property
    def anchor_points(self) -> np.ndarray:
        """Representative anchor points in chart coordinates.

        If no anchor points have been explicitly set, returns the data snapshot
        or falls back to an origin-based default.
        """
        if self._anchor_points_cache is not None:
            return self._anchor_points_cache
        if self._data_snapshot is not None:
            n = min(20, self._data_snapshot.shape[0])
            indices = np.linspace(0, self._data_snapshot.shape[0] - 1, n, dtype=int)
            self._anchor_points_cache = self._data_snapshot[indices]
            return self._anchor_points_cache
        # Origin fallback
        self._anchor_points_cache = np.zeros((1, self.dim), dtype=np.float64)
        return self._anchor_points_cache

    @anchor_points.setter
    def anchor_points(self, value: np.ndarray | None) -> None:
        if value is not None:
            self._anchor_points_cache = np.asarray(value, dtype=np.float64)
            if self._anchor_points_cache.ndim == 1:
                self._anchor_points_cache = self._anchor_points_cache.reshape(1, -1)
            if self._anchor_points_cache.shape[1] != self.dim:
                raise ValueError(
                    f"Anchor points must have shape (N, {self.dim}), "
                    f"got {self._anchor_points_cache.shape}"
                )
        else:
            self._anchor_points_cache = None

    _anchor_points_cache: np.ndarray | None = field(
        default=None, init=False, repr=False
    )

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    def embed(self, data: np.ndarray) -> np.ndarray:
        """Project ambient-space data into chart coordinates.

        Parameters
        ----------
        data : ndarray of shape (N, ambient_dim) or (ambient_dim,)
            Data points in the ambient space.

        Returns
        -------
        ndarray of shape (N, dim)
            Coordinates in the chart's local coordinate system.

        Raises
        ------
        ValueError
            If the data dimensionality does not match ``ambient_dim``.
        """
        if self.embedding_fn is None:
            raise RuntimeError("Chart has no embedding function set.")
        d = np.asarray(data, dtype=np.float64)
        original_ndim = d.ndim
        if d.ndim == 1:
            d = d.reshape(1, -1)
        if d.shape[1] != self.ambient_dim:
            raise ValueError(
                f"Expected data with {self.ambient_dim} features, got {d.shape[1]}"
            )
        result = self.embedding_fn(d)
        result = np.asarray(result, dtype=np.float64)
        # Store snapshot for lazy bounds computation
        if self._data_snapshot is None:
            self._data_snapshot = result.copy()
        else:
            self._data_snapshot = np.vstack([self._data_snapshot, result])
        if original_ndim == 1:
            return result[0]
        return result

    def inverse(self, coords: np.ndarray) -> np.ndarray:
        """Lift chart coordinates back to ambient space.

        Parameters
        ----------
        coords : ndarray of shape (N, dim) or (dim,)
            Points in chart coordinates.

        Returns
        -------
        ndarray of shape (N, ambient_dim)
            Corresponding points in ambient space.
        """
        if self.inverse_fn is None:
            raise RuntimeError("Chart has no inverse function set.")
        c = np.asarray(coords, dtype=np.float64)
        original_ndim = c.ndim
        if c.ndim == 1:
            c = c.reshape(1, -1)
        if c.shape[1] != self.dim:
            raise ValueError(
                f"Expected coordinates with {self.dim} features, got {c.shape[1]}"
            )
        result = self.inverse_fn(c)
        result = np.asarray(result, dtype=np.float64)
        if original_ndim == 1:
            return result[0]
        return result

    def contains(self, coords: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Check which coordinate points lie within the chart bounds.

        Parameters
        ----------
        coords : ndarray of shape (N, dim)
            Points in chart coordinates to test.
        margin : float
            Fractional margin to extend the bounding box.  ``0.0`` means
            exact bounds; ``0.1`` extends by 10 %.

        Returns
        -------
        ndarray of shape (N,), dtype=bool
            Boolean mask – ``True`` where the point is inside.

        Raises
        ------
        RuntimeError
            If bounds have not been set and cannot be inferred.
        """
        b = self.bounds
        if b is None:
            raise RuntimeError(
                f"Chart '{self.name}' has no bounds. Embed data first or set bounds explicitly."
            )
        mn, mx = b
        c = np.asarray(coords, dtype=np.float64)
        if c.ndim == 1:
            c = c.reshape(1, -1)
        if c.shape[1] != self.dim:
            raise ValueError(
                f"Expected coordinates with {self.dim} features, got {c.shape[1]}"
            )
        if margin > 0:
            span = mx - mn
            mn = mn - margin * span
            mx = mx + margin * span
        inside = np.all((c >= mn) & (c <= mx), axis=1)
        return inside

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Serialize the chart to a JSON-serialisable dictionary.

        Callable ``embedding_fn`` / ``inverse_fn`` are stored as string
        descriptors (``None`` or ``"default"``).  For custom functions the
        caller is responsible for separate persistence.
        """
        b = self.bounds
        bounds_list: list[Any] | None = None
        if b is not None:
            bounds_list = [b[0].tolist(), b[1].tolist()]
        ap = self._anchor_points_cache
        anchors_list: list[Any] | None = None
        if ap is not None:
            anchors_list = ap.tolist()
        emb_desc = "default" if self._is_default_embedding else None
        inv_desc = "default" if self._is_default_inverse else None
        return {
            "chart_id": self.chart_id,
            "name": self.name,
            "dim": self.dim,
            "ambient_dim": self.ambient_dim,
            "embedding_fn": emb_desc,
            "inverse_fn": inv_desc,
            "bounds": bounds_list,
            "anchor_points": anchors_list,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Chart:
        """Reconstruct a chart from a serialised dictionary.

        Parameters
        ----------
        d : dict
            Dictionary previously returned by :meth:`to_dict`.

        Returns
        -------
        Chart
        """
        dim = d["dim"]
        ambient_dim = d["ambient_dim"]
        chart = cls(
            name=d["name"],
            dim=dim,
            ambient_dim=ambient_dim,
            chart_id=d["chart_id"],
            metadata=d.get("metadata", {}),
        )
        # Bounds
        bounds_raw = d.get("bounds")
        if bounds_raw is not None:
            chart.bounds = (
                np.asarray(bounds_raw[0], dtype=np.float64),
                np.asarray(bounds_raw[1], dtype=np.float64),
            )
        # Anchor points
        ap_raw = d.get("anchor_points")
        if ap_raw is not None:
            chart.anchor_points = np.asarray(ap_raw, dtype=np.float64)
        return chart

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @property
    def _is_default_embedding(self) -> bool:
        """True if the current embedding is the auto-generated default."""
        return (
            self.embedding_fn is not None
            and getattr(self.embedding_fn, "__name__", "") == "_embed"
        )

    @property
    def _is_default_inverse(self) -> bool:
        """True if the current inverse is the auto-generated default."""
        return (
            self.inverse_fn is not None
            and getattr(self.inverse_fn, "__name__", "") == "_inverse"
        )

    def __repr__(self) -> str:
        return (
            f"Chart(name={self.name!r}, id={self.chart_id!r}, "
            f"dim={self.dim}, ambient_dim={self.ambient_dim})"
        )

    def summary(self) -> dict[str, Any]:
        """Return a human-readable summary dictionary."""
        b = self.bounds
        return {
            "chart_id": self.chart_id,
            "name": self.name,
            "dim": self.dim,
            "ambient_dim": self.ambient_dim,
            "has_bounds": b is not None,
            "n_anchor_points": (
                0
                if self._anchor_points_cache is None
                else self._anchor_points_cache.shape[0]
            ),
            "metadata": self.metadata,
        }
