"""
Tangent Bundle - collection of tangent spaces covering the manifold.

The tangent bundle T(M) is the disjoint union of all tangent spaces
T_p(M) for every point p ∈ M.  In practice we sample a finite set of
anchor points and construct a tangent space at each one, then route
queries to the nearest (or optimally combined) tangent space for
fast local search.

Architecture
------------
Each anchor is identified by a string *anchor_id*.  Anchor coordinates
are stored in a ``scipy.spatial.KDTree`` for O(log N) nearest-lookup.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import numpy as np
from scipy.spatial import KDTree

from .tangent_space import TangentSpace

logger = logging.getLogger(__name__)


class TangentBundle:
    """
    A finite cover of the manifold via multiple tangent spaces.

    Parameters
    ----------
    intrinsic_dim : int | None
        Override for the intrinsic dimension of every tangent space.
        If *None* the dimension is inferred per tangent space from data.
    metric_eps : float
        Epsilon threshold for coverage analysis: a data point is
        "covered" if its projection error is within this value.
    """

    def __init__(
        self,
        intrinsic_dim: int | None = None,
        metric_eps: float = 0.1,
    ) -> None:
        self.intrinsic_dim = intrinsic_dim
        self.metric_eps = metric_eps

        # anchor_id → TangentSpace
        self.tangent_spaces: dict[str, TangentSpace] = {}

        # For fast nearest-anchor lookup
        self._anchor_coords: np.ndarray = np.empty((0, 0), dtype=np.float64)
        self._anchor_ids: list[str] = []
        self._kd_tree: KDTree | None = None

        logger.info("TangentBundle initialised (eps=%.4f)", metric_eps)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_kdtree(self) -> None:
        """Rebuild the KDTree from current anchors."""
        if len(self._anchor_ids) == 0:
            self._anchor_coords = np.empty((0, 0), dtype=np.float64)
            self._kd_tree = None
            return
        self._anchor_coords = np.array(
            [self.tangent_spaces[aid].base_point for aid in self._anchor_ids]
        )
        self._kd_tree = KDTree(self._anchor_coords)

    def _ensure_kdtree(self) -> None:
        """Build KDTree lazily."""
        if self._kd_tree is None:
            self._rebuild_kdtree()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_tangent_space(
        self,
        base_point: np.ndarray,
        data_neighbors: np.ndarray,
        anchor_id: str | None = None,
    ) -> str:
        """
        Create a new tangent space and add it to the bundle.

        Parameters
        ----------
        base_point : np.ndarray, shape (ambient_dim,)
            Anchor point on the manifold.
        data_neighbors : np.ndarray, shape (n, ambient_dim)
            Local neighbourhood used to compute the basis.
        anchor_id : str | None
            Custom id.  If *None* a UUID is generated.

        Returns
        -------
        str
            The anchor id of the newly added tangent space.
        """
        if anchor_id is None:
            anchor_id = str(uuid.uuid4())[:8]

        ts = TangentSpace(
            base_point=np.asarray(base_point, dtype=np.float64),
            data=np.asarray(data_neighbors, dtype=np.float64),
            intrinsic_dim=self.intrinsic_dim,
        )
        self.tangent_spaces[anchor_id] = ts
        self._anchor_ids.append(anchor_id)
        self._rebuild_kdtree()

        logger.debug("Added tangent space %s (dim=%d)", anchor_id, ts.dimension)
        return anchor_id

    def remove_tangent_space(self, anchor_id: str) -> None:
        """
        Remove a tangent space from the bundle.

        Parameters
        ----------
        anchor_id : str
            Id of the tangent space to remove.

        Raises
        ------
        KeyError
            If *anchor_id* does not exist.
        """
        if anchor_id not in self.tangent_spaces:
            raise KeyError(f"Anchor '{anchor_id}' not found in bundle.")
        del self.tangent_spaces[anchor_id]
        self._anchor_ids.remove(anchor_id)
        self._rebuild_kdtree()
        logger.debug("Removed tangent space %s", anchor_id)

    def nearest_anchor(
        self, data_point: np.ndarray, k: int = 1
    ) -> list[tuple[str, float]]:
        """
        Find the nearest anchor(s) to *data_point*.

        Parameters
        ----------
        data_point : np.ndarray, shape (ambient_dim,)
        k : int
            Number of nearest anchors to return.

        Returns
        -------
        list of (anchor_id, distance) tuples, sorted by distance ascending.

        Raises
        ------
        RuntimeError
            If the bundle is empty.
        """
        if len(self.tangent_spaces) == 0:
            raise RuntimeError("TangentBundle is empty; no anchors available.")

        self._ensure_kdtree()
        point = np.asarray(data_point, dtype=np.float64).reshape(1, -1)

        dists, indices = self._kd_tree.query(point, k=min(k, len(self._anchor_ids)))

        # KDTree.query may return (1, k) or scalar; normalise to 1-D
        dists = np.asarray(dists, dtype=np.float64).ravel()
        indices = np.asarray(indices, dtype=np.int64).ravel()

        results = []
        for d, idx in zip(dists, indices):
            aid = self._anchor_ids[int(idx)]
            results.append((aid, float(d)))
        return results

    def get_tangent_space(self, anchor_id: str) -> TangentSpace:
        """Retrieve a tangent space by anchor id."""
        if anchor_id not in self.tangent_spaces:
            raise KeyError(f"Anchor '{anchor_id}' not found.")
        return self.tangent_spaces[anchor_id]

    def get_optimal_tangent_space(
        self, data_point: np.ndarray, k_neighbors: int = 3
    ) -> tuple[TangentSpace, np.ndarray]:
        """
        Compute a weighted combination of nearby tangent spaces to
        produce an "optimal" tangent space for *data_point*.

        The weights are inverse-distance-squared: closer anchors
        contribute more.  The combined basis is the weighted SVD
        of the constituent bases.

        Parameters
        ----------
        data_point : np.ndarray, shape (ambient_dim,)
        k_neighbors : int
            How many nearby tangent spaces to blend.

        Returns
        -------
        TangentSpace
            Synthesised tangent space optimised for *data_point*.
        np.ndarray
            Weights used for each constituent space.
        """
        point = np.asarray(data_point, dtype=np.float64)
        anchors = self.nearest_anchor(point, k=k_neighbors)
        k = len(anchors)

        # Build weighted basis matrix
        ambient_dim = self.tangent_spaces[anchors[0][0]].base_point.shape[0]
        dim = self.tangent_spaces[anchors[0][0]].dimension
        weights = np.zeros(k, dtype=np.float64)

        weighted_basis = np.zeros((ambient_dim, dim), dtype=np.float64)

        for i, (aid, dist) in enumerate(anchors):
            ts = self.tangent_spaces[aid]
            w = 1.0 / (dist**2 + 1e-12)
            weights[i] = w
            # Parallel-transport basis to common frame (first anchor)
            if i == 0:
                weighted_basis += w * ts.basis
            else:
                for col in range(ts.dimension):
                    transported = ts.parallel_transport(
                        np.eye(ts.dimension)[col],
                        self.tangent_spaces[anchors[0][0]],
                    )
                    weighted_basis[:, col] += w * (
                        self.tangent_spaces[anchors[0][0]].basis @ transported
                    )

        # Normalise weights
        weights /= weights.sum() + 1e-15

        # Re-orthogonalise via QR
        Q, _ = np.linalg.qr(weighted_basis)
        combined_basis = Q[:, :dim]

        # Build a synthetic TangentSpace
        combined_ts = TangentSpace(point, data=None, intrinsic_dim=dim)
        combined_ts.basis = combined_basis

        # Blend metric tensors
        blended_metric = np.zeros((dim, dim), dtype=np.float64)
        for i, (aid, dist) in enumerate(anchors):
            ts = self.tangent_spaces[aid]
            # Resize metric if dimensions differ
            m = ts.metric_tensor.copy()
            if m.shape[0] != dim:
                padded = np.eye(dim)
                padded[: m.shape[0], : m.shape[1]] = m
                m = padded
            blended_metric += weights[i] * m
        combined_ts.metric_tensor = 0.5 * (blended_metric + blended_metric.T)

        logger.debug("Optimal TS built from %d anchors, weights=%s", k, weights)
        return combined_ts, weights

    def reindex(self, data: np.ndarray, n_anchors: int | None = None) -> None:
        """
        Rebuild the entire tangent bundle from scratch.

        Anchors are selected via k-means++ style selection on *data*.

        Parameters
        ----------
        data : np.ndarray, shape (N, ambient_dim)
        n_anchors : int | None
            Number of tangent spaces to create.  If *None* a heuristic
            based on data size is used (≈ √N, minimum 3, maximum 200).
        """
        data = np.asarray(data, dtype=np.float64)
        N, D = data.shape

        if n_anchors is None:
            n_anchors = min(200, max(3, int(np.sqrt(N))))

        logger.info("Reindexing bundle: N=%d, D=%d, n_anchors=%d", N, D, n_anchors)

        # Clear existing state
        self.tangent_spaces.clear()
        self._anchor_ids.clear()

        # Select anchor points using farthest-point sampling
        anchor_indices = self._farthest_point_sampling(data, n_anchors)

        # Build tangent spaces at each anchor
        k_nn = min(max(2 * (self.intrinsic_dim or D // 2) + 1, 10), N)
        from scipy.spatial import KDTree as _KDTree

        data_tree = _KDTree(data)

        for idx in anchor_indices:
            base = data[idx]
            _, nn_idx = data_tree.query(base, k=k_nn)
            nn_idx = np.atleast_1d(np.asarray(nn_idx, dtype=np.int64)).ravel()
            neighbors = data[nn_idx]
            self.add_tangent_space(base, neighbors)

        logger.info(
            "Reindex complete: %d tangent spaces created.", len(self.tangent_spaces)
        )

    def coverage_analysis(self, data: np.ndarray) -> dict[str, Any]:
        """
        Analyse how well the tangent bundle covers *data*.

        A data point is "covered" if the reconstruction error after
        projecting to and lifting from the nearest tangent space is
        below ``self.metric_eps``.

        Parameters
        ----------
        data : np.ndarray, shape (N, ambient_dim)

        Returns
        -------
        dict with keys:
            * n_covered : int
            * coverage_ratio : float  (n_covered / N)
            * mean_error : float
            * max_error : float
            * std_error : float
            * per_anchor : dict[str, int]  (anchor_id → n points best-served)
        """
        if len(self.tangent_spaces) == 0:
            return {
                "n_covered": 0,
                "coverage_ratio": 0.0,
                "mean_error": float("inf"),
                "max_error": float("inf"),
                "std_error": float("inf"),
                "per_anchor": {},
            }

        data = np.asarray(data, dtype=np.float64)
        N = data.shape[0]

        errors = np.zeros(N, dtype=np.float64)
        best_anchor = [""] * N

        self._ensure_kdtree()

        # Batch nearest-anchor lookup
        dists, indices = self._kd_tree.query(data, k=1)
        dists = np.asarray(dists, dtype=np.float64).ravel()
        indices = np.asarray(indices, dtype=np.int64).ravel()

        for i in range(N):
            aid = self._anchor_ids[int(indices[i])]
            ts = self.tangent_spaces[aid]
            tc = ts.project(data[i])
            reconstructed = ts.lift(tc)
            errors[i] = np.linalg.norm(data[i] - reconstructed)
            best_anchor[i] = aid

        n_covered = int(np.sum(errors <= self.metric_eps))
        per_anchor: dict[str, int] = {}
        for aid in best_anchor:
            per_anchor[aid] = per_anchor.get(aid, 0) + 1

        result = {
            "n_covered": n_covered,
            "coverage_ratio": n_covered / max(N, 1),
            "mean_error": float(np.mean(errors)),
            "max_error": float(np.max(errors)),
            "std_error": float(np.std(errors)),
            "per_anchor": per_anchor,
        }
        logger.info(
            "Coverage: %.2f%% (mean_err=%.4f)",
            result["coverage_ratio"] * 100,
            result["mean_error"],
        )
        return result

    @staticmethod
    def _farthest_point_sampling(data: np.ndarray, n_samples: int) -> list[int]:
        """
        Select *n_samples* anchor indices using farthest-point sampling.

        Parameters
        ----------
        data : np.ndarray, shape (N, D)
        n_samples : int

        Returns
        -------
        list of int
            Indices into *data*.
        """
        N = data.shape[0]
        n_samples = min(n_samples, N)

        # Start from a random point
        rng = np.random.default_rng(42)
        first = int(rng.integers(0, N))
        selected = [first]
        min_dists = np.full(N, np.inf, dtype=np.float64)

        for _ in range(1, n_samples):
            last = selected[-1]
            # Euclidean distance from last selected to all points
            d = np.sum((data - data[last]) ** 2, axis=1)
            min_dists = np.minimum(min_dists, d)
            # Select the point with the largest minimum distance
            next_idx = int(np.argmax(min_dists))
            selected.append(next_idx)

        return selected

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def serialize(self) -> dict[str, Any]:
        """Serialise the entire bundle to a plain dict."""
        return {
            "intrinsic_dim": self.intrinsic_dim,
            "metric_eps": self.metric_eps,
            "anchor_ids": self._anchor_ids,
            "tangent_spaces": {
                aid: ts.to_dict() for aid, ts in self.tangent_spaces.items()
            },
        }

    @classmethod
    def deserialize(cls, d: dict[str, Any]) -> TangentBundle:
        """Deserialise from a dict produced by ``serialize``."""
        bundle = cls(
            intrinsic_dim=d.get("intrinsic_dim"),
            metric_eps=d.get("metric_eps", 0.1),
        )
        for aid in d["anchor_ids"]:
            ts_dict = d["tangent_spaces"][aid]
            bundle.tangent_spaces[aid] = TangentSpace.from_dict(ts_dict)
            bundle._anchor_ids.append(aid)
        bundle._rebuild_kdtree()
        return bundle

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_spaces(self) -> int:
        """Number of tangent spaces in the bundle."""
        return len(self.tangent_spaces)

    def __repr__(self) -> str:
        return (
            f"TangentBundle(n_spaces={self.n_spaces}, "
            f"intrinsic_dim={self.intrinsic_dim}, eps={self.metric_eps})"
        )

    def __len__(self) -> int:
        return self.n_spaces
