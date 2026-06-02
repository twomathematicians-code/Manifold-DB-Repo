"""
Tangent Space Index - main index structure for query acceleration.

Uses per-tangent-space ``sklearn.neighbors.BallTree`` indices over the
low-dimensional tangent projections for fast local nearest-neighbour
search.  Queries are routed to the nearest (or optimally combined)
tangent space, projected to the low-dimensional frame, searched, and
mapped back to ambient-space identifiers.

Workflow
--------
1. **Build** – ``build_from_data`` selects anchor points via farthest-point
   sampling, builds tangent spaces, and projects all points into their
   nearest tangent space.
2. **Insert** – project to nearest TS and add to the per-TS BallTree.
3. **Search** – find nearest TS, project query, search locally, return
   ambient-space point ids with tangent-space distances.
4. **Delete / Update** – remove or replace entries with lazy reindexing.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.neighbors import BallTree

from .tangent_bundle import TangentBundle
from .tangent_space import TangentSpace

logger = logging.getLogger(__name__)


class TangentSpaceIndex:
    """
    Index for approximate nearest-neighbour search on manifold data.

    Parameters
    ----------
    intrinsic_dim : int | None
        Override for the intrinsic dimension of each tangent space.
        If *None* the dimension is inferred from data.
    metric_eps : float
        Epsilon for coverage analysis in the tangent bundle.
    leaf_size : int
        Leaf size passed to sklearn's BallTree.
    """

    def __init__(
        self,
        intrinsic_dim: Optional[int] = None,
        metric_eps: float = 0.1,
        leaf_size: int = 40,
    ) -> None:
        self.intrinsic_dim = intrinsic_dim
        self.metric_eps = metric_eps
        self.leaf_size = leaf_size

        # Unique identifier for this chart/index
        self.chart_id: str = str(uuid.uuid4())[:12]

        # Tangent bundle (collection of tangent spaces)
        self.bundle = TangentBundle(
            intrinsic_dim=intrinsic_dim,
            metric_eps=metric_eps,
        )

        # Per-anchor data stores
        # anchor_id → {
        #     "point_ids": list of point identifiers,
        #     "tangent_coords": np.ndarray (n, intrinsic_dim),
        #     "balltree": BallTree or None,
        #     "needs_rebuild": bool,
        # }
        self._local_indices: Dict[str, Dict[str, Any]] = {}

        # Global reverse lookup: point_id → (anchor_id, local_index)
        self._point_map: Dict[Any, Tuple[str, int]] = {}

        # Cached stats
        self._size: int = 0
        self._build_time: float = 0.0

        logger.info(
            "TangentSpaceIndex created (chart_id=%s, intrinsic_dim=%s)",
            self.chart_id,
            intrinsic_dim,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_balltree(self, anchor_id: str) -> None:
        """Build or rebuild BallTree for an anchor."""
        store = self._local_indices[anchor_id]
        if store["balltree"] is None or store["needs_rebuild"]:
            coords = store["tangent_coords"]
            if coords.shape[0] > 0:
                store["balltree"] = BallTree(
                    coords, leaf_size=self.leaf_size, metric="euclidean"
                )
            else:
                store["balltree"] = None
            store["needs_rebuild"] = False

    def _get_or_create_local(self, anchor_id: str, intrinsic_dim: int) -> Dict:
        """Get or create the per-anchor store."""
        if anchor_id not in self._local_indices:
            self._local_indices[anchor_id] = {
                "point_ids": [],
                "tangent_coords": np.empty((0, intrinsic_dim), dtype=np.float64),
                "balltree": None,
                "needs_rebuild": True,
            }
        return self._local_indices[anchor_id]

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_from_data(
        self,
        point_ids: List[Any],
        data_points: np.ndarray,
        n_anchors: int = 50,
    ) -> Dict[str, Any]:
        """
        Construct the full index from a set of data points.

        Parameters
        ----------
        point_ids : list
            Unique identifiers for each data point.
        data_points : np.ndarray, shape (N, ambient_dim)
        n_anchors : int
            Number of tangent spaces (anchor points) to create.

        Returns
        -------
        dict
            Build statistics (timing, coverage, etc.)
        """
        t0 = time.time()
        data = np.asarray(data_points, dtype=np.float64)
        N, D = data.shape
        assert len(point_ids) == N, "point_ids length must match data_points rows."

        logger.info("Building index from %d points (D=%d, anchors=%d)", N, D, n_anchors)

        # Reset state
        self._local_indices.clear()
        self._point_map.clear()
        self._size = 0

        # Build the tangent bundle (selects anchors, computes TS bases)
        self.bundle.reindex(data, n_anchors=n_anchors)

        # Determine intrinsic dimension from bundle
        sample_ts = next(iter(self.bundle.tangent_spaces.values()))
        dim = sample_ts.dimension
        logger.info("Intrinsic dimension: %d", dim)

        # Project each point to its nearest tangent space and store
        for i in range(N):
            pid = point_ids[i]
            pt = data[i]

            # Find nearest anchor
            nearest = self.bundle.nearest_anchor(pt, k=1)[0]
            aid = nearest[0]
            ts = self.bundle.get_tangent_space(aid)

            # Project to tangent coordinates
            tc = ts.project(pt)

            # Store in local index
            store = self._get_or_create_local(aid, dim)
            local_idx = len(store["point_ids"])
            store["point_ids"].append(pid)
            if store["tangent_coords"].shape[0] == 0:
                store["tangent_coords"] = tc.reshape(1, -1)
            else:
                store["tangent_coords"] = np.vstack([store["tangent_coords"], tc])
            store["needs_rebuild"] = True

            self._point_map[pid] = (aid, local_idx)
            self._size += 1

        # Build all BallTrees
        for aid in self._local_indices:
            self._ensure_balltree(aid)

        self._build_time = time.time() - t0

        # Coverage analysis
        coverage = self.bundle.coverage_analysis(data)

        stats = {
            "n_points": N,
            "n_anchors": self.bundle.n_spaces,
            "intrinsic_dim": dim,
            "ambient_dim": D,
            "build_time_sec": round(self._build_time, 4),
            "coverage": coverage,
        }
        logger.info("Index built: %s", stats)
        return stats

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert(self, point_id: Any, data_point: np.ndarray) -> None:
        """
        Insert a single point into the index.

        Parameters
        ----------
        point_id : hashable
            Unique identifier.
        data_point : np.ndarray, shape (ambient_dim,)
        """
        if point_id in self._point_map:
            raise ValueError(f"Point {point_id} already exists in index.")

        pt = np.asarray(data_point, dtype=np.float64)

        # Find nearest anchor
        nearest = self.bundle.nearest_anchor(pt, k=1)[0]
        aid = nearest[0]
        ts = self.bundle.get_tangent_space(aid)

        # Project to tangent coords
        tc = ts.project(pt)

        # Store
        dim = ts.dimension
        store = self._get_or_create_local(aid, dim)
        local_idx = len(store["point_ids"])
        store["point_ids"].append(point_id)
        if store["tangent_coords"].shape[0] == 0:
            store["tangent_coords"] = tc.reshape(1, -1)
        else:
            store["tangent_coords"] = np.vstack([store["tangent_coords"], tc])
        store["needs_rebuild"] = True

        self._point_map[point_id] = (aid, local_idx)
        self._size += 1

        logger.debug("Inserted point %s into anchor %s", point_id, aid)

    def batch_insert(
        self, point_ids: List[Any], data_points: np.ndarray
    ) -> None:
        """
        Insert multiple points at once.  More efficient than individual
        inserts because BallTree rebuilds are deferred.

        Parameters
        ----------
        point_ids : list of hashable
        data_points : np.ndarray, shape (N, ambient_dim)
        """
        data = np.asarray(data_points, dtype=np.float64)
        for i, pid in enumerate(point_ids):
            self.insert(pid, data[i])

        # Now rebuild all dirty BallTrees
        for aid, store in self._local_indices.items():
            if store["needs_rebuild"]:
                self._ensure_balltree(aid)

        logger.info("Batch insert complete: %d points", len(point_ids))

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_point: np.ndarray,
        k: int = 10,
        n_candidates: Optional[int] = None,
        search_k_anchors: int = 1,
    ) -> List[Tuple[Any, float]]:
        """
        Search for the *k* nearest neighbours of *query_point*.

        Algorithm
        ---------
        1. Find the nearest tangent space (anchor).
        2. Project the query into tangent coordinates.
        3. Search the per-TS BallTree (low-dimensional, fast).
        4. Map results back to ambient-space point ids.

        Parameters
        ----------
        query_point : np.ndarray, shape (ambient_dim,)
        k : int
            Number of nearest neighbours to return.
        n_candidates : int | None
            If provided, oversample this many candidates from the local
            BallTree and re-rank in ambient space.  If *None*, return
            the top-k from tangent space directly.
        search_k_anchors : int
            Number of nearby tangent spaces to search.  Results from all
            anchors are merged and de-duplicated.

        Returns
        -------
        list of (point_id, distance) tuples, sorted by distance ascending.
            Distance is the tangent-space geodesic distance approximation.
        """
        if self._size == 0:
            return []

        query = np.asarray(query_point, dtype=np.float64)
        n_candidates = n_candidates or max(k * 2, 20)
        effective_k = min(n_candidates, k)

        # Collect candidates from multiple nearby anchors
        candidate_pids: Dict[Any, float] = {}  # pid → best distance

        anchors = self.bundle.nearest_anchor(query, k=search_k_anchors)

        for aid, dist_to_anchor in anchors:
            if aid not in self._local_indices:
                continue

            ts = self.bundle.get_tangent_space(aid)
            store = self._local_indices[aid]
            self._ensure_balltree(aid)

            if store["balltree"] is None:
                continue

            # Project query into this tangent space
            tc = ts.project(query).reshape(1, -1)
            search_k = min(n_candidates, store["balltree"].data.shape[0])

            if search_k == 0:
                continue

            dists, local_indices = store["balltree"].query(
                tc, k=search_k, return_distance=True
            )
            dists = dists.ravel()
            local_indices = local_indices.ravel()

            for d, lidx in zip(dists, local_indices):
                pid = store["point_ids"][int(lidx)]
                # Distance = tangent-space distance (already Euclidean in TS)
                # Add penalty proportional to how far the anchor is from query
                penalty = 0.5 * dist_to_anchor ** 2
                total_dist = float(d) + penalty
                if pid not in candidate_pids or total_dist < candidate_pids[pid]:
                    candidate_pids[pid] = total_dist

        # Sort by distance and return top-k
        results = sorted(candidate_pids.items(), key=lambda x: x[1])[:k]
        return results

    # ------------------------------------------------------------------
    # Delete / Update
    # ------------------------------------------------------------------

    def delete(self, point_id: Any) -> bool:
        """
        Delete a point from the index.

        Parameters
        ----------
        point_id : hashable

        Returns
        -------
        bool
            True if the point was found and deleted, False otherwise.
        """
        if point_id not in self._point_map:
            logger.warning("Point %s not found for deletion.", point_id)
            return False

        aid, local_idx = self._point_map[point_id]
        store = self._local_indices[aid]

        # Remove from local store
        n_local = len(store["point_ids"])
        if local_idx < n_local - 1:
            # Swap with last element for O(1) removal
            last_pid = store["point_ids"][-1]
            store["point_ids"][local_idx] = last_pid
            store["tangent_coords"][local_idx] = store["tangent_coords"][-1]
            # Update reverse map for swapped element
            self._point_map[last_pid] = (aid, local_idx)

        store["point_ids"].pop()
        store["tangent_coords"] = store["tangent_coords"][:-1]
        store["needs_rebuild"] = True

        del self._point_map[point_id]
        self._size -= 1

        logger.debug("Deleted point %s from anchor %s", point_id, aid)
        return True

    def update(self, point_id: Any, new_data_point: np.ndarray) -> bool:
        """
        Update a point's data in the index.

        If the nearest tangent space has changed, the point is moved
        to the new anchor.

        Parameters
        ----------
        point_id : hashable
        new_data_point : np.ndarray, shape (ambient_dim,)

        Returns
        -------
        bool
            True if the point was found and updated.
        """
        if point_id not in self._point_map:
            logger.warning("Point %s not found for update.", point_id)
            return False

        # Delete old entry first
        self.delete(point_id)

        # Re-insert at new location
        self.insert(point_id, np.asarray(new_data_point, dtype=np.float64))

        logger.debug("Updated point %s", point_id)
        return True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Total number of indexed points."""
        return self._size

    @property
    def stats(self) -> Dict[str, Any]:
        """Summary statistics for the index."""
        per_anchor_counts = {
            aid: len(store["point_ids"])
            for aid, store in self._local_indices.items()
        }
        max_per = max(per_anchor_counts.values()) if per_anchor_counts else 0
        min_per = min(per_anchor_counts.values()) if per_anchor_counts else 0
        avg_per = np.mean(list(per_anchor_counts.values())) if per_anchor_counts else 0

        return {
            "chart_id": self.chart_id,
            "size": self._size,
            "n_anchors": self.bundle.n_spaces,
            "intrinsic_dim": self.intrinsic_dim,
            "build_time_sec": round(self._build_time, 4),
            "points_per_anchor": {
                "min": int(min_per),
                "max": int(max_per),
                "mean": round(float(avg_per), 2),
            },
        }

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the index to a plain dict."""
        local_serializable = {}
        for aid, store in self._local_indices.items():
            local_serializable[aid] = {
                "point_ids": store["point_ids"],
                "tangent_coords": store["tangent_coords"].tolist(),
            }

        return {
            "chart_id": self.chart_id,
            "intrinsic_dim": self.intrinsic_dim,
            "metric_eps": self.metric_eps,
            "leaf_size": self.leaf_size,
            "bundle": self.bundle.serialize(),
            "local_indices": local_serializable,
            "point_map": {str(k): v for k, v in self._point_map.items()},
            "size": self._size,
            "build_time": self._build_time,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TangentSpaceIndex":
        """Deserialise from a dict produced by ``to_dict``."""
        idx = cls(
            intrinsic_dim=d.get("intrinsic_dim"),
            metric_eps=d.get("metric_eps", 0.1),
            leaf_size=d.get("leaf_size", 40),
        )
        idx.chart_id = d["chart_id"]
        idx._size = d["size"]
        idx._build_time = d.get("build_time", 0.0)

        # Restore bundle
        idx.bundle = TangentBundle.deserialize(d["bundle"])

        # Restore local indices
        for aid, store in d["local_indices"].items():
            coords = np.asarray(store["tangent_coords"], dtype=np.float64)
            dim = coords.shape[1] if coords.ndim == 2 and coords.shape[0] > 0 else (
                idx.intrinsic_dim or 0
            )
            idx._local_indices[aid] = {
                "point_ids": store["point_ids"],
                "tangent_coords": coords,
                "balltree": None,
                "needs_rebuild": True,
            }

        # Rebuild BallTrees
        for aid in idx._local_indices:
            idx._ensure_balltree(aid)

        # Restore point map
        for k, v in d["point_map"].items():
            idx._point_map[k] = v

        return idx

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TangentSpaceIndex(chart_id={self.chart_id}, "
            f"size={self._size}, anchors={self.bundle.n_spaces})"
        )

    def __len__(self) -> int:
        return self._size
