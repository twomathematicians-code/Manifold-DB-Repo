"""
Atlas Builder - automatic manifold learning and chart discovery.

Uses local PCA eigenvalue analysis, KNN graph construction, and community
detection (Louvain) to discover the intrinsic chart structure of a dataset.
For each pair of overlapping charts an affine transition map is fitted.

The overall pipeline:

1. **Dimensionality estimation** – local PCA variance-ratio analysis.
2. **KNN graph** – k-nearest-neighbour adjacency for topology capture.
3. **Community detection** – graph partitioning into charts.
4. **Chart creation** – local PCA projections define chart coordinate maps.
5. **Overlap detection** – boundary points shared between neighbouring charts.
6. **Transition fitting** – affine (or neural) maps on overlap data.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .chart import Chart
from .transition_map import AffineTransition, NeuralTransition, TransitionMap

if TYPE_CHECKING:
    from .atlas_manager import AtlasManager

logger = logging.getLogger(__name__)


# ======================================================================
# Optional dependency helpers
# ======================================================================
def _import_networkx():
    try:
        import networkx as nx
        from networkx.algorithms.community import louvain_communities

        return nx, louvain_communities
    except ImportError:
        logger.error(
            "networkx is required for community detection. "
            "Install with: pip install networkx"
        )
        raise


def _import_torch():
    try:
        import torch

        return torch
    except ImportError:
        logger.error(
            "PyTorch is required for NeuralTransition. "
            "Install with: pip install torch"
        )
        raise


# ======================================================================
class AtlasBuilder:
    """
    Automatic manifold-learning atlas constructor.

    Parameters
    ----------
    k_neighbors : int
        Number of neighbours for the KNN graph (default 15).
    pca_variance_threshold : float
        Cumulative variance ratio threshold for intrinsic dimension estimation
        (default 0.95).
    min_chart_size : int
        Minimum number of points for a chart to be retained (default 50).
    overlap_margin : float
        Fractional margin added to overlap bounding boxes (default 0.05).
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        k_neighbors: int = 15,
        pca_variance_threshold: float = 0.95,
        min_chart_size: int = 50,
        overlap_margin: float = 0.05,
        random_state: int | None = None,
    ) -> None:
        self.k_neighbors = k_neighbors
        self.pca_variance_threshold = pca_variance_threshold
        self.min_chart_size = min_chart_size
        self.overlap_margin = overlap_margin
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        logger.info(
            "AtlasBuilder created: k=%d, pca_thresh=%.2f, min_size=%d",
            k_neighbors,
            pca_variance_threshold,
            min_chart_size,
        )

    # ------------------------------------------------------------------
    # 1. Intrinsic dimension estimation
    # ------------------------------------------------------------------
    def estimate_intrinsic_dimension(
        self,
        data: np.ndarray,
        n_samples: int = 500,
    ) -> int:
        """Estimate intrinsic dimension via local PCA eigenvalue analysis.

        Sub-samples the data, fits local PCA on each sample's neighbourhood,
        and finds the smallest dimension *d* such that the cumulative
        explained variance ratio exceeds ``pca_variance_threshold`` for the
        majority of samples.

        Parameters
        ----------
        data : ndarray of shape (N, D)
            Ambient-space data.
        n_samples : int
            Number of points to sample for local analysis.

        Returns
        -------
        int
            Estimated intrinsic dimension (at least 1).
        """
        data = np.asarray(data, dtype=np.float64)
        N, D = data.shape
        n_samples = min(n_samples, N)
        k = min(self.k_neighbors, N - 1)

        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
        nn.fit(data)
        distances, indices = nn.kneighbors(data)

        dim_estimates: list[int] = []
        sample_idx = self._rng.choice(N, size=n_samples, replace=False)
        for idx in sample_idx:
            neighbours = data[indices[idx, 1:]]  # exclude self
            if neighbours.shape[0] < 3:
                continue
            max_components = min(neighbours.shape[0], neighbours.shape[1])
            pca = PCA(n_components=max_components)
            pca.fit(neighbours)
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            d = int(np.searchsorted(cumvar, self.pca_variance_threshold) + 1)
            dim_estimates.append(d)

        if not dim_estimates:
            logger.warning("Could not estimate dimension; falling back to 2.")
            return 2

        # Use median of local estimates
        median_dim = int(np.median(dim_estimates))
        median_dim = max(1, min(median_dim, D))
        logger.info(
            "Intrinsic dimension estimate: %d (from %d local estimates).",
            median_dim,
            len(dim_estimates),
        )
        return median_dim

    # ------------------------------------------------------------------
    # 2. KNN graph
    # ------------------------------------------------------------------
    def compute_knn_graph(
        self,
        data: np.ndarray,
        k: int | None = None,
    ) -> sparse.csr_matrix:
        """Construct a symmetric KNN adjacency graph.

        Parameters
        ----------
        data : ndarray of shape (N, D)
        k : int or None
            Number of neighbours (defaults to ``self.k_neighbors``).

        Returns
        -------
        scipy.sparse.csr_matrix of shape (N, N)
            Symmetric binary adjacency matrix.
        """
        data = np.asarray(data, dtype=np.float64)
        N = data.shape[0]
        k = k or self.k_neighbors
        k = min(k, N - 1)

        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
        nn.fit(data)
        distances, indices = nn.kneighbors(data)

        # Build symmetric adjacency
        rows: list[int] = []
        cols: list[int] = []
        weights: list[float] = []
        for i in range(N):
            for j_idx in range(1, k + 1):  # skip self
                j = indices[i, j_idx]
                dist = distances[i, j_idx]
                weight = np.exp(-dist)
                rows.append(i)
                cols.append(j)
                weights.append(weight)
                rows.append(j)
                cols.append(i)
                weights.append(weight)

        adj = sparse.csr_matrix((weights, (rows, cols)), shape=(N, N))
        logger.info("KNN graph built: %d nodes, %d edges.", N, adj.nnz // 2)
        return adj

    # ------------------------------------------------------------------
    # 3. Community detection
    # ------------------------------------------------------------------
    def detect_chart_boundaries(
        self,
        knn_graph: sparse.csr_matrix,
        min_chart_size: int | None = None,
        resolution: float = 1.0,
    ) -> list[np.ndarray]:
        """Partition the KNN graph into communities (chart assignments).

        Uses the Louvain community detection algorithm.

        Parameters
        ----------
        knn_graph : sparse.csr_matrix
            Symmetric adjacency matrix from :meth:`compute_knn_graph`.
        min_chart_size : int or None
            Minimum community size (defaults to ``self.min_chart_size``).
        resolution : float
            Louvain resolution parameter (> 0).  Higher values yield more
            (smaller) communities.

        Returns
        -------
        list of ndarray
            Each element is an int-array of data-point indices belonging to
            one chart.  Small communities below ``min_chart_size`` are merged
            into the nearest neighbour.
        """
        nx, louvain_communities = _import_networkx()
        G = nx.from_scipy_sparse_array(knn_graph)
        communities = louvain_communities(
            G, resolution=resolution, seed=self.random_state
        )
        min_size = min_chart_size or self.min_chart_size

        # Convert to index arrays
        clusters: list[np.ndarray] = [
            np.array(sorted(c), dtype=int) for c in communities
        ]
        logger.info(
            "Louvain found %d communities (resolution=%.2f).",
            len(clusters),
            resolution,
        )

        # Merge small clusters into nearest large cluster
        large: list[np.ndarray] = [c for c in clusters if len(c) >= min_size]
        small: list[np.ndarray] = [c for c in clusters if len(c) < min_size]

        if small:
            logger.info(
                "Merging %d small clusters (< %d points).", len(small), min_size
            )
            # Compute centroids of large clusters
            if large:
                for s in small:
                    # Assign each point in small cluster to nearest large cluster
                    # We use the mean index as representative (approximate)
                    centroid = np.mean(s)
                    best_large_idx = 0
                    best_dist = float("inf")
                    for li, lg in enumerate(large):
                        lg_centroid = np.mean(lg)
                        dist = abs(float(centroid) - float(lg_centroid))
                        if dist < best_dist:
                            best_dist = dist
                            best_large_idx = li
                    large[best_large_idx] = np.sort(
                        np.concatenate([large[best_large_idx], s])
                    )
            else:
                # No large clusters; just keep all as-is
                large = clusters

        logger.info("After merging: %d charts.", len(large))
        return large

    # ------------------------------------------------------------------
    # 4. Overlap detection
    # ------------------------------------------------------------------
    def compute_overlap_regions(
        self,
        chart_a: Chart,
        chart_b: Chart,
        knn_graph: sparse.csr_matrix,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Find data points that lie in the overlap of two charts.

        Uses the KNN graph to identify boundary points: a point is
        considered overlapping if it is in chart A's index set and has
        neighbours in chart B's index set (or vice versa).

        Parameters
        ----------
        chart_a, chart_b : Chart
            Charts with ``metadata["point_indices"]`` set.
        knn_graph : sparse.csr_matrix
            The full KNN adjacency matrix.

        Returns
        -------
        (overlap_indices, source_coords, target_coords) or None
            *overlap_indices* are the ambient-space data indices shared.
            *source_coords* / *target_coords* are the chart-coordinate
            representations.  Returns *None* if no overlap is found.
        """
        idx_a = set(chart_a.metadata.get("point_indices", []))
        idx_b = set(chart_b.metadata.get("point_indices", []))
        if not idx_a or not idx_b:
            return None

        # Graph-based overlap: points in A with neighbours in B
        G_coo = knn_graph.tocoo()
        overlap_set: set = set()
        for i, j in zip(G_coo.row, G_coo.col):
            if i in idx_a and j in idx_b:
                overlap_set.add(i)
            elif i in idx_b and j in idx_a:
                overlap_set.add(i)

        if not overlap_set:
            return None

        overlap_idx = np.array(sorted(overlap_set), dtype=int)
        logger.debug(
            "Overlap between '%s' and '%s': %d points.",
            chart_a.name,
            chart_b.name,
            len(overlap_idx),
        )
        return overlap_idx

    def compute_overlap_bounds(
        self,
        chart_a: Chart,
        chart_b: Chart,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Compute axis-aligned overlap bounds in chart_a coordinates.

        Uses the overlap indices stored in chart metadata to identify shared
        points and derive the bounding box.

        Returns
        -------
        (min_coords, max_coords) or None
            Each of shape ``(chart_a.dim,)``, or *None* if no overlap.
        """
        overlap_idx = self._find_overlap_indices(chart_a, chart_b)
        if overlap_idx is None or len(overlap_idx) == 0:
            return None

        # Re-embed using the data stored in chart_a's anchor snapshot
        try:
            if chart_a._data_snapshot is None:
                return None
            a_coords = chart_a._data_snapshot[overlap_idx]
        except (IndexError, ValueError):
            return None

        mn = a_coords.min(axis=0) - self.overlap_margin * (
            a_coords.max(axis=0) - a_coords.min(axis=0) + 1e-10
        )
        mx = a_coords.max(axis=0) + self.overlap_margin * (
            a_coords.max(axis=0) - a_coords.min(axis=0) + 1e-10
        )
        return (mn, mx)

    def _find_overlap_indices(
        self,
        chart_a: Chart,
        chart_b: Chart,
    ) -> np.ndarray | None:
        """Get indices of points belonging to both charts."""
        idx_a = set(chart_a.metadata.get("point_indices", []))
        idx_b = set(chart_b.metadata.get("point_indices", []))
        if not idx_a or not idx_b:
            return None
        overlap = sorted(idx_a & idx_b)
        if not overlap:
            # Graph-based check (no knn_graph here; return None)
            return None
        return np.array(overlap, dtype=int)

    # ------------------------------------------------------------------
    # 5. Transition fitting
    # ------------------------------------------------------------------
    def fit_transition_map(
        self,
        overlap_data: np.ndarray,
        source_coords: np.ndarray,
        target_coords: np.ndarray,
        map_type: str = "affine",
    ) -> TransitionMap:
        """Fit a transition map on overlap data.

        Parameters
        ----------
        overlap_data : ndarray of shape (N, D)
            Original ambient-space data for the overlap points.
        source_coords : ndarray of shape (N, d)
            Coordinates in the source chart.
        target_coords : ndarray of shape (N, d)
            Coordinates in the target chart.
        map_type : str
            ``"affine"`` or ``"neural"``.

        Returns
        -------
        TransitionMap
        """
        source_coords = np.asarray(source_coords, dtype=np.float64)
        target_coords = np.asarray(target_coords, dtype=np.float64)
        N, d = source_coords.shape
        if target_coords.shape != (N, d):
            raise ValueError(
                "source_coords and target_coords must have the same shape."
            )

        if map_type == "affine":
            # Least-squares affine: y = M @ x + b
            X = np.hstack([source_coords, np.ones((N, 1))])
            Y = target_coords
            params, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            matrix = params[:d, :]
            bias = params[d, :]
            return AffineTransition(
                source_chart_id="",
                target_chart_id="",
                dim=d,
                matrix=matrix,
                bias=bias,
            )
        elif map_type == "neural":
            _import_torch()  # validate availability
            tmap = NeuralTransition(
                source_chart_id="",
                target_chart_id="",
                dim=d,
            )
            tmap.fit(source_coords, target_coords, epochs=300)
            return tmap
        else:
            raise ValueError(
                f"Unknown map_type '{map_type}'. Use 'affine' or 'neural'."
            )

    def fit_transition_map_simple(
        self,
        chart_a: Chart,
        chart_b: Chart,
    ) -> TransitionMap | None:
        """Fit a transition map between two charts using their data snapshots.

        A convenience method used by :meth:`AtlasManager.rebuild_overlaps`.

        Returns
        -------
        TransitionMap or None
            Fitted affine transition, or *None* if no overlap data is available.
        """
        overlap_idx = self._find_overlap_indices(chart_a, chart_b)
        if overlap_idx is None or len(overlap_idx) < 3:
            return None
        if chart_a._data_snapshot is None or chart_b._data_snapshot is None:
            return None
        try:
            src_c = chart_a._data_snapshot[overlap_idx]
            tgt_c = chart_b._data_snapshot[overlap_idx]
        except IndexError:
            return None
        tmap = self.fit_transition_map(
            overlap_data=None,
            source_coords=src_c,
            target_coords=tgt_c,
            map_type="affine",
        )
        tmap.source_chart_id = chart_a.chart_id
        tmap.target_chart_id = chart_b.chart_id
        ov_bounds = self.compute_overlap_bounds(chart_a, chart_b)
        if ov_bounds is not None:
            tmap.overlap_region = ov_bounds
        logger.info(
            "Fitted affine transition '%s' -> '%s' (%d overlap points).",
            chart_a.name,
            chart_b.name,
            len(overlap_idx),
        )
        return tmap

    # ------------------------------------------------------------------
    # 6. Full pipeline
    # ------------------------------------------------------------------
    def build(
        self,
        data: np.ndarray,
        modality_labels: np.ndarray | None = None,
        n_charts_hint: int | None = None,
        resolution: float | None = None,
    ) -> AtlasManager:
        """Full atlas construction pipeline.

        Parameters
        ----------
        data : ndarray of shape (N, D)
            Ambient-space data.
        modality_labels : ndarray of shape (N,) or None
            Optional per-point modality identifiers.  If provided, the
            builder first splits data by modality and builds a sub-atlas
            for each, then merges.
        n_charts_hint : int or None
            If provided, adjusts the Louvain resolution parameter to target
            approximately this many charts.
        modality : str, optional
            Tag applied to generated charts.

        Returns
        -------
        AtlasManager
            Fully initialised atlas with charts and transition maps.
        """
        # Lazy import to avoid circular dependency with atlas_manager
        from .atlas_manager import AtlasManager

        data = np.asarray(data, dtype=np.float64)
        N, D = data.shape
        logger.info("=== Atlas build start ===  N=%d, D=%d", N, D)

        # Standardise data for PCA
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        # 1. Intrinsic dimension
        intrinsic_dim = self.estimate_intrinsic_dimension(data_scaled)

        # 2. KNN graph
        knn_graph = self.compute_knn_graph(data_scaled)

        # 3. Community detection
        if n_charts_hint is not None and n_charts_hint > 0:
            resolution = self._tune_resolution(knn_graph, n_charts_hint)

        clusters = self.detect_chart_boundaries(
            knn_graph,
            resolution=resolution or 1.0,
        )

        # 4. Create charts
        atlas = AtlasManager(name="auto_atlas")
        for ci, indices in enumerate(clusters):
            cluster_data = data_scaled[indices]
            n = len(indices)

            # Local PCA for chart coordinates
            max_comp = min(intrinsic_dim, min(cluster_data.shape))
            pca = PCA(n_components=max_comp)
            coords = pca.fit_transform(cluster_data)

            chart = Chart(
                name=f"chart_{ci}",
                dim=max_comp,
                ambient_dim=D,
            )
            # Set embedding/inverse via PCA
            pca_obj = pca
            scaler_obj = scaler

            def _make_embed(pca_o, sc_o):
                def _embed(d):
                    d = np.asarray(d, dtype=np.float64)
                    if d.ndim == 1:
                        d = d.reshape(1, -1)
                    scaled = sc_o.transform(d)
                    return pca_o.transform(scaled)

                return _embed

            def _make_inverse(pca_o, sc_o):
                def _inv(c):
                    c = np.asarray(c, dtype=np.float64)
                    if c.ndim == 1:
                        c = c.reshape(1, -1)
                    unscaled = pca_o.inverse_transform(c)
                    return sc_o.inverse_transform(unscaled)

                return _inv

            chart.embedding_fn = _make_embed(pca_obj, scaler_obj)
            chart.inverse_fn = _make_inverse(pca_obj, scaler_obj)
            chart.metadata["point_indices"] = indices.tolist()
            chart.metadata["n_points"] = n

            # Embed data to establish bounds
            chart.embed(data[indices])
            chart.anchor_points = coords[np.linspace(0, n - 1, min(20, n), dtype=int)]

            atlas.add_chart(chart)
            logger.info(
                "Chart '%s': %d points, dim=%d.",
                chart.name,
                n,
                chart.dim,
            )

        # 5. Overlap detection & transition fitting
        charts = atlas.get_all_charts()
        for i in range(len(charts)):
            for j in range(i + 1, len(charts)):
                ci, cj = charts[i], charts[j]
                overlap_idx = self.compute_overlap_regions(ci, cj, knn_graph)
                if overlap_idx is not None and len(overlap_idx) >= 3:
                    # Fit affine transition
                    src_coords = (
                        ci._data_snapshot[overlap_idx]
                        if ci._data_snapshot is not None
                        else None
                    )
                    tgt_coords = (
                        cj._data_snapshot[overlap_idx]
                        if cj._data_snapshot is not None
                        else None
                    )
                    if src_coords is not None and tgt_coords is not None:
                        try:
                            tmap = self.fit_transition_map(
                                overlap_data=data[overlap_idx],
                                source_coords=src_coords,
                                target_coords=tgt_coords,
                                map_type="affine",
                            )
                            tmap.source_chart_id = ci.chart_id
                            tmap.target_chart_id = cj.chart_id
                            ov_bounds = self.compute_overlap_bounds(ci, cj)
                            if ov_bounds is not None:
                                tmap.overlap_region = ov_bounds
                            atlas.add_transition_map(tmap)
                        except Exception as e:
                            logger.warning(
                                "Failed to fit transition '%s' -> '%s': %s",
                                ci.name,
                                cj.name,
                                e,
                            )

        logger.info(
            "=== Atlas build complete ===  %d charts, %d transitions.",
            len(atlas._charts),
            len(atlas._transitions),
        )
        return atlas

    # ------------------------------------------------------------------
    # 7. Quality analysis
    # ------------------------------------------------------------------
    def analyze_quality(
        self,
        atlas: AtlasManager,
        data: np.ndarray,
    ) -> dict[str, Any]:
        """Compute quality metrics for an atlas.

        Parameters
        ----------
        atlas : AtlasManager
            The atlas to evaluate.
        data : ndarray of shape (N, D)

        Returns
        -------
        dict with keys:
        - coverage : float – fraction of data points covered by at least one chart
        - avg_overlap_ratio : float – mean overlap size as fraction of chart size
        - dim_estimates : list[int] – per-chart intrinsic dimensions
        - n_isolated_charts : int – charts with zero transitions
        - n_charts : int
        - n_transitions : int
        """
        data = np.asarray(data, dtype=np.float64)
        N = data.shape[0]
        charts = atlas.get_all_charts()
        transitions = atlas.get_all_transition_maps()

        # Coverage: how many data points are assigned to some chart
        covered = set()
        for chart in charts:
            indices = chart.metadata.get("point_indices", [])
            covered.update(indices)
        coverage = len(covered) / max(N, 1)

        # Overlap ratios
        overlap_ratios: list[float] = []
        transition_chart_ids = set()
        for tmap in transitions:
            src_id = tmap.source_chart_id
            tgt_id = tmap.target_chart_id
            transition_chart_ids.add(src_id)
            transition_chart_ids.add(tgt_id)
            # Estimate overlap from transition overlap_region
            ov = tmap.overlap_region
            if ov is not None:
                mn, mx = ov
                span = mx - mn
                vol = np.prod(span) if np.all(span > 0) else 0.0
                # Normalize by "unit" volume per dim (approximate)
                overlap_ratios.append(min(float(vol), 1.0))

        avg_overlap = float(np.mean(overlap_ratios)) if overlap_ratios else 0.0

        # Isolated charts
        connected_chart_ids = set()
        for tmap in transitions:
            connected_chart_ids.add(tmap.source_chart_id)
            connected_chart_ids.add(tmap.target_chart_id)
        n_isolated = sum(1 for c in charts if c.chart_id not in connected_chart_ids)

        dim_estimates = [c.dim for c in charts]

        metrics = {
            "coverage": coverage,
            "avg_overlap_ratio": avg_overlap,
            "dim_estimates": dim_estimates,
            "n_isolated_charts": n_isolated,
            "n_charts": len(charts),
            "n_transitions": len(transitions),
        }
        logger.info("Atlas quality metrics: %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _tune_resolution(
        self,
        knn_graph: sparse.csr_matrix,
        target_n: int,
        max_iter: int = 20,
    ) -> float:
        """Binary-search for a Louvain resolution yielding ≈ target_n charts."""
        nx, louvain_communities = _import_networkx()
        G = nx.from_scipy_sparse_array(knn_graph)

        lo, hi = 0.01, 10.0
        best_res = 1.0
        for _ in range(max_iter):
            mid = (lo + hi) / 2
            comms = louvain_communities(G, resolution=mid, seed=self.random_state)
            n = len(comms)
            if n < target_n:
                lo = mid
            elif n > target_n:
                hi = mid
            else:
                return mid
            best_res = mid
        logger.info(
            "Resolution tuning: target=%d, got=%d (res=%.3f).",
            target_n,
            n,
            best_res,
        )
        return best_res
