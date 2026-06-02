"""
Atlas Manager - orchestrates charts and transition maps.

Manages the complete atlas structure of the manifold database.  An *atlas*
``A = {(U_α, φ_α)}`` is a collection of charts that cover the manifold, together
with transition maps ``ψ_{αβ}: φ_α(U_α ∩ U_β) → φ_β(U_α ∩ U_β)`` defined on
their pairwise overlaps.

The :class:`AtlasManager` provides CRUD operations for charts and transitions,
automatic atlas construction via manifold-learning techniques, and full
persistence support.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .atlas_builder import AtlasBuilder
from .chart import Chart
from .transition_map import TransitionMap, create_transition_map

logger = logging.getLogger(__name__)


class AtlasManager:
    """
    Central orchestrator for a manifold atlas.

    Parameters
    ----------
    name : str
        Human-readable name for this atlas (used in summaries / logging).
    metadata : dict
        Arbitrary metadata attached to the atlas as a whole.

    Examples
    --------
    >>> mgr = AtlasManager("my_manifold")
    >>> chart = Chart(name="patch_0", dim=2, ambient_dim=10)
    >>> mgr.add_chart(chart)
    """

    def __init__(self, name: str = "default_atlas", metadata: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.metadata: Dict[str, Any] = metadata or {}
        self._charts: Dict[str, Chart] = {}
        self._transitions: Dict[Tuple[str, str], TransitionMap] = {}
        self._builder: Optional[AtlasBuilder] = None
        logger.info("AtlasManager '%s' initialised.", self.name)

    # ------------------------------------------------------------------
    # Chart CRUD
    # ------------------------------------------------------------------
    def add_chart(self, chart: Chart) -> None:
        """Register a new chart.

        Parameters
        ----------
        chart : Chart
            The chart to add.

        Raises
        ------
        ValueError
            If a chart with the same ID already exists.
        """
        if chart.chart_id in self._charts:
            raise ValueError(
                f"Chart with id '{chart.chart_id}' already exists in atlas '{self.name}'."
            )
        self._charts[chart.chart_id] = chart
        logger.info("Added chart '%s' (id=%s) to atlas '%s'.", chart.name, chart.chart_id, self.name)

    def remove_chart(self, chart_id: str) -> None:
        """Remove a chart and all associated transition maps.

        Parameters
        ----------
        chart_id : str
            ID of the chart to remove.

        Raises
        ------
        KeyError
            If the chart ID is not found.
        """
        if chart_id not in self._charts:
            raise KeyError(f"Chart '{chart_id}' not found in atlas '{self.name}'.")
        # Remove transitions that reference this chart
        keys_to_remove = [
            k for k in self._transitions if chart_id in k
        ]
        for k in keys_to_remove:
            del self._transitions[k]
            logger.debug("Removed transition map %s (chart %s removed).", k, chart_id)
        del self._charts[chart_id]
        logger.info("Removed chart '%s' from atlas '%s'.", chart_id, self.name)

    def get_chart(self, chart_id: str) -> Chart:
        """Retrieve a chart by ID.

        Raises
        ------
        KeyError
            If the chart ID is not found.
        """
        if chart_id not in self._charts:
            raise KeyError(f"Chart '{chart_id}' not found in atlas '{self.name}'.")
        return self._charts[chart_id]

    def get_all_charts(self) -> List[Chart]:
        """Return all charts in the atlas."""
        return list(self._charts.values())

    # ------------------------------------------------------------------
    # Transition CRUD
    # ------------------------------------------------------------------
    def add_transition_map(self, tmap: TransitionMap) -> None:
        """Register a transition map between two charts.

        Parameters
        ----------
        tmap : TransitionMap
            The transition map to add.

        Raises
        ------
        ValueError
            If source or target chart does not exist in the atlas.
        ValueError
            If a transition between the same ordered pair already exists.
        """
        src = tmap.source_chart_id
        tgt = tmap.target_chart_id
        if src not in self._charts:
            raise ValueError(f"Source chart '{src}' not found in atlas.")
        if tgt not in self._charts:
            raise ValueError(f"Target chart '{tgt}' not found in atlas.")
        key = (src, tgt)
        if key in self._transitions:
            raise ValueError(f"Transition {src} -> {tgt} already exists.")
        self._transitions[key] = tmap
        logger.info(
            "Added transition %s -> %s (type=%s) to atlas '%s'.",
            src, tgt, type(tmap).__name__, self.name,
        )

    def get_transition(self, source_id: str, target_id: str) -> TransitionMap:
        """Retrieve the transition map from *source_id* to *target_id*.

        Raises
        ------
        KeyError
            If no such transition exists.
        """
        key = (source_id, target_id)
        if key not in self._transitions:
            raise KeyError(
                f"No transition map from '{source_id}' to '{target_id}' in atlas '{self.name}'."
            )
        return self._transitions[key]

    def get_all_transition_maps(self) -> List[TransitionMap]:
        """Return all transition maps in the atlas."""
        return list(self._transitions.values())

    # ------------------------------------------------------------------
    # Chart lookup
    # ------------------------------------------------------------------
    def find_chart(
        self,
        data: np.ndarray,
        modality: Optional[str] = None,
    ) -> Optional[Chart]:
        """Locate the best chart for a given data point.

        Strategy:
        1. If ``modality`` is given, prefer charts whose metadata includes
           a matching ``modality`` key.
        2. Among candidates, embed the data and check containment.  Return
           the chart with the smallest bounding-box volume that contains the
           point.
        3. If no chart contains the point, return the chart whose centroid
           is nearest to the embedded point.

        Parameters
        ----------
        data : ndarray of shape (ambient_dim,) or (N, ambient_dim)
        modality : str or None
            Optional modality filter.

        Returns
        -------
        Chart or None
            Best matching chart, or *None* if the atlas is empty.
        """
        if not self._charts:
            logger.warning("Atlas '%s' is empty; cannot find chart.", self.name)
            return None
        d = np.asarray(data, dtype=np.float64)
        if d.ndim == 1:
            d = d.reshape(1, -1)

        candidates: List[Chart] = []
        if modality is not None:
            candidates = [
                c for c in self._charts.values()
                if c.metadata.get("modality") == modality
            ]
            if not candidates:
                logger.info(
                    "No chart with modality='%s'; searching all charts.", modality
                )
                candidates = list(self._charts.values())
        else:
            candidates = list(self._charts.values())

        # Try containment
        for chart in candidates:
            try:
                coords = chart.embed(d)
                if chart.contains(coords).any():
                    logger.debug("find_chart: hit chart '%s'.", chart.name)
                    return chart
            except (ValueError, RuntimeError):
                continue

        # Fallback: nearest centroid
        best_chart: Optional[Chart] = None
        best_dist = float("inf")
        for chart in candidates:
            try:
                coords = chart.embed(d)
                centroid = np.mean(coords, axis=0)
                dist = float(np.linalg.norm(coords[0] - centroid))
                if dist < best_dist:
                    best_dist = dist
                    best_chart = chart
            except (ValueError, RuntimeError):
                continue
        if best_chart is not None:
            logger.debug(
                "find_chart: fallback nearest-centroid to '%s' (dist=%.4f).",
                best_chart.name, best_dist,
            )
        return best_chart

    # ------------------------------------------------------------------
    # Automatic atlas construction
    # ------------------------------------------------------------------
    def build_atlas(
        self,
        data: np.ndarray,
        modality_labels: Optional[np.ndarray] = None,
        modality: Optional[str] = None,
        **builder_kwargs: Any,
    ) -> None:
        """Automatically construct an atlas from data.

        Delegates to :class:`AtlasBuilder` for the heavy lifting (KNN graph,
        community detection, dimensionality estimation, transition fitting).

        Parameters
        ----------
        data : ndarray of shape (N, ambient_dim)
            Raw data in ambient space.
        modality_labels : ndarray of shape (N,) or None
            Optional per-point modality labels.
        modality : str or None
            Modality tag applied to all discovered charts.
        **builder_kwargs
            Forwarded to :meth:`AtlasBuilder.build`.
        """
        data = np.asarray(data, dtype=np.float64)
        logger.info(
            "Building atlas '%s' from data of shape %s.", self.name, data.shape
        )
        self._builder = AtlasBuilder()
        atlas = self._builder.build(
            data, modality_labels=modality_labels, **builder_kwargs
        )
        # Import charts
        for chart in atlas.get_all_charts():
            if modality is not None:
                chart.metadata["modality"] = modality
            self.add_chart(chart)
        # Import transitions
        for tmap in atlas.get_all_transition_maps():
            self.add_transition_map(tmap)
        logger.info(
            "Atlas build complete: %d charts, %d transitions.",
            len(self._charts), len(self._transitions),
        )

    def rebuild_overlaps(self) -> None:
        """Recompute overlap regions between all pairs of charts.

        Uses the embedded data snapshots stored on each chart to identify
        shared data points, then re-derives axis-aligned overlap bounds and
        refits transition maps where overlap exists.

        Only linear/affine fits are performed here; for neural transitions
        re-run :meth:`build_atlas`.
        """
        logger.info("Rebuilding overlaps for atlas '%s' (%d charts).", self.name, len(self._charts))
        charts = list(self._charts.values())
        if self._builder is None:
            self._builder = AtlasBuilder()
        for i in range(len(charts)):
            for j in range(i + 1, len(charts)):
                ci, cj = charts[i], charts[j]
                key_ij = (ci.chart_id, cj.chart_id)
                key_ji = (cj.chart_id, ci.chart_id)
                if key_ij not in self._transitions and key_ji not in self._transitions:
                    # Try to create a new transition if charts share dimensionality
                    if ci.dim != cj.dim:
                        logger.debug(
                            "Charts '%s' and '%s' have different dims (%d vs %d); skipping.",
                            ci.name, cj.name, ci.dim, cj.dim,
                        )
                        continue
                    tmap = self._builder.fit_transition_map_simple(ci, cj)
                    if tmap is not None:
                        self.add_transition_map(tmap)
                else:
                    existing = self._transitions.get(key_ij) or self._transitions.get(key_ji)
                    if existing is not None:
                        ov = self._builder.compute_overlap_bounds(ci, cj)
                        if ov is not None:
                            existing.overlap_region = ov
                            logger.debug(
                                "Updated overlap region for %s -> %s.",
                                existing.source_chart_id, existing.target_chart_id,
                            )
        logger.info("Overlap rebuild complete.")

    # ------------------------------------------------------------------
    # Summary & introspection
    # ------------------------------------------------------------------
    def atlas_summary(self) -> Dict[str, Any]:
        """Return a dictionary of atlas statistics.

        Includes chart count, transition count, dimension ranges, coverage
        estimates, and per-chart summaries.
        """
        charts = self.get_all_charts()
        transitions = self.get_all_transition_maps()
        dims = [c.dim for c in charts]
        ambient_dims = [c.ambient_dim for c in charts]
        return {
            "name": self.name,
            "n_charts": len(charts),
            "n_transitions": len(transitions),
            "dim_range": (min(dims), max(dims)) if dims else (0, 0),
            "ambient_dim_range": (min(ambient_dims), max(ambient_dims)) if ambient_dims else (0, 0),
            "chart_ids": [c.chart_id for c in charts],
            "transition_pairs": [(t.source_chart_id, t.target_chart_id) for t in transitions],
            "charts": [c.summary() for c in charts],
            "metadata": self.metadata,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def serialize(self) -> Dict[str, Any]:
        """Serialize the entire atlas to a JSON-friendly dictionary."""
        charts_serial = [c.to_dict() for c in self._charts.values()]
        transitions_serial = [t.to_dict() for t in self._transitions.values()]
        return {
            "name": self.name,
            "metadata": self.metadata,
            "charts": charts_serial,
            "transitions": transitions_serial,
        }

    def deserialize(self, d: Dict[str, Any]) -> None:
        """Restore atlas state from a serialised dictionary.

        Clears any existing charts/transitions before loading.

        Parameters
        ----------
        d : dict
            Output of :meth:`serialize`.
        """
        self._charts.clear()
        self._transitions.clear()
        self.name = d.get("name", "restored_atlas")
        self.metadata = d.get("metadata", {})
        for cd in d.get("charts", []):
            chart = Chart.from_dict(cd)
            self._charts[chart.chart_id] = chart
        for td in d.get("transitions", []):
            tmap = create_transition_map(td)
            self._transitions[(tmap.source_chart_id, tmap.target_chart_id)] = tmap
        logger.info(
            "Deserialized atlas '%s': %d charts, %d transitions.",
            self.name, len(self._charts), len(self._transitions),
        )

    def save(self, filepath: str) -> None:
        """Write atlas to a JSON file.

        Parameters
        ----------
        filepath : str
            Destination path.
        """
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.serialize(), f, indent=2)
        logger.info("Atlas saved to %s.", filepath)

    def load(self, filepath: str) -> None:
        """Load atlas from a JSON file.

        Parameters
        ----------
        filepath : str
            Source path.
        """
        with open(filepath, "r") as f:
            d = json.load(f)
        self.deserialize(d)
        logger.info("Atlas loaded from %s.", filepath)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._charts)

    def __repr__(self) -> str:
        return (
            f"AtlasManager(name={self.name!r}, "
            f"n_charts={len(self._charts)}, "
            f"n_transitions={len(self._transitions)})"
        )
