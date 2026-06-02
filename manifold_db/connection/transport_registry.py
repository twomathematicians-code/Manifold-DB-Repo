"""
Transport Registry — manages and caches parallel transport computations.

The registry maintains an LRU cache of transport results, allows
registration of custom transport functions between chart pairs,
supports chain composition of multiple transports, and can precompute
a cost heatmap for chart-pair transports.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Transport Registry
# ---------------------------------------------------------------------------


class TransportRegistry:
    """Registry and cache for parallel transport computations.

    Maintains a mapping from (source_chart, target_chart) pairs to
    transport functions, and an LRU cache for computed transport results.

    Parameters
    ----------
    max_cache_size : int
        Maximum number of cached transport results (default 256).
    """

    def __init__(self, max_cache_size: int = 256) -> None:
        self.max_cache_size = max_cache_size
        # Chart-pair → transport function
        self._transports: dict[tuple[str, str], Callable[..., np.ndarray]] = {}
        # LRU cache: key → (result, metadata)
        self._cache: OrderedDict[str, tuple[np.ndarray, dict[str, Any]]] = OrderedDict()
        # Precomputed cost heatmap
        self._heatmap: np.ndarray | None = None
        self._heatmap_chart_ids: list[str] | None = None

    # ---- registration -----------------------------------------------------

    def register_transport(
        self,
        source_chart: str,
        target_chart: str,
        transport_fn: Callable[..., np.ndarray],
    ) -> None:
        """Register a transport function for a chart pair.

        Parameters
        ----------
        source_chart : str
        target_chart : str
        transport_fn : callable
            Function that performs the parallel transport.  Should accept
            a vector (and optionally other arguments) and return the
            transported vector.
        """
        key = (source_chart, target_chart)
        self._transports[key] = transport_fn
        # Invalidate cache entries involving either chart
        self.invalidate(source_chart)
        self.invalidate(target_chart)
        logger.info("Registered transport %s → %s", source_chart, target_chart)

    def get_transport(
        self,
        source_chart: str,
        target_chart: str,
    ) -> Callable[..., np.ndarray] | None:
        """Retrieve the transport function for a chart pair.

        Returns ``None`` if no transport is registered.
        """
        return self._transports.get((source_chart, target_chart))

    def has_transport(self, source_chart: str, target_chart: str) -> bool:
        """Check whether a transport function is registered for the pair."""
        return (source_chart, target_chart) in self._transports

    def list_transports(self) -> list[tuple[str, str]]:
        """List all registered (source, target) chart pairs."""
        return list(self._transports.keys())

    # ---- cache management -------------------------------------------------

    def _make_cache_key(
        self,
        source_chart: str,
        target_chart: str,
        vector: np.ndarray,
        **kwargs: Any,
    ) -> str:
        """Create a deterministic cache key from transport parameters."""
        vec_bytes = vector.tobytes()
        kw_bytes = json.dumps(kwargs, sort_keys=True, default=str).encode()
        raw = f"{source_chart}|{target_chart}|".encode() + vec_bytes + b"|" + kw_bytes
        return hashlib.sha256(raw).hexdigest()

    def get_cached(
        self,
        source_chart: str,
        target_chart: str,
        vector: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray | None:
        """Look up a cached transport result.

        Returns ``None`` on cache miss.
        """
        key = self._make_cache_key(source_chart, target_chart, vector, **kwargs)
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            result, meta = self._cache[key]
            logger.debug("Cache hit for transport %s → %s", source_chart, target_chart)
            return result
        return None

    def put_cached(
        self,
        source_chart: str,
        target_chart: str,
        vector: np.ndarray,
        result: np.ndarray,
        **kwargs: Any,
    ) -> None:
        """Store a transport result in the cache."""
        key = self._make_cache_key(source_chart, target_chart, vector, **kwargs)
        self._cache[key] = (
            result.copy(),
            {
                "source_chart": source_chart,
                "target_chart": target_chart,
            },
        )
        # Evict oldest if over capacity
        while len(self._cache) > self.max_cache_size:
            self._cache.popitem(last=False)
        logger.debug(
            "Cached transport result for %s → %s (cache size=%d)",
            source_chart,
            target_chart,
            len(self._cache),
        )

    # ---- chain composition ------------------------------------------------

    def compute_chain(
        self,
        transport_chain: list[tuple[str, str]],
        vector: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compose multiple transports along a chain of chart pairs.

        Parameters
        ----------
        transport_chain : list of (source, target) chart-pair tuples
            Ordered list defining the transport chain.
        vector : (n,)
            Initial vector to transport.
        **kwargs :
            Additional arguments forwarded to each transport function.

        Returns
        -------
        np.ndarray — the vector after sequential transport through all pairs.

        Raises
        ------
        ValueError
            If any chart pair in the chain is not registered.
        """
        v = np.asarray(vector, dtype=np.float64).copy()
        for i, (src, tgt) in enumerate(transport_chain):
            fn = self._transports.get((src, tgt))
            if fn is None:
                raise ValueError(
                    f"No transport registered for chart pair ({src}, {tgt}) "
                    f"at chain position {i}"
                )
            v = fn(v, **kwargs)
            logger.debug(
                "Chain step %d/%d: %s → %s", i + 1, len(transport_chain), src, tgt
            )
        return v

    # ---- invalidation -----------------------------------------------------

    def invalidate(self, chart_id: str) -> int:
        """Invalidate all cached transports involving *chart_id*.

        Returns the number of entries removed.
        """
        to_remove = [
            key
            for key, (_, meta) in self._cache.items()
            if meta["source_chart"] == chart_id or meta["target_chart"] == chart_id
        ]
        for key in to_remove:
            del self._cache[key]
        logger.info(
            "Invalidated %d cache entries for chart %s", len(to_remove), chart_id
        )
        return len(to_remove)

    def clear_cache(self) -> None:
        """Clear the entire transport cache."""
        self._cache.clear()
        logger.info("Transport cache cleared")

    # ---- heatmap precomputation -------------------------------------------

    def precompute_heatmap(
        self,
        chart_ids: list[str],
        sample_vectors: np.ndarray | None = None,
        n_samples: int = 10,
    ) -> np.ndarray:
        """Precompute a transport cost heatmap between chart pairs.

        For each ordered pair (i, j) of charts, the "cost" is the average
        Frobenius-norm change of randomly sampled vectors after transport.
        If a transport function is not registered, the cost is ``inf``.

        Parameters
        ----------
        chart_ids : list of str
            Charts to include in the heatmap.
        sample_vectors : (n_samples, dim) or None
            Vectors to use for cost estimation.  If None, random unit vectors
            are generated.
        n_samples : int
            Number of random vectors if *sample_vectors* is None.

        Returns
        -------
        np.ndarray, shape (len(chart_ids), len(chart_ids)) — cost matrix.
        """
        n_charts = len(chart_ids)
        n_dim = 3  # default dimension for random vectors
        heatmap = np.full((n_charts, n_charts), np.inf, dtype=np.float64)

        if sample_vectors is not None:
            vectors = np.asarray(sample_vectors, dtype=np.float64)
            n_dim = vectors.shape[1]
        else:
            # Random unit vectors
            vectors = np.random.randn(n_samples, n_dim)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-14)
            vectors = vectors / norms

        for i, src in enumerate(chart_ids):
            for j, tgt in enumerate(chart_ids):
                if i == j:
                    heatmap[i, j] = 0.0
                    continue
                fn = self._transports.get((src, tgt))
                if fn is None:
                    heatmap[i, j] = np.inf
                    continue
                # Average cost over sample vectors
                total_cost = 0.0
                count = 0
                for k in range(len(vectors)):
                    try:
                        transported = fn(vectors[k])
                        cost = float(np.linalg.norm(transported - vectors[k]))
                        total_cost += cost
                        count += 1
                    except Exception:
                        heatmap[i, j] = np.inf
                        break
                if count > 0 and heatmap[i, j] != np.inf:
                    heatmap[i, j] = total_cost / count

        self._heatmap = heatmap
        self._heatmap_chart_ids = chart_ids
        logger.info(
            "Precomputed heatmap for %d charts (%.0f%% reachable)",
            n_charts,
            100.0 * np.sum(np.isfinite(heatmap)) / heatmap.size,
        )
        return heatmap

    def get_heatmap(self) -> tuple[np.ndarray, list[str]] | None:
        """Return the precomputed heatmap and chart labels, or None."""
        if self._heatmap is None or self._heatmap_chart_ids is None:
            return None
        return self._heatmap, self._heatmap_chart_ids

    def get_cost(self, source_chart: str, target_chart: str) -> float:
        """Look up the precomputed transport cost between two charts."""
        if self._heatmap is None or self._heatmap_chart_ids is None:
            return np.inf
        try:
            i = self._heatmap_chart_ids.index(source_chart)
            j = self._heatmap_chart_ids.index(target_chart)
            return float(self._heatmap[i, j])
        except ValueError:
            return np.inf

    # ---- serialization ----------------------------------------------------

    def serialize(self) -> dict[str, Any]:
        """Serialize registry state (excludes non-serialisable transport functions)."""
        data: dict[str, Any] = {
            "max_cache_size": self.max_cache_size,
            "transport_pairs": list(self._transports.keys()),
            "cache_size": len(self._cache),
        }
        if self._heatmap is not None and self._heatmap_chart_ids is not None:
            data["heatmap"] = self._heatmap.tolist()
            data["heatmap_chart_ids"] = self._heatmap_chart_ids
        return data

    def deserialize(self, data: dict[str, Any]) -> None:
        """Restore registry metadata from serialized data.

        Note: Transport functions must be re-registered manually.
        """
        self.max_cache_size = data.get("max_cache_size", 256)
        if "heatmap" in data and "heatmap_chart_ids" in data:
            self._heatmap = np.array(data["heatmap"], dtype=np.float64)
            self._heatmap_chart_ids = data["heatmap_chart_ids"]
        logger.info(
            "Deserialized transport registry: %d pairs known, heatmap=%s",
            len(data.get("transport_pairs", [])),
            "yes" if self._heatmap is not None else "no",
        )
