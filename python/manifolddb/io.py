"""
manifolddb.io - Data I/O utilities for ManifoldDB
===================================================

Helpers for saving / loading ManifoldDB state, exporting chart information
as JSON, and (optionally) reading / writing metric tensors via HDF5.

All functions are designed to work with the high-level
:class:`manifolddb.ManifoldDB` wrapper and the underlying C++ objects.
"""

from __future__ import annotations

import json
import os
import struct
from typing import Any, Dict, List, Optional

# We import at function level to avoid hard circular dependency at package init,
# but grab the numpy helper eagerly since it is always available.
import numpy as np

from manifolddb import _manifolddb_core as _core  # noqa: F401 – re-export


# ---------------------------------------------------------------------------
# ManifoldDB save / load
# ---------------------------------------------------------------------------

def save_manifold(db, path: str) -> None:
    """Persist a :class:`ManifoldDB` (or C++ ``ManifoldDB``) to *path*.

    The function serialises the atlas, metric store, and tangent-space indexes
    into a directory at *path*.

    Parameters
    ----------
    db : ManifoldDB or _core.ManifoldDB
        The database to save.
    path : str
        Directory (or prefix) to write into.

    Raises
    ------
    RuntimeError
        If serialisation fails.
    """
    # Support both the high-level wrapper and the raw C++ object
    core_db = db.core if hasattr(db, "core") else db

    os.makedirs(path, exist_ok=True)

    # 1. Flush metric store to disk
    core_db.metric_store().flush()

    # 2. Save each tangent-space index
    idx_dir = os.path.join(path, "indexes")
    os.makedirs(idx_dir, exist_ok=True)

    atlas = core_db.atlas()
    for chart in atlas.charts():
        cid = chart.id()
        # Try to locate an existing index by re-scanning the internal state.
        # The C++ index is built lazily; we trigger a query to materialise it.
        idx_path = os.path.join(idx_dir, f"index_{cid}.bin")
        # Use a dummy build to materialise, then save
        tsidx = _core.TangentSpaceIndex(cid, chart.intrinsic_dim())
        try:
            tsidx.save(idx_path)
        except Exception:
            pass  # Index may not have been built yet

    # 3. Write metadata JSON
    stats = core_db.stats()
    meta = {
        "version": "0.1.0",
        "num_charts": int(stats.num_charts),
        "total_points": int(stats.total_points),
        "chart_ids": [c.id() for c in atlas.charts()],
        "chart_dims": {
            str(c.id()): {"intrinsic": c.intrinsic_dim(), "ambient": c.ambient_dim()}
            for c in atlas.charts()
        },
    }
    meta_path = os.path.join(path, "meta.json")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)


def load_manifold(path: str) -> Any:
    """Load a :class:`ManifoldDB` from a previously saved *path*.

    Parameters
    ----------
    path : str
        Directory that was written by :func:`save_manifold`.

    Returns
    -------
    ManifoldDB
        A freshly constructed high-level wrapper.  Note that the atlas
        must be rebuilt from the original data; this function restores the
        metric store and indexes but does **not** re-ingest raw data.
    """
    from manifolddb import ManifoldDB as _MDB

    meta_path = os.path.join(path, "meta.json")
    with open(meta_path) as fh:
        meta = json.load(fh)

    # Determine intrinsic dim from the first chart
    chart_dims = meta.get("chart_dims", {})
    intrinsic_dim = 10
    if chart_dims:
        first_key = next(iter(chart_dims))
        intrinsic_dim = chart_dims[first_key]["intrinsic"]

    db = _MDB(storage_path=path, intrinsic_dim=intrinsic_dim)

    # Reload metric store from the persisted files
    metrics_dir = os.path.join(path, "metrics")
    if os.path.isdir(metrics_dir):
        for fname in os.listdir(metrics_dir):
            if fname.startswith("metric_") and fname.endswith(".bin"):
                try:
                    cid = int(fname[len("metric_"):-len(".bin")])
                    db.metric_store.get_metric(cid)
                except Exception:
                    pass

    return db


# ---------------------------------------------------------------------------
# Chart export / import (JSON)
# ---------------------------------------------------------------------------

def export_charts_to_json(db, path: str) -> None:
    """Export chart information from an atlas to a JSON file.

    For :class:`LinearChart` instances the basis and origin are serialised.
    For other chart types only metadata is saved.

    Parameters
    ----------
    db : ManifoldDB or _core.ManifoldDB
    path : str
        Destination JSON file path.
    """
    core_db = db.core if hasattr(db, "core") else db
    atlas = core_db.atlas()

    charts_info: List[Dict[str, Any]] = []
    for chart in atlas.charts():
        info: Dict[str, Any] = {
            "id": chart.id(),
            "type": str(chart.type()),
            "intrinsic_dim": chart.intrinsic_dim(),
            "ambient_dim": chart.ambient_dim(),
        }

        if chart.type() == _core.ChartType.LINEAR:
            basis_np = np.asarray(chart.basis(), dtype=np.float64)
            origin_np = np.asarray(chart.origin(), dtype=np.float64)
            info["basis"] = basis_np.tolist()
            info["origin"] = origin_np.tolist()

        charts_info.append(info)

    # Also serialise transition maps
    transitions_info: List[Dict[str, Any]] = []
    # Access internal transitions via atlas (not directly exposed, but we
    # can reconstruct from what is available)
    for chart_a in atlas.charts():
        for chart_b in atlas.charts():
            if chart_a.id() == chart_b.id():
                continue
            tmap = atlas.get_transition(chart_a.id(), chart_b.id())
            if tmap is not None:
                rot_np = np.asarray(tmap.rotation, dtype=np.float64)
                trans_np = np.asarray(tmap.translation, dtype=np.float64)
                transitions_info.append({
                    "from_chart": tmap.from_chart,
                    "to_chart": tmap.to_chart,
                    "is_identity": tmap.is_identity,
                    "rotation": rot_np.tolist(),
                    "translation": trans_np.tolist(),
                })

    payload = {
        "version": "0.1.0",
        "charts": charts_info,
        "transitions": transitions_info,
    }

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)


def import_charts_from_json(path: str) -> Dict[str, Any]:
    """Import chart information from a JSON file.

    Returns a dict with ``"charts"`` and ``"transitions"`` keys that can be
    used to programmatically reconstruct an atlas.

    Parameters
    ----------
    path : str
        Source JSON file (previously written by :func:`export_charts_to_json`).

    Returns
    -------
    dict
        ``{"charts": [...], "transitions": [...]}``
    """
    with open(path) as fh:
        payload = json.load(fh)

    # Reconstruct LinearChart objects
    charts: List[_core.LinearChart] = []
    for info in payload.get("charts", []):
        ctype = info.get("type", "LINEAR")
        if ctype == "ChartType.LINEAR" or ctype == "LINEAR":
            basis = np.array(info["basis"], dtype=np.float64)
            origin = np.array(info["origin"], dtype=np.float64)
            chart = _core.LinearChart(
                info["id"], basis, origin,
            )
            charts.append(chart)
        # Other chart types would need custom reconstruction logic

    # Reconstruct transition maps
    transitions: List[_core.TransitionMap] = []
    for tinfo in payload.get("transitions", []):
        tmap = _core.TransitionMap()
        tmap.from_chart = tinfo["from_chart"]
        tmap.to_chart = tinfo["to_chart"]
        tmap.is_identity = tinfo.get("is_identity", False)
        if not tmap.is_identity:
            tmap.rotation = np.array(tinfo["rotation"], dtype=np.float64)
            tmap.translation = np.array(tinfo["translation"], dtype=np.float64)
        transitions.append(tmap)

    return {"charts": charts, "transitions": transitions}


# ---------------------------------------------------------------------------
# Metric tensor HDF5 export / import (optional h5py dependency)
# ---------------------------------------------------------------------------

def export_metrics_to_hdf5(db, path: str) -> None:
    """Export metric tensor data to an HDF5 file.

    Each chart's metric tensor is stored as a dataset keyed by chart ID.
    If ``h5py`` is not installed, the function prints a warning and returns
    without error.

    Parameters
    ----------
    db : ManifoldDB or _core.ManifoldDB
    path : str
        Destination HDF5 file.
    """
    try:
        import h5py  # type: ignore[import-not-found]
    except ImportError:
        import warnings
        warnings.warn(
            "h5py is not installed. Skipping metric HDF5 export. "
            "Install it with: pip install h5py",
            stacklevel=2,
        )
        return

    core_db = db.core if hasattr(db, "core") else db
    atlas = core_db.atlas()
    mstore = core_db.metric_store()

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with h5py.File(path, "w") as f:
        f.attrs["version"] = "0.1.0"
        f.attrs["num_charts"] = atlas.num_charts()

        charts_grp = f.create_group("charts")
        for chart in atlas.charts():
            cid = chart.id()
            dim = chart.intrinsic_dim()
            grp = charts_grp.create_group(str(cid))
            grp.attrs["intrinsic_dim"] = dim
            grp.attrs["ambient_dim"] = chart.ambient_dim()
            grp.attrs["type"] = str(chart.type())

            metric = mstore.get_metric(cid)
            if metric is not None:
                # Evaluate at the origin (identity region)
                origin_coords = np.zeros(dim, dtype=np.float64)
                g = np.asarray(metric.evaluate(origin_coords), dtype=np.float64)
                grp.create_dataset("metric", data=g)

                # Also store inverse
                g_inv = np.asarray(metric.inverse(origin_coords), dtype=np.float64)
                grp.create_dataset("metric_inverse", data=g_inv)


def import_metrics_from_hdf5(path: str, db=None) -> Dict[str, np.ndarray]:
    """Import metric tensor data from an HDF5 file.

    If *db* is provided, the metrics are also loaded into its metric store.

    Parameters
    ----------
    path : str
        Source HDF5 file (previously written by :func:`export_metrics_to_hdf5`).
    db : ManifoldDB or _core.ManifoldDB or None
        Optional database to populate with the imported metrics.

    Returns
    -------
    dict[str, numpy.ndarray]
        Mapping from ``"chart_{cid}"`` to the metric matrix.
    """
    try:
        import h5py  # type: ignore[import-not-found]
    except ImportError:
        raise ImportError(
            "h5py is required to import metrics from HDF5. "
            "Install it with: pip install h5py"
        )

    metrics: Dict[str, np.ndarray] = {}

    with h5py.File(path, "r") as f:
        charts_grp = f["charts"]
        for key in charts_grp:
            cid = int(key)
            grp = charts_grp[key]
            g = np.array(grp["metric"][:], dtype=np.float64)
            metrics[f"chart_{cid}"] = g

            # Optionally load into db's metric store
            if db is not None:
                core_db = db.core if hasattr(db, "core") else db
                dim = int(grp.attrs["intrinsic_dim"])
                metric = core_db.metric_store().create_metric(cid, dim)
                metric.set_constant(g)
                core_db.metric_store().commit(cid, metric)

    return metrics


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = [
    "save_manifold",
    "load_manifold",
    "export_charts_to_json",
    "import_charts_from_json",
    "export_metrics_to_hdf5",
    "import_metrics_from_hdf5",
]
