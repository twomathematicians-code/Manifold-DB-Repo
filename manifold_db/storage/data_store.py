"""
Data Store - manages raw data points with metadata, supporting insert,
search, filter, and bulk import/export operations.

Each DataPoint carries an id, vector embedding, metadata dict,
modality tag, chart assignment, and insertion timestamp.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from manifold_db.storage.backend import MemoryStorage, StorageBackend

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# DataPoint
# ──────────────────────────────────────────────────────────────


@dataclass
class DataPoint:
    """
    A single data point in the manifold database.

    Attributes:
        id:        Unique identifier string.
        vector:    Numeric embedding (1-D numpy array).
        metadata:  Arbitrary key-value metadata dict.
        modality:  Modality tag (e.g. "text", "image", "audio").
        chart_id:  ID of the chart this point belongs to (None if unassigned).
        timestamp: Unix timestamp of insertion.
    """

    id: str
    vector: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    modality: str = "default"
    chart_id: str | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "vector": self.vector.tolist(),
            "metadata": self.metadata,
            "modality": self.modality,
            "chart_id": self.chart_id,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DataPoint:
        vec = d.get("vector", [])
        if not isinstance(vec, np.ndarray):
            vec = np.asarray(vec, dtype=np.float64)
        return cls(
            id=str(d["id"]),
            vector=vec,
            metadata=d.get("metadata", {}),
            modality=d.get("modality", "default"),
            chart_id=d.get("chart_id"),
            timestamp=d.get("timestamp", time.time()),
        )

    @property
    def dimension(self) -> int:
        return self.vector.size


# ──────────────────────────────────────────────────────────────
# Data Store
# ──────────────────────────────────────────────────────────────


class DataStore:
    """
    High-level data store for manifold data points.

    Wraps a StorageBackend with an in-memory index for fast search
    and filtering by modality / chart.

    Supports:
        - insert / batch_insert
        - get / get_batch
        - delete / update
        - search (nearest neighbours via cosine / euclidean)
        - filter by modality / chart
        - stats
        - export / import (JSON, CSV, parquet)
    """

    _KEY_PREFIX = "dp:"

    def __init__(
        self,
        backend: StorageBackend | None = None,
        vector_dimension: int | None = None,
    ) -> None:
        if backend is None:
            backend = MemoryStorage()
        self._backend = backend
        self._vector_dimension = vector_dimension
        # In-memory index: chart_id → {point_id: vector}
        self._chart_index: dict[str, dict[str, np.ndarray]] = {}
        # Modality index: modality → set of point_ids
        self._modality_index: dict[str, set] = {}
        # Full vector matrix for fast batch search (populated lazily)
        self._all_vectors: np.ndarray | None = None
        self._all_ids: list[str] = []
        self._dirty = True
        self._lock = asyncio.Lock()
        logger.info("DataStore initialised (backend=%s)", type(backend).__name__)

    # ── insert / update / delete ──────────────────────────────

    async def insert(self, data_point: DataPoint) -> None:
        """Insert a single data point."""
        async with self._lock:
            self._validate_vector(data_point.vector)
            key = f"{self._KEY_PREFIX}{data_point.id}"
            await self._backend.put(key, data_point.to_dict())
            self._add_to_indices(data_point)
            self._dirty = True
        logger.debug("Inserted data point %s", data_point.id)

    async def batch_insert(self, data_points: Sequence[DataPoint]) -> int:
        """Insert multiple data points. Returns number inserted."""
        if not data_points:
            return 0
        items: dict[str, Any] = {}
        for dp in data_points:
            self._validate_vector(dp.vector)
            key = f"{self._KEY_PREFIX}{dp.id}"
            items[key] = dp.to_dict()
            self._add_to_indices(dp)
        async with self._lock:
            await self._backend.put_batch(items)
            self._dirty = True
        logger.info("Batch inserted %d data points", len(data_points))
        return len(data_points)

    async def get(self, point_id: str) -> DataPoint | None:
        """Retrieve a single data point by id."""
        key = f"{self._KEY_PREFIX}{point_id}"
        data = await self._backend.get(key)
        if data is None:
            return None
        return DataPoint.from_dict(data)

    async def get_batch(self, point_ids: Sequence[str]) -> list[DataPoint]:
        """Retrieve multiple data points by ids."""
        if not point_ids:
            return []
        keys = [f"{self._KEY_PREFIX}{pid}" for pid in point_ids]
        data_map = await self._backend.get_batch(keys)
        results: list[DataPoint] = []
        for pid in point_ids:
            key = f"{self._KEY_PREFIX}{pid}"
            if key in data_map:
                results.append(DataPoint.from_dict(data_map[key]))
        return results

    async def delete(self, point_id: str) -> bool:
        """Delete a data point. Returns True if it existed."""
        async with self._lock:
            key = f"{self._KEY_PREFIX}{point_id}"
            data = await self._backend.get(key)
            if data is None:
                return False
            await self._backend.delete(key)
            self._remove_from_indices(point_id, data)
            self._dirty = True
            return True

    async def update(self, point_id: str, new_data: dict[str, Any]) -> DataPoint | None:
        """Update fields of an existing data point."""
        async with self._lock:
            key = f"{self._KEY_PREFIX}{point_id}"
            existing = await self._backend.get(key)
            if existing is None:
                return None
            # Remove from indices (will re-add)
            self._remove_from_indices(point_id, existing)
            # Apply updates
            for k, v in new_data.items():
                if k == "vector":
                    v = np.asarray(v, dtype=np.float64)
                    self._validate_vector(v)
                existing[k] = v
            await self._backend.put(key, existing)
            dp = DataPoint.from_dict(existing)
            self._add_to_indices(dp)
            self._dirty = True
            return dp

    # ── search ────────────────────────────────────────────────

    async def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        chart_id: str | None = None,
        metric: str = "cosine",
    ) -> list[tuple]:
        """
        Search for the k nearest neighbours of query_vector.

        Returns list of (point_id, distance) tuples sorted by distance.
        """
        query_vector = np.asarray(query_vector, dtype=np.float64)
        candidates = self._get_candidate_ids(chart_id)

        if not candidates:
            return []

        # Build matrix of candidate vectors
        vectors = []
        ids = []
        for pid in candidates:
            vec = None
            # Check in-memory chart index first
            for chart_idx in self._chart_index.values():
                if pid in chart_idx:
                    vec = chart_idx[pid]
                    break
            if vec is not None:
                vectors.append(vec)
                ids.append(pid)

        if not vectors:
            return []

        matrix = np.vstack(vectors)  # shape (n, d)
        distances = self._compute_distances(query_vector, matrix, metric)

        # Sort and take top-k
        order = np.argsort(distances)[:k]
        return [(ids[i], float(distances[i])) for i in order]

    # ── filters ───────────────────────────────────────────────

    async def get_by_modality(self, modality: str) -> list[DataPoint]:
        """Get all data points matching a modality."""
        point_ids = self._modality_index.get(modality, set())
        if not point_ids:
            return []
        return await self.get_batch(list(point_ids))

    async def get_by_chart(self, chart_id: str) -> list[DataPoint]:
        """Get all data points belonging to a chart."""
        chart_vectors = self._chart_index.get(chart_id, {})
        if not chart_vectors:
            return []
        return await self.get_batch(list(chart_vectors.keys()))

    # ── stats ────────────────────────────────────────────────

    async def stats(self) -> dict[str, Any]:
        """Return summary statistics of the data store."""
        all_ids = await self._backend.list_keys(prefix=self._KEY_PREFIX)
        chart_dist: dict[str, int] = {
            cid: len(vecs) for cid, vecs in self._chart_index.items()
        }
        modality_dist: dict[str, int] = {
            m: len(s) for m, s in self._modality_index.items()
        }
        return {
            "total_points": len(all_ids),
            "modalities": modality_dist,
            "chart_distribution": chart_dist,
            "vector_dimension": self._vector_dimension,
            "modalities_list": list(self._modality_index.keys()),
            "charts_list": list(self._chart_index.keys()),
        }

    # ── export / import ───────────────────────────────────────

    async def export(self, format: str = "json", path: str | None = None) -> Any:
        """
        Export all data points. Returns export location or data.

        Formats: "json", "csv", "parquet" (requires pyarrow).
        """
        all_ids = await self._backend.list_keys(prefix=self._KEY_PREFIX)
        points = await self.get_batch(all_ids)
        records = [dp.to_dict() for dp in points]

        if format == "json":
            out_path = path or "manifold_export.json"
            with open(out_path, "w") as f:
                json.dump(records, f, indent=2, default=self._json_default)
            logger.info("Exported %d points to %s", len(records), out_path)
            return out_path

        elif format == "csv":
            out_path = path or "manifold_export.csv"
            if not records:
                Path(out_path).touch()
                return out_path
            with open(out_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=records[0].keys())
                writer.writeheader()
                for r in records:
                    row = dict(r)
                    row["vector"] = json.dumps(row["vector"])
                    row["metadata"] = json.dumps(row["metadata"])
                    writer.writerow(row)
            logger.info("Exported %d points to %s", len(records), out_path)
            return out_path

        elif format == "parquet":
            try:
                import pyarrow as pa  # type: ignore
                import pyarrow.parquet as pq  # type: ignore
            except ImportError:
                raise RuntimeError(
                    "pyarrow is required for parquet export. "
                    "Install it with: pip install pyarrow"
                )
            # Flatten vectors and metadata
            table_data: dict[str, list] = {
                "id": [],
                "vector": [],
                "metadata": [],
                "modality": [],
                "chart_id": [],
                "timestamp": [],
            }
            for r in records:
                table_data["id"].append(r["id"])
                table_data["vector"].append(r["vector"])
                table_data["metadata"].append(json.dumps(r["metadata"]))
                table_data["modality"].append(r["modality"])
                table_data["chart_id"].append(r.get("chart_id", ""))
                table_data["timestamp"].append(r["timestamp"])

            table = pa.table(table_data)
            out_path = path or "manifold_export.parquet"
            pq.write_table(table, out_path)
            logger.info("Exported %d points to %s", len(records), out_path)
            return out_path

        else:
            raise ValueError(f"Unsupported export format: {format}")

    async def import_data(
        self,
        file_path: str | Path,
        format: str = "json",
    ) -> int:
        """
        Bulk import data points from a file.

        Formats: "json", "csv", "parquet".
        Returns number of points imported.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Import file not found: {file_path}")

        points: list[DataPoint] = []

        if format == "json":
            with open(file_path) as f:
                records = json.load(f)
            for r in records:
                points.append(DataPoint.from_dict(r))

        elif format == "csv":
            with open(file_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row["vector"] = json.loads(row.get("vector", "[]"))
                    row["metadata"] = json.loads(row.get("metadata", "{}"))
                    points.append(DataPoint.from_dict(row))

        elif format == "parquet":
            try:
                import pyarrow.parquet as pq  # type: ignore
            except ImportError:
                raise RuntimeError("pyarrow required for parquet import.")
            table = pq.read_table(str(file_path))
            for i in range(table.num_rows):
                row = {col: table.column(col)[i].as_py() for col in table.column_names}
                if isinstance(row.get("vector"), str):
                    row["vector"] = json.loads(row["vector"])
                if isinstance(row.get("metadata"), str):
                    row["metadata"] = json.loads(row["metadata"])
                points.append(DataPoint.from_dict(row))

        else:
            raise ValueError(f"Unsupported import format: {format}")

        count = await self.batch_insert(points)
        logger.info("Imported %d points from %s", count, file_path)
        return count

    # ── index management (internal) ───────────────────────────

    def _add_to_indices(self, dp: DataPoint) -> None:
        """Add data point to in-memory indices."""
        if dp.chart_id:
            if dp.chart_id not in self._chart_index:
                self._chart_index[dp.chart_id] = {}
            self._chart_index[dp.chart_id][dp.id] = dp.vector
        if dp.modality not in self._modality_index:
            self._modality_index[dp.modality] = set()
        self._modality_index[dp.modality].add(dp.id)

    def _remove_from_indices(self, point_id: str, data: dict[str, Any]) -> None:
        """Remove data point from in-memory indices."""
        chart_id = data.get("chart_id")
        if chart_id and chart_id in self._chart_index:
            self._chart_index[chart_id].pop(point_id, None)
            if not self._chart_index[chart_id]:
                del self._chart_index[chart_id]
        modality = data.get("modality", "default")
        if modality in self._modality_index:
            self._modality_index[modality].discard(point_id)
            if not self._modality_index[modality]:
                del self._modality_index[modality]

    def _get_candidate_ids(self, chart_id: str | None = None) -> list[str]:
        """Get candidate point IDs for search."""
        if chart_id:
            return list(self._chart_index.get(chart_id, {}).keys())
        # All points from all charts
        ids: list[str] = []
        for chart_vecs in self._chart_index.values():
            ids.extend(chart_vecs.keys())
        return ids

    # ── distance computation ──────────────────────────────────

    @staticmethod
    def _compute_distances(
        query: np.ndarray, matrix: np.ndarray, metric: str
    ) -> np.ndarray:
        """Compute distances between query and each row of matrix."""
        if metric == "cosine":
            query_norm = np.linalg.norm(query)
            if query_norm == 0:
                return np.ones(matrix.shape[0])
            matrix_norms = np.linalg.norm(matrix, axis=1)
            # Avoid division by zero
            safe_norms = np.where(matrix_norms == 0, 1.0, matrix_norms)
            dots = matrix @ query
            cos_sims = dots / (safe_norms * query_norm)
            return 1.0 - cos_sims
        elif metric == "euclidean":
            diff = matrix - query[np.newaxis, :]
            return np.linalg.norm(diff, axis=1)
        elif metric == "manhattan":
            diff = matrix - query[np.newaxis, :]
            return np.sum(np.abs(diff), axis=1)
        else:
            # Default: euclidean
            diff = matrix - query[np.newaxis, :]
            return np.linalg.norm(diff, axis=1)

    # ── validation ────────────────────────────────────────────

    def _validate_vector(self, vector: np.ndarray) -> None:
        if not isinstance(vector, np.ndarray):
            vector = np.asarray(vector)
        if vector.ndim != 1:
            raise ValueError(f"Vector must be 1-D, got ndim={vector.ndim}")
        if self._vector_dimension is not None and vector.size != self._vector_dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self._vector_dimension}, "
                f"got {vector.size}"
            )

    @staticmethod
    def _json_default(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    async def close(self) -> None:
        """Close the underlying backend."""
        await self._backend.close()
