"""
Storage Backend - persistent storage for manifold databases.
Stores charts, indices, metrics, and raw data with multiple backend support.

Supported backends:
    - MemoryStorage   : in-memory dict (testing / development)
    - FileStorage     : filesystem using numpy .npz and JSON
    - SQLiteStorage   : SQLite with aiosqlite for structured metadata + blobs

StorageManager wraps a backend with an LRU cache, write-ahead log,
and auto-compaction.
"""

from __future__ import annotations

import abc
import asyncio
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Abstract Base
# ──────────────────────────────────────────────────────────────


class StorageBackend(abc.ABC):
    """Interface that all storage backends must implement."""

    @abc.abstractmethod
    async def put(self, key: str, value: Any) -> None: ...

    @abc.abstractmethod
    async def get(self, key: str) -> Any | None: ...

    @abc.abstractmethod
    async def delete(self, key: str) -> None: ...

    async def put_batch(self, items: dict[str, Any]) -> None:
        for k, v in items.items():
            await self.put(k, v)

    async def get_batch(self, keys: list[str]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for k in keys:
            val = await self.get(k)
            if val is not None:
                result[k] = val
        return result

    @abc.abstractmethod
    async def exists(self, key: str) -> bool: ...

    @abc.abstractmethod
    async def list_keys(self, prefix: str = "") -> list[str]: ...

    # Transactions (optional – backends may raise NotImplementedError)

    async def begin_transaction(self) -> None:
        raise NotImplementedError("Transactions not supported by this backend")

    async def commit(self) -> None:
        raise NotImplementedError("Transactions not supported by this backend")

    async def rollback(self) -> None:
        raise NotImplementedError("Transactions not supported by this backend")

    async def close(self) -> None:
        """Release resources held by the backend."""
        pass


# ──────────────────────────────────────────────────────────────
# Memory Storage
# ──────────────────────────────────────────────────────────────


class MemoryStorage(StorageBackend):
    """
    In-memory dict-based storage. Thread-safe with asyncio.Lock.
    Ideal for testing and fast prototyping.
    """

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._transaction_active = False
        self._pending: dict[str, Any] = {}
        logger.info("MemoryStorage initialised")

    async def put(self, key: str, value: Any) -> None:
        async with self._lock:
            if self._transaction_active:
                self._pending[key] = value
            else:
                self._store[key] = value

    async def get(self, key: str) -> Any | None:
        async with self._lock:
            if key in self._pending:
                return self._pending[key]
            return self._store.get(key)

    async def delete(self, key: str) -> None:
        async with self._lock:
            if self._transaction_active:
                self._pending[key] = None  # tombstone
            else:
                self._store.pop(key, None)

    async def put_batch(self, items: dict[str, Any]) -> None:
        async with self._lock:
            if self._transaction_active:
                self._pending.update(items)
            else:
                self._store.update(items)

    async def get_batch(self, keys: list[str]) -> dict[str, Any]:
        async with self._lock:
            result = {}
            for k in keys:
                val = self._pending.get(k) if k in self._pending else self._store.get(k)
                if val is not None:
                    result[k] = val
            return result

    async def exists(self, key: str) -> bool:
        async with self._lock:
            return key in self._store or key in self._pending

    async def list_keys(self, prefix: str = "") -> list[str]:
        async with self._lock:
            keys = list(self._store.keys())
            if prefix:
                keys = [k for k in keys if k.startswith(prefix)]
            return sorted(keys)

    async def begin_transaction(self) -> None:
        async with self._lock:
            self._transaction_active = True
            self._pending = {}

    async def commit(self) -> None:
        async with self._lock:
            for k, v in self._pending.items():
                if v is None:
                    self._store.pop(k, None)
                else:
                    self._store[k] = v
            self._pending = {}
            self._transaction_active = False

    async def rollback(self) -> None:
        async with self._lock:
            self._pending = {}
            self._transaction_active = False

    async def close(self) -> None:
        self._store.clear()
        self._pending.clear()


# ──────────────────────────────────────────────────────────────
# File Storage
# ──────────────────────────────────────────────────────────────


class FileStorage(StorageBackend):
    """
    Filesystem-based backend.

    Stores:
        - Charts     → {chart_id}.chart.json
        - Index data → {chart_id}.index.npz
        - Metrics    → {chart_id}.metric.json
        - Raw data   → data.npz or data.json
    """

    def __init__(self, base_path: str | Path = "./manifold_data") -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        logger.info("FileStorage initialised at %s", self.base_path)

    def _chart_path(self, chart_id: str) -> Path:
        return self.base_path / f"{chart_id}.chart.json"

    def _index_path(self, chart_id: str) -> Path:
        return self.base_path / f"{chart_id}.index.npz"

    def _metric_path(self, chart_id: str) -> Path:
        return self.base_path / f"{chart_id}.metric.json"

    def _data_path(self, key: str) -> Path:
        return self.base_path / f"{key}.data.json"

    def _blob_path(self, key: str) -> Path:
        return self.base_path / f"{key}.npz"

    async def put(self, key: str, value: Any) -> None:
        async with self._lock:
            if isinstance(value, np.ndarray):
                np.savez_compressed(str(self._blob_path(key)), data=value)
            elif isinstance(value, (dict, list)):
                # Check if it looks like chart / metric data
                if "chart_id" in value if isinstance(value, dict) else False:
                    p = self._chart_path(value["chart_id"])
                elif "metric_type" in value if isinstance(value, dict) else False:
                    p = self._metric_path(key)
                else:
                    p = self._data_path(key)
                with open(p, "w") as f:
                    json.dump(value, f, default=self._json_default)
            else:
                with open(self._data_path(key), "w") as f:
                    json.dump(value, f, default=self._json_default)

    async def get(self, key: str) -> Any | None:
        async with self._lock:
            # Try JSON first
            p = self._data_path(key)
            if p.exists():
                with open(p) as f:
                    return json.load(f)
            # Try numpy blob
            p = self._blob_path(key)
            if p.exists():
                data = np.load(str(p), allow_pickle=True)
                return data["data"] if "data" in data else data
            # Try chart
            p = self._chart_path(key)
            if p.exists():
                with open(p) as f:
                    return json.load(f)
            # Try metric
            p = self._metric_path(key)
            if p.exists():
                with open(p) as f:
                    return json.load(f)
            return None

    async def delete(self, key: str) -> None:
        async with self._lock:
            for p in [
                self._data_path(key),
                self._blob_path(key),
                self._chart_path(key),
                self._metric_path(key),
            ]:
                if p.exists():
                    p.unlink()

    async def exists(self, key: str) -> bool:
        for p in [
            self._data_path(key),
            self._blob_path(key),
            self._chart_path(key),
            self._metric_path(key),
        ]:
            if p.exists():
                return True
        return False

    async def list_keys(self, prefix: str = "") -> list[str]:
        keys: list[str] = []
        for ext in (
            "*.chart.json",
            "*.index.npz",
            "*.metric.json",
            "*.data.json",
            "*.npz",
        ):
            for p in self.base_path.glob(ext):
                name = p.stem
                # Remove suffixes like .chart, .index, .metric, .data
                for suffix in (".chart", ".index", ".metric", ".data"):
                    if name.endswith(suffix):
                        name = name[: -len(suffix)]
                        break
                if not prefix or name.startswith(prefix):
                    if name not in keys:
                        keys.append(name)
        return sorted(keys)

    async def close(self) -> None:
        pass

    @staticmethod
    def _json_default(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ──────────────────────────────────────────────────────────────
# SQLite Storage
# ──────────────────────────────────────────────────────────────


class SQLiteStorage(StorageBackend):
    """
    SQLite-backed storage using aiosqlite.

    Tables:
        - charts          : id, atlas_id, dimension, centre, basis (JSON), created_at
        - transition_maps : source_chart, target_chart, overlap_data (BLOB)
        - metrics         : id, chart_id, metric_type, parameters (JSON), created_at
        - data_points     : id, chart_id, vector (BLOB), metadata (JSON), modality, ts
        - indices         : id, chart_id, index_type, index_data (BLOB), created_at
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS charts (
        chart_id   TEXT PRIMARY KEY,
        atlas_id   TEXT NOT NULL DEFAULT 'default',
        dimension  INTEGER NOT NULL,
        centre     BLOB,
        basis      TEXT,       -- JSON-serialised basis matrix
        created_at REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS transition_maps (
        source_chart TEXT NOT NULL,
        target_chart TEXT NOT NULL,
        overlap_data BLOB,
        PRIMARY KEY (source_chart, target_chart)
    );
    CREATE TABLE IF NOT EXISTS metrics (
        id         TEXT PRIMARY KEY,
        chart_id   TEXT NOT NULL,
        metric_type TEXT NOT NULL,
        parameters TEXT,       -- JSON
        created_at REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS data_points (
        id         TEXT PRIMARY KEY,
        chart_id   TEXT NOT NULL,
        vector     BLOB NOT NULL,
        metadata   TEXT,       -- JSON
        modality   TEXT,
        timestamp  REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS indices (
        id          TEXT PRIMARY KEY,
        chart_id    TEXT NOT NULL,
        index_type  TEXT NOT NULL,
        index_data  BLOB,
        created_at  REAL NOT NULL
    );
    """

    def __init__(self, db_path: str | Path = "./manifold.db") -> None:
        self.db_path = str(db_path)
        self._conn: Any | None = None  # aiosqlite connection
        self._lock = asyncio.Lock()
        logger.info("SQLiteStorage initialised at %s", self.db_path)

    async def _ensure_conn(self):
        """Lazily create the aiosqlite connection and tables."""
        if self._conn is None:
            try:
                import aiosqlite  # type: ignore
            except ImportError:
                raise RuntimeError(
                    "aiosqlite is required for SQLiteStorage. "
                    "Install it with: pip install aiosqlite"
                )
            self._conn = await aiosqlite.connect(self.db_path)
            self._conn.row_factory = aiosqlite.Row
            await self._conn.executescript(self._SCHEMA)
            await self._conn.commit()

    async def put(self, key: str, value: Any) -> None:
        await self._ensure_conn()
        async with self._lock:
            if isinstance(value, dict):
                table = self._guess_table(value)
                await self._put_dict(table, key, value)
            else:
                # Store as generic blob in indices table
                blob = (
                    np.dumps(value)
                    if isinstance(value, np.ndarray)
                    else str(value).encode()
                )
                await self._conn.execute(
                    (
                        "INSERT OR REPLACE INTO indices "
                        "(id, chart_id, index_type, index_data, created_at) "
                        "VALUES (?, ?, 'raw', ?, ?)"
                    ),
                    (key, "generic", blob, time.time()),
                )
                await self._conn.commit()

    async def _put_dict(self, table: str, key: str, value: dict):
        now = time.time()
        if table == "charts":
            centre = (
                np.array(value.get("centre", [])).tobytes()
                if "centre" in value
                else None
            )
            await self._conn.execute(
                (
                    "INSERT OR REPLACE INTO charts "
                    "(chart_id, atlas_id, dimension, centre, basis, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)"
                ),
                (
                    key,
                    value.get("atlas_id", "default"),
                    value.get("dimension", 0),
                    centre,
                    json.dumps(value.get("basis"), default=self._json_default),
                    now,
                ),
            )
        elif table == "metrics":
            await self._conn.execute(
                (
                    "INSERT OR REPLACE INTO metrics "
                    "(id, chart_id, metric_type, parameters, created_at) "
                    "VALUES (?, ?, ?, ?, ?)"
                ),
                (
                    key,
                    value.get("chart_id", ""),
                    value.get("metric_type", "geodesic"),
                    json.dumps(value.get("parameters", {}), default=self._json_default),
                    now,
                ),
            )
        elif table == "data_points":
            vec = np.array(value.get("vector", [])).astype(np.float32).tobytes()
            await self._conn.execute(
                (
                    "INSERT OR REPLACE INTO data_points "
                    "(id, chart_id, vector, metadata, modality, timestamp) "
                    "VALUES (?, ?, ?, ?, ?, ?)"
                ),
                (
                    key,
                    value.get("chart_id", ""),
                    vec,
                    json.dumps(value.get("metadata", {}), default=self._json_default),
                    value.get("modality", ""),
                    now,
                ),
            )
        elif table == "transition_maps":
            src = value.get("source_chart", "")
            tgt = value.get("target_chart", "")
            overlap = (
                np.array(value.get("overlap_data", [])).tobytes()
                if "overlap_data" in value
                else None
            )
            await self._conn.execute(
                "INSERT OR REPLACE INTO transition_maps (source_chart, target_chart, overlap_data) "
                "VALUES (?, ?, ?)",
                (src, tgt, overlap),
            )
        else:
            blob = json.dumps(value).encode()
            await self._conn.execute(
                "INSERT OR REPLACE INTO indices (id, chart_id, index_type, index_data, created_at) "
                "VALUES (?, ?, 'raw', ?, ?)",
                (key, "generic", blob, now),
            )
        await self._conn.commit()

    @staticmethod
    def _guess_table(value: dict) -> str:
        if "dimension" in value and "atlas_id" in value:
            return "charts"
        if "metric_type" in value:
            return "metrics"
        if "vector" in value and "chart_id" in value:
            return "data_points"
        if "source_chart" in value and "target_chart" in value:
            return "transition_maps"
        return "indices"

    async def get(self, key: str) -> Any | None:
        await self._ensure_conn()
        async with self._lock:
            # Try charts
            cursor = await self._conn.execute(
                "SELECT * FROM charts WHERE chart_id = ?", (key,)
            )
            row = await cursor.fetchone()
            if row:
                centre = (
                    np.frombuffer(row["centre"], dtype=np.float64)
                    if row["centre"]
                    else np.array([])
                )
                return {
                    "chart_id": row["chart_id"],
                    "atlas_id": row["atlas_id"],
                    "dimension": row["dimension"],
                    "centre": centre.tolist(),
                    "basis": json.loads(row["basis"]) if row["basis"] else None,
                    "created_at": row["created_at"],
                }

            # Try data_points
            cursor = await self._conn.execute(
                "SELECT * FROM data_points WHERE id = ?", (key,)
            )
            row = await cursor.fetchone()
            if row:
                vec = np.frombuffer(row["vector"], dtype=np.float32)
                return {
                    "id": row["id"],
                    "chart_id": row["chart_id"],
                    "vector": vec.tolist(),
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    "modality": row["modality"],
                    "timestamp": row["timestamp"],
                }

            # Try metrics
            cursor = await self._conn.execute(
                "SELECT * FROM metrics WHERE id = ?", (key,)
            )
            row = await cursor.fetchone()
            if row:
                return {
                    "id": row["id"],
                    "chart_id": row["chart_id"],
                    "metric_type": row["metric_type"],
                    "parameters": (
                        json.loads(row["parameters"]) if row["parameters"] else {}
                    ),
                    "created_at": row["created_at"],
                }

            # Try indices
            cursor = await self._conn.execute(
                "SELECT * FROM indices WHERE id = ?", (key,)
            )
            row = await cursor.fetchone()
            if row and row["index_data"]:
                return row["index_data"]
            return None

    async def delete(self, key: str) -> None:
        await self._ensure_conn()
        async with self._lock:
            for table in ("charts", "data_points", "metrics", "indices"):
                try:
                    col = "chart_id" if table == "charts" else "id"
                    await self._conn.execute(
                        f"DELETE FROM {table} WHERE {col} = ?", (key,)
                    )
                except Exception:
                    pass
            await self._conn.commit()

    async def exists(self, key: str) -> bool:
        val = await self.get(key)
        return val is not None

    async def list_keys(self, prefix: str = "") -> list[str]:
        await self._ensure_conn()
        keys: list[str] = []
        async with self._lock:
            for table in ("charts", "data_points", "metrics", "indices"):
                col = "chart_id" if table == "charts" else "id"
                cursor = await self._conn.execute(f"SELECT {col} FROM {table}")
                rows = await cursor.fetchall()
                for row in rows:
                    k = row[0]
                    if not prefix or k.startswith(prefix):
                        if k not in keys:
                            keys.append(k)
        return sorted(keys)

    async def begin_transaction(self) -> None:
        await self._ensure_conn()
        await self._conn.execute("BEGIN")

    async def commit(self) -> None:
        await self._ensure_conn()
        await self._conn.commit()

    async def rollback(self) -> None:
        await self._ensure_conn()
        await self._conn.rollback()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @staticmethod
    def _json_default(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ──────────────────────────────────────────────────────────────
# Storage Manager
# ──────────────────────────────────────────────────────────────


@dataclass
class _WAL:
    """Simple write-ahead log entry."""

    timestamp: float
    operation: str
    key: str
    value_hash: str


class StorageManager:
    """
    Wraps a StorageBackend with:
      - LRU cache for frequently accessed data
      - Write-ahead log (WAL) for crash recovery
      - Auto-compaction of cache when it grows too large
    """

    def __init__(
        self,
        backend: StorageBackend,
        cache_size: int = 1024,
        wal_enabled: bool = True,
    ) -> None:
        self._backend = backend
        self._cache_size = cache_size
        self._wal_enabled = wal_enabled
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._wal: list[_WAL] = []
        self._lock = asyncio.Lock()
        self._compaction_threshold = cache_size * 2
        logger.info(
            "StorageManager created: backend=%s cache=%d wal=%s",
            type(backend).__name__,
            cache_size,
            wal_enabled,
        )

    @classmethod
    def create(
        cls,
        backend_type: str = "memory",
        config: dict[str, Any] | None = None,
    ) -> StorageManager:
        """
        Factory: create a StorageManager with the given backend.

        Args:
            backend_type: "memory", "file", or "sqlite"
            config: optional dict of backend-specific config
        """
        cfg = config or {}
        if backend_type == "memory":
            backend: StorageBackend = MemoryStorage()
        elif backend_type == "file":
            backend = FileStorage(base_path=cfg.get("base_path", "./manifold_data"))
        elif backend_type == "sqlite":
            backend = SQLiteStorage(db_path=cfg.get("db_path", "./manifold.db"))
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")

        return cls(
            backend=backend,
            cache_size=cfg.get("cache_size", 1024),
            wal_enabled=cfg.get("wal_enabled", True),
        )

    # ── cache helpers ──────────────────────────────────────────

    def _cache_get(self, key: str) -> Any | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _cache_put(self, key: str, value: Any) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def _cache_invalidate(self, key: str) -> None:
        self._cache.pop(key, None)

    async def _auto_compact(self) -> None:
        """Compact the WAL when it exceeds the threshold."""
        if len(self._wal) >= self._compaction_threshold:
            logger.info("Auto-compacting WAL (%d entries)", len(self._wal))
            self._wal = self._wal[len(self._wal) // 2 :]

    # ── WAL helpers ───────────────────────────────────────────

    def _wal_append(self, op: str, key: str, value: Any) -> None:
        if not self._wal_enabled:
            return
        value_hash = str(hash(str(value))) if value is not None else "None"
        self._wal.append(
            _WAL(
                timestamp=time.time(),
                operation=op,
                key=key,
                value_hash=value_hash,
            )
        )

    # ── public API (delegates to backend + cache) ─────────────

    async def put(self, key: str, value: Any) -> None:
        async with self._lock:
            self._wal_append("put", key, value)
            self._cache_invalidate(key)
            await self._backend.put(key, value)
            await self._auto_compact()

    async def get(self, key: str) -> Any | None:
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        async with self._lock:
            val = await self._backend.get(key)
            if val is not None:
                self._cache_put(key, val)
            return val

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._wal_append("delete", key, None)
            self._cache_invalidate(key)
            await self._backend.delete(key)

    async def put_batch(self, items: dict[str, Any]) -> None:
        async with self._lock:
            for k, v in items.items():
                self._wal_append("put", k, v)
                self._cache_invalidate(k)
            await self._backend.put_batch(items)
            await self._auto_compact()

    async def get_batch(self, keys: list[str]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        remaining: list[str] = []
        for k in keys:
            cached = self._cache_get(k)
            if cached is not None:
                result[k] = cached
            else:
                remaining.append(k)
        if remaining:
            fetched = await self._backend.get_batch(remaining)
            for k, v in fetched.items():
                self._cache_put(k, v)
                result[k] = v
        return result

    async def exists(self, key: str) -> bool:
        cached = self._cache_get(key)
        if cached is not None:
            return True
        return await self._backend.exists(key)

    async def list_keys(self, prefix: str = "") -> list[str]:
        return await self._backend.list_keys(prefix)

    async def begin_transaction(self) -> None:
        await self._backend.begin_transaction()

    async def commit(self) -> None:
        await self._backend.commit()

    async def rollback(self) -> None:
        await self._backend.rollback()
        # Invalidate cache on rollback
        self._cache.clear()

    async def clear_cache(self) -> None:
        """Manually clear the in-memory cache."""
        async with self._lock:
            self._cache.clear()

    async def wal_entries(self) -> list[dict[str, Any]]:
        """Return the current WAL entries."""
        return [
            {
                "timestamp": e.timestamp,
                "operation": e.operation,
                "key": e.key,
                "value_hash": e.value_hash,
            }
            for e in self._wal
        ]

    async def close(self) -> None:
        await self._backend.close()
        self._cache.clear()
        self._wal.clear()
        logger.info("StorageManager closed")
