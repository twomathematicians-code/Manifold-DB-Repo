"""
Manifold Storage Module - persistent storage backends and data store.

Public API:
    StorageBackend      – abstract base class for storage backends
    MemoryStorage       – in-memory dict-based backend
    FileStorage         – filesystem backend (npz + json)
    SQLiteStorage       – SQLite backend with aiosqlite
    StorageManager       – backend wrapper with LRU cache and WAL
    DataPoint           – data point dataclass with vector + metadata
    DataStore           – high-level data store with search and filtering
"""

from manifold_db.storage.backend import (
    FileStorage,
    MemoryStorage,
    SQLiteStorage,
    StorageBackend,
    StorageManager,
)
from manifold_db.storage.data_store import DataPoint, DataStore

__all__ = [
    "StorageBackend",
    "MemoryStorage",
    "FileStorage",
    "SQLiteStorage",
    "StorageManager",
    "DataPoint",
    "DataStore",
]
