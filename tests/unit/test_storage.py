"""
Unit tests for manifold_db.storage — MemoryStorage, FileStorage, StorageManager, DataStore.
"""

import asyncio
import os

import numpy as np
import pytest

from manifold_db.storage import (
    DataPoint,
    DataStore,
    FileStorage,
    MemoryStorage,
    StorageManager,
)

# ---------- Helpers ----------


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------- MemoryStorage ----------


class TestMemoryStorage:
    def test_put_get(self):
        store = MemoryStorage()
        _run(store.put("key1", "value1"))
        val = _run(store.get("key1"))
        assert val == "value1"

    def test_get_missing(self):
        store = MemoryStorage()
        val = _run(store.get("missing"))
        assert val is None

    def test_delete(self):
        store = MemoryStorage()
        _run(store.put("k", "v"))
        _run(store.delete("k"))
        assert _run(store.get("k")) is None

    def test_put_batch_get_batch(self):
        store = MemoryStorage()
        items = {"k1": "v1", "k2": "v2", "k3": "v3"}
        _run(store.put_batch(items))
        batch = _run(store.get_batch(["k1", "k3"]))
        assert batch["k1"] == "v1"
        assert batch["k3"] == "v3"

    def test_exists(self):
        store = MemoryStorage()
        _run(store.put("k", "v"))
        assert _run(store.exists("k"))
        assert not _run(store.exists("missing"))

    def test_list_keys(self):
        store = MemoryStorage()
        for i in range(5):
            _run(store.put(f"key_{i}", f"val_{i}"))
        keys = _run(store.list_keys(prefix="key_"))
        assert len(keys) == 5

    def test_transaction(self):
        store = MemoryStorage()
        _run(store.begin_transaction())
        _run(store.put("t1", "v1"))
        _run(store.commit())
        assert _run(store.get("t1")) == "v1"


# ---------- FileStorage ----------


class TestFileStorage:
    def test_put_get(self, tmp_path):
        store = FileStorage(base_path=str(tmp_path))
        _run(store.put("key1", {"data": [1, 2, 3]}))
        val = _run(store.get("key1"))
        assert val["data"] == [1, 2, 3]

    def test_delete(self, tmp_path):
        store = FileStorage(base_path=str(tmp_path))
        _run(store.put("k", "v"))
        _run(store.delete("k"))
        assert _run(store.get("k")) is None

    def test_list_keys(self, tmp_path):
        store = FileStorage(base_path=str(tmp_path))
        for i in range(3):
            _run(store.put(f"item/{i}", i))
        keys = _run(store.list_keys("item/"))
        assert len(keys) == 3


# ---------- StorageManager ----------


class TestStorageManager:
    def test_create_memory(self):
        sm = StorageManager.create("memory")
        assert isinstance(sm._backend, MemoryStorage)

    def test_create_file(self, tmp_path):
        sm = StorageManager.create("file", {"path": str(tmp_path)})
        _run(sm.put("k", "v"))
        assert _run(sm.get("k")) == "v"
        _run(sm.close())

    def test_cache(self):
        sm = StorageManager(MemoryStorage(), cache_size=10)
        _run(sm.put("k", "v"))
        assert _run(sm.get("k")) == "v"
        _run(sm.close())


# ---------- DataStore ----------


class TestDataPoint:
    def test_creation(self):
        dp = DataPoint(id="p0", vector=np.zeros(5), modality="text")
        assert dp.dimension == 5

    def test_serialization(self):
        dp = DataPoint(id="p0", vector=np.ones(3))
        d = dp.to_dict()
        dp2 = DataPoint.from_dict(d)
        np.testing.assert_allclose(dp2.vector, dp.vector)

    def test_metadata(self):
        dp = DataPoint(id="p0", vector=np.zeros(3), metadata={"key": "val"})
        assert dp.metadata["key"] == "val"


class TestDataStore:
    def test_insert_and_get(self):
        ds = DataStore(backend=MemoryStorage())
        dp = DataPoint(id="p0", vector=np.zeros(10))
        _run(ds.insert(dp))
        retrieved = _run(ds.get("p0"))
        assert retrieved is not None
        assert retrieved.id == "p0"

    def test_batch_insert(self):
        ds = DataStore(backend=MemoryStorage())
        points = [DataPoint(id=f"p{i}", vector=np.random.randn(10)) for i in range(20)]
        count = _run(ds.batch_insert(points))
        assert count == 20

    def test_delete(self):
        ds = DataStore(backend=MemoryStorage())
        dp = DataPoint(id="p0", vector=np.zeros(10))
        _run(ds.insert(dp))
        deleted = _run(ds.delete("p0"))
        assert deleted is True

    def test_update(self):
        ds = DataStore(backend=MemoryStorage())
        dp = DataPoint(id="p0", vector=np.zeros(10))
        _run(ds.insert(dp))
        updated = _run(ds.update("p0", {"vector": np.ones(10)}))
        assert updated is not None

    def test_stats(self):
        ds = DataStore(backend=MemoryStorage())
        points = [DataPoint(id=f"p{i}", vector=np.random.randn(5)) for i in range(10)]
        _run(ds.batch_insert(points))
        stats = _run(ds.stats())
        assert stats["count"] == 10

    def test_get_by_modality(self):
        ds = DataStore(backend=MemoryStorage())
        _run(ds.insert(DataPoint(id="t0", vector=np.zeros(5), modality="text")))
        _run(ds.insert(DataPoint(id="i0", vector=np.zeros(5), modality="image")))
        text_pts = _run(ds.get_by_modality("text"))
        assert len(text_pts) == 1
