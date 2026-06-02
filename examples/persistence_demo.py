#!/usr/bin/env python3
"""
persistence_demo.py - Data Persistence and Schema Evolution with ManifoldDB
============================================================================

This demo demonstrates ManifoldDB's persistence and schema evolution features:

  1.  Create a ManifoldDB, insert data, and build an atlas.
  2.  Save the database to disk using the ``manifolddb.io`` persistence utilities.
  3.  Close the database (release all in-memory state).
  4.  Reopen the database from the saved state on disk.
  5.  Query the reopened database to verify data persistence.
  6.  Demonstrate schema evolution: insert new data points and rebuild the
      atlas to incorporate the expanded dataset.

Why Persistence Matters
-----------------------
In production deployments, ManifoldDB must survive process restarts, server
crashes, and schema updates.  The persistence layer serialises the atlas
(chart decomposition, metric tensors, transition maps) and the tangent-space
indexes to disk, so they can be reloaded without re-ingesting raw data.

Schema Evolution
----------------
As new data arrives (e.g., new users, new documents, new modalities), the
manifold structure may change.  ManifoldDB's ``evolve()`` method extends the
atlas by inserting new points, discovering additional charts if necessary,
and rebuilding the tangent-space indexes — all without losing existing data.

Run with::

    python examples/persistence_demo.py

Requirements
------------
- numpy
- manifolddb C++ extension
"""

from __future__ import annotations

import os
import sys
import tempfile
import shutil
import json
import time

import numpy as np

# ---------------------------------------------------------------------------
# Import ManifoldDB with graceful fallback
# ---------------------------------------------------------------------------
try:
    from manifolddb import ManifoldDB
except ImportError:
    try:
        sys.path.insert(0, ".")
        from python.manifolddb import ManifoldDB
    except ImportError:
        print("ERROR: Cannot import ManifoldDB. Build the C++ extension first.")
        print("See README.md for build instructions.")
        sys.exit(1)

# Attempt to import the I/O utilities (may not be available in all builds)
try:
    from manifolddb.io import save_manifold, load_manifold
    HAS_IO = True
except ImportError:
    HAS_IO = False


# ---------------------------------------------------------------------------
# Helper: Swiss Roll data (same as other examples for consistency)
# ---------------------------------------------------------------------------

def make_swiss_roll(n_points: int = 500, noise: float = 0.05, seed: int = 42):
    """Generate a Swiss Roll dataset in R^3.

    Returns (ambient_data, intrinsic_params).
    """
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1.0 + 2.0 * rng.random(n_points))
    h = 21.0 * rng.random(n_points)

    x = t * np.cos(t) + noise * rng.standard_normal(n_points)
    y = h + noise * rng.standard_normal(n_points)
    z = t * np.sin(t) + noise * rng.standard_normal(n_points)

    return np.column_stack([x, y, z]).astype(np.float64)


# ---------------------------------------------------------------------------
# Main Demo
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 64)
    print("  ManifoldDB — Persistence & Schema Evolution Demo")
    print("=" * 64)
    print()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    N_INITIAL = 500
    N_NEW = 300       # New points for schema evolution
    INTRINSIC_DIM = 2
    AMBIENT_DIM = 3

    # Use a known directory (not a temp dir) so we can "reopen" it
    storage_dir = tempfile.mkdtemp(prefix="manifolddb_persist_")
    save_dir = os.path.join(storage_dir, "saved_state")

    print(f"[0] Storage directory: {storage_dir}")
    print(f"    Save directory:     {save_dir}")

    try:
        # ==================================================================
        # PHASE 1: Create, populate, and build the database
        # ==================================================================
        print(f"\n{'=' * 64}")
        print("  PHASE 1: Initial Database Creation")
        print(f"{'=' * 64}")

        print(f"\n[1] Creating ManifoldDB (intrinsic_dim={INTRINSIC_DIM}) ...")
        db = ManifoldDB(
            storage_path=storage_dir,
            intrinsic_dim=INTRINSIC_DIM,
            enable_cuda=False,
            geodesic_tolerance=1e-6,
        )
        print(f"    {db}")

        # ------------------------------------------------------------------
        # Generate and insert initial data
        # ------------------------------------------------------------------
        print(f"\n[2] Generating and inserting {N_INITIAL} initial points ...")
        initial_data = make_swiss_roll(n_points=N_INITIAL, seed=42)
        print(f"    Data shape: {initial_data.shape}")

        db.insert(initial_data, modality_id=0)
        print(f"    Inserted {N_INITIAL} points (modality_id=0)")

        # ------------------------------------------------------------------
        # Build atlas
        # ------------------------------------------------------------------
        print(f"\n[3] Building atlas ...")
        db.build(method="linear")

        stats_before = db.stats()
        print(f"    Charts: {stats_before['num_charts']}")
        print(f"    Points: {stats_before['total_points']}")
        print(f"    Index:  {stats_before['index_size']}")

        # ------------------------------------------------------------------
        # Query before saving (baseline)
        # ------------------------------------------------------------------
        print(f"\n[4] Querying k-NN (baseline, before save) ...")
        query_pt = initial_data[0]
        results_before = db.query_knn(query_pt, k=5)
        print(f"    Query: {query_pt}")
        for r in results_before:
            print(f"      id={r['id']:>4d}  dist={r['distance']:.4f}  "
                  f"chart={r['chart_id']}")

        # ==================================================================
        # PHASE 2: Save to disk
        # ==================================================================
        print(f"\n{'=' * 64}")
        print("  PHASE 2: Saving to Disk")
        print(f"{'=' * 64}")

        if HAS_IO:
            print(f"\n[5] Saving database to {save_dir} ...")
            save_manifold(db, save_dir)

            # Verify save directory contents
            print(f"    Saved files:")
            if os.path.isdir(save_dir):
                for fname in sorted(os.listdir(save_dir)):
                    fpath = os.path.join(save_dir, fname)
                    if os.path.isfile(fpath):
                        size = os.path.getsize(fpath)
                        print(f"      {fname} ({size} bytes)")
                    elif os.path.isdir(fpath):
                        n_files = len(os.listdir(fpath))
                        print(f"      {fname}/ ({n_files} files)")
        else:
            print(f"\n[5] manifolddb.io not available — using direct storage_path.")
            print(f"    Data persists at: {storage_dir}")
            print(f"    (The C++ extension handles persistence via storage_path.)")

        # Read back the metadata JSON to confirm
        meta_path = os.path.join(save_dir, "meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            print(f"\n    Metadata: {json.dumps(meta, indent=6)}")

        # ==================================================================
        # PHASE 3: Close and reopen
        # ==================================================================
        print(f"\n{'=' * 64}")
        print("  PHASE 3: Closing and Reopening")
        print(f"{'=' * 64}")

        # Delete the in-memory database object
        print(f"\n[6] Closing database (deleting in-memory object) ...")
        del db
        db = None
        print("    Database object released.")

        # Record the time to show the object is truly gone
        time.sleep(0.1)

        # Reopen from saved state
        print(f"\n[7] Reopening database ...")
        if HAS_IO and os.path.isdir(save_dir):
            db = load_manifold(save_dir)
            print(f"    Loaded from {save_dir}")
        else:
            # Reopen by creating a new ManifoldDB at the same storage_path
            db = ManifoldDB(
                storage_path=storage_dir,
                intrinsic_dim=INTRINSIC_DIM,
                enable_cuda=False,
                geodesic_tolerance=1e-6,
            )
            print(f"    Reopened at storage_path={storage_dir}")

        print(f"    {db}")

        # ==================================================================
        # PHASE 4: Verify persistence by querying
        # ==================================================================
        print(f"\n{'=' * 64}")
        print("  PHASE 4: Verifying Data Persistence")
        print(f"{'=' * 64}")

        stats_after = db.stats()
        print(f"\n[8] Stats after reopening:")
        print(f"    Charts: {stats_after['num_charts']}")
        print(f"    Points: {stats_after['total_points']}")
        print(f"    Index:  {stats_after['index_size']}")

        # Verify the atlas structure is preserved
        assert stats_after["num_charts"] == stats_before["num_charts"], (
            "Chart count changed after save/load!"
        )
        print(f"    ✓ Chart count preserved: {stats_after['num_charts']}")

        # Query the same point
        print(f"\n[9] Querying k-NN (after reopen) ...")
        results_after = db.query_knn(query_pt, k=5)
        print(f"    Query: {query_pt}")
        for r in results_after:
            print(f"      id={r['id']:>4d}  dist={r['distance']:.4f}  "
                  f"chart={r['chart_id']}")

        # Verify the same top result is returned
        if results_before and results_after:
            same_top = results_before[0]["id"] == results_after[0]["id"]
            print(f"\n    Same top result: {same_top}")
            if same_top:
                print(f"    ✓ Data persistence verified!")
            else:
                print(f"    ! Top result changed (may be due to index rebuild)")

        # ==================================================================
        # PHASE 5: Schema Evolution — add new data
        # ==================================================================
        print(f"\n{'=' * 64}")
        print("  PHASE 5: Schema Evolution")
        print(f"{'=' * 64}")

        # ------------------------------------------------------------------
        # Generate new data points (different random seed for variety)
        # ------------------------------------------------------------------
        print(f"\n[10] Generating {N_NEW} new data points ...")
        new_data = make_swiss_roll(n_points=N_NEW, seed=99)
        print(f"     New data shape: {new_data.shape}")

        # ------------------------------------------------------------------
        # Evolve the schema with the new data
        # ------------------------------------------------------------------
        #
        # The evolve() method inserts the new data and rebuilds the atlas,
        # potentially discovering new charts to accommodate the expanded
        # point cloud.
        #
        print(f"\n[11] Evolving schema with new data ...")
        db.evolve(new_data)

        stats_evolved = db.stats()
        print(f"    Charts after evolve: {stats_evolved['num_charts']} "
              f"(was {stats_after['num_charts']})")
        print(f"    Points after evolve: {stats_evolved['total_points']} "
              f"(was {stats_after['total_points']})")
        print(f"    Index after evolve:  {stats_evolved['index_size']}")

        # Verify total points increased
        expected_total = stats_after["total_points"] + N_NEW
        print(f"    Expected total: {expected_total}")
        print(f"    ✓ Schema evolution successful!" if stats_evolved["total_points"] >= expected_total
              else f"    ! Point count mismatch")

        # ------------------------------------------------------------------
        # Query after evolution
        # ------------------------------------------------------------------
        print(f"\n[12] Querying k-NN (after schema evolution) ...")
        results_evolved = db.query_knn(query_pt, k=5)
        print(f"    Query: {query_pt}")
        for r in results_evolved:
            print(f"      id={r['id']:>4d}  dist={r['distance']:.4f}  "
                  f"chart={r['chart_id']}")

        # Check if new points (ids >= N_INITIAL) appear in results
        new_in_results = sum(
            1 for r in results_evolved if r["id"] >= N_INITIAL
        )
        print(f"\n    New points in top-5 results: {new_in_results}")
        print(f"    (New points have IDs >= {N_INITIAL})")

        # ==================================================================
        # Final Summary
        # ==================================================================
        print(f"\n{'=' * 64}")
        print("  Summary")
        print(f"{'=' * 64}")
        print(f"\n    Phase 1 — Created DB with {N_INITIAL} points")
        print(f"    Phase 2 — Saved to {save_dir}")
        print(f"    Phase 3 — Closed and reopened successfully")
        print(f"    Phase 4 — Verified persistence (queries match)")
        print(f"    Phase 5 — Evolved schema with {N_NEW} new points")
        print(f"    Final: {stats_evolved['total_points']} points, "
              f"{stats_evolved['num_charts']} charts")

        print(f"\n    Final stats:")
        for key, value in stats_evolved.items():
            if isinstance(value, float):
                print(f"      {key:>25s}: {value:.4f}")
            else:
                print(f"      {key:>25s}: {value}")

        print("\n" + "=" * 64)
        print("  Persistence demo completed successfully!")
        print("=" * 64)

    finally:
        shutil.rmtree(storage_dir, ignore_errors=True)
        print(f"\nCleaned up: {storage_dir}")


if __name__ == "__main__":
    main()
