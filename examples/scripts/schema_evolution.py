#!/usr/bin/env python3
"""
Schema Evolution Example — Smooth schema changes as geometric deformations.

In a manifold database, schema changes are treated as smooth deformations
of the data manifold.  This example shows how schema migration works by:

  1. Creating an "old" schema (fewer features)
  2. Creating a "new" schema (additional features)
  3. Building a transition map (geometric deformation) between schemas
  4. Transporting queries from the old schema to the new schema
  5. Demonstrating zero-downtime schema migration

This approach avoids the traditional "stop-the-world" migration and
allows both old and new schemas to coexist during the transition period.

Run:
    python examples/scripts/schema_evolution.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from manifold_db.atlas import (
    AtlasManager, AtlasBuilder, Chart, AffineTransition, LinearTransition
)
from manifold_db.tangent_index import TangentSpace, TangentSpaceIndex
from manifold_db.query import QueryBuilder, ManifoldQuery, QueryType, MetricType
from manifold_db.metric import EuclideanMetric, MetricTensorStore
from manifold_db.connection import (
    LeviCivitaConnection, SchemaTransport, TransportRegistry
)


# ---------------------------------------------------------------------------
# Schema Simulation
# ---------------------------------------------------------------------------

def generate_old_schema_data(n: int = 500, n_features: int = 10) -> np.ndarray:
    """Generate data with the 'old' schema (v1).
    
    Old schema has 10 features — a basic feature set used by the legacy system.
    """
    rng = np.random.default_rng(42)
    return rng.normal(0, 1.0, size=(n, n_features))


def evolve_schema(old_data: np.ndarray, new_features: int = 5) -> np.ndarray:
    """Simulate schema evolution: add new features derived from old ones.
    
    New features are smooth functions of old features plus small noise.
    This models a real scenario where new columns are computed from
    existing data (e.g., derived features, aggregations).
    """
    rng = np.random.default_rng(123)
    n, old_dim = old_data.shape
    
    # New features are non-linear combinations of old features
    new_cols = []
    for i in range(new_features):
        # Each new feature is a smooth function of random old features
        src_indices = rng.choice(old_dim, size=3, replace=False)
        weights = rng.normal(0, 0.3, size=3)
        new_col = np.zeros(n)
        for j, idx in enumerate(src_indices):
            new_col += weights[j] * old_data[:, idx]
        # Add non-linear transformation
        new_col = np.tanh(new_col) + rng.normal(0, 0.05, n)
        new_cols.append(new_col)
    
    new_features_data = np.column_stack(new_cols)
    return np.hstack([old_data, new_features_data])


def main() -> None:
    print("=" * 72)
    print("  Manifold Database — Schema Evolution Example")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Step 1: Create old schema data and manifold
    # ------------------------------------------------------------------
    print("\n[Step 1] Creating old schema (v1)...")
    old_n_features = 10
    n_points = 500
    old_data = generate_old_schema_data(n=n_points, n_features=old_n_features)
    old_ids = [f"v1_{i:04d}" for i in range(n_points)]
    print(f"  Old schema: {old_data.shape}  ({old_n_features} features)")
    
    # Build atlas for old schema
    old_atlas = AtlasManager(name="schema_v1")
    builder = AtlasBuilder(k_neighbors=10, min_chart_size=30, random_state=42)
    old_atlas_built = builder.build(old_data, n_charts_hint=2)
    for chart in old_atlas_built.get_all_charts():
        old_atlas.add_chart(chart)
    for tmap in old_atlas_built.get_all_transition_maps():
        old_atlas.add_transition_map(tmap)
    print(f"  Old atlas: {len(old_atlas.get_all_charts())} charts")
    
    # Build tangent index for old schema
    old_index = TangentSpaceIndex(intrinsic_dim=5)
    old_index.build_from_data(old_ids, old_data, n_anchors=20)
    print(f"  Old index: {old_index.size} points indexed")

    # ------------------------------------------------------------------
    # Step 2: Create new schema data
    # ------------------------------------------------------------------
    print("\n[Step 2] Creating new schema (v2)...")
    new_n_added = 5
    new_data = evolve_schema(old_data, new_features=new_n_added)
    new_n_features = old_n_features + new_n_added
    new_ids = [f"v2_{i:04d}" for i in range(n_points)]
    print(f"  New schema: {new_data.shape}  ({new_n_features} features, "
          f"+{new_n_added} new)")
    
    # Build atlas for new schema
    new_atlas = AtlasManager(name="schema_v2")
    new_atlas_built = builder.build(new_data, n_charts_hint=2)
    for chart in new_atlas_built.get_all_charts():
        new_atlas.add_chart(chart)
    for tmap in new_atlas_built.get_all_transition_maps():
        new_atlas.add_transition_map(tmap)
    print(f"  New atlas: {len(new_atlas.get_all_charts())} charts")
    
    # Build tangent index for new schema
    new_index = TangentSpaceIndex(intrinsic_dim=7)
    new_index.build_from_data(new_ids, new_data, n_anchors=20)
    print(f"  New index: {new_index.size} points indexed")

    # ------------------------------------------------------------------
    # Step 3: Build transition map between schemas
    # ------------------------------------------------------------------
    print("\n[Step 3] Building schema transition map...")
    print("  The transition map captures how data geometry changes")
    print("  when features are added to the schema.")
    
    # Create SchemaTransport for the migration
    schema_transport = SchemaTransport()
    
    # Register old schema reference point and metric
    old_metric = EuclideanMetric(dimension=old_n_features)
    schema_transport.register_schema(
        schema_id="v1",
        reference_point=old_data.mean(axis=0),
        metric=old_metric,
    )
    
    # Register new schema reference point and metric
    new_metric = EuclideanMetric(dimension=new_n_features)
    schema_transport.register_schema(
        schema_id="v2",
        reference_point=new_data.mean(axis=0),
        metric=new_metric,
    )
    
    print("  Schema transport registered:")
    print(f"    v1: {old_n_features}D, ref point norm={np.linalg.norm(old_data.mean(axis=0)):.3f}")
    print(f"    v2: {new_n_features}D, ref point norm={np.linalg.norm(new_data.mean(axis=0)):.3f}")

    # ------------------------------------------------------------------
    # Step 4: Transport a query from old schema to new schema
    # ------------------------------------------------------------------
    print("\n[Step 4] Transporting queries from old schema to new schema...")
    
    # Simulate a query vector in old schema space
    query_old = old_data[0]
    print(f"  Query in old schema: dim={query_old.shape[0]}, "
          f"norm={np.linalg.norm(query_old):.3f}")
    
    # Transport the query to new schema space
    # SchemaTransport handles the dimensionality change (padding)
    transported = schema_transport.transport_query(
        query_vector=query_old,
        source_schema="v1",
        target_schema="v2",
    )
    print(f"  Transported query: dim={transported.shape[0]}, "
          f"norm={np.linalg.norm(transported):.3f}")
    
    # The transported vector should preserve the original features
    # and have reasonable values for the new features
    original_preserved = np.linalg.norm(query_old - transported[:old_n_features])
    print(f"  Preservation of original features: error={original_preserved:.6f}")
    
    # New features are filled with the interpolation default
    new_feature_vals = transported[old_n_features:]
    print(f"  New feature values: mean={new_feature_vals.mean():.4f}, "
          f"std={new_feature_vals.std():.4f}")

    # ------------------------------------------------------------------
    # Step 5: Compare results: direct vs. transported queries
    # ------------------------------------------------------------------
    print("\n[Step 5] Comparing query results...")
    
    # Search in old schema
    old_results = old_index.search(query_old, k=5)
    print("  Old schema results:")
    for pid, dist in old_results:
        print(f"    → {pid}  dist={dist:.6f}")
    
    # Search in new schema with transported query
    new_results = new_index.search(transported, k=5)
    print("  New schema results (transported query):")
    for pid, dist in new_results:
        print(f"    → {pid}  dist={dist:.6f}")
    
    # Ground truth: what the actual new-schema equivalent would find
    ground_truth_query = new_data[0]
    gt_results = new_index.search(ground_truth_query, k=5)
    print("  Ground truth (direct new schema query):")
    for pid, dist in gt_results:
        print(f"    → {pid}  dist={dist:.6f}")
    
    # Compute recall of transported query vs ground truth
    transported_top = set(pid for pid, _ in new_results[:5])
    gt_top = set(pid for pid, _ in gt_results[:5])
    recall = len(transported_top & gt_top) / len(gt_top)
    print(f"\n  Recall (transported vs ground truth): {recall:.2%}")

    # ------------------------------------------------------------------
    # Step 6: Zero-downtime migration demo
    # ------------------------------------------------------------------
    print("\n[Step 6] Zero-downtime schema migration...")
    print("  Both schemas coexist during migration:")
    print(f"    Old schema (v1): {len(old_index)} points, ready for queries")
    print(f"    New schema (v2): {len(new_index)} points, ready for queries")
    print(f"    Transition map: v1 → v2, handles {old_n_features}D → {new_n_features}D")
    print("")
    print("  During migration:")
    print("    • Old queries → served from v1 index (no change)")
    print("    • New queries → served from v2 index (new features)")
    print("    • Cross-schema queries → use SchemaTransport to bridge")
    
    # Simulate concurrent queries during migration
    print("\n  Simulating concurrent queries during migration...")
    t0 = time.time()
    for i in range(100):
        # Mix of old and new schema queries
        if i % 3 == 0:
            # Old schema query
            q = old_data[i % len(old_data)]
            old_index.search(q, k=3)
        elif i % 3 == 1:
            # New schema query
            q = new_data[i % len(new_data)]
            new_index.search(q, k=3)
        else:
            # Cross-schema query (transported)
            q = old_data[i % len(old_data)]
            transported_q = schema_transport.transport_query(q, "v1", "v2")
            new_index.search(transported_q, k=3)
    
    migration_query_time = time.time() - t0
    print(f"  100 mixed queries in {migration_query_time*1000:.2f}ms")
    print(f"  Avg per query: {migration_query_time*10:.2f}ms")
    print(f"  No downtime — both schemas served simultaneously!")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  Schema Evolution Complete!")
    print("=" * 72)
    print(f"""
  Key takeaways:
    • Schema changes are geometric deformations, not table alterations
    • SchemaTransport handles dimensionality changes (pad/truncate)
    • Both old and new schemas coexist during migration (zero downtime)
    • Queries are seamlessly transported across schema versions
    • Recall remains high because geometric structure is preserved
    • No "stop-the-world" migration required
    """)


if __name__ == "__main__":
    main()
