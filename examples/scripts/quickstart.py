#!/usr/bin/env python3
"""
Quickstart Example — Get started with Manifold Database in 5 minutes.

This script demonstrates the core workflow:
  1. Generate synthetic data (Swiss roll + random clusters)
  2. Build a manifold atlas from raw data
  3. Insert data points into the index
  4. Execute various query types (tangent, geodesic, cross-modal)
  5. Print results with distances
  6. Save and load the full database

Run:
    python examples/scripts/quickstart.py
"""

import sys
import os
import json
import time
import numpy as np

# Add project root to path so we can import manifold_db
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from manifold_db.atlas import AtlasBuilder, AtlasManager, Chart
from manifold_db.tangent_index import TangentSpace, TangentBundle, TangentSpaceIndex
from manifold_db.query import QueryParser, QueryBuilder, QueryEngine, ManifoldQuery
from manifold_db.metric import EuclideanMetric, MetricTensorStore
from manifold_db.connection import LeviCivitaConnection, TransportRegistry

# ---------------------------------------------------------------------------
# 1. Generate Synthetic Data
# ---------------------------------------------------------------------------
# We create two kinds of data:
#   a) A Swiss roll — a classic manifold-structured dataset (2D surface in 3D)
#   b) Random clusters — multi-chart data with distinct regions
# ---------------------------------------------------------------------------

def generate_swiss_roll(n_points: int = 1000, noise: float = 0.1) -> np.ndarray:
    """Generate a Swiss roll: points (x, y, z) where t ∈ [0, 4π].
    
    The Swiss roll is a 2D manifold embedded in 3D space.  It is one of
    the standard benchmark datasets for manifold-learning algorithms because
    its intrinsic dimensionality (2) is much lower than its ambient
    dimensionality (3).
    """
    rng = np.random.default_rng(42)
    t = rng.uniform(0, 4 * np.pi, n_points)
    x = t * np.cos(t) + rng.normal(0, noise, n_points)
    y = t * np.sin(t) + rng.normal(0, noise, n_points)
    z = rng.uniform(-1, 1, n_points) + rng.normal(0, noise, n_points)
    return np.column_stack([x, y, z])


def generate_clusters(n_points: int = 500, n_dims: int = 10,
                     n_clusters: int = 3) -> np.ndarray:
    """Generate clustered data in high-dimensional space.
    
    Each cluster lives in its own local region of R^n_dims, forming a
    multi-chart structure when the manifold atlas is built.
    """
    rng = np.random.default_rng(123)
    centers = rng.normal(0, 3, size=(n_clusters, n_dims))
    points_per_cluster = n_points // n_clusters
    data = []
    for i in range(n_clusters):
        cluster_data = rng.normal(
            loc=centers[i], scale=0.5, size=(points_per_cluster, n_dims)
        )
        data.append(cluster_data)
    return np.vstack(data)


def main() -> None:
    print("=" * 72)
    print("  Manifold Database — Quickstart Example")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Step 1: Generate and combine synthetic data
    # ------------------------------------------------------------------
    print("\n[Step 1] Generating synthetic data...")
    swiss_roll_data = generate_swiss_roll(n_points=800, noise=0.15)
    print(f"  Swiss roll:  {swiss_roll_data.shape}  (2D manifold in 3D space)")

    cluster_data = generate_clusters(n_points=300, n_dims=10, n_clusters=3)
    print(f"  Clusters:    {cluster_data.shape}  (3 clusters in 10D space)")

    # For the atlas, we need a single dataset.  We'll use the Swiss roll.
    data = swiss_roll_data
    N, D = data.shape
    point_ids = [f"pt_{i:04d}" for i in range(N)]
    print(f"  Working dataset: {data.shape}")

    # ------------------------------------------------------------------
    # Step 2: Build the manifold atlas
    # ------------------------------------------------------------------
    # The AtlasBuilder automatically discovers the chart structure:
    #   - Estimates intrinsic dimension via local PCA
    #   - Constructs a KNN graph for topology
    #   - Uses Louvain community detection to find chart boundaries
    #   - Fits affine transition maps between overlapping charts
    print("\n[Step 2] Building manifold atlas...")
    t0 = time.time()
    builder = AtlasBuilder(
        k_neighbors=15,               # KNN graph connectivity
        pca_variance_threshold=0.95, # cumulative variance for dim estimation
        min_chart_size=50,            # minimum points per chart
        overlap_margin=0.05,
        random_state=42,
    )
    atlas = builder.build(data, n_charts_hint=3)
    build_time = time.time() - t0
    print(f"  Atlas built in {build_time:.3f}s")
    print(f"  Charts: {len(atlas.get_all_charts())}")
    print(f"  Transitions: {len(atlas.get_all_transition_maps())}")

    # Print atlas summary
    summary = atlas.atlas_summary()
    for chart_info in summary["charts"]:
        print(f"    Chart '{chart_info['name']}': dim={chart_info['dim']}, "
              f"ambient={chart_info['ambient_dim']}, "
              f"anchors={chart_info['n_anchor_points']}")

    # ------------------------------------------------------------------
    # Step 3: Build tangent-space index for fast queries
    # ------------------------------------------------------------------
    print("\n[Step 3] Building tangent-space index...")
    t0 = time.time()
    index = TangentSpaceIndex(intrinsic_dim=2)
    build_stats = index.build_from_data(
        point_ids=point_ids,
        data_points=data,
        n_anchors=30,
    )
    print(f"  Index built in {build_stats['build_time_sec']:.3f}s")
    print(f"  Anchors: {build_stats['n_anchors']}")
    print(f"  Intrinsic dim: {build_stats['intrinsic_dim']}")
    coverage = build_stats.get("coverage", {})
    print(f"  Coverage: {coverage}")

    # ------------------------------------------------------------------
    # Step 4: Insert new data points
    # ------------------------------------------------------------------
    print("\n[Step 4] Inserting new data points...")
    new_points = np.random.randn(5, D) * 2
    for i, pt in enumerate(new_points):
        pid = f"new_{i:03d}"
        index.insert(pid, pt)
        print(f"  Inserted '{pid}': norm={np.linalg.norm(pt):.3f}")

    print(f"  Total indexed points: {index.size}")

    # ------------------------------------------------------------------
    # Step 5: Execute queries
    # ------------------------------------------------------------------
    print("\n[Step 5] Executing queries...")

    # --- 5a. Tangent-space query (fast, approximate) ---
    print("\n  --- Tangent Query (fast search in tangent space) ---")
    query_point = data[0]
    results = index.search(query_point, k=5, search_k_anchors=1)
    print(f"  Query point: pt_0000, norm={np.linalg.norm(query_point):.3f}")
    for pid, dist in results:
        print(f"    → {pid}  (tangent dist = {dist:.6f})")

    # --- 5b. Geodesic query (via QueryBuilder) ---
    print("\n  --- Geodesic Query (with geodesic refinement) ---")
    query = (
        QueryBuilder()
        .select("*")
        .from_chart(atlas.get_all_charts()[0].chart_id)
        .where_geodesic(query_point, epsilon=2.0)
        .top_k(5)
        .using_metric("geodesic")
        .build()
    )
    print(f"  Query: {query.to_dict()['query_type']}, epsilon={query.epsilon}, k={query.k}")

    # --- 5c. Cross-modal query example ---
    print("\n  --- Cross-Modal Query (via QueryBuilder) ---")
    cross_modal_query = (
        QueryBuilder()
        .cross_modal("modality_a", "modality_b")
        .with_transport("overlap_region")
        .top_k(10)
        .build()
    )
    print(f"  Query: {cross_modal_query.query_type.value}")
    print(f"    Source modality: {cross_modal_query.modality}")
    print(f"    Target modality: {cross_modal_query.target_modality}")
    print(f"    Transport via: {cross_modal_query.transport_via}")

    # --- 5d. Parallel transport query ---
    print("\n  --- Parallel Transport Query ---")
    charts = atlas.get_all_charts()
    if len(charts) >= 2:
        vec = np.random.randn(charts[0].dim)
        transport_query = (
            QueryBuilder()
            .parallel_transport(vec, charts[0].chart_id, charts[1].chart_id)
            .build()
        )
        print(f"  Transport vector: dim={vec.shape[0]}, norm={np.linalg.norm(vec):.3f}")
        print(f"  Source chart: {charts[0].name}")
        print(f"  Target chart: {charts[1].name}")

    # --- 5e. Query from SQL-like DSL ---
    print("\n  --- SQL-like Query Parsing ---")
    parser = QueryParser()
    sql_query = parser.parse(
        "SELECT * FROM observations WHERE geodesic_distance(embedding, [1.0, 2.0, 3.0]) < 0.5"
    )
    print(f"  Parsed: type={sql_query.query_type.value}, "
          f"epsilon={sql_query.epsilon}")
    if sql_query.query_point is not None:
        print(f"  Query point shape: {sql_query.query_point.shape}")

    # ------------------------------------------------------------------
    # Step 6: Tangent space operations (project, lift, log/exp maps)
    # ------------------------------------------------------------------
    print("\n[Step 6] Tangent space operations...")
    ts = TangentSpace(base_point=data[0], data=data[:50], intrinsic_dim=2)
    print(f"  TangentSpace: ambient={ts._ambient_dim}, intrinsic={ts.dimension}")

    # Project data into tangent coordinates
    sample = data[:5]
    tangent_coords = ts.project(sample)
    print(f"  Projected {sample.shape} → {tangent_coords.shape}")

    # Lift back to ambient
    recovered = ts.lift(tangent_coords)
    reconstruction_error = np.mean(np.linalg.norm(sample - recovered, axis=1))
    print(f"  Reconstruction error: {reconstruction_error:.6f}")

    # Log map and exp map
    log_vec = ts.log_map(data[1])
    print(f"  Log map: {log_vec.shape}, norm={np.linalg.norm(log_vec):.4f}")

    exp_pt = ts.exp_map(log_vec)
    exp_error = np.linalg.norm(data[1] - exp_pt)
    print(f"  Exp map roundtrip error: {exp_error:.6f}")

    # ------------------------------------------------------------------
    # Step 7: Metric and distance computations
    # ------------------------------------------------------------------
    print("\n[Step 7] Metric and distance computations...")
    store = MetricTensorStore()
    metric = EuclideanMetric(dimension=D)
    store.register("default_chart", metric)

    # Geodesic distance approximation
    dist = ts.compute_distance(ts.log_map(data[0]), ts.log_map(data[1]))
    print(f"  Tangent-space geodesic distance (pt0 → pt1): {dist:.6f}")

    # Euclidean comparison
    eucl_dist = np.linalg.norm(data[0] - data[1])
    print(f"  Euclidean distance (pt0 → pt1): {eucl_dist:.6f}")
    print(f"  Ratio (geodesic / euclidean): {dist / eucl_dist:.4f}")

    # ------------------------------------------------------------------
    # Step 8: Save and load the database
    # ------------------------------------------------------------------
    print("\n[Step 8] Saving and loading the database...")
    save_path = os.path.join(
        os.path.dirname(__file__), "..", "output", "quickstart_atlas.json"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    atlas.save(save_path)
    print(f"  Atlas saved to: {save_path}")

    # Load into a fresh atlas
    loaded_atlas = AtlasManager(name="loaded")
    loaded_atlas.load(save_path)
    print(f"  Atlas loaded: {len(loaded_atlas.get_all_charts())} charts, "
          f"{len(loaded_atlas.get_all_transition_maps())} transitions")

    # Save index state
    index_dict = index.to_dict()
    index_save_path = os.path.join(
        os.path.dirname(__file__), "..", "output", "quickstart_index.json"
    )
    with open(index_save_path, "w") as f:
        json.dump(index_dict, f, indent=2)
    print(f"  Index saved to: {index_save_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  Quickstart Complete!")
    print("=" * 72)
    print(f"""
  What you learned:
    • Atlas construction from raw data (AtlasBuilder)
    • Tangent-space indexing for fast search (TangentSpaceIndex)
    • Multiple query types: tangent, geodesic, cross-modal, transport
    • SQL-like query DSL parsing (QueryParser)
    • Fluent query builder API (QueryBuilder)
    • Tangent space math: project, lift, log/exp maps, parallel transport
    • Riemannian metrics and geodesic distances
    • Atlas persistence (save/load JSON)
    """)


if __name__ == "__main__":
    main()
