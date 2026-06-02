#!/usr/bin/env python3
"""
Tutorial Walkthrough — Step-by-step guide to manifold database concepts.

Each section explains a concept from differential geometry and demonstrates
it with runnable code using the Manifold Database library.

Sections:
  1. What is a manifold? (Swiss roll visualisation)
  2. Charts and atlases (local coordinate systems)
  3. Tangent spaces (local linear approximation)
  4. Geodesic distances (shortest paths on curves)
  5. Parallel transport (moving vectors across the manifold)
  6. Multi-modal queries (cross-chart retrieval)
  7. Putting it all together (full pipeline)

Run:
    python examples/scripts/tutorial_walkthrough.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from manifold_db.atlas import AtlasBuilder, AtlasManager, Chart, AffineTransition
from manifold_db.tangent_index import TangentSpace, TangentBundle, TangentSpaceIndex
from manifold_db.query import QueryParser, QueryBuilder, ManifoldQuery
from manifold_db.metric import (
    EuclideanMetric, DiagonalMetric, MetricTensorStore
)
from manifold_db.connection import (
    LeviCivitaConnection, TransportRegistry, SchemaTransport
)


# ═══════════════════════════════════════════════════════════════════════
# Section 1: What is a Manifold?
# ═══════════════════════════════════════════════════════════════════════

def section_1_manifold_intuition():
    """
    SECTION 1: What is a Manifold?
    
    A manifold M is a topological space that locally "looks like" Euclidean
    space.  Formally, every point p ∈ M has a neighbourhood U that is
    homeomorphic to an open subset of R^d, where d is the *intrinsic
    dimension* of the manifold.
    
    Examples:
      - A circle (S^1) is a 1D manifold embedded in 2D
      - A sphere (S^2) is a 2D manifold embedded in 3D
      - A Swiss roll is a 2D manifold embedded in 3D
      - Image embeddings live on a low-D manifold in high-D space
    
    Key insight: Real-world high-dimensional data (text, images, genomes)
    often lies on or near a low-dimensional manifold.  A manifold database
    exploits this structure for efficient storage and retrieval.
    """
    print("\n" + "=" * 72)
    print("  SECTION 1: What is a Manifold?")
    print("=" * 72)
    print("""
  A manifold is a space that locally looks flat (Euclidean) but may be
  globally curved.  Think of Earth's surface: locally flat, globally a sphere.
  
  In data science: high-dimensional data often lives on low-dimensional
  manifolds.  A 768-dim text embedding might really only span ~20
  meaningful dimensions.
  
  The Swiss roll is the classic example: a flat 2D rectangle, rolled
  up into 3D space.  The intrinsic dimension is 2, but the ambient
  dimension is 3.
  """)

    # Generate a simple Swiss roll
    n = 500
    rng = np.random.default_rng(42)
    t = rng.uniform(0, 4 * np.pi, n)
    x = t * np.cos(t) + rng.normal(0, 0.1, n)
    y = t * np.sin(t) + rng.normal(0, 0.1, n)
    z = rng.uniform(-0.5, 0.5, n)
    swiss_roll = np.column_stack([x, y, z])

    print(f"  Swiss roll generated: {swiss_roll.shape}")
    print(f"    Ambient dimension:  {swiss_roll.shape[1]} (lives in 3D)")
    print(f"    Intrinsic dim:      2 (is really a 2D surface)")
    print(f"    Extrinsic dims:     {swiss_roll.shape[1] - 2} (wasted dimensions)")

    # Show that Euclidean distance can be misleading
    p = swiss_roll[0]
    q = swiss_roll[200]
    r = swiss_roll[100]
    
    print(f"\n  Distance example:")
    print(f"    Euclidean(p, q) = {np.linalg.norm(p - q):.3f}")
    print(f"    Euclidean(p, r) = {np.linalg.norm(p - r):.3f}")
    print(f"    Euclidean(r, q) = {np.linalg.norm(r - q):.3f}")
    print(f"""
  The Euclidean distance between p and q goes "through the air",
  but the geodesic distance follows the surface of the roll.
  If p and q are near each other on the unrolled sheet but far in
  3D space, Euclidean distance is a poor measure of similarity.
  """)


# ═══════════════════════════════════════════════════════════════════════
# Section 2: Charts and Atlases
# ═══════════════════════════════════════════════════════════════════════

def section_2_charts_and_atlases():
    """
    SECTION 2: Charts and Atlases
    
    A chart (U, φ) is a local coordinate system:
      - U ⊂ M is an open set (a patch of the manifold)
      - φ: U → R^d maps points to coordinates
    
    An atlas A = {(U_α, φ_α)} is a collection of charts that covers the
    entire manifold.  Where charts overlap, transition maps
    ψ_{αβ}: φ_α(U_α ∩ U_β) → φ_β(U_α ∩ U_β) ensure consistency.
    """
    print("\n" + "=" * 72)
    print("  SECTION 2: Charts and Atlases")
    print("=" * 72)
    print("""
  An atlas is like a road map of the Earth: no single flat page can
  show the whole sphere, but a collection of overlapping pages (charts)
  covers it completely.
  
  In Manifold DB:
    - Each chart represents a cluster or region of your data
    - The chart's coordinate system is a reduced-dimension projection
    - Transition maps connect overlapping charts
  """)

    # Create a simple chart
    chart = Chart(
        name="cluster_0",
        dim=5,           # intrinsic dimension (coordinate system)
        ambient_dim=20,   # embedding space dimension
    )
    print(f"\n  Created chart: {chart}")
    print(f"    dim (intrinsic):  {chart.dim}")
    print(f"    ambient_dim:      {chart.ambient_dim}")
    print(f"    chart_id:        {chart.chart_id}")

    # Embed some data
    data = np.random.randn(10, 20)
    coords = chart.embed(data)
    print(f"\n  Embedded {data.shape} → {coords.shape}")
    print(f"    (projected from {chart.ambient_dim}D to {chart.dim}D coords)")

    # Lift back
    recovered = chart.inverse(coords)
    print(f"  Lifted {coords.shape} → {recovered.shape}")

    # Create an atlas
    atlas = AtlasManager(name="tutorial_atlas")
    atlas.add_chart(chart)
    print(f"\n  Atlas: {atlas}")
    print(f"    Charts: {len(atlas.get_all_charts())}")

    # Auto-build atlas from data
    print("\n  Auto-building atlas from data...")
    builder = AtlasBuilder(k_neighbors=10, min_chart_size=20, random_state=42)
    auto_data = np.random.randn(300, 10)
    auto_atlas = builder.build(auto_data, n_charts_hint=3)
    print(f"  Auto-built: {len(auto_atlas.get_all_charts())} charts, "
          f"{len(auto_atlas.get_all_transition_maps())} transitions")

    for chart in auto_atlas.get_all_charts():
        n_pts = chart.metadata.get("n_points", "?")
        print(f"    '{chart.name}': dim={chart.dim}, points={n_pts}")


# ═══════════════════════════════════════════════════════════════════════
# Section 3: Tangent Spaces
# ═══════════════════════════════════════════════════════════════════════

def section_3_tangent_spaces():
    """
    SECTION 3: Tangent Spaces
    
    The tangent space T_p(M) at a point p on the manifold is the best
    linear approximation of M near p.  It's like the tangent plane to
    a surface at a point.
    
    Operations:
      - project(data) → tangent_coords: ambient → tangent
      - lift(tangent_coords) → ambient: tangent → ambient
      - log_map(point) → tangent_vec: point on M → vector in T_p
      - exp_map(tangent_vec) → point: vector in T_p → point on M
    """
    print("\n" + "=" * 72)
    print("  SECTION 3: Tangent Spaces")
    print("=" * 72)
    print("""
  At every point on a manifold, there's a tangent space — a flat plane
  that just touches the manifold at that point.  Near the point, the
  manifold and its tangent plane are almost identical.
  
  This is the key to efficient computation: in the tangent space, we can
  use standard Euclidean operations (dot products, distances, nearest
  neighbours) that would be wrong in the ambient space.
  """)

    # Create tangent space from data
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, size=(100, 20))
    base_point = data[0]
    
    ts = TangentSpace(base_point=base_point, data=data[:50], intrinsic_dim=5)
    print(f"\n  TangentSpace: {ts}")
    print(f"    Ambient dimension:  {ts._ambient_dim}")
    print(f"    Intrinsic dim:     {ts.dimension}")
    print(f"    Basis shape:       {ts.basis.shape}  (ambient × intrinsic)")
    print(f"    Metric tensor:     {ts.metric_tensor.shape}  (intrinsic × intrinsic)")

    # Project data
    sample = data[:5]
    tangent_coords = ts.project(sample)
    print(f"\n  Project {sample.shape} → {tangent_coords.shape}")

    # Lift back
    recovered = ts.lift(tangent_coords)
    recon_error = np.mean(np.linalg.norm(sample - recovered, axis=1))
    print(f"  Lift {tangent_coords.shape} → {recovered.shape}")
    print(f"  Reconstruction error: {recon_error:.6f}")

    # Log map: maps a point on M to a vector in T_p
    target_point = data[10]
    log_vec = ts.log_map(target_point)
    print(f"\n  Log map (point → tangent vector):")
    print(f"    Target point: dim={target_point.shape[0]}")
    print(f"    Log vector:   dim={log_vec.shape[0]}, norm={np.linalg.norm(log_vec):.4f}")

    # Exp map: maps a tangent vector to a point on M
    exp_point = ts.exp_map(log_vec)
    exp_error = np.linalg.norm(target_point - exp_point)
    print(f"\n  Exp map (tangent vector → point):")
    print(f"    Input:  dim={log_vec.shape[0]}")
    print(f"    Output: dim={exp_point.shape[0]}")
    print(f"    Roundtrip error: {exp_error:.6f}")

    # Metric tensor: inner products in tangent space
    v1 = np.random.randn(ts.dimension)
    v2 = np.random.randn(ts.dimension)
    inner = ts.compute_metric(v1, v2)
    eucl_inner = np.dot(v1, v2)
    dist = ts.compute_distance(v1, v2)
    print(f"\n  Riemannian metric:")
    print(f"    <v1, v2>_G = {inner:.6f}  (vs Euclidean {eucl_inner:.6f})")
    print(f"    d(v1, v2)  = {dist:.6f}")


# ═══════════════════════════════════════════════════════════════════════
# Section 4: Geodesic Distances
# ═══════════════════════════════════════════════════════════════════════

def section_4_geodesic_distances():
    """
    SECTION 4: Geodesic Distances
    
    A geodesic is the generalisation of a straight line to curved
    manifolds.  The geodesic distance d_g(p, q) is the length of the
    shortest path from p to q that stays on the manifold.
    
    For a flat manifold, geodesic distance = Euclidean distance.
    For curved manifolds, geodesic distance accounts for curvature.
    """
    print("\n" + "=" * 72)
    print("  SECTION 4: Geodesic Distances")
    print("=" * 72)
    print("""
  On a flat surface, the shortest path is a straight line (Euclidean).
  On a sphere, the shortest path is a great circle arc.
  On a Swiss roll, the shortest path follows the surface of the roll.
  
  Geodesic distance captures "true" similarity that Euclidean distance
  misses when the manifold is curved.
  """)

    # Create a curved metric
    from manifold_db.geodesic.distance import RiemannianDistance, DistanceComputer

    def curved_metric(x: np.ndarray) -> np.ndarray:
        """Non-uniform metric that makes space 'curved'."""
        dim = len(x)
        scale = 1.0 + 0.5 * np.sin(x[0])  # varies with position
        return np.eye(dim) * scale

    # Compare distances
    p = np.array([0.0, 0.0])
    q = np.array([2.0, 1.0])

    dist_comp = DistanceComputer(metric_tensor_fn=curved_metric)

    tangent_dist = dist_comp.compute(p, q, metric_type="tangent")
    eucl_dist = np.linalg.norm(p - q)
    
    print(f"\n  Points: p=({p[0]}, {p[1]}), q=({q[0]}, {q[1]})")
    print(f"  Euclidean distance:    {eucl_dist:.6f}")
    print(f"  Tangent distance:     {tangent_dist:.6f}")
    print(f"  Difference:           {abs(tangent_dist - eucl_dist):.6f}")
    print(f"""
  When the metric varies across space, the "shortest path" is no longer
  a straight line.  The tangent-space distance accounts for this by
  using the local metric at the midpoint between p and q.
  """)

    # Metric tensor store
    print("\n  MetricTensorStore: per-chart metric management")
    store = MetricTensorStore()
    
    # Register Euclidean metric for one chart
    store.register("chart_0", EuclideanMetric(dimension=5))
    
    # Register diagonal metric for another chart
    store.register("chart_1", DiagonalMetric(
        dimension=5,
        weights=np.array([1.0, 2.0, 0.5, 1.0, 1.5]),
    ))
    
    print(f"    Registered metrics: {store.list_charts()}")
    
    g0 = store.get("chart_0")
    g1 = store.get("chart_1")
    print(f"    chart_0 metric (diagonal): {np.diag(g0.evaluate(np.zeros(5)))}")
    print(f"    chart_1 metric (diagonal): {np.diag(g1.evaluate(np.zeros(5)))}")


# ═══════════════════════════════════════════════════════════════════════
# Section 5: Parallel Transport
# ═══════════════════════════════════════════════════════════════════════

def section_5_parallel_transport():
    """
    SECTION 5: Parallel Transport
    
    Parallel transport moves a tangent vector from one point on the manifold
    to another while keeping it "parallel" (preserving its direction and
    magnitude relative to the manifold).
    
    On a flat surface, parallel transport is trivial (just copy the vector).
    On a sphere, transporting a vector around a closed loop rotates it!
    (This is the geometric origin of curvature.)
    """
    print("\n" + "=" * 72)
    print("  SECTION 5: Parallel Transport")
    print("=" * 72)
    print("""
  Imagine walking on the Earth's surface while holding an arrow pointing
  "north".  If you walk from the equator to the North Pole, the arrow
  still points "north" at the pole.  But if you walk along the equator
  first, then north, the arrow points in a different direction!
  
  This is parallel transport: the rule for moving vectors along the
  manifold.  It depends on the PATH taken, not just the endpoints.
  
  In Manifold DB, parallel transport lets us:
    - Move query vectors between charts
    - Compare vectors defined at different points
    - Perform cross-modal retrieval
  """)

    # Create two tangent spaces and transport between them
    rng = np.random.default_rng(42)
    data_a = rng.normal(0, 1, (50, 10))
    data_b = rng.normal(2, 1, (50, 10))  # shifted distribution

    ts_a = TangentSpace(base_point=data_a.mean(axis=0), data=data_a, intrinsic_dim=5)
    ts_b = TangentSpace(base_point=data_b.mean(axis=0), data=data_b, intrinsic_dim=5)

    vec = np.random.randn(5)
    print(f"\n  Source tangent space: base norm = {np.linalg.norm(ts_a.base_point):.3f}")
    print(f"  Target tangent space: base norm = {np.linalg.norm(ts_b.base_point):.3f}")
    print(f"  Vector to transport: norm = {np.linalg.norm(vec):.3f}")

    transported = ts_a.parallel_transport(vec, ts_b)
    print(f"  Transported vector:   norm = {np.linalg.norm(transported):.3f}")

    # Check magnitude preservation
    source_norm = ts_a.compute_metric(vec, vec)
    target_norm = ts_b.compute_metric(transported, transported)
    print(f"  Source metric norm:   {np.sqrt(source_norm):.6f}")
    print(f"  Target metric norm:   {np.sqrt(target_norm):.6f}")
    print(f"  Norm preservation:   {np.sqrt(target_norm)/np.sqrt(source_norm):.6f}")

    # TransportRegistry
    print("\n  TransportRegistry: caching transport operations")
    registry = TransportRegistry(max_size=64)
    
    def transport_fn(v: np.ndarray) -> np.ndarray:
        return ts_a.parallel_transport(v, ts_b)
    
    registry.register_transport("ts_a", "ts_b", transport_fn)
    
    result = registry.get_transport("ts_a", "ts_b")(vec)
    cached = registry.get_cached("ts_a", "ts_b", vec)
    print(f"    Transport executed: {np.allclose(result, transported)}")
    print(f"    Cache available: {cached is not None}")


# ═══════════════════════════════════════════════════════════════════════
# Section 6: Multi-Modal Queries
# ═══════════════════════════════════════════════════════════════════════

def section_6_multi_modal_queries():
    """
    SECTION 6: Multi-Modal Queries
    
    Multi-modal data (text, images, audio, tabular) often lives in
    different charts of the same atlas.  Cross-modal retrieval uses
    parallel transport to bridge between modalities.
    """
    print("\n" + "=" * 72)
    print("  SECTION 6: Multi-Modal Queries")
    print("=" * 72)
    print("""
  In a multi-modal system:
    - Text embeddings live in one chart (e.g., 768D SBERT)
    - Image embeddings live in another chart (e.g., 512D CLIP)
    - Image-caption pairs create overlap between charts
    - Parallel transport bridges the gap for cross-modal retrieval
  """)

    # Build cross-modal query
    query = (
        QueryBuilder()
        .cross_modal("text", "image")
        .with_transport("overlap_region")
        .top_k(10)
        .build()
    )
    print(f"\n  Cross-modal query:")
    print(f"    Type: {query.query_type.value}")
    print(f"    Source modality: {query.modality}")
    print(f"    Target modality: {query.target_modality}")
    print(f"    Transport via:   {query.transport_via}")
    print(f"    k:               {query.k}")
    print(f"    Valid:           {query.validate()}")
    print(f"    Cost tier:      {query.estimate_cost()}")

    # Transport query
    transport_q = (
        QueryBuilder()
        .parallel_transport(
            vector=np.array([1.0, 2.0, 3.0]),
            source="chart_text",
            target="chart_image",
        )
        .build()
    )
    print(f"\n  Transport query:")
    print(f"    Type:   {transport_q.query_type.value}")
    print(f"    Vector: {transport_q.transport_vector}")
    print(f"    Source:  {transport_q.source_chart}")
    print(f"    Target:  {transport_q.target_chart}")

    # SQL-like parsing
    print("\n  SQL-like query parsing:")
    parser = QueryParser()
    
    sql_examples = [
        "SELECT * FROM observations WHERE geodesic_distance(embedding, [1,2,3]) < 0.5",
        "TANGENT_QUERY FROM chart 'text_embeddings' WHERE distance < 0.5",
        "CROSS_MODAL FROM 'text' TO 'image' TRANSPORT VIA 'overlap_region'",
    ]
    
    for sql in sql_examples:
        parsed = parser.parse(sql)
        print(f"    SQL: {sql[:60]}...")
        print(f"      → type={parsed.query_type.value}, k={parsed.k}")


# ═══════════════════════════════════════════════════════════════════════
# Section 7: Full Pipeline
# ═══════════════════════════════════════════════════════════════════════

def section_7_full_pipeline():
    """
    SECTION 7: Putting it All Together
    
    Complete pipeline: data → atlas → index → query → results.
    """
    print("\n" + "=" * 72)
    print("  SECTION 7: Full Pipeline")
    print("=" * 72)
    print("""
  The complete Manifold DB workflow:
    1. Generate/load data
    2. Build atlas (discover charts and transitions)
    3. Build tangent-space index (for fast search)
    4. Insert/query data
    5. Save/load the database
  """)

    # 1. Data
    print("\n  [1] Generating data...")
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, size=(500, 15))
    pids = [f"doc_{i:04d}" for i in range(len(data))]
    print(f"      {data.shape}")

    # 2. Atlas
    print("  [2] Building atlas...")
    t0 = time.time()
    builder = AtlasBuilder(k_neighbors=10, min_chart_size=20, random_state=42)
    atlas = builder.build(data, n_charts_hint=3)
    print(f"      {len(atlas.get_all_charts())} charts, "
          f"{len(atlas.get_all_transition_maps())} transitions "
          f"({time.time()-t0:.3f}s)")

    # 3. Index
    print("  [3] Building tangent-space index...")
    t0 = time.time()
    index = TangentSpaceIndex(intrinsic_dim=5)
    stats = index.build_from_data(pids, data, n_anchors=20)
    print(f"      {stats['n_points']} points, {stats['n_anchors']} anchors "
          f"({stats['build_time_sec']:.3f}s)")

    # 4. Query
    print("  [4] Executing queries...")
    query_point = data[0]
    
    # Tangent query
    results = index.search(query_point, k=5)
    print(f"      Nearest neighbours of 'doc_0000':")
    for pid, dist in results:
        print(f"        → {pid}  dist={dist:.4f}")

    # 5. Save
    print("  [5] Saving database...")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(output_dir, exist_ok=True)
    
    atlas_path = os.path.join(output_dir, "tutorial_atlas.json")
    atlas.save(atlas_path)
    print(f"      Atlas saved: {atlas_path}")

    print("\n  Pipeline complete!")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 72)
    print("  Manifold Database — Tutorial Walkthrough")
    print("  A step-by-step guide to manifold database concepts")
    print("=" * 72)

    section_1_manifold_intuition()
    section_2_charts_and_atlases()
    section_3_tangent_spaces()
    section_4_geodesic_distances()
    section_5_parallel_transport()
    section_6_multi_modal_queries()
    section_7_full_pipeline()

    print("\n" + "=" * 72)
    print("  Tutorial Complete!")
    print("=" * 72)
    print("""
  Congratulations! You've learned:
    ✓ What manifolds are and why they matter for data
    ✓ Charts and atlases: local coordinate systems
    ✓ Tangent spaces: local linear approximations
    ✓ Geodesic distances: true distances on curved spaces
    ✓ Parallel transport: moving vectors across the manifold
    ✓ Multi-modal queries: cross-chart retrieval
    ✓ The full Manifold DB pipeline

  Next steps:
    • Read docs/architecture/overview.md for system design
    • Read docs/tutorials/advanced-queries.md for query patterns
    • Run examples/scripts/multi_modal_rag.py for a deeper example
    • Run examples/scripts/benchmark_suite.py for performance data
    """)


if __name__ == "__main__":
    main()
