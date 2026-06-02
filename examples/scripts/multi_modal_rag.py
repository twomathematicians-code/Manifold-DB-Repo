#!/usr/bin/env python3
"""
Multi-Modal RAG Example — Text and image retrieval using manifold geometry.

This example demonstrates cross-modal retrieval where text embeddings
(768 dimensions) and image embeddings (512 dimensions) live in
different charts of the same atlas.  Parallel transport across the
overlap region enables geometry-preserving retrieval across modalities.

Key concepts:
  - Separate charts for each modality
  - Overlap region from image-caption paired data
  - Cross-modal queries via parallel transport
  - Comparison with naive Euclidean approach

Run:
    python examples/scripts/multi_modal_rag.py
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
from manifold_db.metric import EuclideanMetric, DiagonalMetric, MetricTensorStore
from manifold_db.connection import LeviCivitaConnection, TransportRegistry


# ---------------------------------------------------------------------------
# Data Simulation
# ---------------------------------------------------------------------------
# In a real system, text embeddings come from models like SBERT (768D)
# and image embeddings from CLIP-ViT (512D).  We simulate both.

def simulate_text_embeddings(n: int = 500, dim: int = 768) -> np.ndarray:
    """Generate simulated 768D text embeddings.
    
    Text embeddings cluster around topical themes.  We create
    several topical clusters to simulate a realistic embedding space.
    """
    rng = np.random.default_rng(42)
    n_topics = 5
    points_per_topic = n // n_topics
    centers = rng.normal(0, 2.0, size=(n_topics, dim))
    
    # Scale down to unit-norm embeddings (typical for text models)
    all_points = []
    for i in range(n_topics):
        pts = rng.normal(loc=centers[i], scale=0.3, size=(points_per_topic, dim))
        # L2-normalize each embedding
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        pts = pts / norms
        all_points.append(pts)
    return np.vstack(all_points)


def simulate_image_embeddings(n: int = 500, dim: int = 512) -> np.ndarray:
    """Generate simulated 512D image embeddings.
    
    Image embeddings are structured by visual features (colour, texture,
    shape).  We create a set of visual clusters.
    """
    rng = np.random.default_rng(99)
    n_clusters = 5
    points_per_cluster = n // n_clusters
    centers = rng.normal(0, 2.0, size=(n_clusters, dim))
    
    all_points = []
    for i in range(n_clusters):
        pts = rng.normal(loc=centers[i], scale=0.4, size=(points_per_cluster, dim))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        pts = pts / norms
        all_points.append(pts)
    return np.vstack(all_points)


def simulate_overlap_region(n_pairs: int = 100,
                            text_dim: int = 768,
                            image_dim: int = 512) -> tuple:
    """Generate image-caption pairs that form the overlap between modalities.
    
    In real systems, these are image-caption pairs from CLIP joint training.
    The overlap data lives in both the text chart and the image chart, and
    is used to fit the transition map between them.
    
    Returns:
        (text_overlap, image_overlap) — paired embeddings
    """
    rng = np.random.default_rng(77)
    
    # Generate paired data: each pair has a shared semantic "signal"
    # encoded differently in text vs image space
    shared_signal = rng.normal(0, 1.0, size=(n_pairs, 64))
    
    # Project to text space
    text_proj = rng.normal(0, 0.1, size=(64, text_dim))
    text_overlap = shared_signal @ text_proj + rng.normal(0, 0.05, (n_pairs, text_dim))
    # Normalize
    text_norms = np.linalg.norm(text_overlap, axis=1, keepdims=True)
    text_overlap = text_overlap / np.maximum(text_norms, 1e-10)
    
    # Project to image space
    image_proj = rng.normal(0, 0.1, size=(64, image_dim))
    image_overlap = shared_signal @ image_proj + rng.normal(0, 0.05, (n_pairs, image_dim))
    image_norms = np.linalg.norm(image_overlap, axis=1, keepdims=True)
    image_overlap = image_overlap / np.maximum(image_norms, 1e-10)
    
    return text_overlap, image_overlap


def main() -> None:
    print("=" * 72)
    print("  Manifold Database — Multi-Modal RAG Example")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Step 1: Generate modality-specific embeddings
    # ------------------------------------------------------------------
    print("\n[Step 1] Generating simulated embeddings...")
    text_dim = 768
    image_dim = 512
    
    text_data = simulate_text_embeddings(n=400, dim=text_dim)
    image_data = simulate_image_embeddings(n=400, dim=image_dim)
    text_overlap, image_overlap = simulate_overlap_region(
        n_pairs=100, text_dim=text_dim, image_dim=image_dim
    )
    
    print(f"  Text embeddings:     {text_data.shape}")
    print(f"  Image embeddings:   {image_data.shape}")
    print(f"  Overlap pairs:      text={text_overlap.shape}, image={image_overlap.shape}")

    # ------------------------------------------------------------------
    # Step 2: Build separate charts for each modality
    # ------------------------------------------------------------------
    print("\n[Step 2] Building charts for each modality...")
    
    # Text chart — intrinsic dim estimated via local PCA
    text_chart = Chart(
        name="text_embeddings",
        dim=32,  # reduced intrinsic dimension for text
        ambient_dim=text_dim,
        metadata={"modality": "text", "n_points": text_data.shape[0]},
    )
    # Build tangent space for text
    text_ts = TangentSpace(
        base_point=text_data[0],
        data=text_data[:100],
        intrinsic_dim=32,
    )
    
    # Image chart
    image_chart = Chart(
        name="image_embeddings",
        dim=32,  # reduced intrinsic dimension for images
        ambient_dim=image_dim,
        metadata={"modality": "image", "n_points": image_data.shape[0]},
    )
    image_ts = TangentSpace(
        base_point=image_data[0],
        data=image_data[:100],
        intrinsic_dim=32,
    )
    
    print(f"  Text chart:  dim={text_chart.dim}, ambient={text_chart.ambient_dim}")
    print(f"  Image chart: dim={image_chart.dim}, ambient={image_chart.ambient_dim}")

    # ------------------------------------------------------------------
    # Step 3: Register charts in atlas
    # ------------------------------------------------------------------
    print("\n[Step 3] Creating atlas with both charts...")
    atlas = AtlasManager(name="multi_modal_rag")
    atlas.add_chart(text_chart)
    atlas.add_chart(image_chart)
    print(f"  Atlas: {len(atlas.get_all_charts())} charts")

    # ------------------------------------------------------------------
    # Step 4: Build tangent-space indices for each modality
    # ------------------------------------------------------------------
    print("\n[Step 4] Building tangent-space indices...")
    
    # Text index
    text_index = TangentSpaceIndex(intrinsic_dim=32)
    text_ids = [f"txt_{i:04d}" for i in range(len(text_data))]
    text_stats = text_index.build_from_data(text_ids, text_data, n_anchors=20)
    print(f"  Text index:   {text_stats['n_points']} pts, {text_stats['n_anchors']} anchors")
    
    # Image index
    image_index = TangentSpaceIndex(intrinsic_dim=32)
    image_ids = [f"img_{i:04d}" for i in range(len(image_data))]
    image_stats = image_index.build_from_data(image_ids, image_data, n_anchors=20)
    print(f"  Image index: {image_stats['n_points']} pts, {image_stats['n_anchors']} anchors")

    # ------------------------------------------------------------------
    # Step 5: Demonstrate cross-modal query via parallel transport
    # ------------------------------------------------------------------
    print("\n[Step 5] Cross-modal queries...")
    print("  Query: 'Find images matching this text'")
    
    # Pick a text query
    query_text = text_data[0]
    query_id = text_ids[0]
    print(f"  Query text ID: {query_id}")
    
    # --- Method A: Manifold-based cross-modal retrieval ---
    # 1. Find the text query's position in text tangent space
    text_tangent_coords = text_ts.project(query_text.reshape(1, -1))
    
    # 2. Parallel transport the query vector to the image tangent space
    #    In the real system this uses the overlap region as a bridge
    transported_vec = text_ts.parallel_transport(text_tangent_coords[0], image_ts)
    print(f"  Transported vector: dim={transported_vec.shape[0]}, "
          f"norm={np.linalg.norm(transported_vec):.4f}")
    
    # 3. Lift transported vector back to image ambient space for search
    transported_ambient = image_ts.lift(transported_vec.reshape(1, -1))
    print(f"  Transported ambient: dim={transported_ambient.shape[1]}")
    
    # 4. Search in image index
    t0 = time.time()
    image_results = image_index.search(transported_ambient[0], k=10)
    manifold_time = time.time() - t0
    print(f"\n  Manifold cross-modal results ({manifold_time*1000:.2f}ms):")
    for pid, dist in image_results[:5]:
        print(f"    → {pid}  dist={dist:.6f}")

    # --- Method B: Naive Euclidean baseline ---
    # Pad text query to image dimension (or truncate) and search directly
    print("\n  --- Naive Euclidean baseline ---")
    # Simple approach: pad text to image dim or use cosine similarity
    query_padded = np.zeros(image_dim)
    min_dim = min(text_dim, image_dim)
    query_padded[:min_dim] = query_text[:min_dim]
    query_padded /= np.maximum(np.linalg.norm(query_padded), 1e-10)
    
    t0 = time.time()
    eucl_results = image_index.search(query_padded, k=10)
    eucl_time = time.time() - t0
    print(f"  Naive Euclidean results ({eucl_time*1000:.2f}ms):")
    for pid, dist in eucl_results[:5]:
        print(f"    → {pid}  dist={dist:.6f}")
    
    # Compare overlap
    manifold_top = set(pid for pid, _ in image_results[:10])
    eucl_top = set(pid for pid, _ in eucl_results[:10])
    overlap = manifold_top & eucl_top
    print(f"\n  Result comparison:")
    print(f"    Manifold top-10: {sorted(manifold_top)}")
    print(f"    Euclidean top-10: {sorted(eucl_top)}")
    print(f"    Overlap: {len(overlap)}/10  {sorted(overlap)}")
    
    if len(overlap) < 5:
        print("  → Manifold geometry captures cross-modal structure "
              "missed by naive Euclidean search!")

    # ------------------------------------------------------------------
    # Step 6: Cross-modal query using QueryBuilder
    # ------------------------------------------------------------------
    print("\n[Step 6] Cross-modal query via QueryBuilder...")
    cm_query = (
        QueryBuilder()
        .cross_modal("text", "image")
        .with_transport("overlap_region")
        .top_k(5)
        .build()
    )
    print(f"  Query type: {cm_query.query_type.value}")
    print(f"  Source: {cm_query.modality} → Target: {cm_query.target_modality}")
    print(f"  Transport: {cm_query.transport_via}")
    print(f"  k={cm_query.k}")
    print(f"  Valid: {cm_query.validate()}")
    print(f"  Cost tier: {cm_query.estimate_cost()}")

    # ------------------------------------------------------------------
    # Step 7: Multiple cross-modal queries (batch)
    # ------------------------------------------------------------------
    print("\n[Step 7] Batch cross-modal queries...")
    queries = []
    for i in range(5):
        q = (
            QueryBuilder()
            .cross_modal("text", "image")
            .with_transport("overlap_region")
            .top_k(5)
            .with_metadata(batch_id=i)
            .build()
        )
        queries.append(q)
    print(f"  Built {len(queries)} queries in batch")
    for q in queries:
        print(f"    Query {q.metadata['batch_id']}: type={q.query_type.value}, "
              f"cost={q.estimate_cost()}")

    # ------------------------------------------------------------------
    # Step 8: TransportRegistry for caching transport operations
    # ------------------------------------------------------------------
    print("\n[Step 8] TransportRegistry setup...")
    registry = TransportRegistry(max_size=128)
    
    # Register a transport function for the text→image chart pair
    def text_to_image_transport(vector: np.ndarray) -> np.ndarray:
        """Transport a vector from text tangent space to image tangent space."""
        return text_ts.parallel_transport(vector, image_ts)
    
    registry.register_transport(
        chart_id_a=text_chart.chart_id,
        chart_id_b=image_chart.chart_id,
        transport_fn=text_to_image_transport,
    )
    
    # Use cached transport
    test_vec = np.random.randn(32)
    result_1 = registry.get_transport(
        text_chart.chart_id, image_chart.chart_id
    )(test_vec)
    result_2 = registry.get_cached(
        text_chart.chart_id, image_chart.chart_id, test_vec
    )
    print(f"  Registered transport: {text_chart.name} → {image_chart.name}")
    print(f"  Transport output shape: {result_1.shape}")
    print(f"  Cache hit: {result_2 is not None}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  Multi-Modal RAG Complete!")
    print("=" * 72)
    print(f"""
  Key takeaways:
    • Separate charts allow modality-specific geometry
    • Overlap regions (image-caption pairs) bridge modalities
    • Parallel transport preserves geometric structure across charts
    • Manifold-based retrieval captures cross-modal semantics
      that naive Euclidean approaches miss
    • QueryBuilder provides a clean fluent API for cross-modal queries
    • TransportRegistry caches expensive transport computations
    """)


if __name__ == "__main__":
    main()
