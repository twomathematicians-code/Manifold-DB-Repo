#!/usr/bin/env python3
"""
cross_modal_demo.py - Cross-Modal Retrieval with ManifoldDB
============================================================

This demo demonstrates ManifoldDB's cross-modal search capability, which
enables retrieval across different data modalities (e.g., text embeddings
and image embeddings) using a shared manifold structure.

What this demo covers:

  1.  Generate two sets of embeddings with *different dimensions*:
        - "Text" embeddings: dim = 16  (modality_id = 0)
        - "Image" embeddings: dim = 32 (modality_id = 1)
      The image embeddings are projected into a common latent space so
      both modalities share the same ambient dimensionality for atlas
      construction.

  2.  Insert both sets into ManifoldDB with distinct modality_id values.

  3.  Build a unified atlas that covers both modalities.

  4.  Query with a text embedding and retrieve semantically similar
      image results via cross-modal geodesic transport.

  5.  Query with an image embedding and retrieve matching text results.

  6.  Display results with topic labels and accuracy metrics.

Why Cross-Modal Retrieval Matters
---------------------------------
In real-world applications (e.g., CLIP, image-text search), queries come in
one modality but the desired results are in another.  A naive Euclidean
search across modalities fails because the embeddings may have very different
distributions.  ManifoldDB builds a shared atlas over both modalities, enabling
geodesic-distance-based cross-modal retrieval that respects the geometry of
each modality's embedding space.

Run with::

    python examples/cross_modal_demo.py

Requirements
------------
- numpy
- manifolddb C++ extension
"""

from __future__ import annotations

import sys
import tempfile
import shutil

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


# ---------------------------------------------------------------------------
# Data Generators
# ---------------------------------------------------------------------------

# Topic names for the demo
TOPIC_NAMES = ["animals", "architecture", "food", "vehicles"]

# Shared ambient dimensionality (both modalities are projected here)
SHARED_DIM = 16

# Text modality dimension (original, before projection)
TEXT_DIM = 16

# Image modality dimension (larger than text, simulating real-world asymmetry)
IMAGE_DIM = 32


def generate_text_embeddings(
    n_per_topic: int = 60,
    seed: int = 100,
) -> tuple:
    """Generate synthetic text-like embeddings clustered around semantic topics.

    Each topic is represented by a centroid in R^{TEXT_DIM}.  Points are sampled
    as Gaussian perturbations around their topic centroid.  The centroids are
    arranged in a well-separated circular layout to ensure distinct clusters.

    Parameters
    ----------
    n_per_topic : int
        Number of text embeddings per topic.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray, shape (n_total, SHARED_DIM)
        Text embeddings projected into the shared ambient space.
    list[str]
        Topic label for each embedding.
    """
    rng = np.random.default_rng(seed)
    n_topics = len(TOPIC_NAMES)

    # Arrange topic centroids in a circle for clear separation
    centroid_angles = np.linspace(0, 2 * np.pi, n_topics, endpoint=False)
    centroid_radius = 4.0

    # Centroids in 2-D (the "semantic" plane), padded to SHARED_DIM
    centroids = np.zeros((n_topics, SHARED_DIM), dtype=np.float64)
    for i, angle in enumerate(centroid_angles):
        centroids[i, 0] = centroid_radius * np.cos(angle)
        centroids[i, 1] = centroid_radius * np.sin(angle)
        # Remaining dimensions: small random values to fill the space
        if SHARED_DIM > 2:
            centroids[i, 2:] = rng.standard_normal(SHARED_DIM - 2) * 0.3

    all_points = []
    all_labels = []

    for i in range(n_topics):
        pts = rng.normal(loc=centroids[i], scale=0.4,
                         size=(n_per_topic, SHARED_DIM)).astype(np.float64)
        all_points.append(pts)
        all_labels.extend([TOPIC_NAMES[i]] * n_per_topic)

    data = np.vstack(all_points)
    return data, all_labels


def generate_image_embeddings(
    text_centroids: np.ndarray,
    n_per_topic: int = 60,
    seed: int = 200,
) -> tuple:
    """Generate synthetic image-like embeddings aligned with text topics.

    Image embeddings are created at a *higher original dimension* (IMAGE_DIM)
    and then projected down to SHARED_DIM.  Each image cluster centroid is
    offset from the corresponding text centroid, simulating the modality gap
    seen in real CLIP-like models.

    Parameters
    ----------
    text_centroids : numpy.ndarray, shape (n_topics, SHARED_DIM)
        Text topic centroids (used as anchors for alignment).
    n_per_topic : int
        Number of image embeddings per topic.
    seed : int
        Random seed.

    Returns
    -------
    numpy.ndarray, shape (n_total, SHARED_DIM)
        Image embeddings in the shared ambient space.
    list[str]
        Topic label for each embedding.
    """
    rng = np.random.default_rng(seed)
    n_topics = len(TOPIC_NAMES)

    all_points = []
    all_labels = []

    for i in range(n_topics):
        # Generate in high-dimensional space first
        high_dim_pts = rng.normal(
            loc=0, scale=1.0,
            size=(n_per_topic, IMAGE_DIM),
        )

        # Project to shared dimension via a random linear projection + bias
        # The bias shifts each topic near the text centroid (modality alignment)
        projection = rng.standard_normal((IMAGE_DIM, SHARED_DIM)) * 0.3
        bias = text_centroids[i] + rng.standard_normal(SHARED_DIM) * 0.5

        pts = (high_dim_pts @ projection + bias).astype(np.float64)

        all_points.append(pts)
        all_labels.extend([TOPIC_NAMES[i]] * n_per_topic)

    data = np.vstack(all_points)
    return data, all_labels


# ---------------------------------------------------------------------------
# Cross-Modal Retrieval Evaluation
# ---------------------------------------------------------------------------

def evaluate_retrieval(
    results: list,
    labels: list,
    query_topic: str,
    k: int,
    modality_name: str = "target",
) -> dict:
    """Evaluate cross-modal retrieval results against ground truth.

    Parameters
    ----------
    results : list[dict]
        Retrieval results from ManifoldDB.
    labels : list[str]
        Ground-truth topic labels for the target modality.
    query_topic : str
        The topic of the query (expected label for correct results).
    k : int
        Number of results requested.
    modality_name : str
        Name of the target modality (for display).

    Returns
    -------
    dict
        Evaluation metrics: correct_count, accuracy, per-result details.
    """
    correct = 0
    details = []

    for rank, r in enumerate(results):
        rid = r["id"]
        if 0 <= rid < len(labels):
            label = labels[rid]
        else:
            label = "unknown"
        is_match = label == query_topic
        if is_match:
            correct += 1
        details.append({
            "rank": rank,
            "id": rid,
            "label": label,
            "distance": r["distance"],
            "match": is_match,
        })

    return {
        "correct": correct,
        "total": len(results),
        "accuracy": correct / max(len(results), 1),
        "details": details,
    }


# ---------------------------------------------------------------------------
# Main Demo
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 64)
    print("  ManifoldDB — Cross-Modal Retrieval Demo")
    print(f"  Topics: {TOPIC_NAMES}")
    print(f"  Shared ambient dim: {SHARED_DIM}")
    print(f"  Text dim (orig): {TEXT_DIM}, Image dim (orig): {IMAGE_DIM}")
    print("=" * 64)
    print()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    N_PER_TOPIC = 60
    INTRINSIC_DIM = 4
    K = 10

    TEXT_MODALITY = 0
    IMAGE_MODALITY = 1

    storage_dir = tempfile.mkdtemp(prefix="manifolddb_crossmodal_")
    print(f"[0] Storage: {storage_dir}")

    try:
        # ------------------------------------------------------------------
        # Step 1: Generate synthetic multi-modal data
        # ------------------------------------------------------------------
        print(f"\n[1] Generating synthetic embeddings ...")
        print(f"    Points per topic per modality: {N_PER_TOPIC}")

        text_data, text_labels = generate_text_embeddings(
            n_per_topic=N_PER_TOPIC, seed=100,
        )
        print(f"    Text embeddings:   {text_data.shape} (modality {TEXT_MODALITY})")

        # Compute text centroids for image alignment
        text_centroids = np.zeros((len(TOPIC_NAMES), SHARED_DIM), dtype=np.float64)
        counts = np.zeros(len(TOPIC_NAMES))
        for pt, lbl in zip(text_data, text_labels):
            idx = TOPIC_NAMES.index(lbl)
            text_centroids[idx] += pt
            counts[idx] += 1
        text_centroids /= counts[:, np.newaxis]

        image_data, image_labels = generate_image_embeddings(
            text_centroids, n_per_topic=N_PER_TOPIC, seed=200,
        )
        print(f"    Image embeddings: {image_data.shape} (modality {IMAGE_MODALITY})")

        total_points = len(text_data) + len(image_data)
        print(f"    Total: {total_points} points across 2 modalities")

        # ------------------------------------------------------------------
        # Step 2: Initialise ManifoldDB and insert both modalities
        # ------------------------------------------------------------------
        print(f"\n[2] Initialising ManifoldDB (intrinsic_dim={INTRINSIC_DIM}) ...")
        db = ManifoldDB(
            storage_path=storage_dir,
            intrinsic_dim=INTRINSIC_DIM,
            enable_cuda=False,
            geodesic_tolerance=1e-6,
        )
        print(f"    {db}")

        print("\n    Inserting text embeddings ...")
        db.insert(text_data, modality_id=TEXT_MODALITY)

        print("    Inserting image embeddings ...")
        db.insert(image_data, modality_id=IMAGE_MODALITY)

        # ------------------------------------------------------------------
        # Step 3: Build unified atlas
        # ------------------------------------------------------------------
        print("\n[3] Building unified atlas over both modalities ...")
        db.build(method="linear")

        stats = db.stats()
        print(f"    Charts:  {stats['num_charts']}")
        print(f"    Points:  {stats['total_points']}")
        print(f"    Index:   {stats['index_size']}")

        # ------------------------------------------------------------------
        # Step 4: Cross-modal retrieval: Text → Image
        # ------------------------------------------------------------------
        print(f"\n[4] Cross-modal retrieval: Text → Image")
        print("    " + "=" * 58)

        text_to_image_correct = 0
        text_to_image_total = 0

        for topic_name in TOPIC_NAMES:
            # Find the first text embedding for this topic
            query_idx = text_labels.index(topic_name)
            query = text_data[query_idx]

            results = db.cross_modal_query(
                query,
                source_modality=TEXT_MODALITY,
                target_modality=IMAGE_MODALITY,
                k=K,
            )

            eval_result = evaluate_retrieval(
                results, image_labels, topic_name, K, modality_name="image",
            )
            text_to_image_correct += eval_result["correct"]
            text_to_image_total += eval_result["total"]

            print(f"\n    Query: text '{topic_name}' (id={query_idx})")
            print(f"    Results ({len(results)}/{K} retrieved, "
                  f"{eval_result['correct']} correct):")
            for d in eval_result["details"]:
                match_str = "  <-- MATCH" if d["match"] else ""
                print(f"      [{d['rank']:>2d}] id={d['id']:>3d}  "
                      f"topic={d['label']:<14s}  "
                      f"dist={d['distance']:.4f}{match_str}")

        text_to_image_acc = (text_to_image_correct / text_to_image_total
                             if text_to_image_total > 0 else 0.0)
        print(f"\n    Text→Image accuracy: {text_to_image_correct}/{text_to_image_total} "
              f"= {text_to_image_acc:.1%}")

        # ------------------------------------------------------------------
        # Step 5: Cross-modal retrieval: Image → Text
        # ------------------------------------------------------------------
        print(f"\n[5] Cross-modal retrieval: Image → Text")
        print("    " + "=" * 58)

        image_to_text_correct = 0
        image_to_text_total = 0

        for topic_name in TOPIC_NAMES:
            # Find the first image embedding for this topic
            query_idx = None
            for j, lbl in enumerate(image_labels):
                if lbl == topic_name:
                    query_idx = j
                    break
            if query_idx is None:
                continue

            query = image_data[query_idx]

            results = db.cross_modal_query(
                query,
                source_modality=IMAGE_MODALITY,
                target_modality=TEXT_MODALITY,
                k=K,
            )

            eval_result = evaluate_retrieval(
                results, text_labels, topic_name, K, modality_name="text",
            )
            image_to_text_correct += eval_result["correct"]
            image_to_text_total += eval_result["total"]

            print(f"\n    Query: image '{topic_name}' (id={query_idx})")
            print(f"    Results ({len(results)}/{K} retrieved, "
                  f"{eval_result['correct']} correct):")
            for d in eval_result["details"]:
                match_str = "  <-- MATCH" if d["match"] else ""
                print(f"      [{d['rank']:>2d}] id={d['id']:>3d}  "
                      f"topic={d['label']:<14s}  "
                      f"dist={d['distance']:.4f}{match_str}")

        image_to_text_acc = (image_to_text_correct / image_to_text_total
                             if image_to_text_total > 0 else 0.0)
        print(f"\n    Image→Text accuracy: {image_to_text_correct}/{image_to_text_total} "
              f"= {image_to_text_acc:.1%}")

        # ------------------------------------------------------------------
        # Step 6: Summary
        # ------------------------------------------------------------------
        print(f"\n[6] Summary")
        print("    " + "-" * 58)
        print(f"    Database: {db}")
        print(f"    Text→Image accuracy:  {text_to_image_acc:.1%}")
        print(f"    Image→Text accuracy:  {image_to_text_acc:.1%}")
        print(f"    Mean cross-modal accuracy: "
              f"{(text_to_image_acc + image_to_text_acc) / 2:.1%}")
        print()
        print("    Cross-modal retrieval leverages the shared atlas")
        print("    structure to bridge different embedding spaces, enabling")
        print("    semantically meaningful results across modalities.")

        print("\n" + "=" * 64)
        print("  Cross-modal demo completed successfully!")
        print("=" * 64)

    finally:
        shutil.rmtree(storage_dir, ignore_errors=True)
        print(f"\nCleaned up: {storage_dir}")


if __name__ == "__main__":
    main()
