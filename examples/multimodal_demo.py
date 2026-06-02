#!/usr/bin/env python3
"""
multimodal_demo.py - Cross-modal retrieval with ManifoldDB
============================================================

Demonstrates multi-modal data management:
  1. Generate synthetic text-like and image-like embeddings in a shared
     latent space
  2. Insert both modalities into ManifoldDB
  3. Build a unified atlas over the shared manifold
  4. Perform cross-modal queries (text -> image, image -> text)
  5. Show that semantically similar items are retrieved across modalities

Run with::

    python examples/multimodal_demo.py

Requires the ``_manifolddb_core`` C++ extension to be importable.
"""

from __future__ import annotations

import sys
import tempfile
import shutil

import numpy as np

# ---------------------------------------------------------------------------
# Import ManifoldDB components
# ---------------------------------------------------------------------------
try:
    from manifolddb import ManifoldDB, eigen_to_numpy
except ImportError:
    sys.path.insert(0, ".")
    from python.manifolddb import ManifoldDB, eigen_to_numpy


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def generate_text_embeddings(n_per_topic: int = 50, dim: int = 16, n_topics: int = 4, seed: int = 123):
    """Generate synthetic text-like embeddings clustered around semantic topics.

    Each topic defines a centroid in R^dim.  Points are sampled as Gaussian
    perturbations around their topic centroid, mimicking how real text
    embeddings cluster semantically.

    Parameters
    ----------
    n_per_topic : int
        Number of embeddings per topic.
    dim : int
        Ambient dimensionality.
    n_topics : int
        Number of distinct semantic topics.
    seed : int
        Random seed.

    Returns
    -------
    numpy.ndarray, shape (n_per_topic * n_topics, dim)
    list[str]
        Topic labels for each point.
    """
    rng = np.random.default_rng(seed)

    # Generate well-separated topic centroids
    centroid_angles = np.linspace(0, 2 * np.pi, n_topics, endpoint=False)
    centroid_radius = 3.0
    centroids = np.column_stack([
        centroid_radius * np.cos(centroid_angles),
        centroid_radius * np.sin(centroid_angles),
    ])
    # Pad to dim dimensions
    if dim > 2:
        centroids = np.hstack([
            centroids,
            rng.standard_normal((n_topics, dim - 2)) * 0.5,
        ])

    topic_names = [f"topic_{i}" for i in range(n_topics)]

    all_points = []
    all_labels = []

    for i in range(n_topics):
        pts = rng.normal(loc=centroids[i], scale=0.3, size=(n_per_topic, dim))
        all_points.append(pts)
        all_labels.extend([topic_names[i]] * n_per_topic)

    data = np.vstack(all_points)
    return data, all_labels


def generate_image_embeddings(
    text_centroids: np.ndarray,
    n_per_topic: int = 50,
    dim: int = 16,
    offset: float = 0.5,
    seed: int = 456,
):
    """Generate synthetic image-like embeddings that are semantically aligned
    with text embeddings.

    Each image embedding is created by adding a small offset to the
    corresponding text topic centroid, simulating how image and text
    embeddings occupy nearby (but not identical) regions in a shared
    CLIP-like latent space.

    Parameters
    ----------
    text_centroids : numpy.ndarray, shape (n_topics, dim)
        Topic centroids from the text embeddings (used as anchors).
    n_per_topic : int
        Number of image embeddings per topic.
    dim : int
        Ambient dimensionality.
    offset : float
        Offset magnitude from the text centroid.
    seed : int

    Returns
    -------
    numpy.ndarray, shape (n_per_topic * n_topics, dim)
    list[str]
        Topic labels for each point.
    """
    rng = np.random.default_rng(seed)
    n_topics = text_centroids.shape[0]

    topic_names = [f"topic_{i}" for i in range(n_topics)]

    all_points = []
    all_labels = []

    for i in range(n_topics):
        # Image centroids are offset from text centroids
        img_centroid = text_centroids[i] + rng.standard_normal(dim) * offset

        pts = rng.normal(loc=img_centroid, scale=0.35, size=(n_per_topic, dim))
        all_points.append(pts)
        all_labels.extend([topic_names[i]] * n_per_topic)

    data = np.vstack(all_points)
    return data, all_labels


def compute_topic_centroids(data: np.ndarray, labels: list, dim: int, n_topics: int) -> np.ndarray:
    """Compute the centroid of each topic in the data."""
    centroids = np.zeros((n_topics, dim))
    counts = np.zeros(n_topics)
    for pt, label in zip(data, labels):
        topic_idx = int(label.split("_")[1])
        centroids[topic_idx] += pt
        counts[topic_idx] += 1
    for i in range(n_topics):
        if counts[i] > 0:
            centroids[i] /= counts[i]
    return centroids


def main() -> None:
    print("=" * 64)
    print("ManifoldDB - Multi-Modal Cross-Retrieval Demo")
    print("=" * 64)
    print()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    AMBIENT_DIM = 16
    INTRINSIC_DIM = 4
    N_TOPICS = 4
    N_PER_TOPIC = 50
    TEXT_MODALITY = 0
    IMAGE_MODALITY = 1

    storage_dir = tempfile.mkdtemp(prefix="manifolddb_multimodal_")
    print(f"[0] Storage directory: {storage_dir}")

    try:
        # ------------------------------------------------------------------
        # 1. Generate synthetic multi-modal data
        # ------------------------------------------------------------------
        print(f"\n[1] Generating synthetic embeddings (dim={AMBIENT_DIM}) ...")
        print(f"    Topics: {N_TOPICS}, Points per topic: {N_PER_TOPIC}")

        text_data, text_labels = generate_text_embeddings(
            n_per_topic=N_PER_TOPIC, dim=AMBIENT_DIM, n_topics=N_TOPICS,
        )
        print(f"    Text embeddings:   {text_data.shape}")

        text_centroids = compute_topic_centroids(
            text_data, text_labels, AMBIENT_DIM, N_TOPICS,
        )
        image_data, image_labels = generate_image_embeddings(
            text_centroids,
            n_per_topic=N_PER_TOPIC,
            dim=AMBIENT_DIM,
            offset=0.5,
        )
        print(f"    Image embeddings: {image_data.shape}")

        # ------------------------------------------------------------------
        # 2. Initialise ManifoldDB and insert both modalities
        # ------------------------------------------------------------------
        print("\n[2] Initialising ManifoldDB ...")
        db = ManifoldDB(
            storage_path=storage_dir,
            intrinsic_dim=INTRINSIC_DIM,
            enable_cuda=False,
            geodesic_tolerance=1e-6,
        )
        print(f"    {db}")

        print("\n    Inserting text embeddings (modality 0) ...")
        db.insert(text_data, modality=TEXT_MODALITY)

        print("    Inserting image embeddings (modality 1) ...")
        db.insert(image_data, modality=IMAGE_MODALITY)

        total_points = N_PER_TOPIC * N_TOPICS * 2
        print(f"    Total inserted: {total_points} points across 2 modalities")

        # ------------------------------------------------------------------
        # 3. Build a unified atlas
        # ------------------------------------------------------------------
        print("\n[3] Building unified atlas ...")
        db.build()
        stats = db.stats()
        print(f"    Charts:   {stats['num_charts']}")
        print(f"    Points:   {stats['total_points']}")
        print(f"    Index:    {stats['index_size']}")

        # ------------------------------------------------------------------
        # 4. Cross-modal retrieval: text -> image
        # ------------------------------------------------------------------
        print("\n[4] Cross-modal retrieval: Text -> Image")
        print("    " + "-" * 55)

        k = 5
        for topic_idx in range(N_TOPICS):
            topic_name = f"topic_{topic_idx}"
            # Use the first text embedding of this topic as a query
            query_idx = text_labels.index(topic_name)
            query = text_data[query_idx]

            results = db.cross_modal_query(
                query,
                source=TEXT_MODALITY,
                target=IMAGE_MODALITY,
                k=k,
            )

            # Count how many retrieved images belong to the correct topic
            correct = sum(
                1 for r in results
                if image_labels[r["id"]] == topic_name
            )

            print(f"\n    Query: text '{topic_name}' (id={query_idx})")
            print(f"    Retrieved {k} nearest images (correct: {correct}/{k}):")
            for rank, r in enumerate(results):
                label = image_labels[r["id"]]
                match = "  <- MATCH" if label == topic_name else ""
                print(f"      [{rank}] id={r['id']:>3d}  "
                      f"topic={label:<10s}  dist={r['distance']:.4f}{match}")

        # ------------------------------------------------------------------
        # 5. Cross-modal retrieval: image -> text
        # ------------------------------------------------------------------
        print("\n[5] Cross-modal retrieval: Image -> Text")
        print("    " + "-" * 55)

        for topic_idx in range(N_TOPICS):
            topic_name = f"topic_{topic_idx}"
            # Use the first image embedding of this topic as a query
            query_idx = N_PER_TOPIC * N_TOPICS + topic_idx * N_PER_TOPIC
            # image_labels starts at 0 for images, but image IDs in ManifoldDB
            # are separate. Let's find the first image of this topic.
            img_query_idx = None
            for j, lbl in enumerate(image_labels):
                if lbl == topic_name:
                    img_query_idx = j
                    break

            if img_query_idx is None:
                continue

            query = image_data[img_query_idx]
            results = db.cross_modal_query(
                query,
                source=IMAGE_MODALITY,
                target=TEXT_MODALITY,
                k=k,
            )

            correct = sum(
                1 for r in results
                if text_labels[r["id"]] == topic_name
            )

            print(f"\n    Query: image '{topic_name}' (id={img_query_idx})")
            print(f"    Retrieved {k} nearest texts (correct: {correct}/{k}):")
            for rank, r in enumerate(results):
                label = text_labels[r["id"]]
                match = "  <- MATCH" if label == topic_name else ""
                print(f"      [{rank}] id={r['id']:>3d}  "
                      f"topic={label:<10s}  dist={r['distance']:.4f}{match}")

        # ------------------------------------------------------------------
        # 6. Summary statistics
        # ------------------------------------------------------------------
        print("\n[6] Summary:")
        print(f"    {stats}")
        print(f"\n    Cross-modal retrieval leverages the shared atlas")
        print(f"    structure to bridge text and image embedding spaces.")
        print(f"    Semantically aligned topics are retrieved even when")
        print(f"    the query and target are from different modalities.")

        print("\n" + "=" * 64)
        print("Multi-modal demo completed successfully!")
        print("=" * 64)

    finally:
        # Clean up
        shutil.rmtree(storage_dir, ignore_errors=True)
        print(f"\nCleaned up temporary directory: {storage_dir}")


if __name__ == "__main__":
    main()
