#!/usr/bin/env python3
"""
basic_usage.py - Comprehensive Getting Started Example for ManifoldDB
=====================================================================

This example demonstrates the complete ManifoldDB workflow:

  1.  Import the library (with graceful fallback if C++ extension is absent)
  2.  Generate synthetic data on a known manifold (torus T^2 embedded in R^3)
  3.  Initialise ManifoldDB with appropriate configuration
  4.  Insert data into the database
  5.  Build an atlas (automatic chart decomposition via PCA)
  6.  Query k-nearest neighbours using geodesic distances
  7.  Print comprehensive statistics
  8.  Compare geodesic distances to Euclidean (chord) distances

Mathematical Background
----------------------
A torus T^2 is a 2-dimensional manifold embedded in R^3.  Points on the torus
are parameterised by two angles (θ, φ):

    x = (R + r·cos θ) · cos φ
    y = (R + r·cos θ) · sin φ
    z = r · sin θ

where R is the major radius and r is the minor radius.  Because the torus is
curved, geodesic distances between two points on the torus are generally
*longer* than the straight-line (Euclidean/chord) distance in R^3.  ManifoldDB
exploits this manifold structure to provide geodesically-aware nearest-neighbour
search, which can yield semantically more meaningful results than naive Euclidean
k-NN.

Run with::

    python examples/basic_usage.py

Requirements
------------
- numpy
- manifolddb C++ extension (built from source)
"""

from __future__ import annotations

import sys
import tempfile
import shutil
import math

# ---------------------------------------------------------------------------
# Import ManifoldDB with graceful fallback.
#
# We attempt to import the high-level Python wrapper.  If the C++ extension
# (manifolddb_core) is not available, we print a clear message and exit
# gracefully instead of crashing with an ImportError traceback.
# ---------------------------------------------------------------------------
try:
    from manifolddb import ManifoldDB
except ImportError:
    try:
        # Support running from the project root (in-tree development)
        sys.path.insert(0, ".")
        from python.manifolddb import ManifoldDB
    except ImportError:
        print("=" * 64)
        print("ERROR: Cannot import ManifoldDB.")
        print("")
        print("The C++ extension 'manifolddb_core' is not available.")
        print("Please build ManifoldDB from source first:")
        print("")
        print("    mkdir build && cd build")
        print("    cmake .. && make")
        print("    pip install -e .")
        print("")
        print("Once built, re-run this example.")
        print("=" * 64)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Synthetic Data Generation
# ---------------------------------------------------------------------------

def make_torus(
    n_points: int = 800,
    R: float = 2.0,
    r: float = 0.5,
    noise: float = 0.02,
    seed: int = 42,
):
    """Sample points from a torus T^2 embedded in R^3.

    The torus is parameterised by two angles θ ∈ [0, 2π) and φ ∈ [0, 2π):

        x = (R + r·cos θ) · cos φ
        y = (R + r·cos θ) · sin φ
        z = r · sin θ

    Parameters
    ----------
    n_points : int
        Number of points to sample from the torus surface.
    R : float
        Major radius (distance from the centre of the tube to the centre
        of the torus).
    r : float
        Minor radius (radius of the tube).
    noise : float
        Standard deviation of additive Gaussian noise in R^3, simulating
        measurement error or imperfect manifold alignment.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray, shape (n_points, 3)
        Ambient-space coordinates of points on the noisy torus.
    numpy.ndarray, shape (n_points, 2)
        The intrinsic (θ, φ) parameters used to generate each point.
    """
    import numpy as np

    rng = np.random.default_rng(seed)

    # Uniformly sample angles θ and φ from [0, 2π)
    theta = rng.uniform(0, 2 * np.pi, n_points)
    phi = rng.uniform(0, 2 * np.pi, n_points)

    # Parametric embedding into R^3
    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)

    # Stack into (n_points, 3) ambient coordinates
    ambient = np.column_stack([x, y, z])

    # Add a small amount of Gaussian noise to break the exact manifold
    ambient += rng.normal(loc=0.0, scale=noise, size=ambient.shape)

    # Intrinsic parameters for reference
    intrinsic = np.column_stack([theta, phi])

    return ambient.astype(np.float64), intrinsic.astype(np.float64)


# ---------------------------------------------------------------------------
# Main Example
# ---------------------------------------------------------------------------

def main() -> None:
    import numpy as np

    print("=" * 64)
    print("  ManifoldDB — Basic Usage Example")
    print("  Data: Torus T^2 embedded in R^3")
    print("=" * 64)
    print()

    # ------------------------------------------------------------------
    # Step 1: Configuration
    # ------------------------------------------------------------------
    #
    # The torus has intrinsic dimension 2 (a 2-D surface) embedded in 3-D
    # ambient space.  We tell ManifoldDB the intrinsic dimension so it can
    # build charts of the correct dimensionality during atlas construction.
    #
    INTRINSIC_DIM = 2   # d = 2 for the torus
    AMBIENT_DIM = 3     # D = 3 (points live in R^3)
    N_POINTS = 800

    # ------------------------------------------------------------------
    # Step 2: Create a temporary directory for database storage
    # ------------------------------------------------------------------
    #
    # ManifoldDB stores its atlas, metric tensors, and tangent-space indexes
    # on disk at the specified storage_path.  Using a temp directory ensures
    # cleanup after the example runs.
    #
    storage_dir = tempfile.mkdtemp(prefix="manifolddb_basic_")
    print(f"[1] Storage directory: {storage_dir}")

    try:
        # ------------------------------------------------------------------
        # Step 3: Initialise ManifoldDB
        # ------------------------------------------------------------------
        #
        # ManifoldDB is configured with:
        #   - storage_path:       where to persist data on disk
        #   - intrinsic_dim:      the manifold's intrinsic dimension (d)
        #   - enable_cuda:        whether to use GPU acceleration (off here)
        #   - geodesic_tolerance: solver convergence threshold
        #
        db = ManifoldDB(
            storage_path=storage_dir,
            intrinsic_dim=INTRINSIC_DIM,
            enable_cuda=False,
            geodesic_tolerance=1e-6,
        )
        print(f"[2] ManifoldDB created: {db}")
        print(f"    Intrinsic dim: {INTRINSIC_DIM}")
        print(f"    Ambient dim:   {AMBIENT_DIM}")

        # ------------------------------------------------------------------
        # Step 4: Generate synthetic torus data
        # ------------------------------------------------------------------
        #
        # We sample N_POINTS from a torus surface with small Gaussian noise.
        # The intrinsic parameters (θ, φ) are kept for reference comparisons.
        #
        data, intrinsic_params = make_torus(
            n_points=N_POINTS, R=2.0, r=0.5, noise=0.02, seed=42,
        )
        print(f"\n[3] Generated torus data: {data.shape}")
        print(f"    Major radius R = 2.0, minor radius r = 0.5")
        print(f"    Noise σ = 0.02")

        # ------------------------------------------------------------------
        # Step 5: Insert data into the database
        # ------------------------------------------------------------------
        #
        # The insert() method accepts a 2-D numpy array of shape (N, D) where
        # each row is one D-dimensional ambient-space point.  The modality_id
        # parameter allows storing data from multiple modalities (e.g., text
        # and image embeddings) in the same database.
        #
        print(f"\n[4] Inserting {N_POINTS} points into ManifoldDB ...")
        db.insert(data, modality_id=0)
        print(f"    Inserted successfully (modality_id=0)")

        # ------------------------------------------------------------------
        # Step 6: Build the atlas
        # ------------------------------------------------------------------
        #
        # Atlas construction discovers the manifold's chart structure from the
        # data.  Using method='linear' (PCA-based), ManifoldDB decomposes the
        # point cloud into local affine patches (charts) that cover the
        # manifold.  Each chart provides a local coordinate system in R^d.
        #
        print(f"\n[5] Building atlas (PCA-based linear charts) ...")
        db.build(method="linear")
        print("    Atlas built successfully.")

        stats = db.stats()
        print(f"    Charts discovered:  {stats['num_charts']}")
        print(f"    Total points:       {stats['total_points']}")
        print(f"    Index size:         {stats['index_size']}")
        print(f"    Build time:         {stats['build_time_ms']:.2f} ms")

        # ------------------------------------------------------------------
        # Step 7: Query k-nearest neighbours (geodesic)
        # ------------------------------------------------------------------
        #
        # The query_knn method searches for the k closest points by *geodesic*
        # distance (shortest path along the manifold), not Euclidean distance.
        # This is the key advantage of ManifoldDB over flat-space indexers.
        #
        print(f"\n[6] Querying k-NN (geodesic distance) ...")
        query_point = data[0]
        k = 10
        results = db.query_knn(query_point, k=k)

        print(f"    Query point (first row): {query_point}")
        print(f"    Top-{k} neighbours by geodesic distance:")
        print(f"    {'Rank':>4s}  {'ID':>5s}  {'Chart':>5s}  "
              f"{'Geo Dist':>10s}  {'Euc Resid':>10s}")
        print("    " + "-" * 44)
        for i, r in enumerate(results):
            print(f"    {i:>4d}  {r['id']:>5d}  {r['chart_id']:>5d}  "
                  f"{r['distance']:>10.4f}  {r['euclidean_residual']:>10.4f}")

        # ------------------------------------------------------------------
        # Step 8: Compare geodesic vs. Euclidean distances
        # ------------------------------------------------------------------
        #
        # On a curved manifold, the geodesic (intrinsic) distance is always
        # >= the Euclidean (ambient/chord) distance.  The difference is larger
        # for points that are far apart along the manifold but close through
        # the ambient space (e.g., opposite sides of the torus tube).
        #
        print(f"\n[7] Geodesic vs. Euclidean Distance Comparison")
        print(f"    {'Pair':>8s}  {'Geodesic':>10s}  {'Euclidean':>10s}  "
              f"{'Ratio':>8s}  {'Δ':>10s}")
        print("    " + "-" * 54)

        # Select several point pairs spanning different distances
        pair_indices = [
            (0, 1),
            (0, 10),
            (0, 50),
            (0, 100),
            (0, 400),
            (N_POINTS // 2, N_POINTS // 2 + 25),
        ]

        for idx_a, idx_b in pair_indices:
            pt_a = data[idx_a]
            pt_b = data[idx_b]

            # Euclidean (straight-line) distance in R^3
            euclidean_dist = float(np.linalg.norm(pt_a - pt_b))

            # Query the nearest neighbours of pt_a and check if pt_b appears
            # in the results.  If not, we use a large-k query to find it.
            knn_results = db.query_knn(pt_a, k=N_POINTS)
            geo_dist = None
            for r in knn_results:
                if r["id"] == idx_b:
                    geo_dist = r["distance"]
                    break

            if geo_dist is not None and euclidean_dist > 1e-10:
                ratio = geo_dist / euclidean_dist
                delta = geo_dist - euclidean_dist
                print(f"    ({idx_a:>4d},{idx_b:>4d})  "
                      f"{geo_dist:>10.4f}  {euclidean_dist:>10.4f}  "
                      f"{ratio:>8.4f}  {delta:>10.4f}")
            else:
                print(f"    ({idx_a:>4d},{idx_b:>4d})  "
                      f"{'N/A':>10s}  {euclidean_dist:>10.4f}  "
                      f"{'N/A':>8s}  {'N/A':>10s}")

        # ------------------------------------------------------------------
        # Step 9: Final statistics
        # ------------------------------------------------------------------
        #
        # The stats() method returns a dict with operational metrics about
        # the database: number of charts, total points indexed, index size,
        # and timing information.
        #
        print(f"\n[8] Final Statistics:")
        stats = db.stats()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"    {key:>25s}: {value:.4f}")
            else:
                print(f"    {key:>25s}: {value}")

        # ------------------------------------------------------------------
        # Step 10: Compute a geodesic path (if supported)
        # ------------------------------------------------------------------
        print(f"\n[9] Computing geodesic path between two points ...")
        start_pt = data[0]
        end_pt = data[N_POINTS // 2]
        euclidean_dist = float(np.linalg.norm(start_pt - end_pt))
        print(f"    Start:  {start_pt}")
        print(f"    End:    {end_pt}")
        print(f"    Euclidean distance: {euclidean_dist:.4f}")

        try:
            path_info = db.geodesic_path(start_pt, end_pt)
            print(f"    Geodesic length:   {path_info['total_length']:.4f}")
            print(f"    Converged:         {path_info['converged']}")
            print(f"    Integration steps: {path_info['num_steps']}")
            print(f"    Path sample points: {len(path_info['points'])}")

            if euclidean_dist > 1e-10 and path_info['total_length'] > 0:
                ratio = path_info['total_length'] / euclidean_dist
                print(f"    Geo / Euclidean ratio: {ratio:.4f}")
                print(f"    (Ratio > 1 confirms the manifold is curved)")
        except Exception as exc:
            print(f"    Geodesic path computation: {exc}")

        print("\n" + "=" * 64)
        print("  Example completed successfully!")
        print("=" * 64)

    finally:
        # Clean up temporary storage directory
        shutil.rmtree(storage_dir, ignore_errors=True)
        print(f"\nCleaned up: {storage_dir}")


if __name__ == "__main__":
    main()
