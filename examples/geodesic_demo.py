#!/usr/bin/env python3
"""
geodesic_demo.py - Geodesic Paths and Manifold Structure Demo
===============================================================

This demo illustrates how ManifoldDB leverages manifold geometry for
nearest-neighbour search and distance computation on a Swiss Roll — a classic
2-D manifold embedded in R^3.

What this demo covers:

  1.  Generate a Swiss Roll point cloud in R^3
  2.  Insert points into ManifoldDB and build the atlas
  3.  Compute geodesic paths between several pairs of points
  4.  Compare geodesic distances to Euclidean (chord) distances
  5.  Demonstrate how the manifold structure affects nearest-neighbour queries
      (geodesic neighbours differ from Euclidean neighbours)
  6.  Visualise geodesic paths and distance comparisons with matplotlib
      (saved to ``examples/output/``)

Mathematical Background
----------------------
The Swiss Roll is a 2-D rectangular ribbon that has been rolled up in 3-D
space.  Its parametric form is:

    x(t, h) = t · cos(t)
    y(t, h) = h
    z(t, h) = t · sin(t)

where t ∈ [1.5π, 4.5π] controls the angular position and h ∈ [0, 21] is
the height.  Two points that are close in Euclidean distance may be far
apart in geodesic distance if they lie on different "layers" of the roll.
ManifoldDB's geodesic search correctly handles this ambiguity.

Run with::

    python examples/geodesic_demo.py

Requirements
------------
- numpy
- matplotlib (optional; visualisation is skipped if unavailable)
- manifolddb C++ extension
"""

from __future__ import annotations

import os
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
# Output directory for matplotlib figures
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def ensure_output_dir() -> None:
    """Create the output directory if it does not exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Swiss Roll Data Generation
# ---------------------------------------------------------------------------

def make_swiss_roll(n_points: int = 1000, noise: float = 0.05, seed: int = 42):
    """Generate a Swiss Roll dataset in R^3.

    The Swiss Roll is a 2-D rectangular ribbon (parameters t, h) that has been
    spiralled in 3-D space.  It is a classic test case for manifold learning
    algorithms because points that are nearby in ambient space may be far apart
    along the manifold surface.

    Parameters
    ----------
    n_points : int
        Number of points to sample.
    noise : float
        Standard deviation of Gaussian noise added to the embedding.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray, shape (n_points, 3)
        Ambient-space coordinates.
    numpy.ndarray, shape (n_points, 2)
        Intrinsic parameters (t, h).
    """
    rng = np.random.default_rng(seed)

    # t controls the angular position of the roll
    t = 1.5 * np.pi * (1.0 + 2.0 * rng.random(n_points))
    # h is the height along the roll
    h = 21.0 * rng.random(n_points)

    x = t * np.cos(t) + noise * rng.standard_normal(n_points)
    y = h + noise * rng.standard_normal(n_points)
    z = t * np.sin(t) + noise * rng.standard_normal(n_points)

    ambient = np.column_stack([x, y, z])
    intrinsic = np.column_stack([t, h])

    return ambient.astype(np.float64), intrinsic.astype(np.float64)


# ---------------------------------------------------------------------------
# Main Demo
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 64)
    print("  ManifoldDB — Geodesic Demo")
    print("  Manifold: Swiss Roll (2-D surface in R^3)")
    print("=" * 64)
    print()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    N_POINTS = 1000
    INTRINSIC_DIM = 2
    NOISE = 0.05

    storage_dir = tempfile.mkdtemp(prefix="manifolddb_geodesic_")
    print(f"[0] Storage directory: {storage_dir}")

    try:
        # ------------------------------------------------------------------
        # Step 1: Generate Swiss Roll data
        # ------------------------------------------------------------------
        print(f"\n[1] Generating Swiss Roll data ({N_POINTS} points, noise={NOISE}) ...")
        data, intrinsic = make_swiss_roll(n_points=N_POINTS, noise=NOISE, seed=42)
        print(f"    Ambient shape: {data.shape}")
        print(f"    Intrinsic shape: {intrinsic.shape}")

        # ------------------------------------------------------------------
        # Step 2: Insert and build
        # ------------------------------------------------------------------
        print(f"\n[2] Initialising ManifoldDB (intrinsic_dim={INTRINSIC_DIM}) ...")
        db = ManifoldDB(
            storage_path=storage_dir,
            intrinsic_dim=INTRINSIC_DIM,
            enable_cuda=False,
            geodesic_tolerance=1e-6,
        )
        print(f"    {db}")

        print("\n    Inserting data ...")
        db.insert(data, modality_id=0)

        print("    Building atlas ...")
        db.build(method="linear")

        stats = db.stats()
        print(f"    Charts: {stats['num_charts']}, Points: {stats['total_points']}")

        # ------------------------------------------------------------------
        # Step 3: Geodesic paths between point pairs
        # ------------------------------------------------------------------
        print(f"\n[3] Computing geodesic paths between point pairs ...")
        print(f"    {'Pair':>10s}  {'Geo Dist':>10s}  {'Euc Dist':>10s}  "
              f"{'Ratio':>8s}  {'Converged':>10s}")
        print("    " + "-" * 56)

        # Select pairs that span different geodesic ranges
        pair_indices = [
            (0, 5),             # Very close (same neighbourhood)
            (0, 50),            # Moderate distance
            (0, 200),           # Far apart
            (N_POINTS // 4, N_POINTS // 2),     # Cross-roll pair
            (N_POINTS // 4, 3 * N_POINTS // 4), # Opposite sides of roll
        ]

        geodesic_results = []  # For later visualisation

        for idx_a, idx_b in pair_indices:
            pt_a = data[idx_a]
            pt_b = data[idx_b]
            euc_dist = float(np.linalg.norm(pt_a - pt_b))

            try:
                path_info = db.geodesic_path(pt_a, pt_b)
                geo_dist = path_info["total_length"]
                converged = path_info["converged"]
                ratio = geo_dist / (euc_dist + 1e-30)
            except Exception:
                geo_dist = float("nan")
                converged = False
                ratio = float("nan")

            geodesic_results.append({
                "idx_a": idx_a,
                "idx_b": idx_b,
                "pt_a": pt_a,
                "pt_b": pt_b,
                "geo_dist": geo_dist,
                "euc_dist": euc_dist,
                "ratio": ratio,
                "converged": converged,
            })

            print(f"    ({idx_a:>4d},{idx_b:>4d})  "
                  f"{geo_dist:>10.4f}  {euc_dist:>10.4f}  "
                  f"{ratio:>8.4f}  {str(converged):>10s}")

        # ------------------------------------------------------------------
        # Step 4: Manifold structure affects k-NN queries
        # ------------------------------------------------------------------
        #
        # We compare geodesic k-NN results to Euclidean k-NN.  On a curved
        # manifold, the rankings can differ significantly because Euclidean
        # distance ignores the manifold's intrinsic geometry.
        #
        print(f"\n[4] Geodesic k-NN vs. Euclidean k-NN comparison ...")

        query_idx = 0
        query_pt = data[query_idx]
        k = 10

        # Geodesic k-NN from ManifoldDB
        geo_knn = db.query_knn(query_pt, k=k)
        geo_ids = set(r["id"] for r in geo_knn)

        # Euclidean k-NN (naive brute-force)
        euc_dists = np.linalg.norm(data - query_pt, axis=1)
        euc_knn_ids = set(int(np.argsort(euc_dists)[1:k + 1]))  # exclude self

        # Compare rankings
        overlap = geo_ids & euc_knn_ids
        geo_only = geo_ids - euc_knn_ids
        euc_only = euc_knn_ids - geo_ids

        print(f"    Query point index: {query_idx}")
        print(f"    k = {k}")
        print(f"    Geodesic k-NN IDs:  {sorted(geo_ids)}")
        print(f"    Euclidean k-NN IDs: {sorted(euc_knn_ids)}")
        print(f"    Overlap: {len(overlap)}  |  "
              f"Geodesic-only: {len(geo_only)}  |  "
              f"Euclidean-only: {len(euc_only)}")

        if geo_only:
            print(f"    Points retrieved only by geodesic search: {sorted(geo_only)}")
        if euc_only:
            print(f"    Points retrieved only by Euclidean search: {sorted(euc_only)}")

        # ------------------------------------------------------------------
        # Step 5: Comprehensive geodesic vs. Euclidean distance table
        # ------------------------------------------------------------------
        print(f"\n[5] Distance comparison for {N_POINTS} points (sampled) ...")
        print(f"    {'Idx':>5s}  {'Geodesic':>10s}  {'Euclidean':>10s}  "
              f"{'Ratio':>8s}  {'Notes':>20s}")
        print("    " + "-" * 60)

        sample_ids = np.linspace(1, N_POINTS - 1, 12, dtype=int)
        for idx in sample_ids:
            pt = data[idx]
            euc = float(np.linalg.norm(data[0] - pt))
            try:
                path_info = db.geodesic_path(data[0], pt)
                geo = path_info["total_length"]
                ratio = geo / (euc + 1e-30)
                notes = ""
                if ratio > 1.5:
                    notes = "manifold wraps around"
                elif ratio < 1.01:
                    notes = "nearly flat locally"
                print(f"    {idx:>5d}  {geo:>10.4f}  {euc:>10.4f}  "
                      f"{ratio:>8.4f}  {notes:>20s}")
            except Exception:
                print(f"    {idx:>5d}  {'err':>10s}  {euc:>10.4f}  "
                      f"{'N/A':>8s}")

        # ------------------------------------------------------------------
        # Step 6: Visualisation (matplotlib)
        # ------------------------------------------------------------------
        print(f"\n[6] Generating visualisations ...")
        ensure_output_dir()
        plot_swiss_roll_with_paths(data, geodesic_results, N_POINTS)
        plot_distance_comparison(geodesic_results)
        plot_knn_comparison(data, geo_ids, euc_knn_ids, query_idx)

        print("=" * 64)
        print("  Geodesic demo completed successfully!")
        print(f"  Figures saved to: {OUTPUT_DIR}/")
        print("=" * 64)

    finally:
        shutil.rmtree(storage_dir, ignore_errors=True)
        print(f"\nCleaned up: {storage_dir}")


# ---------------------------------------------------------------------------
# Visualisation Functions
# ---------------------------------------------------------------------------

def plot_swiss_roll_with_paths(
    data: np.ndarray,
    geodesic_results: list,
    n_points: int,
) -> None:
    """Plot the Swiss Roll surface with geodesic paths overlaid.

    Saves to ``examples/output/geodesic_paths.png``.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("    matplotlib not available — skipping geodesic_paths.png")
        return

    fig = plt.figure(figsize=(14, 5))

    # --- Left: 3-D Swiss Roll surface with geodesic paths ---
    ax1 = fig.add_subplot(121, projection="3d")

    # Plot all points as a scatter
    ax1.scatter(
        data[:, 0], data[:, 1], data[:, 2],
        c=data[:, 1], cmap="viridis", s=1, alpha=0.3,
    )

    # Overlay geodesic paths (use Euclidean straight lines as fallback)
    colors = plt.cm.Set1(np.linspace(0, 1, len(geodesic_results)))
    for i, res in enumerate(geodesic_results):
        pt_a, pt_b = res["pt_a"], res["pt_b"]
        # Draw the ambient-space straight line (proxy for geodesic direction)
        ax1.plot(
            [pt_a[0], pt_b[0]],
            [pt_a[1], pt_b[1]],
            [pt_a[2], pt_b[2]],
            "-o", color=colors[i], markersize=4, linewidth=1.5,
            label=f"Pair {i} (geo={res['geo_dist']:.2f})",
        )

    ax1.set_title("Swiss Roll with Geodesic Paths")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.legend(fontsize=6, loc="upper left")

    # --- Right: Distance comparison bar chart ---
    ax2 = fig.add_subplot(122)
    x_pos = np.arange(len(geodesic_results))
    width = 0.35

    geo_dists = [r["geo_dist"] for r in geodesic_results]
    euc_dists = [r["euc_dist"] for r in geodesic_results]
    labels = [f"({r['idx_a']},{r['idx_b']})" for r in geodesic_results]

    bars1 = ax2.bar(x_pos - width / 2, geo_dists, width,
                     label="Geodesic", color="coral", alpha=0.8)
    bars2 = ax2.bar(x_pos + width / 2, euc_dists, width,
                     label="Euclidean", color="steelblue", alpha=0.8)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("Distance")
    ax2.set_title("Geodesic vs. Euclidean Distance")
    ax2.legend()

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "geodesic_paths.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    Saved: {out_path}")


def plot_distance_comparison(geodesic_results: list) -> None:
    """Plot geodesic vs. Euclidean distances as a scatter plot.

    Points above the diagonal y=x indicate that the geodesic distance
    exceeds the Euclidean distance (manifold curvature effect).

    Saves to ``examples/output/distance_comparison.png``.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("    matplotlib not available — skipping distance_comparison.png")
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    euc = [r["euc_dist"] for r in geodesic_results]
    geo = [r["geo_dist"] for r in geodesic_results]

    ax.scatter(euc, geo, s=80, c="coral", edgecolors="darkred", zorder=5)

    # Diagonal reference line (geo = euclidean)
    max_val = max(max(euc), max(geo)) * 1.1
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="y = x (flat)")

    ax.set_xlabel("Euclidean Distance (R^3)")
    ax.set_ylabel("Geodesic Distance (Manifold)")
    ax.set_title("Geodesic vs. Euclidean Distance\n"
                 "(points above diagonal = curved manifold)")
    ax.legend()
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "distance_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    Saved: {out_path}")


def plot_knn_comparison(
    data: np.ndarray,
    geo_ids: set,
    euc_ids: set,
    query_idx: int,
) -> None:
    """Visualise geodesic vs. Euclidean k-NN results on the Swiss Roll.

    Saves to ``examples/output/knn_comparison.png``.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("    matplotlib not available — skipping knn_comparison.png")
        return

    fig = plt.figure(figsize=(14, 5))

    # --- Left: Geodesic k-NN ---
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(data[:, 0], data[:, 1], data[:, 2],
                c="lightgray", s=1, alpha=0.3)
    ax1.scatter(data[query_idx, 0], data[query_idx, 1], data[query_idx, 2],
                c="red", s=100, marker="*", zorder=10, label="Query")

    geo_list = sorted(geo_ids)
    if geo_list:
        geo_pts = data[geo_list]
        ax1.scatter(geo_pts[:, 0], geo_pts[:, 1], geo_pts[:, 2],
                    c="darkorange", s=30, zorder=5, label=f"Geodesic k-NN ({len(geo_list)})")

    ax1.set_title("Geodesic k-NN")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.legend(fontsize=7)

    # --- Right: Euclidean k-NN ---
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(data[:, 0], data[:, 1], data[:, 2],
                c="lightgray", s=1, alpha=0.3)
    ax2.scatter(data[query_idx, 0], data[query_idx, 1], data[query_idx, 2],
                c="red", s=100, marker="*", zorder=10, label="Query")

    euc_list = sorted(euc_ids)
    if euc_list:
        euc_pts = data[euc_list]
        ax2.scatter(euc_pts[:, 0], euc_pts[:, 1], euc_pts[:, 2],
                    c="steelblue", s=30, zorder=5, label=f"Euclidean k-NN ({len(euc_list)})")

    ax2.set_title("Euclidean k-NN")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.legend(fontsize=7)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "knn_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    Saved: {out_path}")


if __name__ == "__main__":
    main()
