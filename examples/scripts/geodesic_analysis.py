#!/usr/bin/env python3
"""
Geodesic Analysis Example — Scientific computing with geodesic queries.

Demonstrates geodesic queries on a 2D potential energy surface,
a common task in computational chemistry and physics.

This example:
  1. Creates a 2D potential energy surface (Mexican hat / Rosenbrock)
  2. Samples points on the surface
  3. Builds a manifold atlas
  4. Computes geodesic paths between minima
  5. Finds all states within energy threshold (geodesic ball query)
  6. Visualises geodesic paths (saves to examples/output/)

Run:
    python examples/scripts/geodesic_analysis.py
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from manifold_db.atlas import AtlasBuilder, AtlasManager, Chart
from manifold_db.tangent_index import TangentSpace, TangentSpaceIndex
from manifold_db.query import QueryBuilder, ManifoldQuery, QueryType
from manifold_db.metric import (
    EuclideanMetric, DiagonalMetric, MetricTensorStore
)
from manifold_db.geodesic.distance import RiemannianDistance, DistanceComputer


# ---------------------------------------------------------------------------
# Potential Energy Surfaces
# ---------------------------------------------------------------------------

def mexican_hat_potential(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mexican hat (sombrero) potential energy surface.
    
    V(r) = (r^2 - 1)^2 where r^2 = x^2 + y^2
    
    Features:
      - Global minimum at r=1 (ring of minima)
      - Local maximum at r=0
      - Energy increases for r >> 1
    """
    r_sq = x**2 + y**2
    return (r_sq - 1.0)**2


def mexican_hat_gradient(x: float, y: float) -> np.ndarray:
    """Gradient of the Mexican hat potential."""
    r_sq = x**2 + y**2
    factor = 4.0 * (r_sq - 1.0)
    return np.array([factor * x, factor * y])


def mexican_hat_metric(x: float, y: float) -> np.ndarray:
    """Riemannian metric tensor derived from the potential.
    
    Uses the Hessian of V as a conformal factor:
      g_{ij} = (1 + α |∇V|^2) δ_{ij}
    
    This makes the metric sensitive to the steepness of the potential,
    so geodesic paths prefer to follow valleys rather than climb ridges.
    """
    grad = mexican_hat_gradient(x, y)
    grad_sq = float(np.dot(grad, grad))
    conformal = 1.0 + 0.5 * grad_sq
    return np.array([[conformal, 0.0],
                     [0.0, conformal]])


def sample_energy_surface(n_points: int = 2000,
                          x_range: tuple = (-2.0, 2.0),
                          y_range: tuple = (-2.0, 2.0)) -> np.ndarray:
    """Sample points on the potential energy surface.
    
    Each point is (x, y, V(x,y)) in 3D, where the third coordinate
    is the energy.  The manifold is the 2D surface embedded in 3D.
    """
    rng = np.random.default_rng(42)
    x = rng.uniform(x_range[0], x_range[1], n_points)
    y = rng.uniform(y_range[0], y_range[1], n_points)
    V = mexican_hat_potential(x, y)
    return np.column_stack([x, y, V])


def compute_geodesic_path(
    start: np.ndarray,
    end: np.ndarray,
    metric_fn,
    n_waypoints: int = 30,
) -> np.ndarray:
    """Compute a geodesic path between two points on the manifold.
    
    Uses energy minimisation: the geodesic is the path of minimum
    length d(γ) = ∫ √(g_{ij} γ̇^i γ̇^j) dt.
    
    Simplified to Euclidean path with metric-weighted energy for speed.
    """
    from scipy.optimize import minimize
    
    # Interpolate initial path as straight line
    t = np.linspace(0, 1, n_waypoints)
    init_path = np.outer(1 - t, start) + np.outer(t, end)
    
    def path_energy(waypoints_flat: np.ndarray) -> float:
        pts = waypoints_flat.reshape(n_waypoints, 2)
        path = np.vstack([start.reshape(1, 2), pts, end.reshape(1, 2)])
        energy = 0.0
        for i in range(len(path) - 1):
            diff = path[i + 1] - path[i]
            mid = 0.5 * (path[i] + path[i + 1])
            g = metric_fn(mid[0], mid[1])
            energy += float(diff @ g @ diff)
        return energy
    
    init_flat = init_path[1:-1].ravel()  # exclude start and end
    result = minimize(path_energy, init_flat, method="L-BFGS-B",
                     options={"maxiter": 300, "ftol": 1e-12})
    
    optimised = result.x.reshape(n_waypoints, 2)
    full_path = np.vstack([start.reshape(1, 2), optimised, end.reshape(1, 2)])
    return full_path


def save_visualisation(path: np.ndarray, filename: str,
                       start: np.ndarray, end: np.ndarray) -> None:
    """Save a simple ASCII visualisation of the geodesic path."""
    output_dir = Path(os.path.join(os.path.dirname(__file__), "..", "output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename
    
    # Create a simple text-based plot
    width, height = 80, 30
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Map (x,y) ∈ [-2,2]² to grid coordinates
    def to_grid(x, y):
        gx = int((x + 2) / 4 * (width - 1))
        gy = int((y + 2) / 4 * (height - 1))
        gx = max(0, min(width - 1, gx))
        gy = max(0, min(height - 1, gy))
        return gy, gx
    
    # Plot path
    for pt in path:
        gy, gx = to_grid(pt[0], pt[1])
        grid[gy][gx] = '·'
    
    # Plot start and end
    gy, gx = to_grid(start[0], start[1])
    grid[gy][gx] = 'S'
    gy, gx = to_grid(end[0], end[1])
    grid[gy][gx] = 'E'
    
    # Add border
    lines = ['+' + '-' * width + '+']
    for row in grid:
        lines.append('|' + ''.join(row) + '|')
    lines.append('+' + '-' * width + '+')
    
    plot_text = '\n'.join(lines)
    with open(filepath, 'w') as f:
        f.write("Geodesic Path Visualisation\n")
        f.write(f"Start: ({start[0]:.2f}, {start[1]:.2f})  "
                f"V={mexican_hat_potential(start[0], start[1]):.4f}\n")
        f.write(f"End:   ({end[0]:.2f}, {end[1]:.2f})  "
                f"V={mexican_hat_potential(end[0], end[1]):.4f}\n")
        f.write(f"Path length: {len(path)} waypoints\n\n")
        f.write(plot_text)
    
    print(f"  Visualisation saved: {filepath}")


def main() -> None:
    print("=" * 72)
    print("  Manifold Database — Geodesic Analysis Example")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Step 1: Create the potential energy surface
    # ------------------------------------------------------------------
    print("\n[Step 1] Creating Mexican hat potential energy surface...")
    
    # Grid for reference
    x_grid = np.linspace(-2, 2, 100)
    y_grid = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = mexican_hat_potential(X, Y)
    
    print(f"  Surface: {X.shape} grid")
    print(f"  V range: [{Z.min():.4f}, {Z.max():.4f}]")
    print(f"  Global minima: V=0 at r=1 (ring)")
    
    # Find minimum energy points on sampled data
    min_idx = np.unravel_index(Z.argmin(), Z.shape)
    print(f"  Grid minimum at: x={X[min_idx]:.3f}, y={Y[min_idx]:.3f}, "
          f"V={Z[min_idx]:.4f}")

    # ------------------------------------------------------------------
    # Step 2: Sample points on the surface
    # ------------------------------------------------------------------
    print("\n[Step 2] Sampling points on the surface...")
    data = sample_energy_surface(n_points=2000)
    N, D = data.shape
    point_ids = [f"state_{i:04d}" for i in range(N)]
    print(f"  Sampled {N} points in {D}D (2 spatial + 1 energy)")
    
    # Energy statistics
    energies = data[:, 2]
    print(f"  Energy: min={energies.min():.4f}, mean={energies.mean():.4f}, "
          f"max={energies.max():.4f}")

    # ------------------------------------------------------------------
    # Step 3: Build manifold atlas
    # ------------------------------------------------------------------
    print("\n[Step 3] Building manifold atlas...")
    t0 = time.time()
    builder = AtlasBuilder(
        k_neighbors=15,
        pca_variance_threshold=0.95,
        min_chart_size=30,
        random_state=42,
    )
    atlas = builder.build(data, n_charts_hint=4)
    build_time = time.time() - t0
    print(f"  Atlas built in {build_time:.3f}s")
    print(f"  Charts: {len(atlas.get_all_charts())}")
    print(f"  Transitions: {len(atlas.get_all_transition_maps())}")
    
    for chart in atlas.get_all_charts():
        print(f"    '{chart.name}': dim={chart.dim}, ambient={chart.ambient_dim}")

    # ------------------------------------------------------------------
    # Step 4: Build tangent-space index
    # ------------------------------------------------------------------
    print("\n[Step 4] Building tangent-space index...")
    intrinsic_dim = 2  # 2D surface
    index = TangentSpaceIndex(intrinsic_dim=intrinsic_dim)
    stats = index.build_from_data(point_ids, data, n_anchors=40)
    print(f"  Index: {stats['n_points']} points, {stats['n_anchors']} anchors")
    print(f"  Build time: {stats['build_time_sec']:.3f}s")

    # ------------------------------------------------------------------
    # Step 5: Compute geodesic paths between minima
    # ------------------------------------------------------------------
    print("\n[Step 5] Computing geodesic paths between minima...")
    
    # Find two low-energy points (near the ring of minima at r=1)
    low_energy_mask = energies < 0.1
    low_energy_indices = np.where(low_energy_mask)[0]
    print(f"  Low-energy states (V < 0.1): {len(low_energy_indices)}")
    
    if len(low_energy_indices) >= 2:
        # Pick two minima on opposite sides of the ring
        idx_a = low_energy_indices[0]
        idx_b = low_energy_indices[len(low_energy_indices) // 2]
        
        start_xy = data[idx_a, :2]
        end_xy = data[idx_b, :2]
        print(f"  Minimum A: ({start_xy[0]:.3f}, {start_xy[1]:.3f}), "
              f"V={energies[idx_a]:.6f}")
        print(f"  Minimum B: ({end_xy[0]:.3f}, {end_xy[1]:.3f}), "
              f"V={energies[idx_b]:.6f}")
        
        # Compute geodesic path
        t0 = time.time()
        geodesic = compute_geodesic_path(
            start_xy, end_xy, mexican_hat_metric, n_waypoints=30
        )
        geo_time = time.time() - t0
        print(f"  Geodesic computed in {geo_time*1000:.2f}ms, "
              f"{len(geodesic)} waypoints")
        
        # Compute Euclidean (straight-line) distance for comparison
        eucl_dist = np.linalg.norm(start_xy - end_xy)
        
        # Compute geodesic distance (path length with metric)
        geo_dist = 0.0
        for i in range(len(geodesic) - 1):
            diff = geodesic[i + 1] - geodesic[i]
            mid = 0.5 * (geodesic[i] + geodesic[i + 1])
            g = mexican_hat_metric(mid[0], mid[1])
            geo_dist += float(np.sqrt(diff @ g @ diff))
        
        print(f"  Euclidean distance: {eucl_dist:.6f}")
        print(f"  Geodesic distance:  {geo_dist:.6f}")
        print(f"  Ratio (geo/eucl):   {geo_dist / eucl_dist:.4f}")
        
        # Save visualisation
        save_visualisation(
            geodesic, "geodesic_path.txt", start_xy, end_xy
        )

    # ------------------------------------------------------------------
    # Step 6: Geodesic ball query — states within energy threshold
    # ------------------------------------------------------------------
    print("\n[Step 6] Geodesic ball query (energy threshold)...")
    
    # Find all states within a geodesic distance threshold of a reference point
    reference = data[idx_a]  # use first minimum as reference
    print(f"  Reference point: {point_ids[idx_a]}")
    print(f"  Energy threshold: V < 0.5")
    
    # Use tangent-space search for approximate geodesic neighbours
    k_search = 50
    results = index.search(reference, k=k_search, search_k_anchors=3)
    
    # Filter by energy
    low_energy_results = []
    for pid, dist in results:
        pid_idx = int(pid.split("_")[1])
        energy = data[pid_idx, 2]
        if energy < 0.5:
            low_energy_results.append((pid, dist, energy))
    
    print(f"  Found {len(low_energy_results)} states within geodesic ball "
          f"AND energy < 0.5:")
    for pid, dist, energy in low_energy_results[:10]:
        print(f"    → {pid}  geo_dist={dist:.4f}  V={energy:.6f}")

    # ------------------------------------------------------------------
    # Step 7: Riemannian distance computation
    # ------------------------------------------------------------------
    print("\n[Step 7] Riemannian distance computations...")
    
    dist_computer = DistanceComputer(
        metric_tensor_fn=lambda x: mexican_hat_metric(x[0], x[1]),
    )
    
    # Compare distances between several pairs
    print("  Pairwise distances (Riemannian vs Euclidean):")
    test_pairs = [
        (low_energy_indices[0], low_energy_indices[1]),
        (low_energy_indices[0], low_energy_indices[2] if len(low_energy_indices) > 2 else low_energy_indices[1]),
    ]
    
    for i_a, i_b in test_pairs:
        p = data[i_a]
        q = data[i_b]
        riemann = dist_computer.compute(p, q, metric_type="tangent")
        eucl = np.linalg.norm(p - q)
        print(f"    {point_ids[i_a]} ↔ {point_ids[i_b]}:")
        print(f"      Riemannian: {riemann:.6f}")
        print(f"      Euclidean:  {eucl:.6f}")
        print(f"      Ratio:      {riemann/eucl:.4f}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  Geodesic Analysis Complete!")
    print("=" * 72)
    print(f"""
  Key takeaways:
    • Potential energy surfaces are natural manifolds
    • Geodesic paths follow valleys (low-energy routes)
    • Geodesic distance differs from Euclidean — captures topology
    • Geodesic ball queries find all reachable states within budget
    • Riemannian metric from potential gradient gives physics-aware distances
    • Tangent-space index provides fast approximate geodesic search
    • Visualisations saved to examples/output/
    """)


if __name__ == "__main__":
    main()
