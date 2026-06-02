#!/usr/bin/env python3
"""
Temporal Manifold Example — Time-series data with drifting manifolds.

Demonstrates how manifold databases handle time-varying data where the
underlying data distribution changes over time (concept drift).

This example:
  1. Generates time-varying data (rotating/clustering distribution)
  2. Builds a time-indexed atlas with charts for each time slice
  3. Queries across time with parallel transport
  4. Detects anomalies (points off the manifold)

Key insight: In a manifold database, temporal drift is a smooth geometric
deformation of the data manifold, not a discrete "version change".

Run:
    python examples/scripts/temporal_manifold.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from manifold_db.atlas import AtlasBuilder, AtlasManager, Chart, AffineTransition
from manifold_db.tangent_index import TangentSpace, TangentSpaceIndex
from manifold_db.query import QueryBuilder, ManifoldQuery, QueryType
from manifold_db.metric import EuclideanMetric, MetricTensorStore
from manifold_db.connection import LeviCivitaConnection, TemporalTransport


# ---------------------------------------------------------------------------
# Time-Varying Data Generation
# ---------------------------------------------------------------------------

def generate_temporal_data(
    n_time_steps: int = 10,
    points_per_step: int = 200,
    n_dims: int = 8,
    drift_type: str = "rotation",
    noise_level: float = 0.2,
) -> list:
    """Generate time-varying data with smooth distribution drift.
    
    Each time step produces a set of points from a distribution that
    gradually changes over time.  The drift is modelled as a smooth
    geometric deformation.
    
    Parameters
    ----------
    drift_type : str
        - "rotation": the distribution rotates in feature space
        - "expansion": the distribution expands/contracts
        - "translation": the distribution shifts in space
    
    Returns
    -------
    list of (time_step, data) tuples
    """
    rng = np.random.default_rng(42)
    datasets = []
    
    for t in range(n_time_steps):
        progress = t / max(n_time_steps - 1, 1)  # 0.0 to 1.0
        angle = progress * np.pi / 3  # 60 degrees total rotation
        
        # Base cluster centers
        n_clusters = 3
        centers = rng.normal(0, 2.0, size=(n_clusters, n_dims))
        
        # Apply temporal drift to centers
        if drift_type == "rotation":
            # Rotate centers using 2D rotation in first two dims
            rot_matrix = np.eye(n_dims)
            rot_matrix[0, 0] = np.cos(angle)
            rot_matrix[0, 1] = -np.sin(angle)
            rot_matrix[1, 0] = np.sin(angle)
            rot_matrix[1, 1] = np.cos(angle)
            drifted_centers = centers @ rot_matrix.T
            
        elif drift_type == "expansion":
            # Scale centers by (1 + sin(progress * pi) * 0.5)
            scale = 1.0 + 0.5 * np.sin(progress * np.pi)
            drifted_centers = centers * scale
            
        elif drift_type == "translation":
            # Shift centers along first dimension
            shift = np.zeros(n_dims)
            shift[0] = progress * 2.0
            drifted_centers = centers + shift
        else:
            drifted_centers = centers
        
        # Generate points around drifted centers
        points_per_cluster = points_per_step // n_clusters
        data_parts = []
        for c in drifted_centers:
            cluster_pts = rng.normal(
                loc=c, scale=noise_level, size=(points_per_cluster, n_dims)
            )
            data_parts.append(cluster_pts)
        
        step_data = np.vstack(data_parts)
        datasets.append((t, step_data))
    
    return datasets


def inject_anomalies(data: np.ndarray, n_anomalies: int = 5,
                     severity: float = 5.0) -> tuple:
    """Inject anomalous points far from the manifold.
    
    Anomalies are points with unusually large norms or in sparse regions.
    
    Returns:
        (data_with_anomalies, anomaly_indices)
    """
    rng = np.random.default_rng(999)
    n_dims = data.shape[1]
    
    anomaly_indices = []
    anomaly_data = []
    
    for i in range(n_anomalies):
        # Anomalies are far from the data distribution
        anom_pt = rng.choice([-1, 1], size=n_dims) * severity
        anom_pt += rng.normal(0, 0.1, n_dims)
        anomaly_data.append(anom_pt)
        anomaly_indices.append(len(data) + i)
    
    anomaly_data = np.array(anomaly_data)
    data_with_anomalies = np.vstack([data, anomaly_data])
    
    return data_with_anomalies, anomaly_indices


def compute_manifold_deviation(point: np.ndarray,
                                tangent_space: TangentSpace) -> float:
    """Compute how far a point deviates from the manifold.
    
    This is measured as the residual after projection into the tangent space.
    Points ON the manifold have small residuals; anomalies have large ones.
    """
    # Project point to tangent space
    projected = tangent_space.project(point.reshape(1, -1))
    # Lift back to ambient
    reconstructed = tangent_space.lift(projected)
    # Residual = distance from original to reconstruction
    deviation = float(np.linalg.norm(point - reconstructed[0]))
    return deviation


def main() -> None:
    print("=" * 72)
    print("  Manifold Database — Temporal Manifold Example")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Step 1: Generate time-varying data
    # ------------------------------------------------------------------
    print("\n[Step 1] Generating time-varying data...")
    n_time_steps = 10
    points_per_step = 200
    n_dims = 8
    
    temporal_data = generate_temporal_data(
        n_time_steps=n_time_steps,
        points_per_step=points_per_step,
        n_dims=n_dims,
        drift_type="rotation",
        noise_level=0.2,
    )
    
    for t, data in temporal_data[:3]:
        print(f"  t={t}: {data.shape}, "
              f"mean_norm={np.mean(np.linalg.norm(data, axis=1)):.3f}")
    print(f"  ... ({n_time_steps} time steps total)")

    # ------------------------------------------------------------------
    # Step 2: Build time-indexed atlas
    # ------------------------------------------------------------------
    print("\n[Step 2] Building time-indexed atlas...")
    atlas = AtlasManager(name="temporal_atlas")
    tangent_spaces = {}
    indices = {}
    
    for t, data in temporal_data:
        step_label = f"t_{t:02d}"
        
        # Build chart for this time step
        chart = Chart(
            name=f"chart_{step_label}",
            dim=5,  # intrinsic dimension
            ambient_dim=n_dims,
            metadata={"time_step": t, "modality": "temporal"},
        )
        atlas.add_chart(chart)
        
        # Build tangent space for this time step
        ts = TangentSpace(
            base_point=data.mean(axis=0),
            data=data[:50],
            intrinsic_dim=5,
        )
        tangent_spaces[t] = ts
        
        # Build index for this time step
        idx = TangentSpaceIndex(intrinsic_dim=5)
        pids = [f"{step_label}_pt_{i:04d}" for i in range(len(data))]
        idx.build_from_data(pids, data, n_anchors=10)
        indices[t] = idx
        
        if t < 3 or t == n_time_steps - 1:
            print(f"  t={t}: chart '{chart.name}', index has {idx.size} points")
    print(f"  Total charts: {len(atlas.get_all_charts())}")

    # ------------------------------------------------------------------
    # Step 3: Query across time with parallel transport
    # ------------------------------------------------------------------
    print("\n[Step 3] Cross-time queries with parallel transport...")
    
    # Pick a query point from time t=0
    t_source = 0
    query_point = temporal_data[t_source][1][0]
    print(f"  Source: t={t_source}, point norm={np.linalg.norm(query_point):.3f}")
    
    # Transport and search at each subsequent time step
    print("\n  Searching across time (transporting query):")
    prev_ts = tangent_spaces[t_source]
    tangent_coords = prev_ts.project(query_point.reshape(1, -1))
    
    for t_target in range(0, n_time_steps, 2):
        target_ts = tangent_spaces[t_target]
        
        # Parallel transport tangent coordinates to target time step
        if t_target != t_source:
            transported = prev_ts.parallel_transport(
                tangent_coords[0], target_ts
            )
        else:
            transported = tangent_coords[0]
        
        # Lift to ambient space
        transported_ambient = target_ts.lift(transported.reshape(1, -1))
        
        # Search in target time step's index
        results = indices[t_target].search(transported_ambient[0], k=3)
        
        top_dist = results[0][1] if results else float('inf')
        print(f"    t={t_target:2d}: best_match_dist={top_dist:.4f}  "
              f"(transported vec norm={np.linalg.norm(transported):.4f})")
        
        # Update previous TS for chain transport
        if t_target != t_source:
            prev_ts = target_ts

    # ------------------------------------------------------------------
    # Step 4: Anomaly detection
    # ------------------------------------------------------------------
    print("\n[Step 4] Anomaly detection...")
    
    # Use a middle time step for anomaly detection
    t_detect = n_time_steps // 2
    detect_data = temporal_data[t_detect][1].copy()
    
    # Inject anomalies
    clean_data, anomaly_indices = inject_anomalies(detect_data, n_anomalies=8)
    print(f"  Detection at t={t_detect}: {clean_data.shape} "
          f"({len(anomaly_indices)} anomalies injected)")
    
    # Build tangent space from clean portion
    detect_ts = TangentSpace(
        base_point=detect_data.mean(axis=0),
        data=detect_data[:100],
        intrinsic_dim=5,
    )
    
    # Compute deviation for all points
    deviations = []
    for i in range(len(clean_data)):
        dev = compute_manifold_deviation(clean_data[i], detect_ts)
        deviations.append((i, dev))
    
    deviations.sort(key=lambda x: x[1], reverse=True)
    
    # Top deviations
    print(f"  Top 10 deviations from manifold:")
    detected_anomalies = 0
    true_anomalies_found = 0
    for rank, (idx, dev) in enumerate(deviations[:10]):
        is_anomaly = "ANOMALY" if idx in anomaly_indices else "normal"
        if idx in anomaly_indices:
            true_anomalies_found += 1
        print(f"    #{rank+1}: point {idx:4d}, deviation={dev:.4f} [{is_anomaly}]")
        if dev > 1.0:  # threshold
            detected_anomalies += 1
    
    # Detection statistics
    print(f"\n  Detection results:")
    print(f"    True anomalies: {len(anomaly_indices)}")
    print(f"    Found in top 10: {true_anomalies_found}")
    print(f"    Detected (dev > 1.0): {detected_anomalies}")
    
    # Compute false positive rate on normal data
    normal_deviations = [d for i, d in deviations if i not in anomaly_indices]
    fp_rate = sum(1 for d in normal_deviations if d > 1.0) / len(normal_deviations)
    print(f"    False positive rate (normal data): {fp_rate:.2%}")

    # ------------------------------------------------------------------
    # Step 5: TemporalTransport for smooth time evolution
    # ------------------------------------------------------------------
    print("\n[Step 5] TemporalTransport setup...")
    
    temporal_transport = TemporalTransport()
    
    # Register time-dependent metric (identity for simplicity)
    def metric_at_time(t: float, x: np.ndarray) -> np.ndarray:
        """Time-dependent metric tensor."""
        dim = len(x)
        # Metric slowly evolves: small perturbation with time
        perturbation = 0.01 * np.sin(t * np.pi / n_time_steps)
        g = np.eye(dim) * (1.0 + perturbation)
        return g
    
    # Reference curve: sequence of centroids over time
    reference_curve = np.array([
        temporal_data[t][1].mean(axis=0) for t in range(n_time_steps)
    ])
    temporal_transport.register_temporal_path(
        times=np.arange(n_time_steps, dtype=float),
        reference_curve=reference_curve,
        metric_fn=metric_at_time,
    )
    print(f"  Registered temporal path: {n_time_steps} time steps")
    print(f"  Reference curve shape: {reference_curve.shape}")

    # Transport a vector from t=0 to t=9
    test_vec = np.random.randn(n_dims)
    result = temporal_transport.transport(
        vector=test_vec,
        source_time=0.0,
        target_time=float(n_time_steps - 1),
    )
    print(f"  Transported vector: t=0 → t={n_time_steps-1}")
    print(f"    Original norm: {np.linalg.norm(test_vec):.4f}")
    print(f"    Transported norm: {np.linalg.norm(result):.4f}")
    print(f"    Norm ratio: {np.linalg.norm(result)/np.linalg.norm(test_vec):.4f}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  Temporal Manifold Analysis Complete!")
    print("=" * 72)
    print(f"""
  Key takeaways:
    • Temporal drift = smooth geometric deformation of the manifold
    • Separate charts per time step capture local structure
    • Parallel transport enables cross-time queries
    • Anomaly detection: large deviation from tangent space = off-manifold
    • TemporalTransport handles time-dependent metric evolution
    • Chain transport: t=0 → t=2 → t=4 → ... for distant time queries
    • False positive rate is low because normals stay near the manifold
    """)


if __name__ == "__main__":
    main()
