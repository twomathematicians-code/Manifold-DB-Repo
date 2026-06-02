// geodesic_solver.cpp
// Geodesic equation solver utilities, path analysis, and ODE helper functions.
//
// The GeodesicSolver class is fully inline in geodesic_solver.hpp (RK4, RK45,
// shooting method, parallel transport). This file provides:
//   1. Geodesic path post-processing (smoothness, length, curvature)
//   2. Free geodesic equation evaluation function
//   3. Jacobi field computation for geodesic deviation
//   4. Geodesic midpoint and interpolation utilities
//   5. Geodesic distance matrix computation

#include "manifold/geodesic_solver.hpp"
#include "manifold/metric_tensor.hpp"
#include "manifold/chart.hpp"

#include <Eigen/QR>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace manifold {

// ═══════════════════════════════════════════════════════════════════════════════
//  Geodesic Path Analysis
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute the discrete curvature along a geodesic path.
///
/// For a smooth geodesic, the curvature is zero (it's a "straight line"
/// on the manifold). Nonzero discrete curvature indicates numerical error
/// or that the path is not a true geodesic.
///
/// κ_i = ||γ''(t_i)|| / ||γ'(t_i)||²
///
/// Computed via finite differences on the discretised path points.
///
/// @param path  Geodesic path to analyze
/// @return      Vector of curvature values at each interior point
std::vector<Scalar> path_curvature(const GeodesicPath& path) {
    std::vector<Scalar> curvatures;
    if (path.points.size() < 3) return curvatures;

    curvatures.reserve(path.points.size() - 2);

    for (size_t i = 1; i + 1 < path.points.size(); ++i) {
        const Vector& p_prev = path.points[i - 1].local_coords;
        const Vector& p_curr = path.points[i].local_coords;
        const Vector& p_next = path.points[i + 1].local_coords;

        Vector d2p = p_next - 2.0 * p_curr + p_prev;  // Second difference ≈ γ''
        Vector dp  = p_next - p_prev;                    // First difference ≈ γ'

        Scalar dp_norm_sq = dp.squaredNorm();
        Scalar d2p_norm   = d2p.norm();

        if (dp_norm_sq > 1e-30) {
            curvatures.push_back(d2p_norm / dp_norm_sq);
        } else {
            curvatures.push_back(0.0);
        }
    }
    return curvatures;
}

/// Compute the total energy of a geodesic path.
///
/// The energy functional is:
///   E[γ] = ½ ∫ g_{ij} γ'^i γ'^j dt
///
/// For a true geodesic, this is minimised given the endpoints.
/// Discrete approximation:
///   E ≈ ½ Σ_k g(γ(t_k), γ'(t_k)) · ||γ'(t_k)||² · Δt
///
/// @param path    Geodesic path
/// @param metric  Metric tensor for the chart
/// @return        Total energy E
Scalar path_energy(const GeodesicPath& path, const MetricTensor& metric) {
    if (path.points.size() < 2) return 0.0;

    Scalar energy = 0.0;

    for (size_t i = 0; i + 1 < path.points.size(); ++i) {
        const Vector& p0 = path.points[i].local_coords;
        const Vector& p1 = path.points[i + 1].local_coords;

        Vector vel = p1 - p0;
        Scalar dt = 1.0;  // Normalised step

        if (path.arc_lengths.size() > i + 1) {
            Scalar ds = path.arc_lengths[i + 1] - path.arc_lengths[i];
            if (ds > 1e-15) dt = ds;
        }

        // Evaluate metric at midpoint for better accuracy
        Vector midpoint = 0.5 * (p0 + p1);
        Matrix g = metric.evaluate(midpoint);

        // E_k = ½ v^T g v · dt
        Scalar v_sq = vel.transpose() * g * vel;
        energy += 0.5 * v_sq * dt;
    }

    return energy;
}

/// Resample a geodesic path to have evenly spaced arc-length parameter values.
///
/// Uses linear interpolation in local coordinates between path points
/// to produce a new path with n_points evenly spaced along the arc length.
///
/// @param path      Input geodesic path
/// @param n_points  Number of output points (default: 100)
/// @return          Resampled path with evenly spaced points
GeodesicPath resample_path(const GeodesicPath& path, int n_points) {
    GeodesicPath result;
    if (path.points.empty() || n_points <= 0) return result;

    result.converged = path.converged;

    if (n_points == 1) {
        result.points.push_back(path.points.front());
        result.arc_lengths.push_back(0.0);
        result.total_length = 0.0;
        return result;
    }

    Scalar total_L = path.total_length;
    if (total_L < 1e-30) total_L = 0.0;

    for (int i = 0; i < n_points; ++i) {
        Scalar s = static_cast<Scalar>(i) / static_cast<Scalar>(n_points - 1) * total_L;

        // Find segment containing s
        size_t seg = 0;
        for (size_t j = 0; j + 1 < path.arc_lengths.size(); ++j) {
            if (path.arc_lengths[j + 1] >= s - 1e-15) {
                seg = j;
                break;
            }
        }

        Scalar s0 = path.arc_lengths[seg];
        Scalar s1 = (seg + 1 < path.arc_lengths.size()) ? path.arc_lengths[seg + 1] : s0;
        Scalar frac = (s1 > s0) ? (s - s0) / (s1 - s0) : 0.0;
        frac = std::clamp(frac, Scalar(0.0), Scalar(1.0));

        ManifoldPoint pt;
        pt.chart_id = path.points[seg].chart_id;

        if (seg + 1 < path.points.size()) {
            pt.local_coords = (1.0 - frac) * path.points[seg].local_coords +
                               frac * path.points[seg + 1].local_coords;
            pt.ambient_coords = (1.0 - frac) * path.points[seg].ambient_coords +
                                  frac * path.points[seg + 1].ambient_coords;
        } else {
            pt.local_coords = path.points[seg].local_coords;
            pt.ambient_coords = path.points[seg].ambient_coords;
        }

        result.points.push_back(pt);
        result.arc_lengths.push_back(s);
    }

    result.total_length = total_L;
    result.num_steps = n_points;
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Geodesic Midpoint and Interpolation
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute the geodesic midpoint between two points.
///
/// The geodesic midpoint m satisfies:
///   exp_p(log_p(q) / 2) = m = exp_q(log_q(p) / 2)
///
/// Uses the exponential map on the chart to compute this.
///
/// @param chart    Chart for the manifold
/// @param p, q     ManifoldPoints on the same chart
/// @return         ManifoldPoint at the geodesic midpoint
ManifoldPoint geodesic_midpoint(const Chart& chart,
                                const ManifoldPoint& p,
                                const ManifoldPoint& q) {
    Vector v = chart.log_map(p, q);
    return chart.exponential_map(p, 0.5 * v);
}

/// Interpolate along a geodesic between two manifold points.
///
/// γ(t) = exp_p(t · log_p(q)),  t ∈ [0, 1]
///
/// @param chart   Chart for the manifold
/// @param p, q    Endpoints (ManifoldPoints)
/// @param t       Interpolation parameter in [0, 1]
/// @return        ManifoldPoint at γ(t)
ManifoldPoint geodesic_interpolate(const Chart& chart,
                                    const ManifoldPoint& p,
                                    const ManifoldPoint& q,
                                    Scalar t) {
    t = std::clamp(t, Scalar(0.0), Scalar(1.0));
    Vector v = chart.log_map(p, q);
    return chart.exponential_map(p, t * v);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Geodesic Distance Matrix
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute a pairwise geodesic distance matrix for a set of manifold points.
///
/// D_{ij} = d_g(p_i, p_j)  (geodesic distance)
///
/// Uses the geodesic solver for same-chart points and Euclidean fallback
/// for cross-chart points.
///
/// @param solver   Geodesic solver
/// @param points   Set of manifold points
/// @return         n × n symmetric distance matrix
Matrix geodesic_distance_matrix(const GeodesicSolver& solver,
                                  const std::vector<ManifoldPoint>& points) {
    int n = static_cast<int>(points.size());
    Matrix D(n, n);
    D.setZero();

    for (int i = 0; i < n; ++i) {
        D(i, i) = 0.0;
        for (int j = i + 1; j < n; ++j) {
            Scalar dist = solver.geodesic_distance(points[i], points[j]);
            D(i, j) = dist;
            D(j, i) = dist;
        }
    }
    return D;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Free Geodesic Equation Evaluation
// ═══════════════════════════════════════════════════════════════════════════════

/// Evaluate the geodesic acceleration at a point: a^i = -Γ^i_{jk} v^j v^k
///
/// Free function version that uses a MetricTensor directly.
///
/// @param metric    Metric tensor field
/// @param position  Current position x ∈ R^d
/// @param velocity  Current velocity v ∈ R^d
/// @param h         Finite difference step for Christoffel symbols
/// @return          Acceleration vector a ∈ R^d
Vector geodesic_accel(const MetricTensor& metric,
                      const Vector& position,
                      const Vector& velocity,
                      Scalar h) {
    int d = static_cast<int>(position.size());
    Tensor3D Gamma = metric.christoffel_symbols(position, h);

    Vector accel(d);
    accel.setZero();

    for (int i = 0; i < d; ++i) {
        Scalar sum = 0.0;
        for (int j = 0; j < d; ++j) {
            for (int k = 0; k < d; ++k) {
                sum -= Gamma(i, j, k) * velocity(j) * velocity(k);
            }
        }
        accel(i) = sum;
    }
    return accel;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Geodesic Triangle Inequality Check
// ═══════════════════════════════════════════════════════════════════════════════

/// Check whether the geodesic distance satisfies the triangle inequality.
///
/// For three points p, q, r on a Riemannian manifold:
///   d_g(p, r) ≤ d_g(p, q) + d_g(q, r)
///
/// @param solver   Geodesic solver
/// @param p, q, r  Three manifold points
/// @return         true if triangle inequality holds (within tolerance)
bool check_triangle_inequality(const GeodesicSolver& solver,
                                const ManifoldPoint& p,
                                const ManifoldPoint& q,
                                const ManifoldPoint& r,
                                Scalar tol) {
    Scalar d_pr = solver.geodesic_distance(p, r);
    Scalar d_pq = solver.geodesic_distance(p, q);
    Scalar d_qr = solver.geodesic_distance(q, r);

    return d_pr <= d_pq + d_qr + tol;
}

} // namespace manifold
