// atlas.cpp
// Atlas utilities, transition map computation, and multi-chart operations.
//
// The Atlas class is fully inline in atlas.hpp (chart management, transport,
// BFS pathfinding, PCA discovery). This file provides:
//   1. Transition map computation for arbitrary chart pairs
//   2. Multi-chart geodesic planning via transition chains
//   3. Atlas consistency validation
//   4. Chart coverage analysis
//   5. Automatic transition discovery with overlap detection

#include "manifold/atlas.hpp"
#include "manifold/linear_chart.hpp"
#include "manifold/parametric_chart.inc"  // not used, just for include consistency

#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <vector>

namespace manifold {

// ═══════════════════════════════════════════════════════════════════════════════
//  Transition Map Computation
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute and register a LinearTransitionMap between two LinearCharts.
///
/// For two linear charts with:
///   Chart A: y = origin_A + B_A · x_A
///   Chart B: y = origin_B + B_B · x_B
///
/// Setting them equal and solving for x_B:
///   x_B = B_B^T · B_A · x_A + B_B^T · (origin_A - origin_B)
///
/// @param atlas    Atlas to register the transition with
/// @param chart_a  Source LinearChart
/// @param chart_b  Target LinearChart
/// @return         true if transition was registered successfully
bool compute_and_register_transition(Atlas& atlas,
                                     const LinearChart& chart_a,
                                     const LinearChart& chart_b) {
    // Verify compatible dimensions
    if (chart_a.intrinsic_dim() != chart_b.intrinsic_dim()) return false;
    if (chart_a.ambient_dim() != chart_b.ambient_dim()) return false;

    int d = static_cast<int>(chart_a.intrinsic_dim());

    // Compute the transition: x_b = R · x_a + t
    Matrix R = chart_b.basis().transpose() * chart_a.basis();
    Vector t = chart_b.basis().transpose() * (chart_a.origin() - chart_b.origin());

    // Forward transition: A → B
    auto tmap_fwd = std::make_unique<LinearTransitionMap>(
        chart_a.id(), chart_b.id(), R, t);
    atlas.add_transition(std::move(tmap_fwd));

    // Reverse transition: B → A
    Eigen::ColPivHouseholderQR<Matrix> qr(R);
    Matrix R_inv = qr.solve(Matrix::Identity(d, d));
    Vector t_inv = -R_inv * t;

    auto tmap_rev = std::make_unique<LinearTransitionMap>(
        chart_b.id(), chart_a.id(), R_inv, t_inv);
    atlas.add_transition(std::move(tmap_rev));

    return true;
}

/// Compute the Jacobian of a transition map numerically.
///
/// J_ij = ∂x_B^i / ∂x_A^j  computed via central finite differences.
///
/// @param tmap       Transition map
/// @param coords_a   Point in chart A coordinates
/// @param h          Finite difference step (default: 1e-5)
/// @return           d × d Jacobian matrix
Matrix numerical_transition_jacobian(const TransitionMap& tmap,
                                     const Vector& coords_a,
                                     Scalar h) {
    int d = static_cast<int>(coords_a.size());
    Matrix J(d, d);

    for (int j = 0; j < d; ++j) {
        Vector xp = coords_a;
        Vector xm = coords_a;
        xp(j) += h;
        xm(j) -= h;

        Vector fp = tmap.forward(xp);
        Vector fm = tmap.forward(xm);

        J.col(j) = (fp - fm) / (2.0 * h);
    }

    return J;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Multi-Chart Geodesic Planning
// ═══════════════════════════════════════════════════════════════════════════════

/// Plan a multi-chart geodesic path between two ambient points.
///
/// When the start and end points lie on different charts, this function:
///   1. Finds the BFS path of chart transitions
///   2. Computes intermediate waypoints at chart boundaries
///   3. Returns the sequence of chart transitions and waypoints
///
/// @param atlas           Atlas for chart management
/// @param start_ambient   Start point in ambient space
/// @param end_ambient     End point in ambient space
/// @return                Vector of {chart_id, local_coords} pairs along the path
struct ChartWaypoint {
    uint32_t chart_id;
    Vector   local_coords;
};

std::vector<ChartWaypoint> plan_multi_chart_path(
        const Atlas& atlas,
        const Vector& start_ambient,
        const Vector& end_ambient) {
    std::vector<ChartWaypoint> path;

    Chart* chart_s = atlas.locate_chart(start_ambient);
    Chart* chart_e = atlas.locate_chart(end_ambient);

    if (!chart_s || !chart_e) return path;

    // Same chart: direct path
    if (chart_s->id() == chart_e->id()) {
        path.push_back({chart_s->id(), chart_s->project(start_ambient)});
        path.push_back({chart_e->id(), chart_e->project(end_ambient)});
        return path;
    }

    // Find chart transition path via BFS
    std::vector<uint32_t> chart_path = atlas.find_path(chart_s->id(), chart_e->id());
    if (chart_path.empty()) return path;

    // Create waypoints at each chart boundary
    for (size_t i = 0; i < chart_path.size(); ++i) {
        auto chart = atlas.get_chart(chart_path[i]);
        if (!chart) continue;

        Vector local;
        if (i == 0) {
            local = chart_s->project(start_ambient);
        } else if (i == chart_path.size() - 1) {
            // Transport to final chart
            ManifoldPoint pt;
            pt.chart_id       = chart_path[i - 1];
            pt.local_coords   = path.back().local_coords;
            pt.ambient_coords = atlas.get_chart(chart_path[i - 1])->embed(pt.local_coords);

            try {
                ManifoldPoint transported = atlas.transport(pt, chart_path[i]);
                local = transported.local_coords;
            } catch (const DBException&) {
                local = chart_e->project(end_ambient);
            }
        } else {
            // Intermediate point: use the chart's origin as a waypoint
            if (auto* lc = dynamic_cast<LinearChart*>(chart.get())) {
                local = Vector::Zero(static_cast<int>(lc->intrinsic_dim()));
            } else {
                local = chart->project(
                    atlas.locate_chart(start_ambient)->embed(path.back().local_coords));
            }
        }

        path.push_back({chart_path[i], local});
    }

    return path;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Atlas Consistency Validation
// ═══════════════════════════════════════════════════════════════════════════════

/// Check the cocycle condition for the atlas:
///   ψ_{α→γ} ≈ ψ_{β→γ} ∘ ψ_{α→β}
///
/// For a C^∞ atlas, this must hold exactly for all overlapping triplets.
/// In practice, we check numerically within a tolerance.
///
/// @param atlas   Atlas to validate
/// @param tol     Tolerance for the check (default: 1e-6)
/// @return        true if the cocycle condition holds for all tested triplets
bool validate_cocycle_condition(const Atlas& atlas, Scalar tol) {
    const auto& charts = atlas.charts();
    if (charts.size() < 3) return true;

    int failures = 0;
    int checks = 0;

    // Check all triplets
    for (size_t i = 0; i < charts.size() && failures == 0; ++i) {
        for (size_t j = 0; j < charts.size() && failures == 0; ++j) {
            for (size_t k = 0; k < charts.size() && failures == 0; ++k) {
                uint32_t id_a = charts[i]->id();
                uint32_t id_b = charts[j]->id();
                uint32_t id_c = charts[k]->id();

                // Get transitions: A→B, B→C, A→C
                const TransitionMap* tab = atlas.get_transition(id_a, id_b);
                const TransitionMap* tbc = atlas.get_transition(id_b, id_c);
                const TransitionMap* tac = atlas.get_transition(id_a, id_c);

                if (!tab || !tbc || !tac) continue;

                // Test at a random point (use zero vector for simplicity)
                int d = static_cast<int>(charts[i]->intrinsic_dim());
                Vector x_a = Vector::Zero(d);

                // Compute ψ_{B→C} ∘ ψ_{A→B}(x_a)
                Vector x_b = tab->forward(x_a);
                Vector x_c_composed = tbc->forward(x_b);

                // Compute ψ_{A→C}(x_a)
                Vector x_c_direct = tac->forward(x_a);

                Scalar error = (x_c_composed - x_c_direct).norm();
                ++checks;

                if (error > tol) {
                    ++failures;
                }
            }
        }
    }

    return failures == 0;
}

/// Check that all charts in the atlas have consistent ambient dimensions.
///
/// @param atlas   Atlas to validate
/// @return        true if all charts have the same ambient dimension
bool validate_consistent_dimensions(const Atlas& atlas) {
    const auto& charts = atlas.charts();
    if (charts.empty()) return true;

    uint32_t D = charts[0]->ambient_dim();
    for (const auto& chart : charts) {
        if (chart->ambient_dim() != D) return false;
    }
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Chart Coverage Analysis
// ═══════════════════════════════════════════════════════════════════════════════

/// Determine which chart best covers a set of ambient points.
///
/// Returns the number of points assigned to each chart.
///
/// @param atlas          Atlas with charts
/// @param ambient_points D × n matrix of ambient points
/// @return               Map of {chart_id → num_assigned_points}
std::unordered_map<uint32_t, size_t> chart_coverage_analysis(
        const Atlas& atlas,
        const Matrix& ambient_points) {
    std::unordered_map<uint32_t, size_t> coverage;

    for (int i = 0; i < ambient_points.cols(); ++i) {
        Vector pt = ambient_points.col(i);
        Chart* chart = atlas.locate_chart(pt);
        if (chart) {
            coverage[chart->id()]++;
        }
    }
    return coverage;
}

/// Compute the coverage overlap between all pairs of charts.
///
/// For each pair (A, B), count how many test points are equally close
/// to both charts (within a tolerance factor).
///
/// @param atlas          Atlas with charts
/// @param test_points    D × m matrix of test points
/// @param tol_factor     Tolerance factor for overlap (default: 1.1)
/// @return               Map of {(chart_a, chart_b) → overlap_count}
std::unordered_map<uint64_t, size_t> pairwise_chart_overlap(
        const Atlas& atlas,
        const Matrix& test_points,
        Scalar tol_factor) {
    std::unordered_map<uint64_t, size_t> overlaps;
    auto pack = [](uint32_t a, uint32_t b) -> uint64_t {
        return (static_cast<uint64_t>(std::min(a, b)) << 32) |
               static_cast<uint64_t>(std::max(a, b));
    };

    for (int i = 0; i < test_points.cols(); ++i) {
        Vector pt = test_points.col(i);

        // Find the best and second-best charts
        Chart* best = nullptr;
        Chart* second = nullptr;
        Scalar best_r = std::numeric_limits<Scalar>::infinity();
        Scalar second_r = std::numeric_limits<Scalar>::infinity();

        for (const auto& chart : atlas.charts()) {
            Vector local = chart->project(pt);
            Vector reembed = chart->embed(local);
            Scalar r = (pt - reembed).norm();

            if (r < best_r) {
                second = best;
                second_r = best_r;
                best = chart;
                best_r = r;
            } else if (r < second_r) {
                second = chart;
                second_r = r;
            }
        }

        if (best && second && second_r < best_r * tol_factor) {
            overlaps[pack(best->id(), second->id())]++;
        }
    }
    return overlaps;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Global Chart Discovery Wrapper
// ═══════════════════════════════════════════════════════════════════════════════

/// Convenience function: create an atlas from ambient data with automatic
/// chart discovery and transition map registration.
///
/// @param data               D × n ambient data matrix
/// @param target_dim         Target intrinsic dimension
/// @param num_charts         Number of charts (0 = auto-detect)
/// @param overlap_threshold  PCA residual threshold for multi-chart
/// @return                   Constructed Atlas
std::unique_ptr<Atlas> build_atlas_from_data(
        const Matrix& data,
        uint32_t target_dim,
        uint32_t num_charts,
        Scalar overlap_threshold) {
    auto atlas = std::make_unique<Atlas>();
    atlas->discover_charts_linear(data, target_dim, num_charts, overlap_threshold);
    return atlas;
}

} // namespace manifold
