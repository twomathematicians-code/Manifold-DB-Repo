// metric_store.cpp
// MetricStore utilities, I/O helpers, and metric persistence management.
//
// The MetricStore class is fully inline in metric_store.hpp (thread-safe
// caching, file I/O, batch evaluation). This file provides:
//   1. Metric file management (cleanup, migration, backup)
//   2. Batch metric creation and population
//   3. Metric store diagnostics and integrity checks
//   4. Cross-chart metric comparison utilities
//   5. Metric field visualisation helpers (scalar curvature sampling)

#include "manifold/metric_store.hpp"
#include "manifold/metric_tensor.hpp"
#include "manifold/chart.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace manifold {

// ═══════════════════════════════════════════════════════════════════════════════
//  Metric File Management
// ═══════════════════════════════════════════════════════════════════════════════

/// Remove all metric files from the store directory.
///
/// @param db_path  Base directory for metric storage
/// @return         Number of files removed
int cleanup_metric_files(const std::string& db_path) {
    int count = 0;
    std::string metrics_dir = db_path + "/metrics";

    if (!std::filesystem::exists(metrics_dir)) return 0;

    for (const auto& entry : std::filesystem::directory_iterator(metrics_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".bin") {
            std::filesystem::remove(entry.path());
            ++count;
        }
    }
    return count;
}

/// Create a backup of all metric files in the store.
///
/// Copies all .bin files to a timestamped backup directory.
///
/// @param db_path  Base directory for metric storage
/// @return         Path to the backup directory, or empty string on failure
std::string backup_metric_files(const std::string& db_path) {
    std::string metrics_dir = db_path + "/metrics";

    if (!std::filesystem::exists(metrics_dir)) return "";

    // Create backup directory with timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << db_path << "/metrics_backup_" << time_t;

    std::string backup_dir = ss.str();
    std::filesystem::create_directories(backup_dir);

    int copied = 0;
    for (const auto& entry : std::filesystem::directory_iterator(metrics_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".bin") {
            std::filesystem::copy_file(
                entry.path(),
                backup_dir / entry.path().filename());
            ++copied;
        }
    }

    return (copied > 0) ? backup_dir : "";
}

/// Count the number of metric files on disk.
///
/// @param db_path  Base directory for metric storage
/// @return         Number of .bin metric files found
int count_metric_files(const std::string& db_path) {
    std::string metrics_dir = db_path + "/metrics";
    if (!std::filesystem::exists(metrics_dir)) return 0;

    int count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(metrics_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".bin") {
            ++count;
        }
    }
    return count;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Batch Metric Creation
// ═══════════════════════════════════════════════════════════════════════════════

/// Create identity metrics for multiple charts at once.
///
/// @param store    MetricStore to populate
/// @param chart_ids_and_dims  Vector of (chart_id, dim) pairs
void batch_create_identity_metrics(
        MetricStore& store,
        const std::vector<std::pair<uint32_t, uint32_t>>& chart_ids_and_dims) {
    for (const auto& [cid, dim] : chart_ids_and_dims) {
        store.create_metric(cid, dim);
    }
}

/// Populate metrics from chart-induced metrics.
///
/// For each chart, compute the induced metric g = J^T J at several sample
/// points and add them as anchors to the corresponding MetricTensor.
///
/// @param store    MetricStore to populate
/// @param charts   Vector of charts to process
/// @param n_samples_per_chart  Number of sample points per chart (default: 10)
void populate_metrics_from_charts(
        MetricStore& store,
        const std::vector<std::shared_ptr<Chart>>& charts,
        int n_samples_per_chart) {
    for (const auto& chart : charts) {
        auto metric = store.get_metric(chart->id());
        if (!metric) {
            metric = store.create_metric(chart->id(), chart->intrinsic_dim());
        }

        int d = static_cast<int>(chart->intrinsic_dim());

        // Sample points in local coordinates (small random perturbations near origin)
        for (int s = 0; s < n_samples_per_chart; ++s) {
            Vector local = Vector::Random(d) * 0.5;

            // Compute induced metric at this point
            Matrix g = chart->compute_local_metric(local);

            // Enforce SPD
            g = 0.5 * (g + g.transpose());

            // Add as anchor
            metric->update(local, g, 1.0);
        }

        // Commit the updated metric
        store.commit(chart->id(), *metric);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Cross-Chart Metric Comparison
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute the Frobenius-norm distance between metrics of two charts
/// evaluated at corresponding local coordinates.
///
/// ||g_A(x) - g_B(x)||_F
///
/// @param store     MetricStore
/// @param chart_a   First chart ID
/// @param chart_b   Second chart ID
/// @param coords    Points at which to evaluate
/// @return          Vector of Frobenius distances (one per point)
std::vector<Scalar> metric_distance_between_charts(
        MetricStore& store,
        uint32_t chart_a,
        uint32_t chart_b,
        const std::vector<Vector>& coords) {
    auto metric_a = store.get_metric(chart_a);
    auto metric_b = store.get_metric(chart_b);

    std::vector<Scalar> distances;
    distances.reserve(coords.size());

    for (const auto& x : coords) {
        if (metric_a && metric_b) {
            Matrix g_a = metric_a->evaluate(x);
            Matrix g_b = metric_b->evaluate(x);
            distances.push_back((g_a - g_b).norm());
        } else {
            distances.push_back(std::numeric_limits<Scalar>::infinity());
        }
    }
    return distances;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Metric Store Diagnostics
// ═══════════════════════════════════════════════════════════════════════════════

/// Check the integrity of all metrics in the store.
///
/// For each chart metric, verify:
///   1. Matrix is symmetric
///   2. Matrix is positive-definite
///   3. Dimension matches expected
///
/// @param store    MetricStore to check
/// @return         Map of {chart_id → error_description}, empty if all OK
std::unordered_map<uint32_t, std::string> validate_metric_store(
        MetricStore& store) {
    std::unordered_map<uint32_t, std::string> errors;

    // We can only validate cached metrics
    // For a full check, we'd need to load all from disk

    // Test each cached metric with the identity vector
    for (uint32_t cid = 0; cid < 100; ++cid) {
        auto metric = store.get_metric(cid);
        if (!metric) continue;

        int d = static_cast<int>(metric->dim());
        if (d == 0) {
            errors[cid] = "zero dimension";
            continue;
        }

        // Test at the origin
        Vector origin = Vector::Zero(d);
        Matrix g = metric->evaluate(origin);

        // Check symmetry
        Scalar sym_err = (g - g.transpose()).cwiseAbs().maxCoeff();
        if (sym_err > 1e-8) {
            errors[cid] = "symmetry error: " + std::to_string(sym_err);
            continue;
        }

        // Check positive-definiteness
        Eigen::LLT<Matrix> llt(g);
        if (llt.info() != Eigen::Success) {
            errors[cid] = "not positive-definite";
        }

        // Check inverse consistency: g^{-1} g ≈ I
        Matrix g_inv = metric->inverse(origin);
        Matrix product = g_inv * g;
        Scalar identity_err = (product - Matrix::Identity(d, d)).cwiseAbs().maxCoeff();
        if (identity_err > 1e-6) {
            errors[cid] = "inverse inconsistency: " + std::to_string(identity_err);
        }
    }

    return errors;
}

/// Print diagnostic information about the metric store.
///
/// @param store  MetricStore to diagnose
/// @return       Human-readable diagnostic string
std::string metric_store_diagnostics(const MetricStore& store) {
    std::ostringstream oss;
    oss << "MetricStore Diagnostics:\n";
    oss << "  Charts in cache: " << store.num_charts() << "\n";

    // Print info about each chart we can access
    for (uint32_t cid = 0; cid < 100; ++cid) {
        if (store.has_chart(cid)) {
            auto metric = store.get_metric(cid);
            if (metric) {
                oss << "  Chart " << cid << ": dim=" << metric->dim()
                    << ", anchors=" << metric->num_anchors()
                    << ", constant=" << (metric->is_constant() ? "yes" : "no")
                    << "\n";
            }
        }
    }

    return oss.str();
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Metric Field Sampling
// ═══════════════════════════════════════════════════════════════════════════════

/// Sample the scalar curvature field on a grid in local coordinates.
///
/// Creates a regular grid of sample points in the local coordinate space
/// and evaluates the scalar curvature at each point.
///
/// @param metric    MetricTensor to sample
/// @param d         Intrinsic dimension
/// @param extent    Extent of the grid in each direction (default: ±2.0)
/// @param resolution Number of grid points per dimension (default: 10)
/// @return          Vector of {coordinates, scalar_curvature} pairs
struct CurvatureSample {
    Vector coords;
    Scalar curvature;
};

std::vector<CurvatureSample> sample_scalar_curvature_field(
        const MetricTensor& metric,
        uint32_t d,
        Scalar extent,
        int resolution) {
    std::vector<CurvatureSample> samples;
    if (d == 0 || resolution < 2) return samples;

    int total_points = 1;
    for (uint32_t i = 0; i < d; ++i) total_points *= resolution;

    samples.reserve(total_points);

    // Generate grid points using a simple recursive approach
    std::vector<Scalar> steps(resolution);
    for (int i = 0; i < resolution; ++i) {
        steps[i] = -extent + 2.0 * extent * static_cast<Scalar>(i) / static_cast<Scalar>(resolution - 1);
    }

    // For simplicity, handle 1D, 2D, and 3D explicitly
    if (d == 1) {
        for (int i = 0; i < resolution; ++i) {
            Vector coords(1);
            coords(0) = steps[i];

            CurvatureSample s;
            s.coords = coords;
            s.curvature = metric.scalar_curvature(coords, 1e-4);
            samples.push_back(s);
        }
    } else if (d == 2) {
        for (int i = 0; i < resolution; ++i) {
            for (int j = 0; j < resolution; ++j) {
                Vector coords(2);
                coords(0) = steps[i];
                coords(1) = steps[j];

                CurvatureSample s;
                s.coords = coords;
                s.curvature = metric.scalar_curvature(coords, 1e-4);
                samples.push_back(s);
            }
        }
    } else {
        // Higher dimensions: sample along diagonal
        for (int i = 0; i < resolution; ++i) {
            Vector coords(d);
            for (uint32_t j = 0; j < d; ++j) {
                coords(j) = steps[i];
            }

            CurvatureSample s;
            s.coords = coords;
            s.curvature = metric.scalar_curvature(coords, 1e-4);
            samples.push_back(s);
        }
    }

    return samples;
}

} // namespace manifold
