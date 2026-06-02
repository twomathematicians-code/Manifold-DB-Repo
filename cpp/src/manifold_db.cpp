// manifold_db.cpp
// ManifoldDB utilities, import/export, and high-level query helpers.
//
// The ManifoldDB class is fully inline in manifold_db.hpp (insert, build_atlas,
// query_geodesic_knn, cross-modal queries). This file provides:
//   1. ManifoldDB factory and configuration helpers
//   2. Data import/export from CSV and binary formats
//   3. Bulk geodesic distance computation utilities
//   4. Query result formatting and analysis
//   5. Database statistics and diagnostic utilities

#include "manifold/manifold_db.hpp"
#include "manifold/linear_chart.hpp"
#include "manifold/tangent_space_index.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace manifold {

// ═══════════════════════════════════════════════════════════════════════════════
//  Configuration Helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// Create a default ManifoldDB configuration.
/// Provides sensible defaults for common use cases.
///
/// @param storage_path   Directory for persistence
/// @param intrinsic_dim Default intrinsic dimension
/// @return              Config struct
ManifoldDB::Config default_config(const std::string& storage_path,
                                   uint32_t intrinsic_dim) {
    ManifoldDB::Config cfg;
    cfg.storage_path          = storage_path;
    cfg.default_intrinsic_dim = intrinsic_dim;
    cfg.enable_cuda           = false;
    cfg.geodesic_tolerance    = 1e-6;
    cfg.index_max_leaf_size   = 16;

    cfg.solver_config.initial_step       = 1e-3;
    cfg.solver_config.min_step          = 1e-8;
    cfg.solver_config.max_step          = 0.1;
    cfg.solver_config.tolerance         = 1e-8;
    cfg.solver_config.max_iterations    = 10000;
    cfg.solver_config.max_bvp_iterations = 50;
    cfg.solver_config.bvp_tolerance     = 1e-6;
    cfg.solver_config.adaptive_step     = true;

    return cfg;
}

/// Create a ManifoldDB with a fast configuration (lower accuracy, faster queries).
///
/// @param storage_path   Directory for persistence
/// @param intrinsic_dim  Default intrinsic dimension
/// @return              Config optimised for speed
ManifoldDB::Config fast_config(const std::string& storage_path,
                                uint32_t intrinsic_dim) {
    auto cfg = default_config(storage_path, intrinsic_dim);
    cfg.solver_config.tolerance         = 1e-4;
    cfg.solver_config.max_iterations    = 1000;
    cfg.solver_config.max_bvp_iterations = 20;
    cfg.solver_config.bvp_tolerance     = 1e-3;
    cfg.solver_config.adaptive_step     = false;
    cfg.solver_config.initial_step       = 1e-2;
    return cfg;
}

/// Create a ManifoldDB with a high-accuracy configuration.
///
/// @param storage_path   Directory for persistence
/// @param intrinsic_dim  Default intrinsic dimension
/// @return              Config optimised for accuracy
ManifoldDB::Config accurate_config(const std::string& storage_path,
                                   uint32_t intrinsic_dim) {
    auto cfg = default_config(storage_path, intrinsic_dim);
    cfg.solver_config.tolerance         = 1e-10;
    cfg.solver_config.max_iterations    = 50000;
    cfg.solver_config.max_bvp_iterations = 100;
    cfg.solver_config.bvp_tolerance     = 1e-8;
    cfg.solver_config.initial_step       = 1e-4;
    cfg.solver_config.min_step          = 1e-12;
    return cfg;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Data Import
// ═══════════════════════════════════════════════════════════════════════════════

/// Import ambient vectors from a CSV file.
///
/// Format: one row per point, comma-separated values.
/// First line may be a header (detected and skipped if non-numeric).
///
/// @param filepath  Path to CSV file
/// @param n_cols    Expected number of columns (0 = auto-detect from first data row)
/// @return          Vector of ambient-space vectors
std::vector<Vector> import_csv(const std::string& filepath, uint32_t n_cols) {
    std::ifstream ifs(filepath);
    if (!ifs.is_open()) {
        throw SerializationError("Cannot open CSV file: " + filepath);
    }

    std::vector<Vector> points;
    std::string line;

    bool first_line = true;

    while (std::getline(ifs, line)) {
        // Skip empty lines
        if (line.empty()) continue;

        // Skip comment lines
        if (line[0] == '#' || line[0] == '%') continue;

        // Parse comma-separated values
        std::stringstream ss(line);
        std::string token;
        std::vector<Scalar> values;

        while (std::getline(ss, token, ',')) {
            // Trim whitespace
            size_t start = token.find_first_not_of(" \t\r\n");
            size_t end = token.find_last_not_of(" \t\r\n");
            if (start != std::string::npos && end != std::string::npos) {
                token = token.substr(start, end - start + 1);
            }

            try {
                values.push_back(std::stod(token));
            } catch (const std::exception&) {
                // Non-numeric value: skip this line (likely a header)
                break;
            }
        }

        // Skip if we couldn't parse any values (header line)
        if (values.empty()) continue;

        // Auto-detect dimension from first valid line
        if (n_cols == 0) {
            n_cols = static_cast<uint32_t>(values.size());
        }

        // Validate dimension
        if (static_cast<uint32_t>(values.size()) != n_cols) {
            continue;  // Skip inconsistent rows
        }

        Vector v(n_cols);
        for (uint32_t i = 0; i < n_cols; ++i) {
            v(i) = values[i];
        }
        points.push_back(v);

        first_line = false;
    }

    return points;
}

/// Import ambient vectors from a binary file.
///
/// Format: [n_points:u32] [ambient_dim:u32] [data: n×D doubles, row-major]
///
/// @param filepath  Path to binary file
/// @return          Vector of ambient-space vectors
std::vector<Vector> import_binary(const std::string& filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs.is_open()) {
        throw SerializationError("Cannot open binary file: " + filepath);
    }

    uint32_t n_points = 0, ambient_dim = 0;
    ifs.read(reinterpret_cast<char*>(&n_points), sizeof(n_points));
    ifs.read(reinterpret_cast<char*>(&ambient_dim), sizeof(ambient_dim));

    std::vector<Vector> points;
    points.reserve(n_points);

    for (uint32_t i = 0; i < n_points; ++i) {
        Vector v(ambient_dim);
        ifs.read(reinterpret_cast<char*>(v.data()), ambient_dim * sizeof(Scalar));
        if (ifs) points.push_back(v);
    }

    return points;
}

/// Conveniently load a CSV file into the database.
///
/// @param db        ManifoldDB to load into
/// @param filepath  Path to CSV file
/// @param modality  Modality ID (default: 0)
/// @return          Number of points loaded
size_t load_csv_into_db(ManifoldDB& db, const std::string& filepath, uint32_t modality) {
    auto points = import_csv(filepath);
    if (points.empty()) return 0;
    db.insert(points, modality);
    return points.size();
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Data Export
// ═══════════════════════════════════════════════════════════════════════════════

/// Export ambient vectors to a CSV file.
///
/// @param filepath  Output file path
/// @param points    Vector of ambient-space vectors
/// @param header    Optional header line (empty = no header)
void export_csv(const std::string& filepath,
                const std::vector<Vector>& points,
                const std::string& header) {
    std::ofstream ofs(filepath);
    if (!ofs.is_open()) {
        throw SerializationError("Cannot open CSV file for writing: " + filepath);
    }

    if (!header.empty()) {
        ofs << header << "\n";
    }

    for (const auto& v : points) {
        for (int i = 0; i < v.size(); ++i) {
            if (i > 0) ofs << ",";
            ofs << v(i);
        }
        ofs << "\n";
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Query Result Analysis
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute the recall of geodesic k-NN vs Euclidean k-NN.
///
/// Recall = |geodesic_knn ∩ euclidean_knn| / k
///
/// A recall of 1.0 means the geodesic re-ranking doesn't change the
/// ordering (manifold is approximately flat locally). Lower recall
/// indicates significant curvature effects.
///
/// @param geodesic_results   Results from geodesic k-NN
/// @param euclidean_results  Results from Euclidean k-NN
/// @return                   Recall in [0, 1]
Scalar compute_knn_recall(const std::vector<NeighborResult>& geodesic_results,
                           const std::vector<NeighborResult>& euclidean_results) {
    if (geodesic_results.empty() || euclidean_results.empty()) return 0.0;

    // Count common global IDs in top-k
    std::unordered_set<uint64_t> geo_ids;
    for (const auto& r : geodesic_results) {
        geo_ids.insert(r.point.global_id);
    }

    size_t common = 0;
    for (const auto& r : euclidean_results) {
        if (geo_ids.count(r.point.global_id)) {
            ++common;
        }
    }

    return static_cast<Scalar>(common) /
           static_cast<Scalar>(std::min(geodesic_results.size(), euclidean_results.size()));
}

/// Compute the geodesic vs Euclidean distance ratio for a set of neighbours.
///
/// ratio = d_geodesic / d_euclidean
///
/// A ratio > 1 indicates the geodesic is longer than Euclidean (curved manifold).
/// A ratio ≈ 1 indicates locally flat geometry.
///
/// @param results  k-NN results with both distances computed
/// @return         Vector of distance ratios
std::vector<Scalar> geodesic_euclidean_ratios(
        const std::vector<NeighborResult>& results) {
    std::vector<Scalar> ratios;
    ratios.reserve(results.size());

    for (const auto& r : results) {
        if (r.euclidean_residual > 1e-15) {
            ratios.push_back(r.geodesic_distance / r.euclidean_residual);
        } else {
            ratios.push_back(1.0);
        }
    }
    return ratios;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Database Diagnostic Utilities
// ═══════════════════════════════════════════════════════════════════════════════

/// Print a summary of the database state to a string.
///
/// @param db  ManifoldDB to summarise
/// @return   Human-readable summary string
std::string database_summary(const ManifoldDB& db) {
    auto stats = db.stats();
    std::ostringstream oss;

    oss << "ManifoldDB Summary:\n";
    oss << "  Charts:            " << stats.num_charts << "\n";
    oss << "  Total Points:      " << stats.total_points << "\n";
    oss << "  Indexed Points:    " << stats.index_size << "\n";
    oss << "  Build Time (ms):   " << stats.build_time_ms << "\n";

    const auto& atlas = db.atlas();
    oss << "  Chart Details:\n";
    for (const auto& chart : atlas.charts()) {
        oss << "    Chart " << chart->id()
            << ": dim=" << chart->intrinsic_dim()
            << " ambient=" << chart->ambient_dim()
            << " type=";
        switch (chart->type()) {
            case ChartType::LINEAR:    oss << "LINEAR"; break;
            case ChartType::NEURAL:    oss << "NEURAL"; break;
            case ChartType::PARAMETRIC:oss << "PARAMETRIC"; break;
            case ChartType::CUSTOM:    oss << "CUSTOM"; break;
        }
        oss << "\n";
    }

    return oss.str();
}

/// Perform a quick self-test of the database: insert points, build atlas,
/// run a query, and verify basic invariants.
///
/// @param db   ManifoldDB to test
/// @param dim  Test dimension (default: 10)
/// @return     true if self-test passes
bool self_test(ManifoldDB& db, uint32_t dim) {
    try {
        // Generate random test data
        int n = 50;
        Matrix data(dim, n);
        for (int i = 0; i < n; ++i) {
            data.col(i) = Vector::Random(dim);
        }

        // Insert
        std::vector<Vector> points;
        points.reserve(n);
        for (int i = 0; i < n; ++i) {
            points.push_back(data.col(i));
        }
        db.insert(points, 0);

        // Build atlas
        uint32_t d = std::min(dim, static_cast<uint32_t>(5));
        db.build_atlas(d);

        // Verify atlas was built
        if (db.atlas().num_charts() == 0) return false;

        // Run a k-NN query
        Vector query = Vector::Random(dim);
        auto results = db.query_geodesic_knn(query, 5);
        if (results.empty()) return false;

        // Verify results are sorted by distance
        for (size_t i = 1; i < results.size(); ++i) {
            if (results[i].geodesic_distance < results[i - 1].geodesic_distance - 1e-10) {
                return false;
            }
        }

        return true;
    } catch (const std::exception&) {
        return false;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Modality Management
// ═══════════════════════════════════════════════════════════════════════════════

/// Get the list of modality IDs currently in the database.
/// Requires access to the internal modality structure.
///
/// @param db  ManifoldDB instance
/// @return   Vector of modality IDs (preserving insertion order)
std::vector<uint32_t> get_modality_ids(const ManifoldDB& /*db*/) {
    // The modality_order_ is private, so this is a placeholder.
    // In production, this would be exposed via the Stats struct or a public method.
    return {0};
}

} // namespace manifold
