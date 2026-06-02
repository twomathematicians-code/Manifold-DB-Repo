#pragma once

/// @file manifold_db.hpp
/// @brief Top-level database API for ManifoldDB – a geometric inference engine.
///
/// ManifoldDB stores data on Riemannian manifolds and supports queries using
/// geodesic distances rather than flat Euclidean metrics. It provides:
///
///   - Data ingestion (ambient-space vectors)
///   - Automatic atlas construction (PCA-based chart decomposition)
///   - Geodesic k-NN and ball queries
///   - Cross-modal search across different data modalities
///   - Schema evolution (extending the manifold structure)
///
/// Architecture:
///   ┌──────────────────────────────────────────────────────┐
///   │                     ManifoldDB                       │
///   │  ┌─────────┐  ┌──────────────┐  ┌────────────────┐  │
///   │  │  Atlas   │  │ MetricStore  │  │ GeodesicSolver │  │
///   │  │ (charts) │  │ (g_ij(x))   │  │ (IVP/BVP)     │  │
///   │  └────┬─────┘  └──────┬──────┘  └───────┬────────┘  │
///   │       │               │                  │           │
///   │  ┌────┴───────────────┴──────────────────┴────────┐ │
///   │  │        TangentSpaceIndex (per-chart R-tree)     │ │
///   │  └────────────────────────────────────────────────┘ │
///   └──────────────────────────────────────────────────────┘
///
/// Mathematical motivation:
///   High-dimensional data (images, embeddings, molecular descriptors) often
///   lies on or near a low-dimensional manifold M ⊂ R^D. ManifoldDB exploits
///   this structure by:
///   1. Covering M with local coordinate charts (atlas)
///   2. Learning the Riemannian metric g_ij(x) from data
///   3. Querying via geodesic distances (shortest paths on M)

#include "atlas.hpp"
#include "geodesic_solver.hpp"
#include "metric_store.hpp"
#include "tangent_space_index.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace manifold {

// ═══════════════════════════════════════════════════════════════════════════════
//  ManifoldDB  –  Top-level API
// ═══════════════════════════════════════════════════════════════════════════════

/// The main database class for manifold-structured data.
///
/// Usage pattern:
///   1. Construct with a Config
///   2. Insert ambient-space data points
///   3. Build atlas (automatic chart decomposition)
///   4. Query using geodesic k-NN or ball search
class ManifoldDB {
public:
    // ── Configuration ───────────────────────────────────────────────────────

    /// Database configuration parameters.
    struct Config {
        std::string  storage_path         = "./manifolddb_data";  ///< Base directory for persistence
        uint32_t     default_intrinsic_dim = 10;                   ///< Default intrinsic dimension d
        bool         enable_cuda          = false;                 ///< Enable GPU-accelerated geodesics
        Scalar       geodesic_tolerance   = 1e-6;                ///< Geodesic solver tolerance
        SolverConfig solver_config;                               ///< Solver tuning parameters
        size_t       index_max_leaf_size  = 16;                   ///< R-tree leaf capacity
    };

    /// Runtime statistics about the database state.
    struct Stats {
        size_t num_charts            = 0;    ///< Number of charts in the atlas
        size_t total_points         = 0;    ///< Total data points across all modalities
        double avg_geodesic_time_ms  = 0.0;  ///< Average geodesic solve time (ms)
        size_t index_size           = 0;    ///< Total points indexed across all charts
        double build_time_ms        = 0.0;  ///< Last atlas build time (ms)
    };

    /// Construct a ManifoldDB with the given configuration.
    /// Creates the storage directory and initialises all subsystems.
    explicit ManifoldDB(const Config& config)
        : config_(config)
        , atlas_(std::make_unique<Atlas>())
        , metric_store_(std::make_unique<MetricStore>(
              config.storage_path + "/metrics"))
        , geodesic_solver_(std::make_unique<GeodesicSolver>(
              std::make_shared<MetricStore>(config.storage_path + "/metrics"),
              config.solver_config))
    {
        std::filesystem::create_directories(config.storage_path);
    }

    ~ManifoldDB() = default;

    // ─── Data Ingestion ─────────────────────────────────────────────────────

    /// Insert a collection of ambient-space points under a modality.
    /// Each point is assigned a global_id and tentatively placed on a chart
    /// if the atlas has already been built.
    ///
    /// @param ambient_points  Vector of D-dimensional ambient vectors
    /// @param modality_id     Modality identifier (default: 0)
    void insert(const std::vector<Vector>& ambient_points, uint32_t modality_id = 0) {
        auto& mod = modalities_[modality_id];
        mod.id = modality_id;

        uint64_t next_id = 0;
        if (!mod.points.empty()) {
            next_id = mod.points.back().global_id + 1;
        }

        for (const auto& pt : ambient_points) {
            ManifoldPoint mp;
            mp.ambient_coords = pt;
            mp.global_id      = next_id++;
            mp.timestamp      = 0.0;
            mp.chart_id       = 0;

            // If atlas exists, assign point to best chart
            if (atlas_->num_charts() > 0) {
                Chart* best = atlas_->locate_chart(pt);
                if (best) {
                    mp.chart_id     = best->id();
                    mp.local_coords = best->project(pt);
                }
            }

            mod.points.push_back(mp);
        }

        rebuild_ambient_matrix(mod);

        // Track modality ordering
        if (std::find(modality_order_.begin(), modality_order_.end(), modality_id)
            == modality_order_.end()) {
            modality_order_.push_back(modality_id);
        }
    }

    /// Insert points as a column-major Eigen matrix (each column = one point).
    void insert(const Matrix& ambient_matrix, uint32_t modality_id = 0) {
        std::vector<Vector> pts;
        pts.reserve(ambient_matrix.cols());
        for (int i = 0; i < ambient_matrix.cols(); ++i) {
            pts.push_back(ambient_matrix.col(i));
        }
        insert(pts, modality_id);
    }

    // ─── Atlas Construction ───────────────────────────────────────────────────

    /// Build the atlas from inserted data with automatic dimension detection.
    /// @param target_intrinsic_dim  Target dimension d (0 = use config default)
    void build_atlas(uint32_t target_intrinsic_dim = 0) {
        if (target_intrinsic_dim == 0) {
            target_intrinsic_dim = config_.default_intrinsic_dim;
        }
        build_atlas_linear(target_intrinsic_dim);
    }

    /// Build atlas using PCA-based linear charts.
    void build_atlas_linear(uint32_t intrinsic_dim) {
        auto t0 = std::chrono::high_resolution_clock::now();

        atlas_ = std::make_unique<Atlas>();

        for (uint32_t mod_id : modality_order_) {
            auto it = modalities_.find(mod_id);
            if (it == modalities_.end() || it->second.raw_ambient.cols() == 0)
                continue;

            atlas_->discover_charts_linear(it->second.raw_ambient, intrinsic_dim);

            // Assign points to charts
            for (auto& pt : it->second.points) {
                Chart* best = atlas_->locate_chart(pt.ambient_coords);
                if (best) {
                    pt.chart_id     = best->id();
                    pt.local_coords = best->project(pt.ambient_coords);
                }
            }
        }

        // Create metric tensors for each chart
        for (const auto& chart : atlas_->charts()) {
            metric_store_->create_metric(chart->id(), chart->intrinsic_dim());
        }

        // Build tangent-space indexes
        indexes_.clear();
        for (const auto& chart : atlas_->charts()) {
            auto idx = std::make_unique<TangentSpaceIndex>(
                chart->id(), chart->intrinsic_dim(), config_.index_max_leaf_size);

            std::vector<ManifoldPoint> chart_points;
            for (auto& [mod_id, mod] : modalities_) {
                for (const auto& pt : mod.points) {
                    if (pt.chart_id == chart->id()) {
                        chart_points.push_back(pt);
                    }
                }
            }
            idx->build(chart_points);
            indexes_[chart->id()] = std::move(idx);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        (void)elapsed_ms;
    }

    // ─── Geodesic Queries ──────────────────────────────────────────────────

    /// k-nearest neighbours by geodesic distance.
    ///
    /// Strategy:
    ///   1. Locate the query point's chart
    ///   2. Retrieve candidate neighbours via R-tree (local Euclidean proxy)
    ///   3. Re-rank candidates by true geodesic distance
    ///   4. Return top-k
    ///
    /// @param query_ambient  Query point in ambient space R^D
    /// @param k              Number of neighbours
    /// @param max_distance   Optional: maximum geodesic distance filter
    /// @return  Vector of NeighborResult sorted by ascending geodesic distance
    [[nodiscard]] std::vector<NeighborResult> query_geodesic_knn(
            const Vector& query_ambient, uint32_t k,
            Scalar max_distance = std::numeric_limits<Scalar>::infinity()) const
    {
        Chart* chart = atlas_->locate_chart(query_ambient);
        if (!chart) return {};

        Vector query_local = chart->project(query_ambient);

        ensure_index(chart->id());
        auto it = indexes_.find(chart->id());
        if (it == indexes_.end() || !it->second) return {};

        // Candidate expansion: retrieve more than k for re-ranking
        size_t n_candidates = std::max<size_t>(k * 3, 50);
        auto candidates = it->second->knn(query_local, n_candidates);

        if (candidates.size() <= static_cast<size_t>(k)) return candidates;

        // Re-rank by true geodesic distance
        std::vector<NeighborResult> results;
        results.reserve(candidates.size());

        ManifoldPoint query_pt;
        query_pt.chart_id       = chart->id();
        query_pt.local_coords   = query_local;
        query_pt.ambient_coords = query_ambient;

        for (auto& cand : candidates) {
            Scalar geo_dist = geodesic_solver_->geodesic_distance(query_pt, cand.point);
            NeighborResult nr;
            nr.point              = cand.point;
            nr.geodesic_distance  = geo_dist;
            nr.euclidean_residual = cand.euclidean_residual;
            results.push_back(nr);
        }

        std::sort(results.begin(), results.end());
        if (results.size() > static_cast<size_t>(k)) {
            results.resize(k);
        }

        if (max_distance < std::numeric_limits<Scalar>::infinity()) {
            results.erase(
                std::remove_if(results.begin(), results.end(),
                    [max_distance](const NeighborResult& r) {
                        return r.geodesic_distance > max_distance;
                    }),
                results.end());
        }

        return results;
    }

    /// All points within a geodesic ball of given radius.
    ///
    /// @param center_ambient  Centre of the ball in ambient space
    /// @param radius          Geodesic radius
    /// @return  All ManifoldPoints with d_g(center, p) ≤ radius
    [[nodiscard]] std::vector<ManifoldPoint> query_geodesic_ball(
            const Vector& center_ambient, Scalar radius) const
    {
        Chart* chart = atlas_->locate_chart(center_ambient);
        if (!chart) return {};

        Vector center_local = chart->project(center_ambient);

        ensure_index(chart->id());
        auto it = indexes_.find(chart->id());
        if (it == indexes_.end() || !it->second) return {};

        // Approximate ball with slightly enlarged Euclidean radius
        auto candidates = it->second->range_search(center_local, radius * 1.5);

        // Filter by true geodesic distance
        std::vector<ManifoldPoint> results;
        ManifoldPoint center_pt;
        center_pt.chart_id       = chart->id();
        center_pt.local_coords   = center_local;
        center_pt.ambient_coords = center_ambient;

        for (const auto& pt : candidates) {
            Scalar geo_dist = geodesic_solver_->geodesic_distance(center_pt, pt);
            if (geo_dist <= radius) {
                results.push_back(pt);
            }
        }
        return results;
    }

    /// Compute the geodesic path between two ambient points.
    [[nodiscard]] GeodesicPath query_geodesic_path(
            const Vector& start_ambient,
            const Vector& end_ambient) const
    {
        Chart* chart_s = atlas_->locate_chart(start_ambient);
        Chart* chart_e = atlas_->locate_chart(end_ambient);

        if (!chart_s || !chart_e) return {};

        // Same chart: direct BVP
        if (chart_s->id() == chart_e->id()) {
            ManifoldPoint ps, pe;
            ps.chart_id       = chart_s->id();
            ps.local_coords   = chart_s->project(start_ambient);
            ps.ambient_coords = start_ambient;

            pe.chart_id       = chart_e->id();
            pe.local_coords   = chart_e->project(end_ambient);
            pe.ambient_coords = end_ambient;

            return geodesic_solver_->solve_bvp(ps, pe, SolverType::SHOOTING);
        }

        // Cross-chart: transport and solve
        ManifoldPoint pe;
        pe.chart_id       = chart_e->id();
        pe.local_coords   = chart_e->project(end_ambient);
        pe.ambient_coords = end_ambient;

        try {
            ManifoldPoint pe_transported = atlas_->transport(pe, chart_s->id());

            ManifoldPoint ps;
            ps.chart_id       = chart_s->id();
            ps.local_coords   = chart_s->project(start_ambient);
            ps.ambient_coords = start_ambient;

            return geodesic_solver_->solve_bvp(ps, pe_transported, SolverType::SHOOTING);
        } catch (const DBException&) {
            // Fallback: ambient straight line
            GeodesicPath path;
            int n_steps = 20;
            for (int i = 0; i <= n_steps; ++i) {
                Scalar alpha = static_cast<Scalar>(i) / n_steps;
                Vector p = start_ambient + alpha * (end_ambient - start_ambient);
                ManifoldPoint pt;
                pt.ambient_coords = p;
                pt.chart_id = chart_s->id();
                if (chart_s) pt.local_coords = chart_s->project(p);
                path.points.push_back(pt);
                path.arc_lengths.push_back(alpha * (end_ambient - start_ambient).norm());
            }
            path.total_length = (end_ambient - start_ambient).norm();
            path.num_steps = n_steps;
            return path;
        }
    }

    // ─── Cross-Modal Queries ────────────────────────────────────────────────

    /// Search points in a target modality using a query from a source modality.
    /// Both modalities share the same atlas, so the query is mapped to the
    /// shared chart space and candidates are retrieved from the target modality.
    ///
    /// @param query_ambient      Query point in ambient space
    /// @param source_modality    Source modality ID
    /// @param target_modality    Target modality ID
    /// @param k                  Number of results
    [[nodiscard]] std::vector<NeighborResult> query_cross_modal(
            const Vector& query_ambient,
            uint32_t source_modality,
            uint32_t target_modality,
            uint32_t k) const
    {
        if (source_modality == target_modality) {
            return query_geodesic_knn(query_ambient, k);
        }

        auto src_it = modalities_.find(source_modality);
        auto tgt_it = modalities_.find(target_modality);
        if (src_it == modalities_.end() || tgt_it == modalities_.end()) return {};

        Chart* chart = atlas_->locate_chart(query_ambient);
        if (!chart) return {};

        Vector query_local = chart->project(query_ambient);

        // Collect target modality points on the same chart
        std::vector<ManifoldPoint> target_pts;
        for (const auto& pt : tgt_it->second.points) {
            if (pt.chart_id == chart->id()) {
                target_pts.push_back(pt);
            }
        }

        // k-NN by local Euclidean (refinable with geodesic re-ranking)
        std::partial_sort(target_pts.begin(),
                          target_pts.begin() + std::min<size_t>(k, target_pts.size()),
                          target_pts.end(),
                          [&query_local](const ManifoldPoint& a, const ManifoldPoint& b) {
                              return (a.local_coords - query_local).norm() <
                                     (b.local_coords - query_local).norm();
                          });

        size_t result_size = std::min<size_t>(k, target_pts.size());
        std::vector<NeighborResult> results;
        results.reserve(result_size);
        for (size_t i = 0; i < result_size; ++i) {
            NeighborResult nr;
            nr.point              = target_pts[i];
            nr.geodesic_distance  = (target_pts[i].local_coords - query_local).norm();
            nr.euclidean_residual = (target_pts[i].ambient_coords - query_ambient).norm();
            results.push_back(nr);
        }

        return results;
    }

    // ─── Schema Evolution ───────────────────────────────────────────────────

    /// Extend the manifold structure to accommodate new data points.
    /// Inserts the points and rebuilds the atlas with the combined dataset.
    void evolve_schema(const std::vector<Vector>& new_ambient_points) {
        uint32_t new_mod = 0;
        for (const auto& id : modality_order_) {
            new_mod = std::max(new_mod, id + 1);
        }
        insert(new_ambient_points, new_mod);

        uint32_t dim = config_.default_intrinsic_dim;
        if (!atlas_->charts().empty()) {
            dim = atlas_->charts()[0]->intrinsic_dim();
        }
        build_atlas(dim);
    }

    // ─── Utility ────────────────────────────────────────────────────────────

    /// Collect runtime statistics about the database.
    [[nodiscard]] Stats stats() const {
        Stats s;
        s.num_charts = atlas_->num_charts();

        size_t total_pts = 0;
        for (const auto& [id, mod] : modalities_) {
            total_pts += mod.points.size();
        }
        s.total_points = total_pts;

        s.index_size = 0;
        for (const auto& [cid, idx] : indexes_) {
            if (idx) s.index_size += idx->size();
        }

        return s;
    }

    // ─── Direct Access (advanced use) ───────────────────────────────────────

    [[nodiscard]] const Atlas& atlas() const              { return *atlas_; }
    [[nodiscard]] Atlas&       atlas()                    { return *atlas_; }
    [[nodiscard]] const MetricStore& metric_store() const { return *metric_store_; }
    [[nodiscard]] MetricStore&       metric_store()       { return *metric_store_; }
    [[nodiscard]] const GeodesicSolver& solver() const    { return *geodesic_solver_; }

private:
    Config config_;
    std::unique_ptr<Atlas>           atlas_;
    std::unique_ptr<MetricStore>     metric_store_;
    std::unique_ptr<GeodesicSolver>  geodesic_solver_;

    /// Per-chart tangent-space indexes (mutable for lazy construction).
    mutable std::unordered_map<uint32_t, std::unique_ptr<TangentSpaceIndex>> indexes_;

    /// Per-modality data storage.
    struct ModalityData {
        uint32_t id = 0;
        std::vector<ManifoldPoint> points;       ///< All points in this modality
        Matrix raw_ambient;                       ///< Ambient data as D × n matrix
        std::vector<uint32_t> chart_ids;           ///< Chart assignment per point
    };
    std::unordered_map<uint32_t, ModalityData> modalities_;
    std::vector<uint32_t> modality_order_;          ///< Preserved insertion order

    /// Rebuild the raw ambient matrix from the points vector.
    void rebuild_ambient_matrix(ModalityData& mod) {
        if (mod.points.empty()) return;

        uint32_t ambient_dim = static_cast<uint32_t>(mod.points[0].ambient_coords.size());
        uint32_t n = static_cast<uint32_t>(mod.points.size());

        mod.raw_ambient.resize(ambient_dim, n);
        for (uint32_t i = 0; i < n; ++i) {
            mod.raw_ambient.col(static_cast<int>(i)) = mod.points[i].ambient_coords;
        }
    }

    /// Lazily build a tangent-space index for a chart if not yet present.
    void ensure_index(uint32_t chart_id) const {
        if (indexes_.find(chart_id) != indexes_.end()) return;

        auto idx = std::make_unique<TangentSpaceIndex>(
            chart_id, config_.default_intrinsic_dim, config_.index_max_leaf_size);

        std::vector<ManifoldPoint> chart_points;
        for (const auto& [mod_id, mod] : modalities_) {
            for (const auto& pt : mod.points) {
                if (pt.chart_id == chart_id) {
                    chart_points.push_back(pt);
                }
            }
        }

        if (!chart_points.empty()) {
            idx->build(chart_points);
        }

        indexes_[chart_id] = std::move(idx);
    }

    /// Convert an ambient vector to a ManifoldPoint (assign to best chart).
    [[nodiscard]] ManifoldPoint ambient_to_manifold_point(
            const Vector& ambient, uint32_t /*modality_id*/) const
    {
        ManifoldPoint mp;
        mp.ambient_coords = ambient;

        Chart* chart = atlas_->locate_chart(ambient);
        if (chart) {
            mp.chart_id     = chart->id();
            mp.local_coords = chart->project(ambient);
        }

        return mp;
    }

    /// Convert a ManifoldPoint back to ambient coordinates.
    [[nodiscard]] Vector manifold_point_to_ambient(const ManifoldPoint& point) const {
        if (point.ambient_coords.size() > 0) {
            return point.ambient_coords;
        }

        auto chart = atlas_->get_chart(point.chart_id);
        if (chart) {
            return chart->embed(point.local_coords);
        }

        return Vector();
    }
};

} // namespace manifold
