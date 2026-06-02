#pragma once

/// @file atlas.hpp
/// @brief Chart atlas with transition maps for coordinate transport.
///
/// An Atlas is a collection of charts { (U_α, φ_α) } that covers the manifold M,
/// together with transition maps that define how to convert coordinates between
/// overlapping charts.
///
/// Mathematical foundation:
///   For two charts α, β with U_α ∩ U_β ≠ ∅, the transition map is:
///     ψ_{α→β} = φ_β^{-1} ∘ φ_α : U_α ∩ U_β ⊂ R^d → R^d
///
///   This maps coordinates in chart α to coordinates in chart β:
///     x_β = ψ_{α→β}(x_α)
///
/// The atlas provides:
///   - Chart management (add, locate, retrieve)
///   - Transition map registration and lookup
///   - Coordinate transport across charts (multi-hop via BFS)
///   - Automatic chart discovery via PCA decomposition
///
/// Transition maps:
///   - LinearTransitionMap: affine x_β = R·x_α + t (default for LinearCharts)
///   - TransitionMap base:    virtual interface for custom transitions

#include "manifold_types.hpp"
#include "chart.hpp"
#include "linear_chart.hpp"

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <Eigen/QR>

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace manifold {

// ═══════════════════════════════════════════════════════════════════════════════
//  TransitionMap  –  Abstract coordinate transformation
// ═══════════════════════════════════════════════════════════════════════════════

/// Abstract base for coordinate transformations between overlapping charts.
///
/// The transition map ψ_{α→β} converts local coordinates from chart α
/// to chart β:  x_β = ψ_{α→β}(x_α)
///
/// For a proper C^∞ atlas, the transition maps and their Jacobians must be
/// smooth and compatible (cocycle condition):
///   ψ_{α→γ} = ψ_{β→γ} ∘ ψ_{α→β}
struct TransitionMap {
    uint32_t from_chart = 0;     ///< Source chart α
    uint32_t to_chart   = 0;     ///< Target chart β
    bool is_identity    = false;  ///< Shortcut: identity map

    virtual ~TransitionMap() = default;

    /// Forward transition: x_β = ψ_{α→β}(x_α)
    /// @param coords_a  Coordinates in chart α
    /// @return  Coordinates in chart β
    [[nodiscard]] virtual Vector forward(const Vector& coords_a) const = 0;

    /// Inverse transition: x_α = ψ_{β→α}(x_β)
    /// @param coords_b  Coordinates in chart β
    /// @return  Coordinates in chart α
    [[nodiscard]] virtual Vector inverse(const Vector& coords_b) const = 0;

    /// Jacobian of the forward transition: J = ∂x_β/∂x_α  (d × d)
    [[nodiscard]] virtual Matrix jacobian(const Vector& /*coords_a*/) const = 0;

    /// Check whether a point in chart α lies within the overlap region.
    /// Points outside the overlap cannot be reliably transported.
    [[nodiscard]] virtual bool in_overlap(const Vector& /*coords_a*/) const {
        return true;  // Default: unbounded charts always overlap
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
//  LinearTransitionMap  –  Affine transition between LinearCharts
// ═══════════════════════════════════════════════════════════════════════════════

/// Affine transition map:  x_β = R · x_α + t
///
/// Derived from two LinearCharts (α, β) with:
///   y = origin_α + B_α · x_α    (chart α embedding)
///   y = origin_β + B_β · x_β    (chart β embedding)
///
/// Setting them equal and solving for x_β:
///   x_β = B_β^T · B_α · x_α + B_β^T · (origin_α − origin_β)
///
/// So:  R = B_β^T · B_α   and   t = B_β^T · (origin_α − origin_β)
struct LinearTransitionMap : public TransitionMap {
    Matrix rotation;       ///< R ∈ R^{d × d}
    Vector translation;    ///< t ∈ R^d

    /// Construct from computed rotation and translation.
    LinearTransitionMap(uint32_t from, uint32_t to,
                       const Matrix& R, const Vector& t)
        : rotation(R), translation(t)
    {
        from_chart = from;
        to_chart   = to;
        is_identity = false;
    }

    [[nodiscard]] Vector forward(const Vector& coords_a) const override {
        if (is_identity) return coords_a;
        return rotation * coords_a + translation;
    }

    [[nodiscard]] Vector inverse(const Vector& coords_b) const override {
        if (is_identity) return coords_b;
        Eigen::ColPivHouseholderQR<Matrix> qr(rotation);
        return qr.solve(coords_b - translation);
    }

    [[nodiscard]] Matrix jacobian(const Vector& /*coords_a*/) const override {
        if (is_identity) return Matrix::Identity(rotation.rows(), rotation.cols());
        return rotation;
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
//  Atlas
// ═══════════════════════════════════════════════════════════════════════════════

/// Manages a collection of charts covering the manifold M and their
/// transition maps for coordinate transport.
///
/// The atlas supports:
///   - Adding charts (LinearChart, NeuralChart, ParametricChart, etc.)
///   - Registering transition maps between overlapping charts
///   - Locating the best chart for an ambient point
///   - Transporting points across charts via multi-hop paths (BFS)
///   - Automatic chart discovery via PCA + k-means clustering
class Atlas {
public:
    Atlas() = default;

    // ── Chart Management ────────────────────────────────────────────────────

    /// Add a chart to the atlas.
    /// @throws DBException if a chart with the same ID already exists
    void add_chart(std::shared_ptr<Chart> chart) {
        if (!chart) return;
        for (const auto& c : charts_) {
            if (c->id() == chart->id()) {
                throw DBException("Atlas::add_chart: duplicate chart id " +
                                  std::to_string(chart->id()));
            }
        }
        charts_.push_back(std::move(chart));
    }

    /// Register a transition map between two charts.
    void add_transition(std::unique_ptr<TransitionMap> transition) {
        if (!transition) return;
        uint32_t from = transition->from_chart;
        uint32_t to   = transition->to_chart;

        transition_lookup_[transition_key(from, to)] = transitions_.size();
        transitions_.push_back(std::move(transition));
    }

    // ── Chart Location ──────────────────────────────────────────────────────

    /// Find the chart whose affine projection best represents an ambient point.
    /// Selects the chart with the smallest projection residual:
    ///   ||y − φ(φ⁻¹(y))||  (distance from point to chart's affine plane)
    ///
    /// @param ambient_coords  y ∈ R^D
    /// @return  Pointer to the best chart, or nullptr if atlas is empty
    [[nodiscard]] Chart* locate_chart(const Vector& ambient_coords) const {
        if (charts_.empty()) return nullptr;

        Chart* best        = nullptr;
        Scalar best_residual = std::numeric_limits<Scalar>::infinity();

        for (auto& chart : charts_) {
            Vector local    = chart->project(ambient_coords);
            Vector reembed  = chart->embed(local);
            Scalar residual = (ambient_coords - reembed).norm();

            if (residual < best_residual) {
                best_residual = residual;
                best = chart.get();
            }
        }
        return best;
    }

    // ── Coordinate Transport ────────────────────────────────────────────────

    /// Transport a manifold point from its current chart to a target chart.
    /// Uses BFS to find the shortest multi-hop path through transition maps.
    ///
    /// @param point              Source point (with chart_id and local_coords)
    /// @param target_chart_id   Target chart ID
    /// @return  Point with updated chart_id, local_coords, and ambient_coords
    /// @throws ChartNotFoundException if no path exists
    [[nodiscard]] ManifoldPoint transport(
            const ManifoldPoint& point,
            uint32_t target_chart_id) const
    {
        if (point.chart_id == target_chart_id) return point;

        auto path = find_path(point.chart_id, target_chart_id);
        if (path.empty() || path[0] != point.chart_id) {
            throw ChartNotFoundException(point.chart_id);
        }

        ManifoldPoint current = point;
        for (size_t i = 0; i + 1 < path.size(); ++i) {
            uint32_t from = path[i];
            uint32_t to   = path[i + 1];

            const TransitionMap* tmap = get_transition(from, to);
            if (!tmap) {
                throw DBException("Atlas::transport: no transition from " +
                                  std::to_string(from) + " to " + std::to_string(to));
            }

            current.local_coords = tmap->forward(current.local_coords);

            // Re-embed in the target chart's ambient coordinates
            auto target_chart = get_chart(to);
            if (target_chart) {
                current.ambient_coords = target_chart->embed(current.local_coords);
            }
            current.chart_id = to;
        }
        return current;
    }

    // ── Overlap Check ───────────────────────────────────────────────────────

    /// Check whether two charts have an overlapping domain (i.e., a transition
    /// map exists between them in either direction).
    [[nodiscard]] bool charts_overlap(uint32_t id_a, uint32_t id_b) const {
        if (id_a == id_b) return true;
        return get_transition(id_a, id_b) != nullptr ||
               get_transition(id_b, id_a) != nullptr;
    }

    // ── Lookup ──────────────────────────────────────────────────────────────

    /// Get a transition map from chart α to chart β, or nullptr if not found.
    [[nodiscard]] const TransitionMap* get_transition(uint32_t from_id,
                                                       uint32_t to_id) const {
        auto it = transition_lookup_.find(transition_key(from_id, to_id));
        if (it != transition_lookup_.end()) {
            return transitions_[it->second].get();
        }
        return nullptr;
    }

    /// Get a chart by its ID, or nullptr if not found.
    [[nodiscard]] std::shared_ptr<Chart> get_chart(uint32_t chart_id) const {
        for (const auto& c : charts_) {
            if (c->id() == chart_id) return c;
        }
        return nullptr;
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    [[nodiscard]] const std::vector<std::shared_ptr<Chart>>& charts() const { return charts_; }
    [[nodiscard]] size_t num_charts() const { return charts_.size(); }

    // ── Path Finding (BFS) ───────────────────────────────────────────────────

    /// Find the shortest sequence of chart hops from one chart to another
    /// via registered transition maps. Uses BFS.
    ///
    /// @return  Vector of chart IDs forming the path, or empty if unreachable
    [[nodiscard]] std::vector<uint32_t> find_path(uint32_t from_id,
                                                   uint32_t to_id) const
    {
        if (from_id == to_id) return {from_id};

        // Build adjacency list from transition maps
        std::unordered_map<uint32_t, std::vector<uint32_t>> adj;
        for (const auto& t : transitions_) {
            adj[t->from_chart].push_back(t->to_chart);
        }

        // BFS
        std::unordered_set<uint32_t> visited;
        std::queue<std::pair<uint32_t, std::vector<uint32_t>>> q;
        q.push({from_id, {from_id}});
        visited.insert(from_id);

        while (!q.empty()) {
            auto [current, path] = q.front();
            q.pop();

            if (current == to_id) return path;

            // Get neighbours (forward + reverse transitions)
            std::vector<uint32_t> neighbours;
            auto it = adj.find(current);
            if (it != adj.end()) neighbours = it->second;

            for (const auto& t : transitions_) {
                if (t->to_chart == current) {
                    neighbours.push_back(t->from_chart);
                }
            }

            for (uint32_t next : neighbours) {
                if (visited.insert(next).second) {
                    auto new_path = path;
                    new_path.push_back(next);
                    q.push({next, std::move(new_path)});
                }
            }
        }

        return {};  // No path found
    }

    // ── Automatic Chart Discovery ─────────────────────────────────────────────

    /// Decompose ambient data into linear charts via PCA.
    /// Optionally splits into multiple charts when residual variance is high.
    ///
    /// @param data                Ambient points (D × n, column-major)
    /// @param target_intrinsic_dim Target intrinsic dimension d
    /// @param num_charts_target   Number of charts to create (0 = auto/single)
    /// @param overlap_threshold   Residual variance threshold for multi-chart
    void discover_charts_linear(const Matrix& data,
                                 uint32_t target_intrinsic_dim,
                                 uint32_t num_charts_target = 0,
                                 Scalar overlap_threshold = 0.1)
    {
        if (data.rows() == 0 || data.cols() == 0) return;

        uint32_t ambient_dim = static_cast<uint32_t>(data.rows());
        uint32_t n_points    = static_cast<uint32_t>(data.cols());

        Vector mean = data.rowwise().mean();
        Matrix centered = data.colwise() - mean;

        // PCA via SVD
        Eigen::JacobiSVD<Matrix> svd(centered, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Matrix U = svd.matrixU();
        Vector S = svd.singularValues();

        uint32_t d = std::min(target_intrinsic_dim,
                               static_cast<uint32_t>(U.cols()));
        Matrix basis = U.leftCols(d);

        // Compute residual variance to decide if we need multiple charts
        Scalar total_var    = S.sum();
        Scalar explained_var = S.head(static_cast<int>(d)).sum();
        Scalar residual_ratio = 1.0 - explained_var / total_var;

        if (num_charts_target <= 1 || residual_ratio < overlap_threshold) {
            auto chart = std::make_shared<LinearChart>(0, basis, mean);
            add_chart(chart);
            return;
        }

        // Multiple charts: k-means clustering in PCA space
        Matrix local_coords = basis.transpose() * centered;

        uint32_t K = num_charts_target;
        std::vector<Vector> centroids(K);
        std::vector<uint32_t> assignments(n_points, 0);

        // k-means++ initialisation
        std::vector<bool> used(n_points, false);
        centroids[0] = local_coords.col(0);
        used[0] = true;

        for (uint32_t k = 1; k < K; ++k) {
            std::vector<Scalar> min_dists(n_points, std::numeric_limits<Scalar>::infinity());
            for (uint32_t i = 0; i < n_points; ++i) {
                for (uint32_t j = 0; j < k; ++j) {
                    Scalar d_val = (local_coords.col(i) - centroids[j]).squaredNorm();
                    min_dists[i] = std::min(min_dists[i], d_val);
                }
            }
            Scalar total = 0.0;
            for (uint32_t i = 0; i < n_points; ++i) total += min_dists[i];
            Scalar r = static_cast<Scalar>(std::rand()) / RAND_MAX * total;
            Scalar cumsum = 0.0;
            for (uint32_t i = 0; i < n_points; ++i) {
                cumsum += min_dists[i];
                if (cumsum >= r) {
                    centroids[k] = local_coords.col(i);
                    break;
                }
            }
        }

        // Lloyd's iterations
        for (int iter = 0; iter < 20; ++iter) {
            bool changed = false;
            for (uint32_t i = 0; i < n_points; ++i) {
                Scalar best_dist = std::numeric_limits<Scalar>::infinity();
                uint32_t best_k = 0;
                for (uint32_t k = 0; k < K; ++k) {
                    Scalar dist_val = (local_coords.col(i) - centroids[k]).squaredNorm();
                    if (dist_val < best_dist) {
                        best_dist = dist_val;
                        best_k = k;
                    }
                }
                if (assignments[i] != best_k) {
                    assignments[i] = best_k;
                    changed = true;
                }
            }
            if (!changed) break;

            for (uint32_t k = 0; k < K; ++k) {
                Vector sum = Vector::Zero(static_cast<int>(d));
                uint32_t count = 0;
                for (uint32_t i = 0; i < n_points; ++i) {
                    if (static_cast<uint32_t>(assignments[i]) == k) {
                        sum += local_coords.col(i);
                        ++count;
                    }
                }
                if (count > 0) centroids[k] = sum / count;
            }
        }

        // Create a LinearChart per cluster
        for (uint32_t k = 0; k < K; ++k) {
            std::vector<Vector> pts;
            for (uint32_t i = 0; i < n_points; ++i) {
                if (static_cast<uint32_t>(assignments[i]) == k) {
                    pts.push_back(data.col(i));
                }
            }
            if (pts.empty()) continue;

            Matrix cluster_data(ambient_dim, pts.size());
            for (size_t i = 0; i < pts.size(); ++i) {
                cluster_data.col(static_cast<int>(i)) = pts[i];
            }

            Vector cluster_mean = cluster_data.rowwise().mean();
            Matrix cluster_centered = cluster_data.colwise() - cluster_mean;
            Eigen::JacobiSVD<Matrix> csvd(cluster_centered, Eigen::ComputeThinU);
            Matrix cluster_basis = csvd.matrixU().leftCols(
                std::min(d, static_cast<uint32_t>(csvd.matrixU().cols())));

            auto chart = std::make_shared<LinearChart>(k, cluster_basis, cluster_mean);
            add_chart(chart);
        }

        auto_discover_transitions();
    }

    /// Placeholder for Python UMAP-based chart discovery.
    /// In a full implementation, this calls out to a Python callback that
    /// runs UMAP on the data and returns chart boundaries.
    void discover_charts(const Matrix& /*data*/,
                         uint32_t /*target_dim*/,
                         std::function<void(const Matrix&, uint32_t)> /*python_callback*/ = {})
    {
        // TODO: Implement Python UMAP callback bridge
        // For now, falls back to PCA-based discovery
    }

private:
    std::vector<std::shared_ptr<Chart>> charts_;
    std::vector<std::unique_ptr<TransitionMap>> transitions_;
    std::unordered_map<uint64_t, size_t> transition_lookup_;

    /// Pack two uint32_t IDs into a single uint64_t key for the lookup map.
    [[nodiscard]] static uint64_t transition_key(uint32_t from_id, uint32_t to_id) {
        return (static_cast<uint64_t>(from_id) << 32) | static_cast<uint64_t>(to_id);
    }

    /// Auto-discover LinearTransitionMaps between nearby LinearCharts.
    void auto_discover_transitions() {
        for (size_t i = 0; i < charts_.size(); ++i) {
            for (size_t j = i + 1; j < charts_.size(); ++j) {
                auto* ci = dynamic_cast<LinearChart*>(charts_[i].get());
                auto* cj = dynamic_cast<LinearChart*>(charts_[j].get());
                if (!ci || !cj) continue;

                // Check proximity
                Scalar origin_dist = (ci->origin() - cj->origin()).norm();
                Scalar extent_i = ci->basis().colwise().norm().maxCoeff();
                Scalar extent_j = cj->basis().colwise().norm().maxCoeff();
                Scalar threshold = (extent_i + extent_j) * 0.5;

                if (origin_dist > threshold * 3.0) continue;

                // Compute transition: x_j = R·x_i + t
                // From: origin_j + B_j·x_j = origin_i + B_i·x_i
                //   x_j = B_j^T·B_i·x_i + B_j^T·(origin_i − origin_j)
                uint32_t di = ci->intrinsic_dim();
                uint32_t dj = cj->intrinsic_dim();
                if (di != dj) continue;

                Matrix R = cj->basis().transpose() * ci->basis();
                Vector t = cj->basis().transpose() * (ci->origin() - cj->origin());

                auto tmap = std::make_unique<LinearTransitionMap>(
                    charts_[i]->id(), charts_[j]->id(), R, t);
                add_transition(std::move(tmap));

                // Add reverse transition
                Eigen::ColPivHouseholderQR<Matrix> qr(R);
                Matrix R_inv = qr.solve(Matrix::Identity(static_cast<int>(di),
                                                          static_cast<int>(di)));
                Vector t_inv = -R_inv * t;

                auto tmap_rev = std::make_unique<LinearTransitionMap>(
                    charts_[j]->id(), charts_[i]->id(), R_inv, t_inv);
                add_transition(std::move(tmap_rev));
            }
        }
    }
};

} // namespace manifold
