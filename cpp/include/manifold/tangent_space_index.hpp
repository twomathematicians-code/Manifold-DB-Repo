#pragma once

/// @file tangent_space_index.hpp
/// @brief R-tree spatial index over the tangent bundle for fast local search.
///
/// TangentSpaceIndex provides efficient k-NN and range queries within a single
/// chart's local coordinate system. It maintains an R-tree (axis-aligned bounding
/// box hierarchy) over ManifoldPoints, enabling:
///
///   - k-nearest-neighbour search (local Euclidean distance)
///   - Range search (ball queries)
///   - Bulk build via STR (Sort-Tile-Recursive) packing
///   - Incremental insert with node splitting
///   - Persistence to/from binary files
///
/// Performance characteristics:
///   - k-NN query:     O(log n + k) average (branch-and-bound pruning)
///   - Range query:    O(log n + m) where m = result count
///   - Bulk build:     O(n log n) via STR
///   - Insert:         O(log n + split) amortised
///
/// Note: This index uses LOCAL Euclidean distance as a proxy for geodesic
/// distance. For exact geodesic k-NN, re-rank results using GeodesicSolver.

#include "manifold_types.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <utility>
#include <vector>

namespace manifold {

// ═══════════════════════════════════════════════════════════════════════════════
//  RTreeNode  –  Axis-Aligned Bounding Box node
// ═══════════════════════════════════════════════════════════════════════════════

/// A node in the R-tree, either an internal node (with children) or a leaf
/// (with stored ManifoldPoints). Each node maintains an AABB for pruning.
///
/// The R-tree is not balanced in the strict sense (unlike an R*-tree),
/// but provides good average-case performance for spatial queries.
struct RTreeNode {
    Vector min_corner;                             ///< AABB lower-left corner
    Vector max_corner;                             ///< AABB upper-right corner
    std::vector<ManifoldPoint> points;              ///< Leaf points (only if is_leaf)
    std::vector<std::unique_ptr<RTreeNode>> children; ///< Child nodes (only if !is_leaf)
    bool   is_leaf = true;                          ///< True if this is a leaf node
    size_t node_id = 0;                             ///< Node identifier (for diagnostics)

    /// Test whether this node's AABB intersects a query AABB.
    [[nodiscard]] bool intersects(const Vector& query_min,
                                   const Vector& query_max) const {
        if (min_corner.size() == 0 || query_min.size() == 0) return false;
        int d = static_cast<int>(min_corner.size());
        for (int i = 0; i < d; ++i) {
            if (max_corner(i) < query_min(i) || min_corner(i) > query_max(i))
                return false;
        }
        return true;
    }

    /// Test whether a point lies inside this node's AABB (with epsilon tolerance).
    [[nodiscard]] bool contains_point(const Vector& point, Scalar eps = 1e-10) const {
        if (min_corner.size() == 0 || point.size() == 0) return false;
        int d = static_cast<int>(min_corner.size());
        for (int i = 0; i < d; ++i) {
            if (point(i) < min_corner(i) - eps || point(i) > max_corner(i) + eps)
                return false;
        }
        return true;
    }

    /// Expand the AABB to include a new point.
    void expand_to_include(const Vector& point) {
        if (min_corner.size() == 0) {
            min_corner = point;
            max_corner = point;
            return;
        }
        int d = static_cast<int>(point.size());
        for (int i = 0; i < d; ++i) {
            min_corner(i) = std::min(min_corner(i), point(i));
            max_corner(i) = std::max(max_corner(i), point(i));
        }
    }

    /// Minimum Euclidean distance from a query point to the AABB.
    /// Used for branch-and-bound pruning in k-NN queries.
    [[nodiscard]] Scalar min_distance(const Vector& query) const {
        if (min_corner.size() == 0) return std::numeric_limits<Scalar>::infinity();
        Scalar dist_sq = 0.0;
        int d = static_cast<int>(min_corner.size());
        for (int i = 0; i < d; ++i) {
            Scalar delta = 0.0;
            if (query(i) < min_corner(i))
                delta = min_corner(i) - query(i);
            else if (query(i) > max_corner(i))
                delta = query(i) - max_corner(i);
            dist_sq += delta * delta;
        }
        return std::sqrt(dist_sq);
    }

    /// AABB volume (used for split quality evaluation).
    [[nodiscard]] Scalar volume() const {
        if (min_corner.size() == 0) return 0.0;
        Scalar vol = 1.0;
        int d = static_cast<int>(min_corner.size());
        for (int i = 0; i < d; ++i) {
            vol *= (max_corner(i) - min_corner(i));
        }
        return vol;
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
//  TangentSpaceIndex  –  R-tree over local coordinates
// ═══════════════════════════════════════════════════════════════════════════════

/// Spatial index over manifold points within a single chart's local coordinates.
///
/// The index is organised as an R-tree over the tangent bundle TM restricted
/// to one chart. Points are indexed by their local coordinates x ∈ R^d,
/// with the AABB hierarchy enabling efficient spatial queries.
///
/// For geodesic queries, this index serves as a candidate generator:
///   1. Retrieve k candidates by local Euclidean distance (fast)
///   2. Re-rank by true geodesic distance via GeodesicSolver (accurate)
class TangentSpaceIndex {
public:
    /// Construct an empty index for a chart.
    /// @param chart_id       Chart this index belongs to
    /// @param intrinsic_dim  Dimension d of the chart's local space
    /// @param max_leaf_size  Maximum points per leaf before splitting (default: 16)
    TangentSpaceIndex(uint32_t chart_id, uint32_t intrinsic_dim,
                      size_t max_leaf_size = 16)
        : chart_id_(chart_id)
        , dim_(intrinsic_dim)
        , max_leaf_size_(max_leaf_size)
        , num_points_(0)
    {}

    // ── Insert ──────────────────────────────────────────────────────────────

    /// Insert a single manifold point into the index.
    /// The point's local_coords are used for spatial placement.
    void insert(const ManifoldPoint& point) {
        if (root_ == nullptr) {
            root_ = std::make_unique<RTreeNode>();
            root_->node_id = 0;
        }
        insert_recursive(root_.get(), point, 0);
        ++num_points_;
    }

    // ── Bulk Build ───────────────────────────────────────────────────────────

    /// Build the index from a collection of points using STR packing.
    /// Clears any existing data before building.
    /// @param points  Collection of manifold points (local_coords must be set)
    void build(const std::vector<ManifoldPoint>& points) {
        clear();
        if (points.empty()) return;

        std::vector<ManifoldPoint> sorted_pts = points;
        root_ = build_recursive(sorted_pts, 0);
        num_points_ = points.size();
    }

    // ── k-NN Search ─────────────────────────────────────────────────────────

    /// Find the k nearest neighbours to a query point in local coordinates.
    /// Uses branch-and-bound with a max-heap for efficient pruning.
    ///
    /// @param query_local  Query point x_q ∈ R^d
    /// @param k            Number of neighbours to retrieve
    /// @param max_radius   Optional: maximum local Euclidean distance (filter)
    /// @return  Vector of NeighborResult sorted by ascending local distance
    [[nodiscard]] std::vector<NeighborResult> knn(
            const Vector& query_local,
            size_t k,
            Scalar max_radius = std::numeric_limits<Scalar>::infinity()) const
    {
        if (!root_) return {};

        // Max-heap: pop farthest when we exceed k
        std::priority_queue<NeighborResult> heap;

        knn_recursive(root_.get(), query_local, heap, k);

        std::vector<NeighborResult> results;
        results.reserve(heap.size());
        while (!heap.empty()) {
            results.push_back(heap.top());
            heap.pop();
        }
        std::reverse(results.begin(), results.end());

        // Optional radius filter
        if (max_radius < std::numeric_limits<Scalar>::infinity()) {
            results.erase(
                std::remove_if(results.begin(), results.end(),
                    [max_radius](const NeighborResult& r) {
                        return r.geodesic_distance > max_radius;
                    }),
                results.end());
        }
        return results;
    }

    /// Alias for knn() – k-NN search in tangent space coordinates.
    [[nodiscard]] std::vector<NeighborResult> knn_tangent(
            const Vector& query_local,
            size_t k) const
    {
        return knn(query_local, k);
    }

    // ── Range Search ──────────────────────────────────────────────────────────

    /// Find all points within a given radius of the query point.
    /// @param query_local  Query point x_q ∈ R^d
    /// @param radius      Search radius r
    /// @return  All ManifoldPoints with ||x - x_q|| ≤ r
    [[nodiscard]] std::vector<ManifoldPoint> range_search(
            const Vector& query_local,
            Scalar radius) const
    {
        std::vector<ManifoldPoint> results;
        if (!root_) return results;

        Scalar radius_sq = radius * radius;
        range_search_recursive(root_.get(), query_local, radius_sq, results);
        return results;
    }

    // ── Clear ─────────────────────────────────────────────────────────────────

    /// Remove all points and reset the index.
    void clear() {
        root_.reset();
        num_points_ = 0;
    }

    // ── Statistics ───────────────────────────────────────────────────────────

    [[nodiscard]] size_t   size()     const { return num_points_; }
    [[nodiscard]] uint32_t chart_id() const { return chart_id_; }
    [[nodiscard]] bool     is_built() const { return root_ != nullptr; }

    // ── Persistence ──────────────────────────────────────────────────────────

    /// Save all indexed points to a binary file.
    /// The R-tree structure is NOT preserved – points are stored linearly
    /// and rebuilt on load. (TODO: serialise tree structure for faster reload.)
    void save(const std::string& path) const {
        std::ofstream ofs(path, std::ios::binary);
        if (!ofs) throw SerializationError("Cannot open file for writing: " + path);

        // Header: chart_id, dim, num_points
        uint32_t cid = chart_id_;
        uint32_t dim = dim_;
        size_t   n   = num_points_;
        ofs.write(reinterpret_cast<const char*>(&cid), sizeof(cid));
        ofs.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        ofs.write(reinterpret_cast<const char*>(&n), sizeof(n));

        // Collect all points via tree traversal
        std::vector<ManifoldPoint> all_pts;
        collect_points(root_.get(), all_pts);

        uint32_t np = static_cast<uint32_t>(all_pts.size());
        ofs.write(reinterpret_cast<const char*>(&np), sizeof(np));

        for (const auto& pt : all_pts) {
            ofs.write(reinterpret_cast<const char*>(&pt.chart_id), sizeof(pt.chart_id));
            ofs.write(reinterpret_cast<const char*>(&pt.global_id), sizeof(pt.global_id));
            ofs.write(reinterpret_cast<const char*>(&pt.timestamp), sizeof(pt.timestamp));

            uint32_t lcd = static_cast<uint32_t>(pt.local_coords.size());
            uint32_t acd = static_cast<uint32_t>(pt.ambient_coords.size());
            ofs.write(reinterpret_cast<const char*>(&lcd), sizeof(lcd));
            ofs.write(reinterpret_cast<const char*>(&acd), sizeof(acd));

            if (lcd > 0)
                ofs.write(reinterpret_cast<const char*>(pt.local_coords.data()),
                          lcd * sizeof(Scalar));
            if (acd > 0)
                ofs.write(reinterpret_cast<const char*>(pt.ambient_coords.data()),
                          acd * sizeof(Scalar));
        }
    }

    /// Load points from a binary file and rebuild the index.
    void load(const std::string& path) {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs) throw SerializationError("Cannot open file for reading: " + path);

        uint32_t cid, dim;
        size_t   n;
        ifs.read(reinterpret_cast<char*>(&cid), sizeof(cid));
        ifs.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        ifs.read(reinterpret_cast<char*>(&n), sizeof(n));

        if (cid != chart_id_ || dim != dim_) {
            throw DimensionMismatchError(
                "TangentSpaceIndex::load: file metadata mismatch");
        }

        uint32_t np;
        ifs.read(reinterpret_cast<char*>(&np), sizeof(np));

        std::vector<ManifoldPoint> points;
        points.reserve(np);

        for (uint32_t i = 0; i < np; ++i) {
            ManifoldPoint pt;
            ifs.read(reinterpret_cast<char*>(&pt.chart_id), sizeof(pt.chart_id));
            ifs.read(reinterpret_cast<char*>(&pt.global_id), sizeof(pt.global_id));
            ifs.read(reinterpret_cast<char*>(&pt.timestamp), sizeof(pt.timestamp));

            uint32_t lcd, acd;
            ifs.read(reinterpret_cast<char*>(&lcd), sizeof(lcd));
            ifs.read(reinterpret_cast<char*>(&acd), sizeof(acd));

            if (lcd > 0) {
                pt.local_coords.resize(lcd);
                ifs.read(reinterpret_cast<char*>(pt.local_coords.data()),
                         lcd * sizeof(Scalar));
            }
            if (acd > 0) {
                pt.ambient_coords.resize(acd);
                ifs.read(reinterpret_cast<char*>(pt.ambient_coords.data()),
                         acd * sizeof(Scalar));
            }
            points.push_back(std::move(pt));
        }

        build(points);
    }

private:
    uint32_t chart_id_;
    uint32_t dim_;
    size_t   max_leaf_size_;
    size_t   num_points_;
    std::unique_ptr<RTreeNode> root_;

    // ── Recursive Insert ──────────────────────────────────────────────────────

    void insert_recursive(RTreeNode* node, const ManifoldPoint& point, int depth) {
        node->expand_to_include(point.local_coords);

        if (node->is_leaf) {
            node->points.push_back(point);
            if (node->points.size() > max_leaf_size_) {
                split_node(node);
            }
            return;
        }

        // Choose child with minimum volume expansion (R-tree insertion heuristic)
        Scalar min_vol_inc = std::numeric_limits<Scalar>::infinity();
        size_t best_child = 0;
        for (size_t c = 0; c < node->children.size(); ++c) {
            Scalar old_vol = node->children[c]->volume();
            node->children[c]->expand_to_include(point.local_coords);
            Scalar new_vol = node->children[c]->volume();
            Scalar inc = new_vol - old_vol;
            if (inc < min_vol_inc) {
                min_vol_inc = inc;
                best_child = c;
            }
        }
        insert_recursive(node->children[best_child].get(), point, depth + 1);
    }

    // ── Split an overflowing leaf node ────────────────────────────────────────

    void split_node(RTreeNode* node) {
        if (node->points.size() <= 1) return;

        // Choose split axis: longest extent of the AABB
        Vector extent = node->max_corner - node->min_corner;
        size_t split_axis = 0;
        if (extent.size() > 0) {
            int max_idx = 0;
            for (int i = 1; i < extent.size(); ++i) {
                if (extent(i) > extent(max_idx)) max_idx = i;
            }
            split_axis = static_cast<size_t>(max_idx);
        }

        // Sort points along split axis
        std::sort(node->points.begin(), node->points.end(),
            [split_axis](const ManifoldPoint& a, const ManifoldPoint& b) {
                return a.local_coords(split_axis) < b.local_coords(split_axis);
            });

        size_t mid = node->points.size() / 2;

        auto left  = std::make_unique<RTreeNode>();
        auto right = std::make_unique<RTreeNode>();
        left->is_leaf = true;
        right->is_leaf = true;

        for (size_t i = 0; i < mid; ++i) {
            left->points.push_back(node->points[i]);
            left->expand_to_include(node->points[i].local_coords);
        }
        for (size_t i = mid; i < node->points.size(); ++i) {
            right->points.push_back(node->points[i]);
            right->expand_to_include(node->points[i].local_coords);
        }

        node->is_leaf = false;
        node->points.clear();
        node->children.push_back(std::move(left));
        node->children.push_back(std::move(right));
    }

    // ── Recursive k-NN with Branch-and-Bound ─────────────────────────────────

    void knn_recursive(const RTreeNode* node, const Vector& query,
                        std::priority_queue<NeighborResult>& heap,
                        size_t k) const
    {
        if (!node) return;

        if (node->is_leaf) {
            for (const auto& pt : node->points) {
                Scalar dist = (pt.local_coords - query).norm();
                NeighborResult nr;
                nr.point = pt;
                nr.geodesic_distance = dist;     // Local Euclidean (proxy)
                nr.euclidean_residual = dist;

                if (heap.size() < k) {
                    heap.push(nr);
                } else if (dist < heap.top().geodesic_distance) {
                    heap.pop();
                    heap.push(nr);
                }
            }
            return;
        }

        // Visit children ordered by min-distance (branch-and-bound pruning)
        struct ChildDist {
            const RTreeNode* node;
            Scalar dist;
            bool operator<(const ChildDist& o) const { return dist > o.dist; }
        };
        std::priority_queue<ChildDist> pq;
        for (const auto& child : node->children) {
            pq.push({child.get(), child->min_distance(query)});
        }

        while (!pq.empty()) {
            auto [child, dist] = pq.top();
            pq.pop();

            // Prune: if heap is full and this child's min distance exceeds
            // the current k-th nearest distance, skip it
            if (heap.size() >= k && dist > heap.top().geodesic_distance) {
                continue;
            }
            knn_recursive(child, query, heap, k);
        }
    }

    // ── Recursive Range Search ────────────────────────────────────────────────

    void range_search_recursive(const RTreeNode* node, const Vector& query,
                                Scalar radius_sq,
                                std::vector<ManifoldPoint>& results) const
    {
        if (!node) return;

        // Prune: AABB entirely outside the query ball
        Scalar d_sq = 0.0;
        int d = static_cast<int>(query.size());
        for (int i = 0; i < d; ++i) {
            Scalar delta = 0.0;
            if (query(i) < node->min_corner(i))
                delta = node->min_corner(i) - query(i);
            else if (query(i) > node->max_corner(i))
                delta = query(i) - node->max_corner(i);
            d_sq += delta * delta;
        }
        if (d_sq > radius_sq) return;

        if (node->is_leaf) {
            for (const auto& pt : node->points) {
                Scalar dist_sq = (pt.local_coords - query).squaredNorm();
                if (dist_sq <= radius_sq) {
                    results.push_back(pt);
                }
            }
            return;
        }

        for (const auto& child : node->children) {
            range_search_recursive(child.get(), query, radius_sq, results);
        }
    }

    // ── Bulk Build (STR-style recursive partitioning) ───────────────────────

    std::unique_ptr<RTreeNode> build_recursive(
            std::vector<ManifoldPoint>& points, int depth)
    {
        auto node = std::make_unique<RTreeNode>();
        node->node_id = static_cast<size_t>(depth);

        if (points.empty()) return node;

        // Compute AABB over all points
        for (const auto& pt : points) {
            node->expand_to_include(pt.local_coords);
        }

        // Base case: leaf node
        if (points.size() <= max_leaf_size_ || dim_ == 0) {
            node->is_leaf = true;
            node->points = std::move(points);
            return node;
        }

        // Recursive case: split along axis = depth % dim (STR heuristic)
        size_t axis = static_cast<size_t>(depth % static_cast<int>(dim_));
        std::sort(points.begin(), points.end(),
            [axis](const ManifoldPoint& a, const ManifoldPoint& b) {
                return a.local_coords(axis) < b.local_coords(axis);
            });

        size_t mid = points.size() / 2;
        std::vector<ManifoldPoint> left_points(points.begin(), points.begin() + mid);
        std::vector<ManifoldPoint> right_points(points.begin() + mid, points.end());

        node->is_leaf = false;
        node->children.push_back(build_recursive(left_points, depth + 1));
        node->children.push_back(build_recursive(right_points, depth + 1));

        return node;
    }

    // ── Collect all points from the tree ─────────────────────────────────────

    void collect_points(const RTreeNode* node,
                        std::vector<ManifoldPoint>& all) const
    {
        if (!node) return;
        if (node->is_leaf) {
            for (const auto& pt : node->points) {
                all.push_back(pt);
            }
            return;
        }
        for (const auto& child : node->children) {
            collect_points(child.get(), all);
        }
    }
};

} // namespace manifold
