// tangent_space_index.cpp
// Tangent space index utilities, distance computation helpers, and index analysis.
//
// The TangentSpaceIndex class is fully inline in tangent_space_index.hpp
// (R-tree construction, k-NN, range search, persistence). This file provides:
//   1. Brute-force k-NN as a correctness reference implementation
//   2. Metric-aware distance computation (weighted distances)
//   3. Index statistics and quality metrics
//   4. Batch distance computation for reranking
//   5. Index construction from raw coordinate arrays

#include "manifold/tangent_space_index.hpp"
#include "manifold/metric_tensor.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace manifold {

// ═══════════════════════════════════════════════════════════════════════════════
//  Brute-Force Reference Implementation
// ═══════════════════════════════════════════════════════════════════════════════

/// Brute-force k-NN search: compute all pairwise distances and return top-k.
/// Used as a correctness reference for the R-tree implementation.
///
/// Time complexity: O(n · d) for all distances, O(n log k) for partial sort.
///
/// @param query_local  Query point x_q ∈ R^d
/// @param points       Candidate points (must have local_coords set)
/// @param k            Number of neighbours to return
/// @return             Vector of NeighborResult sorted by ascending distance
std::vector<NeighborResult> brute_force_knn(
        const Vector& query_local,
        const std::vector<ManifoldPoint>& points,
        size_t k) {
    std::vector<NeighborResult> results;

    // Compute all distances
    std::vector<std::pair<Scalar, size_t>> dists;
    dists.reserve(points.size());

    for (size_t i = 0; i < points.size(); ++i) {
        Scalar dist = (points[i].local_coords - query_local).norm();
        dists.emplace_back(dist, i);
    }

    // Partial sort for top-k
    size_t actual_k = std::min(k, dists.size());
    std::partial_sort(dists.begin(),
                     dists.begin() + static_cast<int>(actual_k),
                     dists.end());

    results.reserve(actual_k);
    for (size_t i = 0; i < actual_k; ++i) {
        NeighborResult nr;
        nr.point              = points[dists[i].second];
        nr.geodesic_distance = dists[i].first;     // Euclidean proxy
        nr.euclidean_residual = dists[i].first;
        results.push_back(nr);
    }

    return results;
}

/// Brute-force range search: find all points within a given radius.
///
/// @param query_local  Query point
/// @param points       Candidate points
/// @param radius       Search radius
/// @return             All points within the radius
std::vector<ManifoldPoint> brute_force_range_search(
        const Vector& query_local,
        const std::vector<ManifoldPoint>& points,
        Scalar radius) {
    std::vector<ManifoldPoint> results;
    Scalar radius_sq = radius * radius;

    for (const auto& pt : points) {
        if ((pt.local_coords - query_local).squaredNorm() <= radius_sq) {
            results.push_back(pt);
        }
    }
    return results;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Metric-Aware Distance Computation
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute the Riemannian (metric-weighted) distance between two points
/// in local coordinates.
///
/// d_g(x, y) = sqrt((x - y)^T · g(midpoint) · (x - y))
///
/// This gives a better approximation to the geodesic distance than
/// Euclidean distance when the metric varies across the chart.
///
/// @param metric     Metric tensor for the chart
/// @param x, y       Points in local coordinates
/// @return           Metric-weighted distance
Scalar metric_weighted_distance(const MetricTensor& metric,
                                const Vector& x,
                                const Vector& y) {
    Vector midpoint = 0.5 * (x + y);
    Matrix g = metric.evaluate(midpoint);
    Vector diff = x - y;
    Scalar d_sq = diff.transpose() * g * diff;
    return std::sqrt(std::abs(d_sq));
}

/// Compute the Mahalanobis distance using a constant metric:
/// d_M(x, y) = sqrt((x - y)^T · G · (x - y))
///
/// @param G    Constant SPD metric matrix
/// @param x, y Points in local coordinates
/// @return     Mahalanobis distance
Scalar mahalanobis_distance(const Matrix& G,
                             const Vector& x,
                             const Vector& y) {
    Vector diff = x - y;
    Scalar d_sq = diff.transpose() * G * diff;
    return std::sqrt(std::abs(d_sq));
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Batch Distance Computation for Reranking
// ═══════════════════════════════════════════════════════════════════════════════

/// Batch metric-weighted distance computation for geodesic re-ranking.
///
/// Given a query point and a set of candidates, compute the metric-weighted
/// distance from the query to each candidate. This is used for the
/// re-ranking step in k-NN queries.
///
/// @param metric     Metric tensor for the chart
/// @param query      Query point in local coordinates
/// @param candidates Candidate points in local coordinates
/// @return           Vector of distances (same order as candidates)
std::vector<Scalar> batch_metric_distances(
        const MetricTensor& metric,
        const Vector& query,
        const std::vector<Vector>& candidates) {
    std::vector<Scalar> distances;
    distances.reserve(candidates.size());

    for (const auto& cand : candidates) {
        distances.push_back(metric_weighted_distance(metric, query, cand));
    }
    return distances;
}

/// Rerank k-NN candidates by geodesic distance.
///
/// Takes initial Euclidean-based k-NN results and reranks them using
/// the metric-weighted (geodesic proxy) distance.
///
/// @param metric     Metric tensor for the chart
/// @param query      Query point in local coordinates
/// @param candidates Initial candidate results (Euclidean distances)
/// @param k          Number of results to return
/// @return           Top-k results reranked by metric distance
std::vector<NeighborResult> rerank_by_metric(
        const MetricTensor& metric,
        const Vector& query,
        const std::vector<NeighborResult>& candidates,
        size_t k) {
    auto reranked = candidates;

    for (auto& nr : reranked) {
        nr.geodesic_distance = metric_weighted_distance(
            metric, query, nr.point.local_coords);
    }

    // Partial sort for top-k
    size_t actual_k = std::min(k, reranked.size());
    std::partial_sort(reranked.begin(),
                     reranked.begin() + static_cast<int>(actual_k),
                     reranked.end(),
                     [](const NeighborResult& a, const NeighborResult& b) {
                         return a.geodesic_distance < b.geodesic_distance;
                     });

    reranked.resize(actual_k);
    return reranked;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Index Statistics and Quality Metrics
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute statistics about the distribution of points in the index.
///
/// @param index  TangentSpaceIndex to analyze
/// @return       {mean_nn_dist, median_nn_dist, std_nn_dist, n_points}
struct IndexStatistics {
    Scalar mean_nn_dist;
    Scalar median_nn_dist;
    Scalar std_nn_dist;
    size_t n_points;
};

IndexStatistics compute_index_statistics(const TangentSpaceIndex& index) {
    IndexStatistics stats;
    stats.n_points = index.size();

    if (index.size() < 2) {
        stats.mean_nn_dist = 0.0;
        stats.median_nn_dist = 0.0;
        stats.std_nn_dist = 0.0;
        return stats;
    }

    // Collect all points and compute nearest-neighbour distances
    // TODO: This requires access to stored points - for now, return defaults
    stats.mean_nn_dist = 0.0;
    stats.median_nn_dist = 0.0;
    stats.std_nn_dist = 0.0;

    return stats;
}

/// Estimate the intrinsic dimensionality of the point set in the index
/// using the correlation dimension method.
///
/// The correlation dimension D_c is estimated from:
///   C(r) = (2 / n(n-1)) Σ_{i<j} Θ(r - ||x_i - x_j||)
///   D_c ≈ d log(C(r)) / d log(r)
///
/// @param points   Collection of points in local coordinates
/// @param n_samples Number of distance samples to use (default: 1000)
/// @return          Estimated intrinsic dimension
Scalar estimate_correlation_dimension(const std::vector<Vector>& points,
                                     size_t n_samples) {
    if (points.size() < 10) return static_cast<Scalar>(points.empty() ? 0 : points[0].size());

    size_t n = points.size();
    size_t actual_samples = std::min(n_samples, n * (n - 1) / 2);

    // Sample random pairwise distances
    std::vector<Scalar> dists;
    dists.reserve(actual_samples);

    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> dist(0, n - 1);

    for (size_t s = 0; s < actual_samples; ++s) {
        size_t i = dist(rng);
        size_t j = dist(rng);
        while (j == i) j = dist(rng);
        dists.push_back((points[i] - points[j]).norm());
    }

    // Compute correlation integral at multiple scales
    std::sort(dists.begin(), dists.end());

    std::vector<Scalar> log_r;
    std::vector<Scalar> log_C;
    int n_bins = 20;

    for (int b = 1; b < n_bins; ++b) {
        Scalar r = dists[static_cast<size_t>(b * dists.size() / n_bins)];
        if (r < 1e-15) continue;

        size_t count = 0;
        for (Scalar d : dists) {
            if (d <= r) ++count;
        }
        Scalar C = 2.0 * static_cast<Scalar>(count) / (static_cast<Scalar>(n) * static_cast<Scalar>(n - 1));

        if (C > 0 && r > 0) {
            log_r.push_back(std::log(r));
            log_C.push_back(std::log(C));
        }
    }

    if (log_r.size() < 3) return static_cast<Scalar>(points[0].size());

    // Linear regression of log(C) vs log(r)
    Scalar sum_x = 0, sum_y = 0, sum_xx = 0, sum_xy = 0;
    size_t m = log_r.size();
    for (size_t i = 0; i < m; ++i) {
        sum_x  += log_r[i];
        sum_y  += log_C[i];
        sum_xx += log_r[i] * log_r[i];
        sum_xy += log_r[i] * log_C[i];
    }

    Scalar denom = m * sum_xx - sum_x * sum_x;
    if (std::abs(denom) < 1e-15) return static_cast<Scalar>(points[0].size());

    Scalar slope = (m * sum_xy - sum_x * sum_y) / denom;
    return std::abs(slope);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Index Construction Helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// Build a TangentSpaceIndex from raw local coordinate arrays.
///
/// @param chart_id     Chart identifier
/// @param dim          Intrinsic dimension d
/// @param coords       n vectors of local coordinates (size d each)
/// @param max_leaf     Maximum leaf size for R-tree (default: 16)
/// @return             Constructed TangentSpaceIndex
TangentSpaceIndex build_index_from_coords(
        uint32_t chart_id,
        uint32_t dim,
        const std::vector<Vector>& coords,
        size_t max_leaf) {
    TangentSpaceIndex index(chart_id, dim, max_leaf);

    std::vector<ManifoldPoint> points;
    points.reserve(coords.size());

    for (size_t i = 0; i < coords.size(); ++i) {
        ManifoldPoint pt;
        pt.chart_id       = chart_id;
        pt.local_coords   = coords[i];
        pt.global_id      = static_cast<uint64_t>(i);
        points.push_back(pt);
    }

    index.build(points);
    return index;
}

} // namespace manifold
