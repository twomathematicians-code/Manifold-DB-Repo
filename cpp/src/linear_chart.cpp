// linear_chart.cpp
// LinearChart utilities, PCA factory, and affine chart construction helpers.
//
// The core LinearChart class (embed, project, jacobian, exponential_map, log_map)
// is fully inline in linear_chart.hpp since it's a trivial affine map.
// This file provides:
//   1. PCA-based LinearChart factory (automated chart construction from data)
//   2. Affine registration between two point sets (Procrustes alignment)
//   3. Basis quality analysis (condition number, coverage, residual)
//   4. Chart merging and splitting utilities

#include "manifold/linear_chart.hpp"

#include <Eigen/SVD>
#include <Eigen/QR>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

namespace manifold {

// ═══════════════════════════════════════════════════════════════════════════════
//  PCA-based LinearChart Factory
// ═══════════════════════════════════════════════════════════════════════════════

/// Create a LinearChart from ambient data via PCA decomposition.
///
/// Algorithm:
///   1. Compute the centroid μ = (1/n) Σ y_i
///   2. Centre the data: Y_c = Y - μ·1^T
///   3. SVD: Y_c = U Σ V^T
///   4. Take the top-d columns of U as the basis matrix B
///   5. Return LinearChart(id, B, μ)
///
/// The resulting chart satisfies:
///   - embed(x) = μ + B·x  (affine embedding)
///   - project(y) = B^T·(y - μ)  (orthogonal projection)
///   - Metric g = B^T·B = I_d  (if columns of B are orthonormal from SVD)
///
/// @param data   D × n matrix of ambient points (column-major)
/// @param id     Chart identifier
/// @param d      Target intrinsic dimension
/// @return       Shared pointer to the constructed LinearChart
std::shared_ptr<LinearChart> create_pca_chart(const Matrix& data,
                                                uint32_t id,
                                                uint32_t d) {
    if (data.rows() == 0 || data.cols() == 0) {
        throw DBException("create_pca_chart: empty data matrix");
    }

    uint32_t D = static_cast<uint32_t>(data.rows());
    uint32_t n = static_cast<uint32_t>(data.cols());

    // Clamp intrinsic dimension to valid range
    d = std::min(d, std::min(D, n));
    if (d == 0) d = 1;

    // Step 1: Centroid
    Vector centroid = data.rowwise().mean();

    // Step 2: Centre data
    Matrix centered = data.colwise() - centroid;

    // Step 3: SVD decomposition
    Eigen::JacobiSVD<Matrix> svd(centered, Eigen::ComputeThinU);
    Matrix U = svd.matrixU();
    Vector singular_values = svd.singularValues();

    // Step 4: Extract top-d basis vectors
    Matrix basis = U.leftCols(static_cast<int>(d));

    // Step 5: Construct chart
    return std::make_shared<LinearChart>(id, basis, centroid);
}

/// Create a LinearChart from a subset of ambient points, with variance
/// reporting for automatic dimension selection.
///
/// @param data        D × n matrix of ambient points
/// @param id          Chart identifier
/// @param d           Target intrinsic dimension
/// @param[out] explained_variance_ratio  Fraction of variance captured by top-d PCs
/// @return            Shared pointer to the constructed LinearChart
std::shared_ptr<LinearChart> create_pca_chart_with_variance(
        const Matrix& data,
        uint32_t id,
        uint32_t d,
        Scalar& explained_variance_ratio) {
    if (data.rows() == 0 || data.cols() == 0) {
        explained_variance_ratio = 0.0;
        throw DBException("create_pca_chart_with_variance: empty data matrix");
    }

    uint32_t D = static_cast<uint32_t>(data.rows());
    uint32_t n = static_cast<uint32_t>(data.cols());
    d = std::min(d, std::min(D, n));
    if (d == 0) d = 1;

    Vector centroid = data.rowwise().mean();
    Matrix centered = data.colwise() - centroid;

    Eigen::JacobiSVD<Matrix> svd(centered, Eigen::ComputeThinU);
    Vector singular_values = svd.singularValues();

    // Compute explained variance ratio
    Scalar total_var = singular_values.squaredNorm();
    Scalar explained_var = singular_values.head(static_cast<int>(d)).squaredNorm();
    explained_variance_ratio = (total_var > 1e-30) ? explained_var / total_var : 0.0;

    Matrix basis = svd.matrixU().leftCols(static_cast<int>(d));
    return std::make_shared<LinearChart>(id, basis, centroid);
}

/// Automatically determine the intrinsic dimension from data using
/// the eigenvalue gap heuristic.
///
/// Find d such that σ_{d+1}/σ_d is minimised (largest relative gap).
/// This is the "scree test" for dimensionality estimation.
///
/// @param data      D × n matrix of ambient points
/// @param max_dim   Maximum dimension to consider (default: min(D, n))
/// @return          Estimated intrinsic dimension
uint32_t estimate_intrinsic_dim(const Matrix& data, uint32_t max_dim) {
    if (data.rows() == 0 || data.cols() == 0) return 1;

    uint32_t D = static_cast<uint32_t>(data.rows());
    uint32_t n = static_cast<uint32_t>(data.cols());
    uint32_t max_d = std::min(max_dim, std::min(D, n));
    if (max_d < 2) return 1;

    Vector centroid = data.rowwise().mean();
    Matrix centered = data.colwise() - centroid;

    Eigen::JacobiSVD<Matrix> svd(centered, Eigen::ComputeThinU);
    Vector svs = svd.singularValues();

    // Find the largest ratio gap σ_{k+1}/σ_k
    Scalar max_gap = 0.0;
    uint32_t best_d = 1;

    for (uint32_t k = 1; k < max_d; ++k) {
        Scalar ratio = (svs(k) > 1e-15) ? svs(k) / svs(k - 1) : 0.0;
        if (ratio > max_gap) {
            max_gap = ratio;
            best_d = k;
        }
    }

    return best_d;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Chart Quality Analysis
// ═══════════════════════════════════════════════════════════════════════════════

/// Evaluate the quality of a linear chart's basis matrix.
///
/// Returns:
///   - condition_number: σ_max / σ_min of B (measures orthogonality quality)
///   - mean_residual:   average projection residual for a set of test points
///   - coverage_ratio:  fraction of test points with residual < threshold
///
/// @param chart      LinearChart to evaluate
/// @param test_pts   D × m matrix of ambient test points
/// @param threshold  Residual threshold for coverage (default: 0.1)
/// @return           {condition_number, mean_residual, coverage_ratio}
struct ChartQuality {
    Scalar condition_number;
    Scalar mean_residual;
    Scalar coverage_ratio;
};

ChartQuality evaluate_chart_quality(const LinearChart& chart,
                                    const Matrix& test_pts,
                                    Scalar threshold) {
    ChartQuality quality;
    quality.condition_number = 0.0;
    quality.mean_residual    = 0.0;
    quality.coverage_ratio   = 0.0;

    // Condition number from SVD of basis
    Eigen::JacobiSVD<Matrix> svd(chart.basis());
    Vector svs = svd.singularValues();
    if (svs.size() > 0) {
        quality.condition_number = svs(0) / std::max(svs(svs.size() - 1), 1e-15);
    }

    // Projection residuals
    if (test_pts.cols() == 0) return quality;

    Scalar total_residual = 0.0;
    int covered = 0;

    for (int i = 0; i < test_pts.cols(); ++i) {
        Vector pt = test_pts.col(i);
        Scalar residual = chart.projection_residual(pt);
        total_residual += residual;
        if (residual < threshold) ++covered;
    }

    quality.mean_residual  = total_residual / static_cast<Scalar>(test_pts.cols());
    quality.coverage_ratio = static_cast<Scalar>(covered) / static_cast<Scalar>(test_pts.cols());

    return quality;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Affine Registration (Procrustes Alignment)
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute the optimal affine transform (R, t) that maps points from chart A
/// to chart B in ambient space.
///
/// Given matched point pairs {(a_i, b_i)}, find R, t minimising:
///   Σ ||b_i - (R·a_i + t)||²
///
/// This is the orthogonal Procrustes problem:
///   1. Centre both point sets: A_c = A - μ_A,  B_c = B - μ_B
///   2. Compute cross-covariance: C = B_c · A_c^T
///   3. SVD: C = U Σ V^T
///   4. R = V·U^T  (with reflection correction if det(R) < 0)
///   5. t = μ_B - R·μ_A
///
/// @param pts_a  D × n matched points in chart A
/// @param pts_b  D × n matched points in chart B
/// @return       {R (D×D rotation), t (D translation)}
struct AffineTransform {
    Matrix R;    ///< Rotation / orthogonal matrix D × D
    Vector t;    ///< Translation vector D × 1
};

AffineTransform procrustes_alignment(const Matrix& pts_a, const Matrix& pts_b) {
    int D = pts_a.rows();
    int n = std::min(pts_a.cols(), pts_b.cols());

    AffineTransform result;
    result.R = Matrix::Identity(D, D);
    result.t = Vector::Zero(D);

    if (n < 1) return result;

    // Extract matching columns
    Matrix A = pts_a.leftCols(n);
    Matrix B = pts_b.leftCols(n);

    // Centre both sets
    Vector mu_a = A.rowwise().mean();
    Vector mu_b = B.rowwise().mean();
    Matrix A_c = A.colwise() - mu_a;
    Matrix B_c = B.colwise() - mu_b;

    // Cross-covariance
    Matrix C = B_c * A_c.transpose();

    // SVD
    Eigen::JacobiSVD<Matrix> svd(C, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Matrix U = svd.matrixU();
    Matrix V = svd.matrixV();

    // Rotation with reflection correction
    Matrix R = V * U.transpose();
    Scalar det = R.determinant();
    if (det < 0) {
        // Correct for reflection: flip sign of last column of V
        Matrix V_corrected = V;
        V_corrected.col(V.cols() - 1) *= -1.0;
        R = V_corrected * U.transpose();
    }

    result.R = R;
    result.t = mu_b - R * mu_a;

    return result;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Chart Merging Utilities
// ═══════════════════════════════════════════════════════════════════════════════

/// Merge two linear charts that represent overlapping regions of the manifold.
///
/// Strategy:
///   1. Collect local coordinates from both charts
///   2. Re-embed to ambient space using each chart's origin + basis
///   3. Perform PCA on the combined ambient points
///   4. Return a new LinearChart with the merged basis and centroid
///
/// @param chart_a  First chart
/// @param chart_b  Second chart
/// @param merged_id  ID for the new merged chart
/// @return          Shared pointer to the merged LinearChart
std::shared_ptr<LinearChart> merge_linear_charts(const LinearChart& chart_a,
                                                  const LinearChart& chart_b,
                                                  uint32_t merged_id) {
    // Collect sample points from both charts
    // Sample a grid in each chart's local coordinates
    int d_a = static_cast<int>(chart_a.intrinsic_dim());
    int d_b = static_cast<int>(chart_b.intrinsic_dim());
    int d   = std::max(d_a, d_b);

    // Collect origin points from both charts
    std::vector<Vector> ambient_pts;
    ambient_pts.push_back(chart_a.origin());
    ambient_pts.push_back(chart_b.origin());

    // Add some sample points along the basis directions
    for (int i = 0; i < d_a; ++i) {
        Vector v = Vector::Zero(d_a);
        v(i) = 1.0;
        ambient_pts.push_back(chart_a.embed(v));
        v(i) = -1.0;
        ambient_pts.push_back(chart_a.embed(v));
    }
    for (int i = 0; i < d_b; ++i) {
        Vector v = Vector::Zero(d_b);
        v(i) = 1.0;
        ambient_pts.push_back(chart_b.embed(v));
        v(i) = -1.0;
        ambient_pts.push_back(chart_b.embed(v));
    }

    // Build ambient matrix
    int D = static_cast<int>(chart_a.ambient_dim());
    Matrix data(D, static_cast<int>(ambient_pts.size()));
    for (size_t i = 0; i < ambient_pts.size(); ++i) {
        data.col(static_cast<int>(i)) = ambient_pts[i];
    }

    return create_pca_chart(data, merged_id, static_cast<uint32_t>(d));
}

/// Compute the overlap between two linear charts.
///
/// The overlap is estimated by projecting chart_a's origin onto chart_b and
/// checking if the projection residual is below a threshold.
/// A more robust estimate would sample multiple points.
///
/// @param chart_a    First chart
/// @param chart_b    Second chart
/// @param threshold  Max residual to consider overlapping (default: 1.0)
/// @return           Estimated overlap measure in [0, 1]
Scalar estimate_chart_overlap(const LinearChart& chart_a,
                              const LinearChart& chart_b,
                              Scalar threshold) {
    // Project origin_a onto chart_b and measure residual
    Scalar r_ab = chart_b.projection_residual(chart_a.origin());
    Scalar r_ba = chart_a.projection_residual(chart_b.origin());

    // Combine residuals: harmonic mean of complement
    Scalar overlap_a = (r_ab < threshold) ? 1.0 - r_ab / threshold : 0.0;
    Scalar overlap_b = (r_ba < threshold) ? 1.0 - r_ba / threshold : 0.0;

    return 0.5 * (overlap_a + overlap_b);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Coordinate Conversion Utilities
// ═══════════════════════════════════════════════════════════════════════════════

/// Convert ambient-space points to local coordinates on a linear chart.
/// Batch version for efficiency.
///
/// @param chart    Target LinearChart
/// @param ambient  D × n matrix of ambient points
/// @return         d × n matrix of local coordinates
Matrix batch_project(const LinearChart& chart, const Matrix& ambient) {
    Vector origin = chart.origin();
    Matrix basis_T = chart.basis().transpose();  // d × D

    // local = B^T * (ambient - origin)
    Matrix local = basis_T * ambient.colwise() - basis_T * origin;
    return local;
}

/// Convert local coordinates to ambient space on a linear chart.
/// Batch version.
///
/// @param chart    Source LinearChart
/// @param local    d × n matrix of local coordinates
/// @return         D × n matrix of ambient coordinates
Matrix batch_embed(const LinearChart& chart, const Matrix& local) {
    return chart.basis() * local + chart.origin() * Matrix::Ones(1, local.cols());
}

} // namespace manifold
