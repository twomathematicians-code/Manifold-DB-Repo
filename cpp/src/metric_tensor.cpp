// metric_tensor.cpp
// Metric tensor field utilities, RBF kernel helpers, and metric algebra operations.
//
// The MetricTensor class is fully inline in metric_tensor.hpp (RBF interpolation,
// Christoffel symbols, sectional curvature, serialization). This file provides:
//   1. RBF kernel factory functions for different kernel types
//   2. Metric tensor algebra (Weyl transform, conformal metric operations)
//   3. Metric learning from point neighbourhoods
//   4. Scalar curvature computation helpers
//   5. Metric field validation and consistency checks

#include "manifold/metric_tensor.hpp"
#include "manifold/chart.hpp"

#include <Eigen/SVD>
#include <Eigen/Cholesky>

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <vector>

namespace manifold {

// ═══════════════════════════════════════════════════════════════════════════════
//  RBF Kernel Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Gaussian RBF kernel: k(r) = exp(-r² / (2σ²))
/// This is the default kernel used by MetricTensor for interpolation.
///
/// @param r     Distance value
/// @param sigma Kernel bandwidth parameter
/// @return      Kernel weight in (0, 1]
Scalar gaussian_rbf(Scalar r, Scalar sigma) {
    Scalar ratio = r / sigma;
    return std::exp(-0.5 * ratio * ratio);
}

/// Thin-plate spline RBF kernel: k(r) = r² · log(r)
/// Suitable for interpolating smooth metric fields with less localisation.
///
/// @param r     Distance value
/// @param sigma Scaling parameter (used as k(r) = (r/σ)² · log(r/σ + ε))
/// @return      Kernel value
Scalar thin_plate_spline_rbf(Scalar r, Scalar sigma) {
    Scalar s = r / sigma;
    if (s < 1e-15) return 0.0;
    return s * s * std::log(s + 1e-15);
}

/// Inverse quadratic RBF kernel: k(r) = 1 / (1 + (r/σ)²)
/// Has broader support than Gaussian.
///
/// @param r     Distance value
/// @param sigma Kernel bandwidth parameter
/// @return      Kernel weight in (0, 1]
Scalar inverse_quadratic_rbf(Scalar r, Scalar sigma) {
    Scalar s = r / sigma;
    return Scalar(1.0) / (Scalar(1.0) + s * s);
}

/// Adaptive bandwidth estimation for RBF kernels.
/// Uses the distance to the k-th nearest anchor to set the bandwidth.
///
/// @param query     Query point
/// @param anchors   Vector of anchor locations
/// @param k         Number of neighbours for bandwidth estimation (default: 3)
/// @param min_sigma Minimum bandwidth clamp (default: 1e-6)
/// @return          Estimated bandwidth σ
Scalar estimate_adaptive_bandwidth(const Vector& query,
                                  const std::vector<Vector>& anchors,
                                  size_t k,
                                  Scalar min_sigma) {
    if (anchors.empty()) return 1.0;
    if (anchors.size() <= k) k = anchors.size();

    // Compute distances to all anchors
    std::vector<Scalar> dists;
    dists.reserve(anchors.size());
    for (const auto& a : anchors) {
        dists.push_back((query - a).norm());
    }

    // Partial sort to find k-th nearest
    std::partial_sort(dists.begin(),
                     dists.begin() + static_cast<int>(k),
                     dists.end());

    return std::max(dists[k - 1], min_sigma);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Metric Learning from Data
// ═══════════════════════════════════════════════════════════════════════════════

/// Learn a local metric tensor from a neighbourhood of ambient points.
///
/// Given a set of points near a chart centre, compute the induced metric:
///   1. Project points to local coordinates
///   2. Compute the Jacobian numerically (finite differences)
///   3. g_local = J^T · J
///
/// This produces the pullback of the ambient Euclidean metric onto the
/// local tangent space, which is the natural Riemannian metric for the chart.
///
/// @param chart           Chart for projection and Jacobian
/// @param centre_local    Centre point in local coordinates
/// @param neighbourhood   Neighbouring points in local coordinates
/// @param h               Finite difference step for Jacobian (default: 1e-5)
/// @return                Estimated metric tensor g ∈ R^{d×d} (SPD)
Matrix learn_local_metric_from_neighbours(const Chart& chart,
                                          const Vector& centre_local,
                                          const std::vector<Vector>& neighbourhood,
                                          Scalar h) {
    // Use the chart's induced metric: g = J^T · J
    // This is already computed by chart.compute_local_metric()
    // But we can refine it using neighbourhood statistics

    Matrix g_baseline = chart.compute_local_metric(centre_local);
    int d = static_cast<int>(chart.intrinsic_dim());

    if (neighbourhood.empty()) return g_baseline;

    // Compute covariance of local coordinates to estimate local anisotropy
    Vector mean = Vector::Zero(d);
    for (const auto& v : neighbourhood) mean += v;
    mean /= static_cast<Scalar>(neighbourhood.size());

    Matrix cov = Matrix::Zero(d, d);
    for (const auto& v : neighbourhood) {
        Vector diff = v - mean;
        cov += diff * diff.transpose();
    }
    cov /= static_cast<Scalar>(neighbourhood.size());

    // Blend the baseline metric with the data-driven covariance
    // g_refined = (1 - α) · g_baseline + α · (g_baseline^{1/2} · cov · g_baseline^{1/2})
    // This preserves the chart structure while adapting to data distribution
    Scalar alpha = 0.3;

    Eigen::LLT<Matrix> llt(g_baseline);
    if (llt.info() != Eigen::Success) return g_baseline;

    Matrix L = llt.matrixL();
    Matrix g_sqrt = L;  // g_baseline = L · L^T, so L is the "square root"

    Matrix adapted = g_sqrt * cov * g_sqrt.transpose();
    Matrix g_refined = (1.0 - alpha) * g_baseline + alpha * adapted;

    // Enforce SPD
    g_refined = 0.5 * (g_refined + g_refined.transpose());

    Eigen::LLT<Matrix> check(g_refined);
    if (check.info() != Eigen::Success) {
        return g_baseline;
    }

    return g_refined;
}

/// Learn a conformal factor field from local residual information.
///
/// A conformal metric has the form g(x) = e^{2φ(x)} · g_0(x) where
/// g_0 is a base metric and φ(x) is a scalar conformal factor.
///
/// This function estimates φ at a point from the ratio of observed
/// geodesic distances to Euclidean distances in the chart:
///   φ ≈ ½ log(d_geodesic / d_euclidean)
///
/// @param geodesic_dist   Observed geodesic distance
/// @param euclidean_dist  Euclidean distance in chart coordinates
/// @return                Conformal factor e^{φ}
Scalar learn_conformal_factor(Scalar geodesic_dist, Scalar euclidean_dist) {
    if (euclidean_dist < 1e-15 || geodesic_dist < 1e-15) return 1.0;

    Scalar ratio = geodesic_dist / euclidean_dist;
    // Clamp to reasonable range [0.01, 100]
    ratio = std::clamp(ratio, Scalar(0.01), Scalar(100.0));

    // φ = log(ratio)  →  conformal factor = e^{φ} = ratio
    return ratio;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Metric Tensor Algebra
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute the trace of the Ricci tensor (scalar curvature) from a metric tensor.
/// Uses the full formula: S = g^{ij} R_{ij}.
///
/// This delegates to MetricTensor::scalar_curvature() but also provides
/// a free function interface for use outside the class.
///
/// @param metric   MetricTensor object
/// @param coords   Local coordinates
/// @param h        Finite difference step
/// @return         Scalar curvature S
Scalar compute_scalar_curvature(const MetricTensor& metric,
                                const Vector& coords,
                                Scalar h) {
    return metric.scalar_curvature(coords, h);
}

/// Compute the metric determinant: det(g_{ij}).
/// The determinant of the metric is important for:
///   - Volume form: dV = sqrt(det g) dx^1 ... dx^d
///   - Divergence: div(X) = (1/√g) ∂_i (√g X^i)
///
/// @param metric   MetricTensor object
/// @param coords   Local coordinates
/// @return         Determinant of the metric tensor
Scalar metric_determinant(const MetricTensor& metric, const Vector& coords) {
    Matrix g = metric.evaluate(coords);
    return g.determinant();
}

/// Compute the volume element √(det g) at a point.
/// This is the factor needed for integration on the manifold.
///
/// @param metric   MetricTensor object
/// @param coords   Local coordinates
/// @return         Volume element √(det g)
Scalar volume_element(const MetricTensor& metric, const Vector& coords) {
    Scalar det = metric_determinant(metric, coords);
    return std::sqrt(std::abs(det));
}

/// Raise an index on a covariant tensor using the inverse metric.
///
/// Given v_i (covariant), compute v^j = g^{ji} v_i
///
/// @param metric     MetricTensor object
/// @param covariant  Covariant vector v_i
/// @param coords     Local coordinates where to evaluate the metric
/// @return           Contravariant vector v^j
Vector raise_index(const MetricTensor& metric,
                   const Vector& covariant,
                   const Vector& coords) {
    Matrix g_inv = metric.inverse(coords);
    return g_inv * covariant;
}

/// Lower an index on a contravariant tensor using the metric.
///
/// Given v^j (contravariant), compute v_i = g_{ij} v^j
///
/// @param metric         MetricTensor object
/// @param contravariant  Contravariant vector v^j
/// @param coords         Local coordinates where to evaluate the metric
/// @return               Covariant vector v_i
Vector lower_index(const MetricTensor& metric,
                   const Vector& contravariant,
                   const Vector& coords) {
    Matrix g = metric.evaluate(coords);
    return g * contravariant;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Metric Field Validation
// ═══════════════════════════════════════════════════════════════════════════════

/// Validate that a metric tensor field produces SPD matrices at a set of
/// sample points. Returns the number of non-SPD points found.
///
/// @param metric      MetricTensor to validate
/// @param sample_pts  Points in local coordinates to test
/// @param tol         Tolerance for symmetry check (default: 1e-8)
/// @return            Number of points where the metric is not SPD
int validate_metric_spd(const MetricTensor& metric,
                        const std::vector<Vector>& sample_pts,
                        Scalar tol) {
    int failures = 0;

    for (const auto& pt : sample_pts) {
        Matrix g = metric.evaluate(pt);

        // Check square
        if (g.rows() != g.cols()) { ++failures; continue; }

        // Check symmetry
        Scalar sym_err = (g - g.transpose()).cwiseAbs().maxCoeff();
        if (sym_err > tol) { ++failures; continue; }

        // Check positive-definiteness via Cholesky
        Eigen::LLT<Matrix> llt(g);
        if (llt.info() != Eigen::Success) { ++failures; continue; }
    }

    return failures;
}

/// Compute the maximum condition number of the metric across sample points.
/// A large condition number indicates near-degeneracy.
///
/// @param metric      MetricTensor to analyze
/// @param sample_pts  Sample points in local coordinates
/// @return            Maximum condition number across samples
Scalar max_condition_number(const MetricTensor& metric,
                            const std::vector<Vector>& sample_pts) {
    Scalar max_cond = 0.0;

    for (const auto& pt : sample_pts) {
        Matrix g = metric.evaluate(pt);
        Eigen::JacobiSVD<Matrix> svd(g);
        Vector svs = svd.singularValues();

        if (svs.size() > 0 && svs(svs.size() - 1) > 1e-15) {
            Scalar cond = svs(0) / svs(svs.size() - 1);
            max_cond = std::max(max_cond, cond);
        }
    }

    return max_cond;
}

/// Interpolate between two metric tensors using the matrix geometric mean.
///
/// The geometric mean of two SPD matrices A and B is defined as:
///   A #_t B = A^{1/2} (A^{-1/2} B A^{-1/2})^t A^{1/2}
///
/// For t=0 this gives A, for t=1 this gives B.
/// The geometric mean preserves the SPD property.
///
/// @param g_a    First SPD metric tensor
/// @param g_b    Second SPD metric tensor
/// @param t      Interpolation parameter in [0, 1]
/// @return       Interpolated SPD metric tensor
Matrix interpolate_metrics_geometric(const Matrix& g_a, const Matrix& g_b, Scalar t) {
    int d = g_a.rows();

    // For simplicity, use the log-Euclidean interpolation:
    // log(g_interp) = (1-t) log(g_a) + t log(g_b)
    // This approximates the geometric mean and is easier to compute.

    Eigen::SelfAdjointEigenSolver<Matrix> eig_a(g_a);
    Eigen::SelfAdjointEigenSolver<Matrix> eig_b(g_b);

    if (eig_a.info() != Eigen::Success || eig_b.info() != Eigen::Success) {
        // Fallback: linear interpolation with symmetrisation
        return 0.5 * ((1.0 - t) * g_a + t * g_b + ((1.0 - t) * g_b + t * g_a).transpose());
    }

    // Log of eigenvalues
    Vector log_eig_a = eig_a.eigenvalues().array().log().matrix();
    Vector log_eig_b = eig_b.eigenvalues().array().log().matrix();

    // Interpolate in log-space
    Vector log_eig_interp = (1.0 - t) * log_eig_a + t * log_eig_b;
    Vector eig_interp = log_eig_interp.array().exp().matrix();

    // Reconstruct using eigenvectors of g_a
    return eig_a.eigenvectors() * eig_interp.asDiagonal() * eig_a.eigenvectors().transpose();
}

} // namespace manifold
