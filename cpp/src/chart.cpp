// chart.cpp
// Chart base class utilities and NeuralChart implementation.
//
// The Chart abstract base and LinearChart/ParametricChart are fully inline in
// the headers. This file provides:
//   1. NeuralChart concrete implementation (ONNX-backed or fallback)
//   2. Free utility functions for chart operations
//   3. Metric validation and symmetry enforcement helpers

#include "manifold/chart.hpp"

#include <Eigen/QR>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace manifold {

// ═══════════════════════════════════════════════════════════════════════════════
//  NeuralChart  –  ONNX-based neural network embedding chart
// ═══════════════════════════════════════════════════════════════════════════════

// NeuralChart::Impl holds the ONNX Runtime session (opaque to header).
// When ONNX Runtime is not available, falls back to a simple MLP
// implemented with raw Eigen operations for prototyping.

struct NeuralChart::Impl {
    bool onnx_available = false;
    std::string model_path;

    // Fallback MLP parameters (used when ONNX is unavailable)
    struct Layer {
        Matrix W;      // Weight matrix
        Vector b;      // Bias vector
    };
    std::vector<Layer> layers;

    // Load a simple text-format model when ONNX is not available.
    // Format per layer: [rows cols] [weights row-major] [biases]
    void load_fallback(const std::string& path, uint32_t input_dim, uint32_t output_dim) {
        std::ifstream ifs(path);
        if (!ifs.good()) {
            // Initialise a random two-layer MLP: input_dim -> hidden -> output_dim
            uint32_t hidden = std::max(input_dim, output_dim);
            layers.resize(2);

            // Layer 0: input_dim x hidden
            layers[0].W = Matrix::Random(static_cast<int>(hidden),
                                        static_cast<int>(input_dim)) * 0.1;
            layers[0].b = Vector::Zero(static_cast<int>(hidden));

            // Layer 1: output_dim x hidden
            layers[1].W = Matrix::Random(static_cast<int>(output_dim),
                                        static_cast<int>(hidden)) * 0.1;
            layers[1].b = Vector::Zero(static_cast<int>(output_dim));

            return;
        }

        // Parse text format
        uint32_t num_layers = 0;
        ifs >> num_layers;
        layers.resize(num_layers);

        for (auto& layer : layers) {
            int rows = 0, cols = 0;
            ifs >> rows >> cols;
            layer.W.resize(rows, cols);
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    ifs >> layer.W(i, j);

            layer.b.resize(rows);
            for (int i = 0; i < rows; ++i)
                ifs >> layer.b(i);
        }
    }

    // Forward pass through the fallback MLP: x -> activation(W*x + b)
    // Uses ReLU activation for hidden layers, identity for output.
    Vector forward(const Vector& x) const {
        Vector h = x;
        for (size_t i = 0; i < layers.size(); ++i) {
            h = layers[i].W * h + layers[i].b;

            // ReLU for all layers except the last
            if (i + 1 < layers.size()) {
                for (int j = 0; j < h.size(); ++j) {
                    h(j) = std::max(Scalar(0.0), h(j));
                }
            }
        }
        return h;
    }
};

NeuralChart::NeuralChart(uint32_t id, uint32_t intrinsic_dim, uint32_t ambient_dim,
                         const std::string& model_path)
    : Chart(id, intrinsic_dim, ambient_dim)
    , model_path_(model_path)
    , impl_(std::make_unique<Impl>())
{
    impl_->model_path = model_path;

    // Attempt to initialise ONNX Runtime session
    // If ONNX Runtime headers are available at compile time, create session here.
    // Otherwise, fall back to simple Eigen-based MLP.
    impl_->onnx_available = false;  // TODO: #ifdef MANIFOLDDB_HAS_ONNX
    impl_->load_fallback(model_path, intrinsic_dim, ambient_dim);
}

NeuralChart::~NeuralChart() = default;

Vector NeuralChart::embed(const Vector& local_coords) const {
    // Sanity check: local_coords must match intrinsic_dim
    if (static_cast<uint32_t>(local_coords.size()) != intrinsic_dim_) {
        throw DimensionMismatchError(
            "NeuralChart::embed: input dimension " +
            std::to_string(local_coords.size()) +
            " does not match intrinsic_dim " + std::to_string(intrinsic_dim_));
    }

    if (impl_->onnx_available) {
        // ONNX Runtime forward pass would go here
        // Run the ONNX model and return the output tensor
        throw DBException("NeuralChart::embed: ONNX Runtime not linked");
    }

    // Fallback: Eigen MLP forward pass
    return impl_->forward(local_coords);
}

Vector NeuralChart::project(const Vector& ambient_coords) const {
    if (static_cast<uint32_t>(ambient_coords.size()) != ambient_dim_) {
        throw DimensionMismatchError(
            "NeuralChart::project: input dimension " +
            std::to_string(ambient_coords.size()) +
            " does not match ambient_dim " + std::to_string(ambient_dim_));
    }

    // Approximate inverse via Gauss-Newton optimisation:
    //   minimise ||embed(x) - ambient_coords||^2
    //   starting from x = J^T * (ambient - origin_proxy)
    //
    // This is a nonlinear least-squares problem solved iteratively.

    int d = static_cast<int>(intrinsic_dim_);
    int D = static_cast<int>(ambient_dim_);

    // Initial guess: try projecting via the Jacobian at origin
    Vector x0 = Vector::Zero(d);
    Vector f0 = embed(x0);
    Matrix J0 = jacobian(x0);
    Vector residual = ambient_coords - f0;

    // Least-squares initial guess: x = (J^T J)^{-1} J^T r
    Eigen::ColPivHouseholderQR<Matrix> qr(J0.transpose() * J0);
    Vector x = qr.solve(J0.transpose() * residual);

    // Gauss-Newton iterations
    for (int iter = 0; iter < 50; ++iter) {
        Vector f_x = embed(x);
        Vector r   = ambient_coords - f_x;
        Scalar r_norm = r.norm();

        if (r_norm < 1e-10) break;

        Matrix J = jacobian(x);
        Matrix JtJ = J.transpose() * J;

        // Regularise if near-singular
        JtJ += 1e-6 * Matrix::Identity(d, d);

        Eigen::ColPivHouseholderQR<Matrix> gn_qr(JtJ);
        Vector dx = gn_qr.solve(J.transpose() * r);

        // Backtracking line search
        Scalar alpha = 1.0;
        Scalar r2_current = r_norm * r_norm;

        for (int ls = 0; ls < 8; ++ls) {
            Vector x_trial = x + alpha * dx;
            Vector r_trial = ambient_coords - embed(x_trial);
            Scalar r2_trial = r_trial.squaredNorm();

            if (r2_trial < r2_current) break;
            alpha *= 0.5;
        }

        x += alpha * dx;

        if (dx.norm() < 1e-10) break;
    }

    return x;
}

Matrix NeuralChart::jacobian(const Vector& local_coords) const {
    // Numerical Jacobian via central finite differences.
    //
    // J_ij = (embed(x + h*e_j)_i - embed(x - h*e_j)_i) / (2h)
    //
    // This avoids needing automatic differentiation or ONNX gradient support.

    int d = static_cast<int>(intrinsic_dim_);
    int D = static_cast<int>(ambient_dim_);
    Scalar h = 1e-5;

    Matrix J(D, d);

    for (int j = 0; j < d; ++j) {
        Vector xp = local_coords;
        Vector xm = local_coords;
        xp(j) += h;
        xm(j) -= h;

        Vector fp = embed(xp);
        Vector fm = embed(xm);

        J.col(j) = (fp - fm) / (2.0 * h);
    }

    return J;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Free Utility Functions for Chart Operations
// ═══════════════════════════════════════════════════════════════════════════════

/// Orthonormalise the columns of a matrix using modified Gram-Schmidt.
/// Ensures B^T B = I_d (identity metric).
///
/// @param B  Input matrix with column vectors to orthonormalise
/// @return   Orthonormalised matrix (same dimensions as B)
Matrix orthonormalise_basis(const Matrix& B) {
    int n_cols = B.cols();
    if (n_cols == 0) return B;

    Matrix Q = B;
    for (int j = 0; j < n_cols; ++j) {
        // Subtract projections onto previous columns
        for (int i = 0; i < j; ++i) {
            Scalar proj = Q.col(i).dot(Q.col(j));
            Q.col(j) -= proj * Q.col(i);
        }
        // Normalise
        Scalar norm = Q.col(j).norm();
        if (norm < 1e-12) {
            // Degenerate column: set to random unit vector orthogonal to previous
            Q.col(j) = Vector::Random(B.rows()).normalized();
            for (int i = 0; i < j; ++i) {
                Scalar proj = Q.col(i).dot(Q.col(j));
                Q.col(j) -= proj * Q.col(i);
            }
            Q.col(j).normalize();
        } else {
            Q.col(j) /= norm;
        }
    }
    return Q;
}

/// Verify that a matrix is symmetric positive-definite (SPD).
/// Checks:
///   1. Square matrix
///   2. Symmetric within tolerance
///   3. All eigenvalues positive
///
/// @param M      Matrix to check
/// @param tol    Symmetry tolerance (default: 1e-10)
/// @return       True if M is SPD
bool is_spd(const Matrix& M, Scalar tol) {
    if (M.rows() != M.cols()) return false;

    // Check symmetry: M - M^T ≈ 0
    Scalar sym_err = (M - M.transpose()).cwiseAbs().maxCoeff();
    if (sym_err > tol) return false;

    // Check positive eigenvalues via Cholesky decomposition
    Eigen::LLT<Matrix> llt(M);
    return llt.info() == Eigen::Success;
}

/// Project a matrix onto the SPD cone via eigendecomposition.
/// Clamps negative eigenvalues to a small positive value.
///
/// g_spd = V * diag(max(λ_i, ε)) * V^T
///
/// @param g     Input matrix (ideally symmetric)
/// @param eps   Minimum eigenvalue clamp (default: 1e-8)
/// @return      Nearest SPD matrix
Matrix project_to_spd(const Matrix& g, Scalar eps) {
    // Symmetrise first
    Matrix sym = 0.5 * (g + g.transpose());

    Eigen::SelfAdjointEigenSolver<Matrix> eigensolver(sym);
    if (eigensolver.info() != Eigen::Success) {
        // Fallback: return identity
        return Matrix::Identity(g.rows(), g.cols());
    }

    Vector eigenvalues = eigensolver.eigenvalues();
    Matrix eigenvectors = eigensolver.eigenvectors();

    // Clamp eigenvalues to [eps, inf)
    for (int i = 0; i < eigenvalues.size(); ++i) {
        eigenvalues(i) = std::max(eigenvalues(i), eps);
    }

    return eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
}

/// Compute the geodesic normal coordinate distance between two ManifoldPoints
/// on the same chart using the log-map approximation.
///
/// d ≈ ||log_p(q)||_g = sqrt(v^T g(p) v) where v = log_p(q)
///
/// @param chart  Chart object
/// @param p, q   ManifoldPoints on the same chart
/// @return       Approximate geodesic distance
Scalar approximate_geodesic_distance(const Chart& chart,
                                     const ManifoldPoint& p,
                                     const ManifoldPoint& q) {
    Vector v = chart.log_map(p, q);
    Matrix g = chart.compute_local_metric(p.local_coords);
    Scalar d_sq = v.transpose() * g * v;
    return std::sqrt(std::abs(d_sq));
}

/// Compute the projection residual for a point onto a chart:
///   r = ||y - φ(φ⁻¹(y))||²
///
/// This measures how well the chart represents the ambient point.
/// A residual of zero means the point lies exactly on the chart's affine plane.
///
/// @param chart           Target chart
/// @param ambient_coords  Ambient point y ∈ R^D
/// @return                Projection residual (scalar)
Scalar compute_chart_residual(const Chart& chart, const Vector& ambient_coords) {
    try {
        Vector local = chart.project(ambient_coords);
        Vector reembedded = chart.embed(local);
        return (ambient_coords - reembedded).norm();
    } catch (const DBException&) {
        return std::numeric_limits<Scalar>::infinity();
    }
}

/// Sample points along a geodesic path for visualisation or interpolation.
///
/// Given a base point and tangent direction, sample n_points evenly spaced
/// points along the exponential map trajectory:
///   γ(t_i) = exp_p(t_i * v / ||v||),  t_i = i / (n-1)
///
/// @param chart        Chart to evaluate on
/// @param base         Base manifold point
/// @param tangent_vec  Tangent direction vector
/// @param n_points     Number of sample points (default: 20)
/// @return             Vector of sampled ManifoldPoints
std::vector<ManifoldPoint> sample_geodesic_arc(const Chart& chart,
                                                const ManifoldPoint& base,
                                                const Vector& tangent_vec,
                                                int n_points) {
    std::vector<ManifoldPoint> arc;
    if (n_points <= 0) return arc;
    if (n_points == 1) {
        arc.push_back(base);
        return arc;
    }

    Scalar v_norm = tangent_vec.norm();
    if (v_norm < 1e-30) {
        for (int i = 0; i < n_points; ++i) arc.push_back(base);
        return arc;
    }

    for (int i = 0; i < n_points; ++i) {
        Scalar t = static_cast<Scalar>(i) / static_cast<Scalar>(n_points - 1);
        Vector v_scaled = tangent_vec * t;
        ManifoldPoint pt = chart.exponential_map(base, v_scaled, 1e-3, 500);
        arc.push_back(pt);
    }
    return arc;
}

/// Compute the mean curvature vector H at a point on the chart.
/// For an embedded submanifold, H = g^{ij} Γ^i_{jk} e_k represents
/// the trace of the second fundamental form.
///
/// @param chart    Chart object
/// @param point    ManifoldPoint with local_coords
/// @param h        Finite difference step (default: 1e-5)
/// @return         Mean curvature vector H ∈ R^d
Vector mean_curvature_vector(const Chart& chart, const ManifoldPoint& point, Scalar h) {
    Tensor3D G2 = chart.christoffel_second_kind(point.local_coords, h);
    Matrix ginv = chart.compute_inverse_metric(point.local_coords);

    int d = static_cast<int>(chart.intrinsic_dim());
    Vector H(d);
    H.setZero();

    // H^k = g^{ij} Γ^i_{jk}  (trace of Christoffel symbols)
    for (int k = 0; k < d; ++k) {
        Scalar sum = 0.0;
        for (int i = 0; i < d; ++i) {
            for (int j = 0; j < d; ++j) {
                sum += ginv(i, j) * G2(i, j, k);
            }
        }
        H(k) = sum;
    }
    return H;
}

} // namespace manifold
