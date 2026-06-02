#pragma once

/// @file chart.hpp
/// @brief Abstract Chart base class and concrete chart implementations.
///
/// A chart (U, φ) provides a local coordinate system on the manifold:
///   φ : U ⊂ R^d  →  M ⊂ R^D
///
/// Each chart exposes:
///   - embed(x):          Forward map  x ↦ φ(x)  (local → ambient)
///   - project(y):        Inverse map  y ↦ φ⁻¹(y)  (ambient → local, best-effort)
///   - jacobian(x):       Pushforward dφ = J(x)   (D × d matrix)
///   - compute_local_metric(x): Induced Riemannian metric g_ij = J^T J
///   - exponential_map(p, v): Geodesic in direction of tangent vector v at base p
///   - log_map(p, q):     Inverse of exponential: tangent vector from p to q
///
/// Concrete implementations:
///   - LinearChart  (linear_chart.hpp): Affine PCA patch
///   - NeuralChart   (ONNX-based):      Neural network embedding
///   - ParametricChart:                   User-supplied callbacks

#include "manifold_types.hpp"

#include <Eigen/SVD>
#include <Eigen/QR>
#include <Eigen/Geometry>

#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace manifold {

// ═══════════════════════════════════════════════════════════════════════════════
//  Chart  –  Abstract Base Class
// ═══════════════════════════════════════════════════════════════════════════════

/// Abstract base class for a coordinate chart (U, φ) on the manifold.
///
/// Mathematical foundation:
///   Let M ⊂ R^D be a d-dimensional submanifold. A chart is a homeomorphism
///   φ : U → φ(U) ⊂ M  where U is an open set of R^d.
///
///   The pushforward (Jacobian) dφ maps tangent vectors from R^d to R^D.
///   The pullback metric g_ij = (dφ)^T (dφ) gives the induced Riemannian
///   metric in local coordinates, making (U, g) a Riemannian chart.
class Chart {
public:
    /// Construct a chart with identity metadata.
    /// @param id            Unique chart identifier within the atlas
    /// @param intrinsic_dim Dimension d of the chart's local coordinate space R^d
    /// @param ambient_dim   Dimension D of the embedding space R^D
    Chart(uint32_t id, uint32_t intrinsic_dim, uint32_t ambient_dim)
        : id_(id)
        , intrinsic_dim_(intrinsic_dim)
        , ambient_dim_(ambient_dim)
    {}

    virtual ~Chart() = default;

    // ── Core Geometry (pure virtual) ──────────────────────────────────────────

    /// Embedding φ : U ⊂ R^d → M ⊂ R^D
    /// Maps local coordinates to ambient (embedding) coordinates.
    /// @param local_coords  x ∈ R^d
    /// @return  y = φ(x) ∈ R^D
    [[nodiscard]] virtual Vector embed(const Vector& local_coords) const = 0;

    /// Projection φ⁻¹ : M ⊂ R^D → U ⊂ R^d  (best-effort right inverse)
    /// Maps ambient coordinates back to local chart coordinates.
    /// For points outside the chart's image, returns the nearest projection.
    /// @param ambient_coords  y ∈ R^D
    /// @return  x ≈ φ⁻¹(y) ∈ R^d
    [[nodiscard]] virtual Vector project(const Vector& ambient_coords) const = 0;

    /// Pushforward (Jacobian)  dφ/dx : R^d → R^D
    /// Returns the D × d matrix of partial derivatives J_ij = ∂y^i/∂x^j
    /// @param local_coords  x ∈ R^d
    /// @return  J(x) ∈ R^{D × d}
    [[nodiscard]] virtual Matrix jacobian(const Vector& local_coords) const = 0;

    // ── Induced Riemannian Metric ─────────────────────────────────────────────

    /// Compute the local Riemannian metric tensor g_ij(x) at a chart point.
    ///
    /// For an embedded submanifold, the pullback of the ambient Euclidean metric is:
    ///   g_ij(x) = J(x)^T J(x)
    ///
    /// This gives a d × d symmetric positive-definite (SPD) matrix.
    /// @param local_coords  x ∈ R^d
    /// @return  g(x) ∈ R^{d × d}, SPD
    [[nodiscard]] virtual Matrix compute_local_metric(const Vector& local_coords) const {
        Matrix J = jacobian(local_coords);
        return J.transpose() * J;
    }

    /// Compute the inverse metric g^{ij}(x).
    /// Uses Cholesky decomposition for SPD matrices, with SVD pseudo-inverse
    /// fallback for numerically near-singular cases.
    /// @param local_coords  x ∈ R^d
    /// @return  g^{-1}(x) ∈ R^{d × d}
    [[nodiscard]] Matrix compute_inverse_metric(const Vector& local_coords) const {
        Matrix g = compute_local_metric(local_coords);
        Eigen::LLT<Matrix> llt(g);
        if (llt.info() == Eigen::Success) {
            Matrix L = llt.matrixL();
            return L.inverse().transpose() * L.inverse();
        }
        // Fallback: pseudo-inverse via SVD
        Eigen::JacobiSVD<Matrix> svd(g,
            Eigen::ComputeFullU | Eigen::ComputeFullV);
        Vector svs = svd.singularValues();
        Scalar thresh = static_cast<Scalar>(intrinsic_dim_) * svs(0) * 1e-12;
        for (int i = 0; i < svs.size(); ++i) {
            svs(i) = (svs(i) > thresh) ? Scalar(1.0) / svs(i) : Scalar(0.0);
        }
        return svd.matrixV() * svs.asDiagonal() * svd.matrixU().transpose();
    }

    // ── Christoffel Symbols (numerical via finite differences) ──────────────

    /// Compute Christoffel symbols of the FIRST kind Γ_{ijk} via central differences.
    ///
    /// Γ_{ijk} = ½ (∂g_{ij}/∂x^k + ∂g_{ik}/∂x^j − ∂g_{jk}/∂x^i)
    ///
    /// @param local_coords  Evaluation point x ∈ R^d
    /// @param h             Finite-difference step size (default: 1e-5)
    /// @return  Tensor3D of shape (d, d, d): Γ(i,j,k) = Γ_{ijk}
    [[nodiscard]] Tensor3D christoffel_first_kind(
            const Vector& local_coords,
            Scalar h = 1e-5) const
    {
        const int d = static_cast<int>(intrinsic_dim_);
        Tensor3D Gamma(d, d, d);
        Gamma.setZero();

        // Precompute ∂g/∂x^a for all directions a
        std::vector<Matrix> dg_dx(d);
        for (int a = 0; a < d; ++a) {
            Vector xp = local_coords, xm = local_coords;
            xp(a) += h;
            xm(a) -= h;
            dg_dx[a] = (compute_local_metric(xp) - compute_local_metric(xm)) / (2.0 * h);
        }

        // Γ_{ijk} = ½ (∂g_{ij}/∂x^k + ∂g_{ik}/∂x^j − ∂g_{jk}/∂x^i)
        for (int i = 0; i < d; ++i) {
            for (int j = 0; j < d; ++j) {
                for (int k = 0; k < d; ++k) {
                    Gamma(i, j, k) = 0.5 * (
                        dg_dx[k](i, j) +
                        dg_dx[j](i, k) -
                        dg_dx[i](j, k)
                    );
                }
            }
        }
        return Gamma;
    }

    /// Compute Christoffel symbols of the SECOND kind Γ^i_{jk}.
    ///
    /// Γ^i_{jk} = g^{il} Γ_{ljk}
    ///
    /// These appear directly in the geodesic equation:
    ///   d²x^i/dt² + Γ^i_{jk} dx^j/dt dx^k/dt = 0
    ///
    /// @param local_coords  Evaluation point x ∈ R^d
    /// @param h             Finite-difference step size
    /// @return  Tensor3D of shape (d, d, d): result(i,j,k) = Γ^i_{jk}
    [[nodiscard]] Tensor3D christoffel_second_kind(
            const Vector& local_coords,
            Scalar h = 1e-5) const
    {
        const int d = static_cast<int>(intrinsic_dim_);
        Tensor3D Gamma1 = christoffel_first_kind(local_coords, h);
        Matrix ginv     = compute_inverse_metric(local_coords);

        Tensor3D Gamma2(d, d, d);
        Gamma2.setZero();

        for (int i = 0; i < d; ++i) {
            for (int j = 0; j < d; ++j) {
                for (int k = 0; k < d; ++k) {
                    Scalar sum = 0.0;
                    for (int l = 0; l < d; ++l) {
                        sum += ginv(i, l) * Gamma1(l, j, k);
                    }
                    Gamma2(i, j, k) = sum;
                }
            }
        }
        return Gamma2;
    }

    // ── Sectional Curvature ─────────────────────────────────────────────────

    /// Compute sectional curvature K(u, v) at a point for a 2-plane spanned by
    /// linearly independent tangent vectors u, v ∈ T_pM.
    ///
    /// K(u, v) = R(u, v, v, u) / (g(u,u)·g(v,v) − g(u,v)²)
    ///
    /// where R is the Riemann curvature tensor:
    ///   R^i_{jkl} = ∂_k Γ^i_{jl} − ∂_l Γ^i_{jk}
    ///              + Γ^i_{mk} Γ^m_{jl} − Γ^i_{ml} Γ^m_{jk}
    ///
    /// @param local_coords  Chart point x
    /// @param u, v          Tangent vectors spanning the 2-plane
    /// @param h             Finite-difference step for Christoffel derivatives
    /// @return  Sectional curvature K
    [[nodiscard]] Scalar sectional_curvature(
            const Vector& local_coords,
            const Vector& u,
            const Vector& v,
            Scalar h = 1e-5) const
    {
        const int d = static_cast<int>(intrinsic_dim_);
        Tensor3D G2 = christoffel_second_kind(local_coords, h);

        // Contract Riemann tensor: R(u,v,v,u)
        auto R_contract = [&](const Vector& uu, const Vector& vv) -> Scalar {
            // Numerical derivatives of Γ^i_{jk}
            std::vector<Tensor3D> dGamma(d);
            for (int a = 0; a < d; ++a) {
                Vector xp = local_coords, xm = local_coords;
                xp(a) += h;
                xm(a) -= h;
                Tensor3D Gp = christoffel_second_kind(xp, h);
                Tensor3D Gm = christoffel_second_kind(xm, h);
                dGamma[a] = Tensor3D(d, d, d);
                for (int i = 0; i < d; ++i)
                    for (int j = 0; j < d; ++j)
                        for (int k = 0; k < d; ++k)
                            dGamma[a](i, j, k) = (Gp(i, j, k) - Gm(i, j, k)) / (2.0 * h);
            }

            // R^i_{jkl} u^j v^k v^l u^i
            Scalar result = 0.0;
            for (int i = 0; i < d; ++i) {
                for (int j = 0; j < d; ++j) {
                    for (int k = 0; k < d; ++k) {
                        for (int l = 0; l < d; ++l) {
                            Scalar Rijkl = dGamma[k](i, j, l) - dGamma[l](i, j, k);
                            for (int m = 0; m < d; ++m) {
                                Rijkl += G2(i, m, k) * G2(m, j, l)
                                       - G2(i, m, l) * G2(m, j, k);
                            }
                            result += Rijkl * uu(i) * vv(j) * vv(k) * uu(l);
                        }
                    }
                }
            }
            return result;
        };

        Scalar numerator   = R_contract(u, v);
        Scalar denom_inner = u.dot(u) * v.dot(v) - std::pow(u.dot(v), 2);
        if (std::abs(denom_inner) < 1e-30) return 0.0;
        return numerator / denom_inner;
    }

    // ── Exponential Map ─────────────────────────────────────────────────────

    /// Compute the exponential map exp_p(v) at base point p in direction v.
    ///
    /// The exponential map follows the geodesic γ(t) with:
    ///   γ(0) = p,  γ'(0) = v / ||v|| · ||v||
    ///
    /// until the arc length equals ||v||_g (the Riemannian norm of v).
    /// Uses RK4 integration of the geodesic equation.
    ///
    /// @param base       Base manifold point p
    /// @param tangent_vec Tangent vector v ∈ T_pM
    /// @param step_size  Integration step (default: 1e-3)
    /// @param max_steps  Maximum integration steps (default: 1000)
    /// @return  ManifoldPoint at exp_p(v)
    [[nodiscard]] virtual ManifoldPoint exponential_map(
            const ManifoldPoint& base,
            const Vector& tangent_vec,
            Scalar step_size  = 1e-3,
            int    max_steps  = 1000) const
    {
        Vector pos = base.local_coords;
        Vector vel = tangent_vec;

        // Normalise velocity so geodesic length equals |tangent_vec|
        Scalar v_norm = vel.norm();
        if (v_norm < 1e-30) return base;

        int d = static_cast<int>(intrinsic_dim_);

        for (int step = 0; step < max_steps; ++step) {
            // RK4 geodesic step: d²x/dt² = -Γ(x)(dx/dt, dx/dt)
            auto accel = [&](const Vector& p, const Vector& v_vec) -> Vector {
                Tensor3D G = christoffel_second_kind(p, 1e-5);
                Vector a(d);
                a.setZero();
                for (int i = 0; i < d; ++i)
                    for (int j = 0; j < d; ++j)
                        for (int k = 0; k < d; ++k)
                            a(i) -= G(i, j, k) * v_vec(j) * v_vec(k);
                return a;
            };

            Vector k1v = accel(pos, vel) * step_size;
            Vector k1x = vel * step_size;

            Vector k2v = accel(pos + 0.5 * k1x, vel + 0.5 * k1v) * step_size;
            Vector k2x = (vel + 0.5 * k1v) * step_size;

            Vector k3v = accel(pos + 0.5 * k2x, vel + 0.5 * k2v) * step_size;
            Vector k3x = (vel + 0.5 * k2v) * step_size;

            Vector k4v = accel(pos + k3x, vel + k3v) * step_size;
            Vector k4x = (vel + k3v) * step_size;

            pos = pos + (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0;
            vel = vel + (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0;

            // Stop when arc length ≈ ||tangent_vec||
            Scalar dist = (pos - base.local_coords).norm();
            if (dist >= v_norm * 0.999) break;
        }

        ManifoldPoint result;
        result.chart_id       = id_;
        result.local_coords   = pos;
        result.ambient_coords = embed(pos);
        return result;
    }

    // ── Logarithmic Map (inverse exponential) ───────────────────────────────

    /// Compute the logarithmic map log_p(q): the tangent vector v ∈ T_pM
    /// such that exp_p(v) = q.
    ///
    /// Uses Newton's method on the residual exp_p(v) − q, with finite-difference
    /// Jacobian approximation.
    ///
    /// @param base     Base point p
    /// @param target   Target point q
    /// @param tolerance Convergence tolerance (default: 1e-8)
    /// @param max_iterations Maximum Newton iterations (default: 100)
    /// @return  Tangent vector v = log_p(q)
    [[nodiscard]] virtual Vector log_map(
            const ManifoldPoint& base,
            const ManifoldPoint& target,
            Scalar tolerance    = 1e-8,
            int    max_iterations = 100) const
    {
        Vector tgt = target.local_coords;
        int d = static_cast<int>(intrinsic_dim_);

        // Initial guess: straight line in local coordinates
        Vector v = tgt - base.local_coords;

        for (int iter = 0; iter < max_iterations; ++iter) {
            ManifoldPoint reached = exponential_map(base, v, 1e-3, 2000);
            Vector residual = reached.local_coords - tgt;

            if (residual.norm() < tolerance) break;

            // Approximate Jacobian of exp via finite differences
            Matrix J(d, d);
            Scalar eps = 1e-7;
            for (int col = 0; col < d; ++col) {
                Vector vp = v;
                vp(col) += eps;
                ManifoldPoint ep = exponential_map(base, vp, 1e-3, 2000);
                J.col(col) = (ep.local_coords - reached.local_coords) / eps;
            }

            // Solve  J δv = −residual
            Eigen::ColPivHouseholderQR<Matrix> qr(J);
            Vector dv = qr.solve(-residual);
            v += dv;

            if (dv.norm() < tolerance) break;
        }
        return v;
    }

    // ── Boundary Check ───────────────────────────────────────────────────────

    /// Test whether a point in local coordinates lies within the chart's domain U.
    /// Default: unbounded (always true). Subclasses may enforce finite domains.
    /// @param local_coords  x ∈ R^d
    /// @return  true if x is within the chart's valid domain
    [[nodiscard]] virtual bool contains(const Vector& /*local_coords*/) const {
        return true;
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    [[nodiscard]] uint32_t id() const            { return id_; }
    [[nodiscard]] uint32_t intrinsic_dim() const { return intrinsic_dim_; }
    [[nodiscard]] uint32_t ambient_dim() const   { return ambient_dim_; }

    /// Runtime chart type identification.
    [[nodiscard]] virtual ChartType type() const = 0;

protected:
    uint32_t id_;              ///< Unique chart identifier
    uint32_t intrinsic_dim_;   ///< d: dimension of local coordinate space
    uint32_t ambient_dim_;     ///< D: dimension of embedding space
};

// ═══════════════════════════════════════════════════════════════════════════════
//  NeuralChart  –  ONNX-based neural network embedding chart
// ═══════════════════════════════════════════════════════════════════════════════

/// Forward declaration of a neural network chart backed by an ONNX runtime model.
///
/// This chart learns a non-linear embedding φ_θ : R^d → R^D parameterised by
/// neural network weights θ, enabling representation of curved manifolds
/// that cannot be captured by affine (LinearChart) patches.
///
/// Mathematical model:
///   φ_θ(x) = f_L ∘ f_{L-1} ∘ … ∘ f_1(x) + b
///
/// where each f_l is a learned affine + nonlinearity layer.
///
/// Implementation notes:
///   - Requires ONNX Runtime at link time (optional dependency).
///   - Jacobian computed via automatic differentiation or numerical FD.
///   - The ONNX model file path and session are managed internally.
class NeuralChart : public Chart {
public:
    /// Construct a NeuralChart from an ONNX model file.
    /// @param id            Chart identifier
    /// @param intrinsic_dim  Input dimension d
    /// @param ambient_dim    Output dimension D
    /// @param model_path     Path to the ONNX model file (.onnx)
    NeuralChart(uint32_t id, uint32_t intrinsic_dim, uint32_t ambient_dim,
                const std::string& model_path);

    ~NeuralChart() override;

    /// Forward pass through the ONNX model: x ↦ φ_θ(x).
    [[nodiscard]] Vector embed(const Vector& local_coords) const override;

    /// Approximate inverse via optimisation (minimise ||φ_θ(x) − y||²).
    [[nodiscard]] Vector project(const Vector& ambient_coords) const override;

    /// Jacobian via numerical central differences of embed().
    [[nodiscard]] Matrix jacobian(const Vector& local_coords) const override;

    [[nodiscard]] ChartType type() const override { return ChartType::NEURAL; }

    /// Path to the underlying ONNX model file.
    [[nodiscard]] const std::string& model_path() const { return model_path_; }

private:
    std::string model_path_;
    // Opaque pointer to ONNX Runtime session (defined in .cpp to avoid
    // exposing onnxruntime headers in the public API).
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ═══════════════════════════════════════════════════════════════════════════════
//  ParametricChart  –  user-supplied callback chart
// ═══════════════════════════════════════════════════════════════════════════════

/// A chart defined by user-supplied std::function callbacks.
/// Useful for custom or procedural manifolds where a closed-form
/// embedding is available.
class ParametricChart : public Chart {
public:
    using EmbedFunc    = std::function<Vector(const Vector&)>;
    using ProjectFunc  = std::function<Vector(const Vector&)>;
    using JacobianFunc = std::function<Matrix(const Vector&)>;

    /// @param id            Chart identifier
    /// @param intrinsic_dim  Local dimension d
    /// @param ambient_dim    Ambient dimension D
    /// @param embed_fn       φ(x) callback
    /// @param project_fn     φ⁻¹(y) callback (best-effort)
    /// @param jacobian_fn    J(x) callback
    ParametricChart(uint32_t    id,
                    uint32_t    intrinsic_dim,
                    uint32_t    ambient_dim,
                    EmbedFunc    embed_fn,
                    ProjectFunc  project_fn,
                    JacobianFunc jacobian_fn)
        : Chart(id, intrinsic_dim, ambient_dim)
        , embed_fn_(std::move(embed_fn))
        , project_fn_(std::move(project_fn))
        , jacobian_fn_(std::move(jacobian_fn))
    {}

    [[nodiscard]] Vector embed(const Vector& local_coords) const override {
        return embed_fn_(local_coords);
    }

    [[nodiscard]] Vector project(const Vector& ambient_coords) const override {
        return project_fn_(ambient_coords);
    }

    [[nodiscard]] Matrix jacobian(const Vector& local_coords) const override {
        return jacobian_fn_(local_coords);
    }

    [[nodiscard]] ChartType type() const override { return ChartType::PARAMETRIC; }

private:
    EmbedFunc    embed_fn_;
    ProjectFunc  project_fn_;
    JacobianFunc jacobian_fn_;
};

} // namespace manifold
