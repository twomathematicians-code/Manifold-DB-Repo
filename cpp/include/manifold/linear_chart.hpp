#pragma once

/// @file linear_chart.hpp
/// @brief LinearChart – affine (PCA) coordinate patch on the manifold.
///
/// The linear chart models a flat patch of the manifold as an affine subspace:
///
///   φ(x) = origin + B · x
///
/// where B ∈ R^{D × d} is an orthonormal basis matrix (columns span the
/// tangent plane) and origin ∈ R^D is the centre of the patch.
///
/// This corresponds to a first-order (linear) Taylor approximation of the
/// manifold near the origin. For a linear chart:
///   - Jacobian J = B  (constant, position-independent)
///   - Metric g = B^T B = I_d  (identity if B is orthonormal)
///   - Christoffel symbols Γ^i_{jk} = 0  (flat geometry)
///   - Geodesics are straight lines: γ(t) = p + t·v
///
/// Linear charts are typically obtained via PCA decomposition of local
/// neighbourhoods in ambient space.

#include "chart.hpp"

#include <Eigen/QR>

#include <stdexcept>
#include <string>

namespace manifold {

// ═══════════════════════════════════════════════════════════════════════════════
//  LinearChart  –  φ(x) = origin + basis · x
// ═══════════════════════════════════════════════════════════════════════════════

/// Affine chart embedding R^d → R^D via a orthonormal basis matrix.
///
/// Construction typically proceeds from PCA:
///   1. Compute the covariance of local ambient points.
///   2. Take the top-d eigenvectors as columns of B.
///   3. Set origin = local centroid.
///
/// Properties:
///   embed(x)    = origin + B · x               (affine map)
///   project(y)  = B^T · (y − origin)          (orthogonal projection)
///   jacobian(x) = B                            (constant D × d)
///
/// If B has orthonormal columns (B^T B = I_d), then the induced metric
/// is the identity: g_ij = δ_ij, and all Christoffel symbols vanish.
class LinearChart : public Chart {
public:
    /// Construct a linear chart with an orthonormal (or at least full-rank)
    /// basis matrix whose columns span the tangent plane.
    ///
    /// @param id      Unique chart identifier
    /// @param basis   D × d matrix (ambient_dim × intrinsic_dim).
    ///                Columns should be orthonormal for a proper isometric chart.
    /// @param origin  D-vector centre in ambient space R^D
    ///
    /// @throws DimensionMismatchError if basis dimensions are inconsistent
    LinearChart(uint32_t id, const Matrix& basis, const Vector& origin)
        : Chart(id,
                static_cast<uint32_t>(basis.cols()),
                static_cast<uint32_t>(basis.rows()))
        , basis_(basis)
        , origin_(origin)
    {
        if (static_cast<uint32_t>(basis.rows()) != ambient_dim_ ||
            static_cast<uint32_t>(basis.cols()) != intrinsic_dim_) {
            throw DimensionMismatchError(
                "LinearChart: basis dimensions (" +
                std::to_string(basis.rows()) + "×" + std::to_string(basis.cols()) +
                ") do not match declared dims (ambient=" +
                std::to_string(ambient_dim_) + ", intrinsic=" +
                std::to_string(intrinsic_dim_) + ")");
        }
        if (static_cast<uint32_t>(origin.size()) != ambient_dim_) {
            throw DimensionMismatchError(
                "LinearChart: origin dimension (" +
                std::to_string(origin.size()) +
                ") does not match ambient_dim (" +
                std::to_string(ambient_dim_) + ")");
        }
    }

    // ── Core Geometry ─────────────────────────────────────────────────────────

    /// Embedding: φ(x) = origin + B · x
    ///
    /// Maps a point x ∈ R^d in local chart coordinates to its ambient
    /// position y = origin + B·x ∈ R^D on the affine tangent plane.
    [[nodiscard]] Vector embed(const Vector& local_coords) const override {
        return origin_ + basis_ * local_coords;
    }

    /// Projection: φ⁻¹(y) = B^T · (y − origin)
    ///
    /// Orthogonally projects an ambient point onto the affine plane and
    /// returns the local coordinates. If B has orthonormal columns, this
    /// is the minimum-norm projection.
    [[nodiscard]] Vector project(const Vector& ambient_coords) const override {
        return basis_.transpose() * (ambient_coords - origin_);
    }

    /// Pushforward (Jacobian): J(x) = B  (constant)
    ///
    /// Since the chart is affine, the Jacobian is constant (position-independent).
    /// Returns the D × d basis matrix.
    [[nodiscard]] Matrix jacobian(const Vector& /*local_coords*/) const override {
        return basis_;
    }

    // ── Flat Geometry Optimisations ─────────────────────────────────────────

    /// The induced metric for a linear chart with orthonormal basis is identity:
    ///   g_ij = (B^T B)_{ij} = δ_ij
    ///
    /// For a non-orthonormal basis, returns B^T B (still constant).
    [[nodiscard]] Matrix compute_local_metric(const Vector& /*local_coords*/) const override {
        return basis_.transpose() * basis_;
    }

    /// Exponential map on a flat chart is a straight line in local coords:
    ///
    ///   exp_p(v) = p_local + v
    ///
    /// Then re-embedded: φ(p_local + v) = origin + B·(p_local + v)
    ///
    /// This is exact (no numerical integration needed) because Γ ≡ 0.
    [[nodiscard]] ManifoldPoint exponential_map(
            const ManifoldPoint& base,
            const Vector& tangent_vec,
            Scalar /*step_size*/  = 1e-3,
            int    /*max_steps*/  = 1000) const override
    {
        Vector new_local = base.local_coords + tangent_vec;
        ManifoldPoint result;
        result.chart_id       = id_;
        result.local_coords   = new_local;
        result.ambient_coords = embed(new_local);
        return result;
    }

    /// Logarithmic map on a flat chart is simply:
    ///
    ///   log_p(q) = q_local − p_local
    ///
    /// No iteration needed (Γ ≡ 0 ⇒ geodesics are straight lines).
    [[nodiscard]] Vector log_map(
            const ManifoldPoint& base,
            const ManifoldPoint& target,
            Scalar /*tolerance*/    = 1e-8,
            int    /*max_iterations*/ = 100) const override
    {
        return target.local_coords - base.local_coords;
    }

    // ── Boundary Check ──────────────────────────────────────────────────────

    /// A linear chart accepts any point whose local coordinate dimension matches.
    /// (Unbounded affine patch – no finite boundary restriction.)
    [[nodiscard]] bool contains(const Vector& local_coords) const override {
        return static_cast<uint32_t>(local_coords.size()) == intrinsic_dim_;
    }

    // ── Type & Accessors ─────────────────────────────────────────────────────

    [[nodiscard]] ChartType type() const override { return ChartType::LINEAR; }

    /// Access the D × d basis matrix B.
    [[nodiscard]] const Matrix& basis()  const { return basis_; }

    /// Access the D-dimensional origin vector.
    [[nodiscard]] const Vector& origin() const { return origin_; }

    /// Compute the residual of projecting an ambient point onto this chart.
    /// The residual measures how far the point is from the affine plane:
    ///   r = y − (origin + B · B^T(y − origin))
    [[nodiscard]] Scalar projection_residual(const Vector& ambient_coords) const {
        Vector local = project(ambient_coords);
        Vector reembedded = embed(local);
        return (ambient_coords - reembedded).norm();
    }

private:
    Matrix basis_;   ///< B ∈ R^{D × d}  (ambient_dim × intrinsic_dim)
    Vector origin_;  ///< origin ∈ R^D   (centre of the affine patch)
};

} // namespace manifold
