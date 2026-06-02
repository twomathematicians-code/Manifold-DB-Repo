#pragma once

/// @file metric_tensor.hpp
/// @brief Riemannian metric tensor field with RBF interpolation and curvature.
///
/// The MetricTensor class stores and evaluates the Riemannian metric g_ij(x)
/// on a chart. The metric is represented by a sparse set of anchor points
/// with associated metric values, and evaluation at arbitrary locations is
/// performed via Gaussian RBF (Radial Basis Function) interpolation:
///
///   g(x) = Σ_k w_k(x) · g_k   /   Σ_k w_k(x)
///
/// where  w_k(x) = α_k · exp(−||x − x_k||² / (2σ_k²))  is the Gaussian
/// kernel weight from anchor k at location x_k with bandwidth σ_k and
/// weight α_k.
///
/// Mathematical background:
///   The Riemannian metric g_ij(x) is a smooth, symmetric, positive-definite
///   (SPD) bilinear form on the tangent space T_xM at each point x.
///   It defines inner products:  ⟨u, v⟩_g = u^i g_{ij} v^j
///   and arc lengths:           ds² = g_{ij} dx^i dx^j
///
///   From the metric we derive:
///   - Inverse metric g^{ij} (contravariant form)
///   - Christoffel symbols Γ^i_{jk} (Levi-Civita connection coefficients)
///   - Sectional curvature K(u,v)
///   - Scalar curvature S = g^{ij} R_{ij}

#include "manifold_types.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <vector>

namespace manifold {

// ═══════════════════════════════════════════════════════════════════════════════
//  MetricTensor
// ═══════════════════════════════════════════════════════════════════════════════

/// Represents the Riemannian metric tensor field g_ij(x) on a single chart.
///
/// Supports two modes:
///   1. Constant metric: g_ij(x) = G  (identity or user-specified SPD matrix)
///   2. Interpolated metric: g(x) from RBF-weighted anchor point values
///
/// The interpolated mode is used for online metric learning, where the metric
/// is refined as new data points are observed.
class MetricTensor {
public:
    /// Construct a metric tensor for a given chart.
    /// Initially set to identity metric (flat space).
    /// @param chart_id  ID of the chart this metric belongs to
    /// @param dim       Intrinsic dimension d of the chart
    explicit MetricTensor(uint32_t chart_id, uint32_t dim)
        : chart_id_(chart_id)
        , dim_(dim)
        , is_constant_(true)
        , constant_metric_(Matrix::Identity(static_cast<int>(dim),
                                            static_cast<int>(dim)))
    {}

    // ── Metric Evaluation ───────────────────────────────────────────────────

    /// Evaluate the metric tensor g_ij at a chart point x.
    ///
    /// In constant mode: returns g(x) = G  (identity by default).
    /// In interpolated mode: RBF-weighted average of anchor metrics.
    ///
    /// @param local_coords  x ∈ R^d  (chart coordinates)
    /// @return  g(x) ∈ R^{d × d}  (symmetric positive-definite)
    [[nodiscard]] Matrix evaluate(const Vector& local_coords) const {
        if (is_constant_) {
            return constant_metric_;
        }
        return interpolate_metric(local_coords);
    }

    /// Evaluate the inverse metric g^{ij}(x).
    ///
    /// Uses Cholesky decomposition (LLT) for SPD inversion, with
    /// SVD pseudo-inverse fallback for near-singular cases.
    ///
    /// @param local_coords  x ∈ R^d
    /// @return  g^{-1}(x) ∈ R^{d × d}
    [[nodiscard]] Matrix inverse(const Vector& local_coords) const {
        Matrix g = evaluate(local_coords);
        int d = static_cast<int>(dim_);
        Eigen::LLT<Matrix> llt(g);
        if (llt.info() == Eigen::Success) {
            return llt.solve(Matrix::Identity(d, d));
        }
        // Fallback: pseudo-inverse via SVD
        Eigen::JacobiSVD<Matrix> svd(g, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Vector svs = svd.singularValues();
        Scalar thresh = static_cast<Scalar>(dim_) * svs(0) * 1e-12;
        for (int i = 0; i < svs.size(); ++i) {
            svs(i) = (svs(i) > thresh) ? Scalar(1.0) / svs(i) : Scalar(0.0);
        }
        return svd.matrixV() * svs.asDiagonal() * svd.matrixU().transpose();
    }

    // ── Christoffel Symbols ─────────────────────────────────────────────────

    /// Compute Christoffel symbols of the second kind Γ^k_{ij}(x) via
    /// central finite differences of the metric tensor.
    ///
    /// Algorithm:
    ///   1. Compute ∂g_{ij}/∂x^a  for all directions a (central differences)
    ///   2. First kind:  Γ_{ijk} = ½ (∂g_{ij}/∂x^k + ∂g_{ik}/∂x^j − ∂g_{jk}/∂x^i)
    ///   3. Second kind: Γ^k_{ij} = g^{kl} Γ_{lij}
    ///
    /// The Christoffel symbols appear in the geodesic equation:
    ///   d²x^k/dt² + Γ^k_{ij} dx^i/dt dx^j/dt = 0
    ///
    /// @param local_coords  Evaluation point x ∈ R^d
    /// @param h             Central-difference step size (default: 1e-5)
    /// @return  Tensor3D of shape (d, d, d): result(k,i,j) = Γ^k_{ij}
    [[nodiscard]] Tensor3D christoffel_symbols(
            const Vector& local_coords,
            Scalar h = 1e-5) const
    {
        int d = static_cast<int>(dim_);

        // Step 1: Numerical partial derivatives ∂g_{ij}/∂x^a
        std::vector<Matrix> dg(d);
        for (int a = 0; a < d; ++a) {
            Vector xp = local_coords, xm = local_coords;
            xp(a) += h;
            xm(a) -= h;
            dg[a] = (evaluate(xp) - evaluate(xm)) / (2.0 * h);
        }

        // Step 2: First kind Γ_{ijk}
        //   Γ_{ijk} = ½ (∂g_{ij}/∂x^k + ∂g_{ik}/∂x^j − ∂g_{jk}/∂x^i)
        Tensor3D G1(d, d, d);
        G1.setZero();
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j)
                for (int k = 0; k < d; ++k)
                    G1(i, j, k) = 0.5 * (
                        dg[k](i, j) +
                        dg[j](i, k) -
                        dg[i](j, k)
                    );

        // Step 3: Second kind Γ^k_{ij} = g^{kl} Γ_{lij}
        Matrix ginv = inverse(local_coords);
        Tensor3D G2(d, d, d);
        G2.setZero();
        for (int k = 0; k < d; ++k)
            for (int i = 0; i < d; ++i)
                for (int j = 0; j < d; ++j)
                    for (int l = 0; l < d; ++l)
                        G2(k, i, j) += ginv(k, l) * G1(l, i, j);

        return G2;
    }

    // ── Sectional Curvature ─────────────────────────────────────────────────

    /// Compute the sectional curvature K(u, v) at a point for the 2-plane
    /// spanned by tangent vectors u, v ∈ T_pM.
    ///
    /// K(u, v) = R(u, v, v, u) / (g(u,u)·g(v,v) − g(u,v)²)
    ///
    /// where R(u,v,w,z) = R^i_{jkl} u^j v^k w^l z^i is the (0,4) Riemann
    /// tensor, computed from:
    ///   R^i_{jkl} = ∂_k Γ^i_{jl} − ∂_l Γ^i_{jk}
    ///              + Γ^i_{mk} Γ^m_{jl} − Γ^i_{ml} Γ^m_{jk}
    ///
    /// @param u, v  Linearly independent tangent vectors
    /// @return  Sectional curvature K(u, v)
    [[nodiscard]] Scalar sectional_curvature(const Vector& u, const Vector& v) const {
        // Flat metric ⇒ zero curvature
        if (is_constant_) return 0.0;

        int d = static_cast<int>(dim_);
        Scalar h = 1e-4;

        // Evaluate at the midpoint of u and v (for finite differences)
        Vector x = 0.5 * (u + v);

        Tensor3D G = christoffel_symbols(x, h);

        // Numerical derivatives of Christoffel symbols
        std::vector<Tensor3D> dG(d);
        for (int a = 0; a < d; ++a) {
            Vector xp = x, xm = x;
            xp(a) += h;
            xm(a) -= h;
            Tensor3D Gp = christoffel_symbols(xp, h);
            Tensor3D Gm = christoffel_symbols(xm, h);
            dG[a] = Tensor3D(d, d, d);
            for (int i = 0; i < d; ++i)
                for (int j = 0; j < d; ++j)
                    for (int k = 0; k < d; ++k)
                        dG[a](i, j, k) = (Gp(i, j, k) - Gm(i, j, k)) / (2.0 * h);
        }

        // Contract Riemann tensor: R(u,v,v,u) = R^i_{jkl} u^j v^k v^l u^i
        Scalar R = 0.0;
        for (int i = 0; i < d; ++i) {
            for (int j = 0; j < d; ++j) {
                for (int k = 0; k < d; ++k) {
                    for (int l = 0; l < d; ++l) {
                        Scalar Rijkl = dG[k](i, j, l) - dG[l](i, j, k);
                        for (int m = 0; m < d; ++m) {
                            Rijkl += G(i, m, k) * G(m, j, l)
                                   - G(i, m, l) * G(m, j, k);
                        }
                        R += Rijkl * u(j) * v(k) * v(l) * u(i);
                    }
                }
            }
        }

        // Denominator: g(u,u)·g(v,v) − g(u,v)²
        Matrix g  = evaluate(x);
        Scalar guu = u.transpose() * g * u;
        Scalar gvv = v.transpose() * g * v;
        Scalar guv = u.transpose() * g * v;
        Scalar denom = guu * gvv - guv * guv;

        if (std::abs(denom) < 1e-30) return 0.0;
        return R / denom;
    }

    // ── Scalar Curvature ─────────────────────────────────────────────────────

    /// Compute the scalar curvature S(x) = g^{ij} R_{ij}(x).
    ///
    /// The Ricci tensor is:  R_{ij} = R^k_{ikj}
    /// and the scalar curvature is its trace w.r.t. the metric.
    [[nodiscard]] Scalar scalar_curvature(
            const Vector& local_coords,
            Scalar h = 1e-5) const
    {
        if (is_constant_) return 0.0;

        int d = static_cast<int>(dim_);
        Tensor3D G = christoffel_symbols(local_coords, h);

        std::vector<Tensor3D> dG(d);
        for (int a = 0; a < d; ++a) {
            Vector xp = local_coords, xm = local_coords;
            xp(a) += h;
            xm(a) -= h;
            Tensor3D Gp = christoffel_symbols(xp, h);
            Tensor3D Gm = christoffel_symbols(xm, h);
            dG[a] = Tensor3D(d, d, d);
            for (int i = 0; i < d; ++i)
                for (int j = 0; j < d; ++j)
                    for (int k = 0; k < d; ++k)
                        dG[a](i, j, k) = (Gp(i, j, k) - Gm(i, j, k)) / (2.0 * h);
        }

        Matrix ginv = inverse(local_coords);
        Scalar S = 0.0;

        // S = g^{ij} R_{ij},  R_{ij} = R^k_{ikj}
        for (int i = 0; i < d; ++i) {
            for (int j = 0; j < d; ++j) {
                Scalar Rij = 0.0;
                for (int k = 0; k < d; ++k) {
                    Rij += dG[j](k, i, k) - dG[k](k, i, j);
                    for (int m = 0; m < d; ++m) {
                        Rij += G(k, m, j) * G(m, i, k)
                             - G(k, m, k) * G(m, i, j);
                    }
                }
                S += ginv(i, j) * Rij;
            }
        }
        return S;
    }

    // ── Online Metric Update ────────────────────────────────────────────────

    /// Add an anchor point with an associated local metric value.
    ///
    /// The bandwidth σ is automatically estimated from the distance to the
    /// k-nearest existing anchors (adaptive kernel width).
    ///
    /// @param local_coords  Anchor location x_k ∈ R^d
    /// @param local_metric  Metric value g_k ∈ R^{d × d} at this anchor
    /// @param weight        Scalar weight α_k (default: 1.0)
    void update(const Vector& local_coords, const Matrix& local_metric, double weight = 1.0) {
        is_constant_ = false;

        Anchor anchor;
        anchor.coords = local_coords;
        anchor.metric = local_metric;
        anchor.weight = weight;
        anchor.sigma  = 1.0;

        // Estimate bandwidth from k-nearest anchor distances
        if (!anchors_.empty()) {
            std::vector<Scalar> dists;
            dists.reserve(anchors_.size());
            for (const auto& a : anchors_) {
                dists.push_back((local_coords - a.coords).norm());
            }
            std::sort(dists.begin(), dists.end());
            size_t k = std::min<size_t>(3, dists.size());
            anchor.sigma = std::max(dists[k - 1], 1e-6);
        }

        anchors_.push_back(std::move(anchor));
    }

    // ── Metric Configuration ─────────────────────────────────────────────────

    /// Set a constant (position-independent) metric tensor.
    /// @param metric  d × d SPD matrix
    /// @throws DimensionMismatchError if dimensions don't match
    void set_constant(const Matrix& metric) {
        if (metric.rows() != static_cast<int>(dim_) ||
            metric.cols() != static_cast<int>(dim_)) {
            throw DimensionMismatchError(
                "MetricTensor::set_constant: metric dimensions must match dim");
        }
        is_constant_ = true;
        constant_metric_ = metric;
    }

    /// Reset to the identity metric g_{ij} = δ_{ij}.
    void set_identity() {
        constant_metric_ = Matrix::Identity(static_cast<int>(dim_),
                                            static_cast<int>(dim_));
        is_constant_ = true;
    }

    /// Clear all anchor points and reset to identity.
    void clear() {
        anchors_.clear();
        is_constant_ = true;
        constant_metric_ = Matrix::Identity(static_cast<int>(dim_),
                                            static_cast<int>(dim_));
    }

    // ── Metadata Accessors ─────────────────────────────────────────────────

    [[nodiscard]] uint32_t chart_id()    const { return chart_id_; }
    [[nodiscard]] uint32_t dim()         const { return dim_; }
    [[nodiscard]] size_t   num_anchors() const { return anchors_.size(); }
    [[nodiscard]] bool     is_constant() const { return is_constant_; }

    // ── Serialization ──────────────────────────────────────────────────────

    /// Serialize the metric tensor to a binary byte buffer.
    /// Format: [chart_id][dim][flags][constant_metric or anchors...]
    [[nodiscard]] std::vector<uint8_t> serialize() const {
        std::vector<uint8_t> buf;
        auto write = [&buf](const void* data, size_t n) {
            const auto* bytes = static_cast<const uint8_t*>(data);
            buf.insert(buf.end(), bytes, bytes + n);
        };

        uint32_t flags = is_constant_ ? 1u : 0u;
        write(&chart_id_, sizeof(chart_id_));
        write(&dim_, sizeof(dim_));
        write(&flags, sizeof(flags));

        if (is_constant_) {
            for (int i = 0; i < constant_metric_.rows(); ++i)
                for (int j = 0; j < constant_metric_.cols(); ++j)
                    write(&constant_metric_(i, j), sizeof(Scalar));
        }

        uint32_t na = static_cast<uint32_t>(anchors_.size());
        write(&na, sizeof(na));

        for (const auto& a : anchors_) {
            uint32_t cd = static_cast<uint32_t>(a.coords.size());
            write(&cd, sizeof(cd));
            for (int i = 0; i < a.coords.size(); ++i)
                write(&a.coords(i), sizeof(Scalar));

            for (int i = 0; i < a.metric.rows(); ++i)
                for (int j = 0; j < a.metric.cols(); ++j)
                    write(&a.metric(i, j), sizeof(Scalar));

            write(&a.weight, sizeof(a.weight));
            write(&a.sigma, sizeof(a.sigma));
        }

        return buf;
    }

    /// Deserialize a metric tensor from a binary byte buffer.
    /// @throws SerializationError on buffer underflow or dimension mismatch
    void deserialize(const std::vector<uint8_t>& data) {
        size_t offset = 0;
        auto read = [&](void* dest, size_t n) {
            if (offset + n > data.size())
                throw SerializationError("MetricTensor::deserialize: buffer underflow");
            std::memcpy(dest, data.data() + offset, n);
            offset += n;
        };

        read(&chart_id_, sizeof(chart_id_));
        read(&dim_, sizeof(dim_));

        uint32_t flags = 0;
        read(&flags, sizeof(flags));
        is_constant_ = (flags & 1u) != 0;

        if (is_constant_) {
            constant_metric_.resize(static_cast<int>(dim_),
                                   static_cast<int>(dim_));
            for (int i = 0; i < static_cast<int>(dim_); ++i)
                for (int j = 0; j < static_cast<int>(dim_); ++j)
                    read(&constant_metric_(i, j), sizeof(Scalar));
        }

        anchors_.clear();
        uint32_t na = 0;
        read(&na, sizeof(na));

        for (uint32_t a_idx = 0; a_idx < na; ++a_idx) {
            Anchor anchor;
            uint32_t cd = 0;
            read(&cd, sizeof(cd));
            anchor.coords.resize(cd);
            for (uint32_t i = 0; i < cd; ++i)
                read(&anchor.coords(i), sizeof(Scalar));

            anchor.metric.resize(static_cast<int>(dim_),
                                static_cast<int>(dim_));
            for (int i = 0; i < static_cast<int>(dim_); ++i)
                for (int j = 0; j < static_cast<int>(dim_); ++j)
                    read(&anchor.metric(i, j), sizeof(Scalar));

            read(&anchor.weight, sizeof(anchor.weight));
            read(&anchor.sigma, sizeof(anchor.sigma));
            anchors_.push_back(std::move(anchor));
        }
    }

private:
    uint32_t chart_id_;     ///< Chart this metric belongs to
    uint32_t dim_;          ///< Intrinsic dimension d
    bool     is_constant_;  ///< If true, g(x) = constant_metric_ everywhere
    Matrix   constant_metric_;  ///< Constant SPD metric (used when is_constant_)

    /// An anchor point for RBF interpolation.
    struct Anchor {
        Vector coords;       ///< Location x_k ∈ R^d
        Matrix metric;       ///< Metric value g_k ∈ R^{d × d}
        double weight;       ///< Scalar weight α_k
        double sigma;        ///< Gaussian kernel bandwidth σ_k
    };
    std::vector<Anchor> anchors_;

    // ── RBF Interpolation ───────────────────────────────────────────────────

    /// Evaluate the metric at x by Gaussian RBF interpolation from anchors.
    ///
    /// Uses the k=5 nearest anchors with adaptive bandwidth:
    ///   g(x) = Σ_k w_k(x) · g_k  /  Σ_k w_k(x)
    ///
    /// where  w_k(x) = α_k · exp(−||x − x_k||² / (2σ_k²))
    ///
    /// The result is symmetrised to enforce SPD: g ← ½(g + g^T)
    [[nodiscard]] Matrix interpolate_metric(const Vector& local_coords) const {
        if (anchors_.empty()) {
            return Matrix::Identity(static_cast<int>(dim_),
                                   static_cast<int>(dim_));
        }

        // Use k nearest anchors for efficiency
        size_t k = std::min<size_t>(5, anchors_.size());
        auto indices = nearest_anchors(local_coords, k);

        Scalar w_sum = 0.0;
        Matrix result = Matrix::Zero(static_cast<int>(dim_),
                                     static_cast<int>(dim_));

        for (size_t idx : indices) {
            const auto& a = anchors_[idx];
            Scalar dist = (local_coords - a.coords).norm();
            // Gaussian RBF kernel
            Scalar w = a.weight * std::exp(-0.5 * (dist / a.sigma) * (dist / a.sigma));
            result  += w * a.metric;
            w_sum   += w;
        }

        if (w_sum < 1e-30) {
            return Matrix::Identity(static_cast<int>(dim_),
                                   static_cast<int>(dim_));
        }
        result /= w_sum;

        // Symmetrise to enforce SPD
        return 0.5 * (result + result.transpose());
    }

    /// Find the indices of the k nearest anchor points to a query location.
    /// Uses partial sort for O(n + k log n) performance.
    [[nodiscard]] std::vector<size_t> nearest_anchors(
            const Vector& local_coords, size_t k = 5) const
    {
        std::vector<std::pair<Scalar, size_t>> dists;
        dists.reserve(anchors_.size());
        for (size_t i = 0; i < anchors_.size(); ++i) {
            Scalar d = (local_coords - anchors_[i].coords).norm();
            dists.emplace_back(d, i);
        }
        std::partial_sort(dists.begin(),
                           dists.begin() + std::min(k, dists.size()),
                           dists.end());

        std::vector<size_t> result;
        size_t count = std::min(k, dists.size());
        result.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            result.push_back(dists[i].second);
        }
        return result;
    }
};

} // namespace manifold
