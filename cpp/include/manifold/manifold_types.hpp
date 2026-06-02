#pragma once

/// @file manifold_types.hpp
/// @brief Core type system and primitives for the ManifoldDB geometric inference engine.
///
/// Defines all fundamental algebraic types (via Eigen), manifold data structures
/// (ManifoldPoint, GeodesicPath, NeighborResult), enumerations, and exceptions
/// used throughout the manifold database library.
///
/// Mathematical background:
///   The engine models data living on a d-dimensional Riemannian manifold M
///   embedded in D-dimensional ambient Euclidean space R^D (typically D >> d).
///   Each point is represented in both local chart coordinates x ∈ U ⊂ R^d
///   and ambient coordinates y = φ(x) ∈ R^D.

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/CXX11/Tensor>

#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <ostream>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace manifold {

// ═══════════════════════════════════════════════════════════════════════════════
//  Fundamental Scalar & Linear Algebra Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Floating-point scalar (double precision for numerical stability).
using Scalar      = double;

/// Column vector in R^n (dynamically sized).
/// Used for local coordinates x ∈ R^d and ambient coordinates y ∈ R^D.
using Vector      = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

/// General dense matrix (dynamically sized).
/// Used for Jacobians J (D × d), metric tensors g (d × d), etc.
using Matrix      = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

/// 3rd-order tensor (rank-3).
/// Used for Christoffel symbols Γ^k_{ij} (d × d × d).
using Tensor3D    = Eigen::Tensor<Scalar, 3>;

/// Sparse matrix for large-scale metric representations.
using SparseMatrix = Eigen::SparseMatrix<Scalar>;

/// Integer vector for indexing operations.
using IntVector   = Eigen::Matrix<int, Eigen::Dynamic, 1>;

// ═══════════════════════════════════════════════════════════════════════════════
//  ManifoldPoint  –  A point on the Riemannian manifold
// ═══════════════════════════════════════════════════════════════════════════════

/// Represents a point p ∈ M on the manifold, with dual representation:
///   - Local coordinates:  x = (x^1, …, x^d)  in chart (U, φ)
///   - Ambient coordinates: y = φ(x) = (y^1, …, y^D) in embedding space R^D
///
/// The chart_id identifies which coordinate patch the point belongs to,
/// enabling the atlas to manage transitions between overlapping charts.
struct ManifoldPoint {
    uint32_t chart_id       = 0;          ///< ID of the home chart
    Vector   local_coords;              ///< x ∈ R^d (chart coordinates)
    Vector   ambient_coords;            ///< y = φ(x) ∈ R^D (embedding coordinates)
    uint64_t global_id     = 0;          ///< Unique identifier across the database
    double   timestamp     = 0.0;        ///< Insertion / modification time

    ManifoldPoint() = default;

    /// Construct a manifold point with full dual representation.
    /// @param cid   Chart identifier this point belongs to
    /// @param lc    Local coordinates in chart R^d
    /// @param ac    Ambient (embedding) coordinates in R^D
    /// @param gid   Globally unique point identifier
    /// @param ts    Timestamp (default: epoch 0)
    ManifoldPoint(uint32_t cid,
                  const Vector& lc,
                  const Vector& ac,
                  uint64_t gid,
                  double ts = 0.0)
        : chart_id(cid)
        , local_coords(lc)
        , ambient_coords(ac)
        , global_id(gid)
        , timestamp(ts)
    {}

    /// Human-readable string representation for diagnostics.
    [[nodiscard]] std::string to_string() const {
        std::ostringstream oss;
        oss << "ManifoldPoint(chart=" << chart_id
            << ", global_id=" << global_id
            << ", local_dim=" << local_coords.size()
            << ", ambient_dim=" << ambient_coords.size()
            << ")";
        return oss.str();
    }

    /// Euclidean norm of the local coordinate vector: ||x||_2
    [[nodiscard]] Scalar local_norm() const {
        return local_coords.norm();
    }

    /// Euclidean norm of the ambient coordinate vector: ||y||_2
    [[nodiscard]] Scalar ambient_norm() const {
        return ambient_coords.norm();
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
//  GeodesicPath  –  Discretised geodesic curve
// ═══════════════════════════════════════════════════════════════════════════════

/// A discrete approximation of a geodesic γ(t) between two manifold points.
/// The geodesic equation (in local coordinates) is:
///
///   d²x^i/dt² + Γ^i_{jk}(x) dx^j/dt dx^k/dt = 0
///
/// where Γ^i_{jk} are the Christoffel symbols of the second kind.
struct GeodesicPath {
    std::vector<ManifoldPoint> points;        ///< Sampled points along γ(t)
    std::vector<Scalar>        arc_lengths;   ///< Cumulative arc length s(t_i) = ∫_0^{t_i} ||γ'(τ)|| dτ
    Scalar total_length  = 0.0;               ///< s(t_final) – total geodesic length
    bool   converged     = false;             ///< Whether the solver converged
    int    num_steps     = 0;                 ///< Number of integration steps taken

    /// Recompute total_length from the cumulative arc_lengths vector.
    void compute_total_length() {
        if (arc_lengths.empty()) {
            total_length = 0.0;
        } else {
            total_length = arc_lengths.back();
        }
    }

    [[nodiscard]] bool is_empty() const {
        return points.empty();
    }

    [[nodiscard]] size_t size() const {
        return points.size();
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
//  NeighborResult  –  k-NN query result
// ═══════════════════════════════════════════════════════════════════════════════

/// Result entry from a geodesic k-nearest-neighbour query.
/// Stores both the true geodesic distance and the Euclidean residual
/// (difference between geodesic and Euclidean distances) for quality assessment.
struct NeighborResult {
    ManifoldPoint point;                                      ///< The neighbour point
    Scalar geodesic_distance  = std::numeric_limits<Scalar>::infinity();  ///< d_g(p, q)
    Scalar euclidean_residual = 0.0;                          ///< |d_g(p,q) - ||y_p - y_q|||

    /// Comparison for sorting by ascending geodesic distance.
    [[nodiscard]] bool operator<(const NeighborResult& other) const {
        return geodesic_distance < other.geodesic_distance;
    }

    [[nodiscard]] bool operator>(const NeighborResult& other) const {
        return geodesic_distance > other.geodesic_distance;
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
//  Enumerations
// ═══════════════════════════════════════════════════════════════════════════════

/// Supported ODE integration methods for the geodesic equation.
enum class SolverType {
    RK4,           ///< Classical 4th-order Runge-Kutta
    RK45,          ///< Adaptive Dormand-Prince (RK4(5))
    RK4_CUDA,      ///< GPU-accelerated RK4 (requires CUDA build)
    SYMPLECTIC,    ///< Symplectic (Störmer-Verlet) integrator
    SHOOTING       ///< Newton shooting for boundary value problems
};

/// Distance metric options for queries.
enum class DistanceType {
    GEODESIC,      ///< True Riemannian distance d_g(p,q)
    EUCLIDEAN,     ///< Ambient Euclidean ||y_p - y_q||
    LOG_CHORDAL    ///< Chordal distance in log-map coordinates
};

/// Chart type classification.
enum class ChartType {
    LINEAR,        ///< Affine (PCA) chart: φ(x) = origin + B·x
    NEURAL,        ///< Neural network chart (ONNX-based): φ = f_θ(x)
    PARAMETRIC,    ///< User-supplied callback-based chart
    CUSTOM         ///< User-defined chart subclass
};

// ═══════════════════════════════════════════════════════════════════════════════
//  Exceptions
// ═══════════════════════════════════════════════════════════════════════════════

/// Base exception for all ManifoldDB runtime errors.
struct DBException : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

/// Thrown when a requested chart ID is not found in the atlas.
struct ChartNotFoundException : public DBException {
    uint32_t chart_id;

    explicit ChartNotFoundException(uint32_t cid)
        : DBException("Chart " + std::to_string(cid) + " not found")
        , chart_id(cid)
    {}
};

/// Thrown when the geodesic solver fails to converge within tolerance.
struct GeodesicSolverError : public DBException {
    using DBException::DBException;
};

/// Thrown when an index build step fails.
struct IndexBuildError : public DBException {
    using DBException::DBException;
};

/// Thrown on serialisation / deserialization errors.
struct SerializationError : public DBException {
    using DBException::DBException;
};

/// Thrown when matrix/vector dimensions are inconsistent.
struct DimensionMismatchError : public DBException {
    using DBException::DBException;
};

// ═══════════════════════════════════════════════════════════════════════════════
//  Stream Overload
// ═══════════════════════════════════════════════════════════════════════════════

/// Stream insertion for ManifoldPoint (diagnostic output).
inline std::ostream& operator<<(std::ostream& os, const ManifoldPoint& p) {
    return os << p.to_string();
}

} // namespace manifold
