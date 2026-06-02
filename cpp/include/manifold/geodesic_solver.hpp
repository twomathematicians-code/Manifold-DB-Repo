#pragma once

/// @file geodesic_solver.hpp
/// @brief Geodesic equation solver with multiple integration methods.
///
/// Solves the geodesic equation on a Riemannian manifold:
///
///   d²x^i/dt² + Γ^i_{jk}(x) dx^j/dt dx^k/dt = 0
///
/// where Γ^i_{jk} are the Christoffel symbols of the second kind (from MetricTensor).
///
/// Supports:
///   - IVP (Initial Value Problem):  Given x(0), x'(0), integrate forward.
///   - BVP (Boundary Value Problem): Given x(0) and x(T), find x'(0) via shooting.
///   - Parallel transport:          Levi-Civita transport along a geodesic.
///   - Geodesic distance:           Arc length of the shortest geodesic.
///
/// Integration methods:
///   - RK4:     Classical 4th-order Runge-Kutta (fixed step)
///   - RK45:    Adaptive Dormand-Prince (variable step, error control)
///   - SYMPLECTIC: Störmer-Verlet (energy-preserving)
///   - SHOOTING: Newton shooting for BVP

#include "manifold_types.hpp"
#include "metric_store.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <vector>

namespace manifold {

// ═══════════════════════════════════════════════════════════════════════════════
//  Solver Configuration
// ═══════════════════════════════════════════════════════════════════════════════

/// Tunable parameters for the geodesic solver.
struct SolverConfig {
    Scalar initial_step      = 1e-3;     ///< Default integration step size dt
    Scalar min_step         = 1e-8;     ///< Minimum allowed step (prevents division by zero)
    Scalar max_step         = 0.1;      ///< Maximum allowed step
    Scalar tolerance         = 1e-8;     ///< Convergence tolerance for IVP/BVP
    int    max_iterations   = 10000;    ///< Max IVP integration steps
    int    max_bvp_iterations = 50;     ///< Max Newton iterations for shooting
    Scalar bvp_tolerance    = 1e-6;     ///< Shooting method residual tolerance
    bool   adaptive_step    = true;     ///< Enable adaptive step for RK45
};

// ═══════════════════════════════════════════════════════════════════════════════
//  GeodesicSolver
// ═══════════════════════════════════════════════════════════════════════════════

/// Solves geodesic equations on Riemannian manifolds using the metric
/// tensor field provided by a MetricStore.
///
/// The geodesic equation in local coordinates is:
///
///   d²x^i/dt² = −Γ^i_{jk}(x) (dx^j/dt) (dx^k/dt)
///
/// This is a 2nd-order ODE that we reformulate as a 1st-order system:
///   dx/dt = v
///   dv/dt = −Γ(x)(v, v)     [geodesic acceleration]
class GeodesicSolver {
public:
    /// Construct a solver backed by a metric store.
    /// @param metric_store  Shared MetricStore providing g_ij(x) per chart
    /// @param config        Solver parameters
    explicit GeodesicSolver(std::shared_ptr<MetricStore> metric_store,
                            SolverConfig config = {})
        : metric_store_(std::move(metric_store))
        , config_(std::move(config))
    {}

    // ── Solve Initial Value Problem ──────────────────────────────────────────
    //
    // Given initial position x(0) and velocity v(0), integrate the geodesic
    // equation forward to parameter t = t_max.
    //
    //   d²x^i/dt² + Γ^i_{jk}(x) dx^j/dt dx^k/dt = 0
    //   x(0) = start.local_coords,   x'(0) = initial_velocity

    /// Solve the geodesic IVP and return the discretised path.
    /// @param start              Starting manifold point (with chart_id, local_coords)
    /// @param initial_velocity   Initial tangent vector v(0) ∈ T_{start}M
    /// @param t_max              Maximum integration parameter (arc length proxy)
    /// @param method             Integration method (RK4, RK45, SYMPLECTIC)
    /// @return  GeodesicPath with points, arc lengths, and convergence flag
    [[nodiscard]] GeodesicPath solve_ivp(
            const ManifoldPoint& start,
            const Vector& initial_velocity,
            Scalar t_max = 1.0,
            SolverType method = SolverType::RK45) const
    {
        GeodesicPath path;
        path.converged = false;

        uint32_t cid = start.chart_id;
        Vector pos = start.local_coords;
        Vector vel = initial_velocity;

        int d = static_cast<int>(pos.size());
        Scalar t = 0.0;
        Scalar dt = config_.initial_step;
        Scalar total_arc = 0.0;

        path.points.push_back(start);
        path.arc_lengths.push_back(0.0);

        int step = 0;
        while (step < config_.max_iterations && t < t_max) {
            Scalar dt_eff = std::min(dt, t_max - t);
            if (dt_eff < config_.min_step) break;

            if (method == SolverType::RK45 && config_.adaptive_step) {
                Vector err(d);
                Scalar new_dt = rk45_step(cid, pos, vel, dt_eff, err);

                // Adaptive step control (Dormand-Prince)
                Scalar err_norm = err.norm();
                if (err_norm > 0.0) {
                    Scalar err_ratio = err_norm / (config_.tolerance + 1e-30);
                    Scalar safety = 0.9;
                    Scalar min_factor = 0.2;
                    Scalar factor = safety * std::pow(1.0 / (err_ratio + 1e-10), 0.2);
                    factor = std::max(factor, min_factor);
                    new_dt = dt_eff * std::min(factor, 5.0);
                }
                dt = std::clamp(new_dt, config_.min_step, config_.max_step);
            } else if (method == SolverType::SYMPLECTIC) {
                symplectic_step(cid, pos, vel, dt_eff);
            } else {
                // Default: RK4
                rk4_step(cid, pos, vel, dt_eff);
            }

            t += dt_eff;

            // Accumulate metric-weighted arc length: ds = sqrt(v^T g v) · dt
            auto metric = metric_store_->get_metric(cid);
            if (metric) {
                Matrix g = metric->evaluate(pos);
                Scalar ds = std::sqrt(std::abs(vel.transpose() * g * vel)) * dt_eff;
                total_arc += ds;
            } else {
                total_arc += vel.norm() * dt_eff;
            }

            ManifoldPoint pt;
            pt.chart_id     = cid;
            pt.local_coords = pos;
            pt.global_id    = 0;
            path.points.push_back(pt);
            path.arc_lengths.push_back(total_arc);

            ++step;
        }

        path.num_steps = step;
        path.total_length = total_arc;
        path.compute_total_length();
        path.converged = (t >= t_max - config_.min_step);

        return path;
    }

    // ── Solve Boundary Value Problem ──────────────────────────────────────────
    //
    // Given start and end points, find the geodesic connecting them.
    // Uses Newton's shooting method: find v(0) such that γ_v(T) = end.

    /// Solve the geodesic BVP via shooting.
    /// @param start  Starting manifold point
    /// @param end    Target manifold point
    /// @param method Must be SHOOTING (other methods fall back to shooting)
    /// @return  GeodesicPath from start to end
    [[nodiscard]] GeodesicPath solve_bvp(
            const ManifoldPoint& start,
            const ManifoldPoint& end,
            SolverType method = SolverType::SHOOTING) const
    {
        if (method == SolverType::SHOOTING) {
            return shooting_method(start, end);
        }
        return shooting_method(start, end);
    }

    // ── Parallel Transport ──────────────────────────────────────────────────
    //
    // Levi-Civita parallel transport preserves the inner product along the curve:
    //   Dv^i/dt = −Γ^i_{jk} v^j dx^k/dt = 0
    //
    // This means ∇_{γ'} V = 0, i.e., V is "parallel" along γ.

    /// Transport a tangent vector along a geodesic path.
    /// Uses RK4 integration of the parallel transport equation.
    ///
    /// @param path              Geodesic path to transport along
    /// @param vector_at_start   Initial tangent vector V(0) ∈ T_{γ(0)}M
    /// @return  Vector of transported tangent vectors at each path point
    [[nodiscard]] std::vector<Vector> parallel_transport(
            const GeodesicPath& path,
            const Vector& vector_at_start) const
    {
        if (path.points.empty()) return {};

        std::vector<Vector> transported;
        transported.push_back(vector_at_start);

        uint32_t cid = path.points[0].chart_id;
        Vector V = vector_at_start;
        int d = static_cast<int>(V.size());

        for (size_t i = 0; i + 1 < path.points.size(); ++i) {
            const Vector& pos = path.points[i].local_coords;

            // Estimate velocity from finite differences of path
            Vector vel;
            if (path.arc_lengths.size() > i + 1) {
                Scalar dt = path.arc_lengths[i + 1] - path.arc_lengths[i];
                if (dt > 1e-30) {
                    vel = (path.points[i + 1].local_coords - pos) / dt;
                } else {
                    vel = Vector::Zero(d);
                }
            } else {
                vel = path.points[i + 1].local_coords - pos;
            }

            // RK4 transport step
            Scalar dt = 1e-3;
            Vector k1 = transport_derivative(cid, pos, vel, V) * dt;
            Vector k2 = transport_derivative(cid, pos + 0.5 * vel * dt,
                                              vel, V + 0.5 * k1) * dt;
            Vector k3 = transport_derivative(cid, pos + 0.5 * vel * dt,
                                              vel, V + 0.5 * k2) * dt;
            Vector k4 = transport_derivative(cid, pos + vel * dt,
                                              vel, V + k3) * dt;

            V = V + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
            transported.push_back(V);
        }

        return transported;
    }

    // ── Geodesic Distance ─────────────────────────────────────────────────────

    /// Compute the geodesic distance between two manifold points.
    /// For points on the same chart: solve the BVP.
    /// For cross-chart points: Euclidean fallback (TODO: multi-chart solver).
    ///
    /// @param p, q  Manifold points
    /// @return  Geodesic distance d_g(p, q)
    [[nodiscard]] Scalar geodesic_distance(
            const ManifoldPoint& p,
            const ManifoldPoint& q) const
    {
        if (p.chart_id == q.chart_id) {
            GeodesicPath path = shooting_method(p, q);
            return path.total_length;
        }
        // Cross-chart: Euclidean fallback
        return (p.ambient_coords - q.ambient_coords).norm();
    }

    // ── Batch Geodesic Distances ────────────────────────────────────────────

    /// Compute geodesic distances from a single query point to multiple
    /// candidate points on the same chart (for k-NN reranking).
    ///
    /// @param chart_id           Chart identifier
    /// @param query_local        Query point in local coordinates
    /// @param candidates_local   Candidate points in local coordinates
    /// @return  Vector of geodesic distances (same order as candidates)
    [[nodiscard]] std::vector<Scalar> batch_geodesic_distance(
            uint32_t chart_id,
            const Vector& query_local,
            const std::vector<Vector>& candidates_local) const
    {
        std::vector<Scalar> distances;
        distances.reserve(candidates_local.size());

        ManifoldPoint query_pt;
        query_pt.chart_id     = chart_id;
        query_pt.local_coords = query_local;

        for (const auto& cand : candidates_local) {
            ManifoldPoint cand_pt;
            cand_pt.chart_id     = chart_id;
            cand_pt.local_coords = cand;

            Scalar dist = geodesic_distance(query_pt, cand_pt);
            distances.push_back(dist);
        }
        return distances;
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    [[nodiscard]] const SolverConfig& config() const { return config_; }
    void set_config(const SolverConfig& cfg) { config_ = cfg; }

private:
    std::shared_ptr<MetricStore> metric_store_;
    SolverConfig config_;

    // ── Geodesic Acceleration ──────────────────────────────────────────────
    //
    // Computes the RHS of the geodesic equation:
    //   a^i = −Γ^i_{jk}(x) v^j v^k
    //
    // This is the "force" that keeps a geodesic on the manifold.
    // For a flat metric (Γ ≡ 0), this is zero (straight lines in local coords).

    /// Compute geodesic acceleration at a point given position and velocity.
    /// @param chart_id  Chart for metric lookup
    /// @param position  Current position x ∈ R^d
    /// @param velocity  Current velocity v ∈ R^d
    /// @return  Acceleration a ∈ R^d where a^i = −Γ^i_{jk} v^j v^k
    [[nodiscard]] Vector geodesic_acceleration(
            uint32_t chart_id,
            const Vector& position,
            const Vector& velocity) const
    {
        auto metric = metric_store_->get_metric(chart_id);
        int d = static_cast<int>(position.size());

        if (!metric) {
            return Vector::Zero(d);  // Flat metric: no acceleration
        }

        Tensor3D Gamma = metric->christoffel_symbols(position, 1e-5);
        Vector accel(d);
        accel.setZero();

        // a^i = −Γ^i_{jk} v^j v^k
        for (int i = 0; i < d; ++i) {
            Scalar sum = 0.0;
            for (int j = 0; j < d; ++j) {
                for (int k = 0; k < d; ++k) {
                    sum -= Gamma(i, j, k) * velocity(j) * velocity(k);
                }
            }
            accel(i) = sum;
        }
        return accel;
    }

    // ── RK4 Single Step ───────────────────────────────────────────────────

    /// Classical 4th-order Runge-Kutta step for the geodesic ODE system.
    void rk4_step(uint32_t chart_id,
                  Vector& pos, Vector& vel,
                  Scalar dt) const
    {
        auto accel = [this, chart_id](const Vector& p, const Vector& v) -> Vector {
            return geodesic_acceleration(chart_id, p, v);
        };

        Vector a1 = accel(pos, vel);
        Vector v1 = vel;

        Vector a2 = accel(pos + 0.5 * dt * v1, vel + 0.5 * dt * a1);
        Vector v2 = vel + 0.5 * dt * a1;

        Vector a3 = accel(pos + 0.5 * dt * v2, vel + 0.5 * dt * a2);
        Vector v3 = vel + 0.5 * dt * a2;

        Vector a4 = accel(pos + dt * v3, vel + dt * a3);
        Vector v4 = vel + dt * a3;

        pos = pos + (dt / 6.0) * (v1 + 2.0 * v2 + 2.0 * v3 + v4);
        vel = vel + (dt / 6.0) * (a1 + 2.0 * a2 + 2.0 * a3 + a4);
    }

    // ── RK45 Dormand-Prince Adaptive Step ──────────────────────────────────

    /// Adaptive RK4(5) step using the Dormand-Prince method.
    /// Provides both a 5th-order solution and an embedded 4th-order solution
    /// for error estimation, enabling automatic step-size control.
    ///
    /// @param chart_id         Chart for metric lookup
    /// @param pos, vel         Current state (modified in-place to 5th-order result)
    /// @param dt               Current step size
    /// @param error_estimate   Output: estimated local truncation error
    /// @return  Suggested next step size
    [[nodiscard]] Scalar rk45_step(uint32_t chart_id,
                                   Vector& pos, Vector& vel,
                                   Scalar dt,
                                   Vector& error_estimate) const
    {
        // Dormand-Prince Butcher tableau coefficients
        constexpr Scalar a21 = 1.0 / 5.0;
        constexpr Scalar a31 = 3.0 / 40.0,      a32 = 9.0 / 40.0;
        constexpr Scalar a41 = 44.0 / 45.0,      a42 = -56.0 / 15.0,    a43 = 32.0 / 9.0;
        constexpr Scalar a51 = 19372.0 / 6561.0, a52 = -25360.0 / 2187.0;
        constexpr Scalar a53 = 64448.0 / 6561.0, a54 = -212.0 / 729.0;
        constexpr Scalar a61 = 9017.0 / 3168.0,  a62 = -355.0 / 33.0;
        constexpr Scalar a63 = 46732.0 / 5247.0, a64 = 49.0 / 176.0;
        constexpr Scalar a65 = -5103.0 / 18656.0;
        constexpr Scalar a71 = 35.0 / 384.0;
        constexpr Scalar a73 = 500.0 / 1113.0,   a74 = 125.0 / 192.0;
        constexpr Scalar a75 = -2187.0 / 6784.0,  a76 = 11.0 / 84.0;

        auto accel = [this, chart_id](const Vector& p, const Vector& v) -> Vector {
            return geodesic_acceleration(chart_id, p, v);
        };

        // 7 stages of the Dormand-Prince method
        Vector k1v = accel(pos, vel);
        Vector k1x = vel;

        Vector k2v = accel(pos + dt * a21 * k1x, vel + dt * a21 * k1v);
        Vector k2x = vel + dt * a21 * k1v;

        Vector k3v = accel(pos + dt * (a31 * k1x + a32 * k2x),
                           vel + dt * (a31 * k1v + a32 * k2v));
        Vector k3x = vel + dt * (a31 * k1v + a32 * k2v);

        Vector k4v = accel(pos + dt * (a41 * k1x + a42 * k2x + a43 * k3x),
                           vel + dt * (a41 * k1v + a42 * k2v + a43 * k3v));
        Vector k4x = vel + dt * (a41 * k1v + a42 * k2v + a43 * k3v);

        Vector k5v = accel(pos + dt * (a51 * k1x + a52 * k2x + a53 * k3x + a54 * k4x),
                           vel + dt * (a51 * k1v + a52 * k2v + a53 * k3v + a54 * k4v));
        Vector k5x = vel + dt * (a51 * k1v + a52 * k2v + a53 * k3v + a54 * k4v);

        Vector k6v = accel(pos + dt * (a61 * k1x + a62 * k2x + a63 * k3x + a64 * k4x + a65 * k5x),
                           vel + dt * (a61 * k1v + a62 * k2v + a63 * k3v + a64 * k4v + a65 * k5v));
        Vector k6x = vel + dt * (a61 * k1v + a62 * k2v + a63 * k3v + a64 * k4v + a65 * k5v);

        Vector k7v = accel(pos + dt * (a71 * k1x + a73 * k3x + a74 * k4x + a75 * k5x + a76 * k6x),
                           vel + dt * (a71 * k1v + a73 * k3v + a74 * k4v + a75 * k5v + a76 * k6v));
        Vector k7x = vel + dt * (a71 * k1v + a73 * k3v + a74 * k4v + a75 * k5v + a76 * k6v);

        int d = static_cast<int>(pos.size());

        // 5th-order solution (FSAL: reuse k7 as k1 of next step)
        Vector pos_new = pos + dt * (a71 * k1x + a73 * k3x + a74 * k4x + a75 * k5x + a76 * k6x);
        Vector vel_new = vel + dt * (a71 * k1v + a73 * k3v + a74 * k4v + a75 * k5v + a76 * k6v);

        // Error estimate: difference between 5th and embedded 4th order
        constexpr Scalar e1 =  71.0 / 57600.0;
        constexpr Scalar e3 = -71.0 / 16695.0;
        constexpr Scalar e4 =  71.0 / 1920.0;
        constexpr Scalar e5 = -17253.0 / 339200.0;
        constexpr Scalar e6 =  22.0 / 525.0;
        constexpr Scalar e7 = -1.0 / 40.0;

        error_estimate = Vector::Zero(d);
        error_estimate += dt * (e1 * k1v + e3 * k3v + e4 * k4v + e5 * k5v + e6 * k6v + e7 * k7v);

        pos = std::move(pos_new);
        vel = std::move(vel_new);

        return dt;
    }

    // ── Symplectic (Störmer-Verlet) Step ───────────────────────────────────

    /// Symplectic Euler / Störmer-Verlet integrator.
    /// Half-kick → drift → half-kick. Preserves the symplectic structure
    /// of the Hamiltonian formulation of geodesic motion.
    void symplectic_step(uint32_t chart_id,
                         Vector& pos, Vector& vel,
                         Scalar dt) const
    {
        Vector a1 = geodesic_acceleration(chart_id, pos, vel);
        vel += 0.5 * dt * a1;       // half kick
        pos += dt * vel;             // drift
        Vector a2 = geodesic_acceleration(chart_id, pos, vel);
        vel += 0.5 * dt * a2;       // half kick
    }

    // ── Shooting Method for BVP ─────────────────────────────────────────────
    //
    // Newton's method on the shooting map F(v) = γ_v(T) − target:
    //   1. Guess v₀ ≈ target − start  (flat-space estimate)
    //   2. Solve IVP with v_n
    //   3. Compute residual r = γ_{v_n}(T) − target
    //   4. Approximate Jacobian J via finite differences
    //   5. Newton update: v_{n+1} = v_n − J⁻¹ r
    //   6. Repeat until ||r|| < tolerance

    /// Solve the geodesic BVP using Newton shooting.
    [[nodiscard]] GeodesicPath shooting_method(
            const ManifoldPoint& start,
            const ManifoldPoint& end) const
    {
        int d = static_cast<int>(start.local_coords.size());

        Vector target = end.local_coords;
        Vector origin = start.local_coords;
        Vector delta  = target - origin;

        Scalar T_est = delta.norm();
        if (T_est < 1e-30) {
            // Trivial case: start ≈ end
            GeodesicPath path;
            path.points.push_back(start);
            path.points.push_back(end);
            path.arc_lengths = {0.0, 0.0};
            path.total_length = 0.0;
            path.converged = true;
            path.num_steps = 0;
            return path;
        }

        // Initial guess: straight-line velocity in flat space
        Vector v = delta;

        GeodesicPath best_path;
        Scalar best_error = std::numeric_limits<Scalar>::infinity();

        for (int iter = 0; iter < config_.max_bvp_iterations; ++iter) {
            GeodesicPath path = solve_ivp(start, v, T_est, SolverType::RK4);

            if (path.points.empty()) continue;

            Vector reached = path.points.back().local_coords;
            Vector residual = reached - target;
            Scalar err = residual.norm();

            if (err < best_error) {
                best_path  = path;
                best_error = err;
            }

            if (err < config_.bvp_tolerance) {
                best_path.converged = true;
                return best_path;
            }

            // Jacobian of shooting map via finite differences
            Matrix J(d, d);
            Scalar eps = 1e-7;
            for (int col = 0; col < d; ++col) {
                Vector vp = v;
                vp(col) += eps;
                GeodesicPath pp = solve_ivp(start, vp, T_est, SolverType::RK4);
                if (!pp.points.empty()) {
                    J.col(col) = (pp.points.back().local_coords - reached) / eps;
                } else {
                    J.col(col) = Vector::Zero(d);
                }
            }

            // Newton update: δv = −J⁻¹ residual
            Eigen::ColPivHouseholderQR<Matrix> qr(J);
            Vector dv = qr.solve(-residual);

            // Line search with backtracking
            Scalar alpha = 1.0;
            for (int ls = 0; ls < 5; ++ls) {
                GeodesicPath trial = solve_ivp(start, v + alpha * dv, T_est, SolverType::RK4);
                if (!trial.points.empty()) {
                    Scalar trial_err = (trial.points.back().local_coords - target).norm();
                    if (trial_err < err) break;
                }
                alpha *= 0.5;
            }

            v = v + alpha * dv;

            if (dv.norm() < config_.bvp_tolerance * 0.01) {
                best_path.converged = true;
                return best_path;
            }
        }

        best_path.converged = (best_error < config_.bvp_tolerance * 10.0);
        return best_path;
    }

    // ── Parallel Transport Derivative ─────────────────────────────────────
    //
    // Dv^i/dt = −Γ^i_{jk} v^j (dx^k/dt)
    //
    // This is the covariant derivative along the curve γ, enforcing
    // that V remains "parallel" with respect to the Levi-Civita connection.

    [[nodiscard]] Vector transport_derivative(
            uint32_t chart_id,
            const Vector& position,
            const Vector& velocity,
            const Vector& transported_vec) const
    {
        auto metric = metric_store_->get_metric(chart_id);
        int d = static_cast<int>(position.size());

        if (!metric) {
            return Vector::Zero(d);  // Flat: no connection
        }

        Tensor3D Gamma = metric->christoffel_symbols(position, 1e-5);
        Vector deriv(d);
        deriv.setZero();

        for (int i = 0; i < d; ++i) {
            Scalar sum = 0.0;
            for (int j = 0; j < d; ++j) {
                for (int k = 0; k < d; ++k) {
                    sum -= Gamma(i, j, k) * transported_vec(j) * velocity(k);
                }
            }
            deriv(i) = sum;
        }
        return deriv;
    }
};

} // namespace manifold
