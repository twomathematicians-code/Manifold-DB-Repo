// bindings.cpp
// PyBind11 bindings for ManifoldDB – geometric inference engine.
// Exposes all C++ classes to Python via pybind11 with Eigen and torch::Tensor support.

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <torch/torch.h>
#include <torch/types.h>

#include "manifold/atlas.hpp"
#include "manifold/chart.hpp"
#include "manifold/geodesic_solver.hpp"
#include "manifold/linear_chart.hpp"
#include "manifold/manifold_db.hpp"
#include "manifold/manifold_types.hpp"
#include "manifold/metric_store.hpp"
#include "manifold/tangent_space_index.hpp"

#include <sstream>

namespace py = pybind11;
using namespace manifold;

// ═════════════════════════════════════════════════════════════════════════════════
//  Helper: torch::Tensor [N, D] → std::vector<Vector>
// ═════════════════════════════════════════════════════════════════════════════════

/// Convert a contiguous float32/float64 PyTorch tensor of shape [N, D] into
/// a std::vector of Eigen column vectors (one per row).
static std::vector<Vector> tensor_to_vectors(const torch::Tensor& tensor) {
    TORCH_CHECK(tensor.dim() == 2,
                "Expected a 2-D tensor [N, D], got dim=", tensor.dim());
    TORCH_CHECK(
        tensor.dtype() == at::kFloat || tensor.dtype() == at::kDouble,
        "Expected float32 or float64 tensor, got ", tensor.dtype().name());
    TORCH_CHECK(tensor.is_contiguous(),
                "Tensor must be contiguous; call .contiguous() before passing");

    auto contiguous = tensor.to(at::kDouble).contiguous();
    int64_t N = contiguous.size(0);
    int64_t D = contiguous.size(1);
    auto acc  = contiguous.accessor<double, 2>();

    std::vector<Vector> points;
    points.reserve(static_cast<size_t>(N));
    for (int64_t i = 0; i < N; ++i) {
        Vector v(static_cast<int>(D));
        for (int64_t j = 0; j < D; ++j) {
            v(j) = acc[i][j];
        }
        points.push_back(std::move(v));
    }
    return points;
}

// ═════════════════════════════════════════════════════════════════════════════════
//  Trampolines (allow Python subclassing of abstract C++ classes)
// ═════════════════════════════════════════════════════════════════════════════════

/// Trampoline for Chart – enables Python subclasses to override pure-virtuals.
class PyChart : public Chart {
public:
    using Chart::Chart;

    Vector embed(const Vector& local_coords) const override {
        PYBIND11_OVERRIDE_PURE(Vector, Chart, embed, local_coords);
    }

    Vector project(const Vector& ambient_coords) const override {
        PYBIND11_OVERRIDE_PURE(Vector, Chart, project, ambient_coords);
    }

    Matrix jacobian(const Vector& local_coords) const override {
        PYBIND11_OVERRIDE_PURE(Matrix, Chart, jacobian, local_coords);
    }

    ChartType type() const override {
        PYBIND11_OVERRIDE_PURE(ChartType, Chart, type);
    }

    ManifoldPoint exponential_map(
            const ManifoldPoint& base,
            const Vector& tangent_vec,
            Scalar step_size,
            int max_steps) const override
    {
        PYBIND11_OVERRIDE(ManifoldPoint, Chart, exponential_map,
                          base, tangent_vec, step_size, max_steps);
    }

    Vector log_map(
            const ManifoldPoint& base,
            const ManifoldPoint& target,
            Scalar tolerance,
            int max_iterations) const override
    {
        PYBIND11_OVERRIDE(Vector, Chart, log_map,
                          base, target, tolerance, max_iterations);
    }

    bool contains(const Vector& local_coords) const override {
        PYBIND11_OVERRIDE(bool, Chart, contains, local_coords);
    }
};

/// Trampoline for TransitionMap – enables Python subclasses for custom
/// coordinate transformations between charts.
class PyTransitionMap : public TransitionMap {
public:
    using TransitionMap::TransitionMap;

    Vector forward(const Vector& coords_a) const override {
        PYBIND11_OVERRIDE_PURE(Vector, TransitionMap, forward, coords_a);
    }

    Vector inverse(const Vector& coords_b) const override {
        PYBIND11_OVERRIDE_PURE(Vector, TransitionMap, inverse, coords_b);
    }

    Matrix jacobian(const Vector& coords_a) const override {
        PYBIND11_OVERRIDE_PURE(Matrix, TransitionMap, jacobian, coords_a);
    }
};

// ═════════════════════════════════════════════════════════════════════════════════
//  Module definition
// ═════════════════════════════════════════════════════════════════════════════════

PYBIND11_MODULE(manifolddb_core, m) {
    m.doc() =
        "ManifoldDB – Riemannian geometric inference engine.\n\n"
        "Provides manifold-aware data structures including charts, geodesic solvers,\n"
        "tangent-space indexes, metric tensors, atlas management, and the top-level\n"
        "ManifoldDB database API for geodesic queries over Riemannian manifolds.\n\n"
        "Supports torch::Tensor input for insert() and evolve_schema().";

    // ═══════════════════════════════════════════════════════════════════════════
    //  Enums
    // ═══════════════════════════════════════════════════════════════════════════

    py::enum_<SolverType>(m, "SolverType")
        .value("RK4", SolverType::RK4)
        .value("RK45", SolverType::RK45)
        .value("RK4_CUDA", SolverType::RK4_CUDA)
        .value("SYMPLECTIC", SolverType::SYMPLECTIC)
        .value("SHOOTING", SolverType::SHOOTING)
        .export_values();

    py::enum_<DistanceType>(m, "DistanceType")
        .value("GEODESIC", DistanceType::GEODESIC)
        .value("EUCLIDEAN", DistanceType::EUCLIDEAN)
        .value("LOG_CHORDAL", DistanceType::LOG_CHORDAL)
        .export_values();

    py::enum_<ChartType>(m, "ChartType")
        .value("LINEAR", ChartType::LINEAR)
        .value("NEURAL", ChartType::NEURAL)
        .value("PARAMETRIC", ChartType::PARAMETRIC)
        .value("CUSTOM", ChartType::CUSTOM)
        .export_values();

    // ═══════════════════════════════════════════════════════════════════════════
    //  Exceptions
    // ═══════════════════════════════════════════════════════════════════════════

    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const DBException& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
    });

    py::register_exception<ChartNotFoundException>(m, "ChartNotFoundError",
                                                    PyExc_RuntimeError);
    py::register_exception<GeodesicSolverError>(m, "GeodesicSolverError",
                                                PyExc_RuntimeError);
    py::register_exception<IndexBuildError>(m, "IndexBuildError",
                                           PyExc_RuntimeError);
    py::register_exception<SerializationError>(m, "SerializationError",
                                               PyExc_RuntimeError);
    py::register_exception<DimensionMismatchError>(m, "DimensionMismatchError",
                                                   PyExc_RuntimeError);

    // ═══════════════════════════════════════════════════════════════════════════
    //  ManifoldPoint
    // ═══════════════════════════════════════════════════════════════════════════

    py::class_<ManifoldPoint>(m, "ManifoldPoint",
        "A point on the Riemannian manifold with dual representation:\n"
        "  - local_coords:  chart coordinates x ∈ R^d\n"
        "  - ambient_coords: embedding coordinates y ∈ R^D")
        .def(py::init<>())
        .def(py::init<uint32_t, const Vector&, const Vector&, uint64_t, double>(),
             py::arg("chart_id"), py::arg("local_coords"),
             py::arg("ambient_coords"), py::arg("global_id"),
             py::arg("timestamp") = 0.0)
        .def_readonly("chart_id", &ManifoldPoint::chart_id,
                      "Home chart identifier")
        .def_readonly("global_id", &ManifoldPoint::global_id,
                      "Unique identifier across the database")
        .def_readwrite("local_coords", &ManifoldPoint::local_coords,
                       "Local chart coordinates x ∈ R^d")
        .def_readwrite("ambient_coords", &ManifoldPoint::ambient_coords,
                       "Ambient embedding coordinates y ∈ R^D")
        .def_readwrite("timestamp", &ManifoldPoint::timestamp,
                       "Insertion / modification time")
        .def("local_norm", &ManifoldPoint::local_norm,
             "Euclidean norm of local coordinate vector")
        .def("ambient_norm", &ManifoldPoint::ambient_norm,
             "Euclidean norm of ambient coordinate vector")
        .def("__repr__", [](const ManifoldPoint& p) {
            return p.to_string();
        });

    // ═══════════════════════════════════════════════════════════════════════════
    //  GeodesicPath
    // ═══════════════════════════════════════════════════════════════════════════

    py::class_<GeodesicPath>(m, "GeodesicPath",
        "Discrete approximation of a geodesic curve between two manifold points.")
        .def(py::init<>())
        .def_readonly("points", &GeodesicPath::points,
                      "Sampled points along the geodesic")
        .def_readonly("arc_lengths", &GeodesicPath::arc_lengths,
                      "Cumulative arc lengths at each sample point")
        .def_readonly("total_length", &GeodesicPath::total_length,
                      "Total geodesic arc length")
        .def_readonly("converged", &GeodesicPath::converged,
                      "Whether the solver converged")
        .def_readonly("num_steps", &GeodesicPath::num_steps,
                      "Number of integration steps taken")
        .def("compute_total_length", &GeodesicPath::compute_total_length,
             "Recompute total_length from arc_lengths")
        .def("is_empty", &GeodesicPath::is_empty,
             "True if the path contains no points")
        .def("size", &GeodesicPath::size,
             "Number of sample points along the path")
        .def("__repr__", [](const GeodesicPath& gp) {
            std::ostringstream oss;
            oss << "GeodesicPath(points=" << gp.points.size()
                << ", total_length=" << gp.total_length
                << ", converged=" << (gp.converged ? "True" : "False")
                << ")";
            return oss.str();
        });

    // ═══════════════════════════════════════════════════════════════════════════
    //  NeighborResult
    // ═══════════════════════════════════════════════════════════════════════════

    py::class_<NeighborResult>(m, "NeighborResult",
        "Result entry from a geodesic k-nearest-neighbour query.")
        .def(py::init<>())
        .def_readonly("point", &NeighborResult::point,
                      "The neighbour manifold point")
        .def_readonly("geodesic_distance", &NeighborResult::geodesic_distance,
                      "True geodesic distance d_g(p, q)")
        .def_readonly("euclidean_residual", &NeighborResult::euclidean_residual,
                      "|d_g(p,q) - ||y_p - y_q|||")
        .def("__lt__", [](const NeighborResult& a, const NeighborResult& b) {
            return a < b;
        }, py::is_operator())
        .def("__gt__", [](const NeighborResult& a, const NeighborResult& b) {
            return a > b;
        }, py::is_operator())
        .def("__repr__", [](const NeighborResult& nr) {
            std::ostringstream oss;
            oss << "NeighborResult(geodesic_distance=" << nr.geodesic_distance
                << ", euclidean_residual=" << nr.euclidean_residual
                << ")";
            return oss.str();
        });

    // ═══════════════════════════════════════════════════════════════════════════
    //  Chart (abstract base)
    // ═══════════════════════════════════════════════════════════════════════════

    py::class_<Chart, PyChart, std::shared_ptr<Chart>>(m, "Chart",
        "Abstract base class for a coordinate chart (U, phi) on the manifold.\n"
        "Provides embedding, projection, Jacobian, metric, exponential/log maps.")
        // No direct constructor – abstract base class
        .def("embed", &Chart::embed, py::arg("local_coords"),
             "Embed local coordinates into ambient space: phi(x) -> R^D")
        .def("project", &Chart::project, py::arg("ambient_coords"),
             "Project ambient coordinates back to local chart: "
             "phi^{-1}(y) -> R^d")
        .def("jacobian", &Chart::jacobian, py::arg("local_coords"),
             "Compute the Jacobian (pushforward) d(phi)/dx: D x d matrix")
        .def("compute_local_metric", &Chart::compute_local_metric,
             py::arg("local_coords"),
             "Compute the induced Riemannian metric g_ij = J^T J")
        .def("compute_inverse_metric", &Chart::compute_inverse_metric,
             py::arg("local_coords"),
             "Compute the inverse metric g^{ij}")
        .def("christoffel_first_kind", &Chart::christoffel_first_kind,
             py::arg("local_coords"), py::arg("h") = 1e-5,
             "Christoffel symbols of the first kind Gamma_{ijk} "
             "(via central differences)")
        .def("christoffel_second_kind", &Chart::christoffel_second_kind,
             py::arg("local_coords"), py::arg("h") = 1e-5,
             "Christoffel symbols of the second kind Gamma^i_{jk}")
        .def("sectional_curvature",
             [](const Chart& c, const Vector& coords,
                const Vector& u, const Vector& v, Scalar h) {
                 return c.sectional_curvature(coords, u, v, h);
             },
             py::arg("local_coords"), py::arg("u"), py::arg("v"),
             py::arg("h") = 1e-5,
             "Sectional curvature K(u, v) for a 2-plane")
        .def("exponential_map",
             [](const Chart& c, const ManifoldPoint& base,
                const Vector& tangent,
                Scalar step_size, int max_steps) {
                 return c.exponential_map(base, tangent, step_size, max_steps);
             },
             py::arg("base"), py::arg("tangent_vec"),
             py::arg("step_size") = 1e-3, py::arg("max_steps") = 1000,
             "Compute the exponential map exp_base(tangent_vec)")
        .def("log_map",
             [](const Chart& c, const ManifoldPoint& base,
                const ManifoldPoint& target,
                Scalar tolerance, int max_iterations) {
                 return c.log_map(base, target, tolerance, max_iterations);
             },
             py::arg("base"), py::arg("target"),
             py::arg("tolerance") = 1e-8, py::arg("max_iterations") = 100,
             "Compute the logarithmic map log_base(target)")
        .def("contains", &Chart::contains, py::arg("local_coords"),
             "Check whether a point lies within the chart's domain")
        .def("id", &Chart::id,
             "Chart identifier")
        .def("intrinsic_dim", &Chart::intrinsic_dim,
             "Intrinsic dimension d of the chart")
        .def("ambient_dim", &Chart::ambient_dim,
             "Ambient dimension D of the embedding")
        .def("type", &Chart::type,
             "Chart type enum (LINEAR, NEURAL, PARAMETRIC, CUSTOM)");

    // ═══════════════════════════════════════════════════════════════════════════
    //  LinearChart
    // ═══════════════════════════════════════════════════════════════════════════

    py::class_<LinearChart, Chart, std::shared_ptr<LinearChart>>(
            m, "LinearChart",
        "Affine (PCA) chart: phi(x) = origin + basis * x.\n"
        "basis is D x d (ambient_dim x intrinsic_dim), origin is D-vector.")
        .def(py::init<uint32_t, const Matrix&, const Vector>(),
             py::arg("id"), py::arg("basis"), py::arg("origin"),
             "Construct a linear chart with orthonormal basis and origin.")
        .def("basis", &LinearChart::basis,
             py::return_value_policy::reference_internal,
             "Basis matrix B (D x d)")
        .def("origin", &LinearChart::origin,
             py::return_value_policy::reference_internal,
             "Origin vector (D)")
        .def("projection_residual", &LinearChart::projection_residual,
             py::arg("ambient_coords"),
             "Residual of projecting an ambient point onto this chart's plane");

    // ═══════════════════════════════════════════════════════════════════════════
    //  ParametricChart
    // ═══════════════════════════════════════════════════════════════════════════

    py::class_<ParametricChart, Chart, std::shared_ptr<ParametricChart>>(
            m, "ParametricChart",
        "Chart with user-supplied Python callbacks for embedding, "
        "projection, and Jacobian computation.")
        .def(py::init<uint32_t, uint32_t, uint32_t,
                      ParametricChart::EmbedFunc,
                      ParametricChart::ProjectFunc,
                      ParametricChart::JacobianFunc>(),
             py::arg("id"), py::arg("intrinsic_dim"), py::arg("ambient_dim"),
             py::arg("embed_fn"), py::arg("project_fn"), py::arg("jacobian_fn"),
             "Construct with callback functions for embed, project, jacobian.");

    // ═══════════════════════════════════════════════════════════════════════════
    //  MetricTensor
    // ═══════════════════════════════════════════════════════════════════════════

    py::class_<MetricTensor, std::shared_ptr<MetricTensor>>(
            m, "MetricTensor",
        "Riemannian metric tensor field g_ij(x) on a single chart.\n"
        "Supports constant or RBF-interpolated metrics.")
        .def(py::init<uint32_t, uint32_t>(),
             py::arg("chart_id"), py::arg("dim"),
             "Construct a metric tensor for a chart (initially identity).")
        .def("evaluate", &MetricTensor::evaluate,
             py::arg("local_coords"),
             "Evaluate g_ij(x) -> d x d symmetric positive-definite matrix")
        .def("inverse", &MetricTensor::inverse,
             py::arg("local_coords"),
             "Evaluate inverse metric g^{ij}(x)")
        .def("christoffel_symbols", &MetricTensor::christoffel_symbols,
             py::arg("local_coords"), py::arg("h") = 1e-5,
             "Christoffel symbols (2nd kind) Gamma^k_{ij} via finite differences")
        .def("sectional_curvature", &MetricTensor::sectional_curvature,
             py::arg("u"), py::arg("v"),
             "Sectional curvature K(u, v)")
        .def("scalar_curvature", &MetricTensor::scalar_curvature,
             py::arg("local_coords"), py::arg("h") = 1e-5,
             "Scalar curvature S = g^{ij} R_{ij}")
        .def("update", &MetricTensor::update,
             py::arg("local_coords"), py::arg("local_metric"),
             py::arg("weight") = 1.0,
             "Add an anchor point to the RBF interpolation")
        .def("set_constant", &MetricTensor::set_constant,
             py::arg("metric"),
             "Set a constant position-independent metric")
        .def("set_identity", &MetricTensor::set_identity,
             "Reset metric to the identity matrix")
        .def("clear", &MetricTensor::clear,
             "Clear all anchor points and reset to identity")
        .def("is_constant", &MetricTensor::is_constant,
             "True if the metric is constant (no anchors)")
        .def("chart_id", &MetricTensor::chart_id,
             "Chart this metric belongs to")
        .def("dim", &MetricTensor::dim,
             "Intrinsic dimension d")
        .def("num_anchors", &MetricTensor::num_anchors,
             "Number of RBF interpolation anchors");

    // ═══════════════════════════════════════════════════════════════════════════
    //  MetricStore
    // ═══════════════════════════════════════════════════════════════════════════

    py::class_<MetricStore, std::shared_ptr<MetricStore>>(
            m, "MetricStore",
        "Thread-safe persistent storage and caching layer for MetricTensors.")
        .def(py::init<const std::string&>(),
             py::arg("db_path"),
             "Construct a metric store backed by a filesystem directory.")
        .def("get_metric", &MetricStore::get_metric,
             py::arg("chart_id"),
             py::return_value_policy::reference_internal,
             "Get a cached or loaded metric tensor (None if not found).")
        .def("create_metric", &MetricStore::create_metric,
             py::arg("chart_id"), py::arg("dim"),
             "Create a new identity metric for the chart.")
        .def("commit", &MetricStore::commit,
             py::arg("chart_id"), py::arg("metric"),
             "Persist a metric tensor to storage and update cache.")
        .def("batch_evaluate", &MetricStore::batch_evaluate,
             py::arg("chart_id"), py::arg("points"),
             "Evaluate the metric at multiple points on the same chart.")
        .def("num_charts", &MetricStore::num_charts,
             "Number of charts with cached metrics.")
        .def("has_chart", &MetricStore::has_chart,
             py::arg("chart_id"),
             "Check whether a metric is cached for a chart.")
        .def("flush", &MetricStore::flush,
             "Flush all cached metrics to persistent storage.");

    // ═══════════════════════════════════════════════════════════════════════════
    //  SolverConfig
    // ═══════════════════════════════════════════════════════════════════════════

    py::class_<SolverConfig>(m, "SolverConfig",
        "Tunable parameters for the geodesic solver.")
        .def(py::init<>())
        .def_readwrite("initial_step", &SolverConfig::initial_step)
        .def_readwrite("min_step", &SolverConfig::min_step)
        .def_readwrite("max_step", &SolverConfig::max_step)
        .def_readwrite("tolerance", &SolverConfig::tolerance)
        .def_readwrite("max_iterations", &SolverConfig::max_iterations)
        .def_readwrite("max_bvp_iterations", &SolverConfig::max_bvp_iterations)
        .def_readwrite("bvp_tolerance", &SolverConfig::bvp_tolerance)
        .def_readwrite("adaptive_step", &SolverConfig::adaptive_step);

    // ═══════════════════════════════════════════════════════════════════════════
    //  GeodesicSolver
    // ═══════════════════════════════════════════════════════════════════════════

    py::class_<GeodesicSolver>(m, "GeodesicSolver",
        "Geodesic equation solver supporting IVP, BVP, parallel transport, "
        "and distance computation.")
        .def(py::init<std::shared_ptr<MetricStore>, SolverConfig>(),
             py::arg("metric_store"), py::arg("config") = SolverConfig(),
             "Construct a solver backed by a metric store.")
        .def("solve_ivp", &GeodesicSolver::solve_ivp,
             py::arg("start"), py::arg("initial_velocity"),
             py::arg("t_max") = 1.0, py::arg("method") = SolverType::RK45,
             "Solve the geodesic initial value problem.")
        .def("solve_bvp", &GeodesicSolver::solve_bvp,
             py::arg("start"), py::arg("end"),
             py::arg("method") = SolverType::SHOOTING,
             "Solve the geodesic boundary value problem (shooting method).")
        .def("parallel_transport", &GeodesicSolver::parallel_transport,
             py::arg("path"), py::arg("vector_at_start"),
             "Levi-Civita parallel-transport a vector along a geodesic path.")
        .def("geodesic_distance", &GeodesicSolver::geodesic_distance,
             py::arg("p"), py::arg("q"),
             "Compute geodesic distance between two manifold points.")
        .def("batch_geodesic_distance",
             &GeodesicSolver::batch_geodesic_distance,
             py::arg("chart_id"), py::arg("query_local"),
             py::arg("candidates_local"),
             "Batch geodesic distances from a query to multiple candidates.")
        .def("config", &GeodesicSolver::config,
             py::return_value_policy::reference,
             "Get the solver configuration (mutable reference).")
        .def("set_config", &GeodesicSolver::set_config,
             py::arg("cfg"),
             "Update the solver configuration.");

    // ═══════════════════════════════════════════════════════════════════════════
    //  TangentSpaceIndex
    // ═══════════════════════════════════════════════════════════════════════════

    py::class_<TangentSpaceIndex>(m, "TangentSpaceIndex",
        "R-tree spatial index over the tangent bundle for fast local search.\n"
        "Provides k-NN and range queries in local chart coordinates.")
        .def(py::init<uint32_t, uint32_t, size_t>(),
             py::arg("chart_id"), py::arg("intrinsic_dim"),
             py::arg("max_leaf_size") = 16,
             "Construct an empty index for a chart.")
        .def("insert", &TangentSpaceIndex::insert,
             py::arg("point"),
             "Insert a single manifold point into the index.")
        .def("build", &TangentSpaceIndex::build,
             py::arg("points"),
             "Bulk-build the index from a list of manifold points (STR packing).")
        .def("knn", &TangentSpaceIndex::knn,
             py::arg("query_local"), py::arg("k"),
             py::arg("max_radius") = std::numeric_limits<Scalar>::infinity(),
             "k-nearest neighbour search in local coordinates.")
        .def("knn_tangent", &TangentSpaceIndex::knn_tangent,
             py::arg("query_local"), py::arg("k"),
             "Alias for knn() – k-NN search in tangent space coordinates.")
        .def("range_search", &TangentSpaceIndex::range_search,
             py::arg("query_local"), py::arg("radius"),
             "Find all points within a Euclidean radius in local coordinates.")
        .def("clear", &TangentSpaceIndex::clear,
             "Remove all points and reset the index.")
        .def("size", &TangentSpaceIndex::size,
             "Number of indexed points.")
        .def("is_built", &TangentSpaceIndex::is_built,
             "Whether the index has been built.")
        .def("chart_id", &TangentSpaceIndex::chart_id,
             "Chart ID this index belongs to.")
        .def("save", &TangentSpaceIndex::save,
             py::arg("path"),
             "Serialize indexed points to a binary file.")
        .def("load", &TangentSpaceIndex::load,
             py::arg("path"),
             "Load points from a binary file and rebuild the index.");

    // ═══════════════════════════════════════════════════════════════════════════
    //  TransitionMap (abstract base)
    // ═══════════════════════════════════════════════════════════════════════════

    py::class_<TransitionMap, PyTransitionMap>(
            m, "TransitionMap",
        "Abstract base for coordinate transformations between overlapping charts.\n"
        "Subclass in Python or use LinearTransitionMap for affine transforms.")
        .def_readwrite("from_chart", &TransitionMap::from_chart,
                       "Source chart ID")
        .def_readwrite("to_chart", &TransitionMap::to_chart,
                       "Target chart ID")
        .def_readwrite("is_identity", &TransitionMap::is_identity,
                       "Shortcut: identity map flag")
        .def("forward", &TransitionMap::forward,
             py::arg("coords_a"),
             "Forward transition: x_beta = psi(x_alpha)")
        .def("inverse", &TransitionMap::inverse,
             py::arg("coords_b"),
             "Inverse transition: x_alpha = psi^{-1}(x_beta)")
        .def("jacobian", &TransitionMap::jacobian,
             py::arg("coords_a"),
             "Jacobian of forward transition: d x_beta / d x_alpha")
        .def("in_overlap", &TransitionMap::in_overlap,
             py::arg("coords_a"),
             "Check whether a point lies within the overlap region.");

    // ═══════════════════════════════════════════════════════════════════════════
    //  LinearTransitionMap
    // ═══════════════════════════════════════════════════════════════════════════

    py::class_<LinearTransitionMap, TransitionMap>(
            m, "LinearTransitionMap",
        "Affine transition map: x_beta = R * x_alpha + t.\n"
        "Derived from two LinearCharts sharing the same ambient space.")
        .def(py::init<uint32_t, uint32_t, const Matrix&, const Vector>(),
             py::arg("from"), py::arg("to"), py::arg("rotation"),
             py::arg("translation"),
             "Construct with source/target chart IDs, rotation R, and translation t.")
        .def_readwrite("rotation", &LinearTransitionMap::rotation,
                       "Rotation matrix R ∈ R^{d x d}")
        .def_readwrite("translation", &LinearTransitionMap::translation,
                       "Translation vector t ∈ R^d");

    // ═══════════════════════════════════════════════════════════════════════════
    //  Atlas
    // ═══════════════════════════════════════════════════════════════════════════

    py::class_<Atlas>(m, "Atlas",
        "Manages a collection of charts covering the manifold M, "
        "with transition maps for coordinate transport.")
        .def(py::init<>(), "Construct an empty atlas.")
        // ── Chart management ────────────────────────────────────────────────
        .def("add_chart", &Atlas::add_chart,
             py::arg("chart"),
             "Register a chart in the atlas.")
        .def("add_transition", &Atlas::add_transition,
             py::arg("transition"),
             "Register a transition map between two charts.")
        // ── Chart location ──────────────────────────────────────────────────
        .def("locate_chart", &Atlas::locate_chart,
             py::arg("ambient_coords"),
             py::return_value_policy::reference,
             "Find the best chart for an ambient point (min projection residual).")
        // ── Coordinate transport ─────────────────────────────────────────────
        .def("transport", &Atlas::transport,
             py::arg("point"), py::arg("target_chart_id"),
             "Transport a manifold point to a target chart via multi-hop BFS.")
        // ── Overlap & lookup ──────────────────────────────────────────────────
        .def("charts_overlap", &Atlas::charts_overlap,
             py::arg("id_a"), py::arg("id_b"),
             "Check whether two charts have overlapping coverage.")
        .def("get_chart", &Atlas::get_chart,
             py::arg("chart_id"),
             "Get a chart by its ID (shared_ptr or None).")
        .def("get_transition", &Atlas::get_transition,
             py::arg("from_id"), py::arg("to_id"),
             py::return_value_policy::reference,
             "Get a transition map from chart A to chart B (None if not found).")
        .def("find_path", &Atlas::find_path,
             py::arg("from_id"), py::arg("to_id"),
             "BFS shortest multi-hop path between charts.")
        // ── Accessors ─────────────────────────────────────────────────────────
        .def("num_charts", &Atlas::num_charts,
             "Number of charts in the atlas.")
        .def("charts", &Atlas::charts,
             py::return_value_policy::reference_internal,
             "List of all chart shared_ptrs.")
        // ── Chart discovery ──────────────────────────────────────────────────
        .def("discover_charts_linear",
             &Atlas::discover_charts_linear,
             py::arg("data"),
             py::arg("target_intrinsic_dim"),
             py::arg("num_charts_target") = 0,
             py::arg("overlap_threshold") = 0.1,
             "Auto-discover linear charts via PCA on ambient data.\n"
             "data: D x n matrix (column-major, each column = one point).")
        .def("discover_charts",
             &Atlas::discover_charts,
             py::arg("data"),
             py::arg("target_dim"),
             py::arg("python_callback") = std::function<void(
                 const Matrix&, uint32_t)>(),
             "Placeholder for Python UMAP-based chart discovery.");

    // ═══════════════════════════════════════════════════════════════════════════
    //  ManifoldDB::Config
    // ═══════════════════════════════════════════════════════════════════════════

    py::class_<ManifoldDB::Config>(m, "ManifoldDBConfig",
        "Configuration parameters for ManifoldDB.")
        .def(py::init<>())
        .def_readwrite("storage_path",
                        &ManifoldDB::Config::storage_path,
                        "Base directory for persistent storage")
        .def_readwrite("default_intrinsic_dim",
                        &ManifoldDB::Config::default_intrinsic_dim,
                        "Default intrinsic dimension d")
        .def_readwrite("enable_cuda",
                        &ManifoldDB::Config::enable_cuda,
                        "Enable GPU-accelerated geodesics")
        .def_readwrite("geodesic_tolerance",
                        &ManifoldDB::Config::geodesic_tolerance,
                        "Geodesic solver convergence tolerance")
        .def_readwrite("solver_config",
                        &ManifoldDB::Config::solver_config,
                        "Solver tuning parameters")
        .def_readwrite("index_max_leaf_size",
                        &ManifoldDB::Config::index_max_leaf_size,
                        "R-tree leaf capacity for tangent-space index");

    // ═══════════════════════════════════════════════════════════════════════════
    //  ManifoldDB::Stats
    // ═══════════════════════════════════════════════════════════════════════════

    py::class_<ManifoldDB::Stats>(m, "ManifoldDBStats",
        "Runtime statistics about the database state.")
        .def(py::init<>())
        .def_readwrite("num_charts",
                        &ManifoldDB::Stats::num_charts)
        .def_readwrite("total_points",
                        &ManifoldDB::Stats::total_points)
        .def_readwrite("avg_geodesic_time_ms",
                        &ManifoldDB::Stats::avg_geodesic_time_ms)
        .def_readwrite("index_size",
                        &ManifoldDB::Stats::index_size)
        .def_readwrite("build_time_ms",
                        &ManifoldDB::Stats::build_time_ms)
        .def("__repr__", [](const ManifoldDB::Stats& s) {
            std::ostringstream oss;
            oss << "ManifoldDBStats(num_charts=" << s.num_charts
                << ", total_points=" << s.total_points
                << ", index_size=" << s.index_size
                << ", build_time_ms=" << s.build_time_ms
                << ")";
            return oss.str();
        });

    // ═══════════════════════════════════════════════════════════════════════════
    //  ManifoldDB (top-level API)
    // ═══════════════════════════════════════════════════════════════════════════

    py::class_<ManifoldDB>(m, "ManifoldDB",
        "Top-level database for manifold-structured data.\n\n"
        "Usage pattern:\n"
        "  1. Construct with a Config\n"
        "  2. Insert ambient-space data points (list of vectors or torch.Tensor)\n"
        "  3. Build atlas (automatic chart decomposition)\n"
        "  4. Query using geodesic k-NN or ball search")
        .def(py::init<const ManifoldDB::Config&>(),
             py::arg("config"),
             "Construct a ManifoldDB instance with the given configuration.")

        // ── Data ingestion: std::vector<Vector> ───────────────────────────────
        .def("insert",
             static_cast<void (ManifoldDB::*)(const std::vector<Vector>&,
                                              uint32_t)>(
                 &ManifoldDB::insert),
             py::arg("ambient_points"), py::arg("modality_id") = 0,
             "Insert a list of ambient-space vectors.")
        .def("insert",
             static_cast<void (ManifoldDB::*)(const Matrix&, uint32_t)>(
                 &ManifoldDB::insert),
             py::arg("ambient_matrix"), py::arg("modality_id") = 0,
             "Insert points as a column-major Eigen matrix (each column = one point).")

        // ── Data ingestion: torch::Tensor [N, D] ──────────────────────────
        .def("insert",
             [](ManifoldDB& self,
                const torch::Tensor& tensor,
                uint32_t modality_id) {
                 auto points = tensor_to_vectors(tensor);
                 self.insert(points, modality_id);
             },
             py::arg("tensor"), py::arg("modality_id") = 0,
             "Insert points from a PyTorch tensor of shape [N, D] "
             "(float32 or float64, contiguous on CPU).")

        // ── Atlas construction ────────────────────────────────────────────────
        .def("build_atlas", &ManifoldDB::build_atlas,
             py::arg("target_intrinsic_dim") = 0,
             "Build atlas with automatic intrinsic dimension detection.")
        .def("build_atlas_linear", &ManifoldDB::build_atlas_linear,
             py::arg("intrinsic_dim"),
             "Build atlas using PCA-based linear charts with given dimension.")

        // ── Geodesic queries ──────────────────────────────────────────────────
        .def("query_geodesic_knn", &ManifoldDB::query_geodesic_knn,
             py::arg("query_ambient"), py::arg("k"),
             py::arg("max_distance") = std::numeric_limits<Scalar>::infinity(),
             "k-nearest neighbours by geodesic distance.")
        .def("query_geodesic_ball", &ManifoldDB::query_geodesic_ball,
             py::arg("center_ambient"), py::arg("radius"),
             "All points within a geodesic radius.")
        .def("query_geodesic_path", &ManifoldDB::query_geodesic_path,
             py::arg("start_ambient"), py::arg("end_ambient"),
             "Compute the geodesic path between two ambient points.")

        // ── Cross-modal queries ─────────────────────────────────────────────
        .def("query_cross_modal", &ManifoldDB::query_cross_modal,
             py::arg("query_ambient"),
             py::arg("source_modality"),
             py::arg("target_modality"),
             py::arg("k"),
             "Cross-modal nearest-neighbour search.")

        // ── Schema evolution: std::vector<Vector> ────────────────────────────
        .def("evolve_schema",
             static_cast<void (ManifoldDB::*)(
                 const std::vector<Vector>&)>(
                 &ManifoldDB::evolve_schema),
             py::arg("new_ambient_points"),
             "Extend the manifold structure to accommodate new data "
             "(std::vector<Vector>).")

        // ── Schema evolution: torch::Tensor [N, D] ──────────────────────────
        .def("evolve_schema",
             [](ManifoldDB& self, const torch::Tensor& tensor) {
                 auto points = tensor_to_vectors(tensor);
                 self.evolve_schema(points);
             },
             py::arg("new_ambient_tensor"),
             "Extend the manifold structure to accommodate new data "
             "(torch::Tensor [N, D], float32 or float64).")

        // ── Utility ──────────────────────────────────────────────────────────
        .def("stats", &ManifoldDB::stats,
             "Collect runtime statistics about the database.")

        // ── Direct access (advanced use) ─────────────────────────────────────
        .def("atlas",
             static_cast<Atlas& (ManifoldDB::*)()>(&ManifoldDB::atlas),
             py::return_value_policy::reference,
             "Direct access to the atlas.")
        .def("metric_store",
             static_cast<MetricStore& (ManifoldDB::*)()>(
                 &ManifoldDB::metric_store),
             py::return_value_policy::reference,
             "Direct access to the metric store.")
        .def("solver",
             static_cast<const GeodesicSolver& (ManifoldDB::*)() const>(
                 &ManifoldDB::solver),
             py::return_value_policy::reference,
             "Direct access to the geodesic solver (const).");
}
