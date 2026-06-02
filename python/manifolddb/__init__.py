"""
ManifoldDB — Riemannian Geometric Inference Engine
==================================================

A Python library for manifold-aware data management over Riemannian manifolds.
Provides data ingestion, atlas construction, geodesic queries, cross-modal search,
metric tensor management, and schema evolution.

The C++ core is exposed via the ``manifolddb_core`` extension module, and this
package wraps it with high-level helpers for numpy and PyTorch interop.

Typical usage::

    import manifolddb as mdb
    import numpy as np

    db = mdb.ManifoldDB("./my_manifold", intrinsic_dim=3)
    data = np.random.randn(500, 10)           # 500 points in R^10
    db.insert(data)
    db.build(method='linear')

    results = db.query_knn(data[0], k=5)
    for r in results:
        print(f"  id={r['id']}  dist={r['distance']:.4f}")

    # Cross-modal retrieval
    db.insert(text_embeddings, modality_id=1)
    hits = db.cross_modal_query(query_vec, source_modality=0,
                                 target_modality=1, k=10)

    # Geodesic path
    path = db.geodesic_path(data[0], data[1])
    print(f"Geodesic length = {path['total_length']:.4f}")
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# C++ extension module
# ---------------------------------------------------------------------------
try:
    from manifolddb import manifolddb_core as _core  # type: ignore[import-not-found]
except ImportError:
    try:
        import manifolddb_core as _core  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "Could not import the C++ extension 'manifolddb_core'.  "
            "Make sure the ManifoldDB package is built correctly.  "
            "See the installation instructions for details."
        ) from exc

# ---------------------------------------------------------------------------
# Re-export C++ enums & exceptions
# ---------------------------------------------------------------------------
SolverType = _core.SolverType
DistanceType = _core.DistanceType
ChartType = _core.ChartType

ChartNotFoundError = _core.ChartNotFoundError
GeodesicSolverError = _core.GeodesicSolverError
IndexBuildError = _core.IndexBuildError
SerializationError = _core.SerializationError
DimensionMismatchError = _core.DimensionMismatchError

# Re-export low-level C++ classes for advanced users
Config = _core.Config
Stats = _core.Stats
SolverConfig = _core.SolverConfig
Atlas = _core.Atlas
TransitionMap = _core.TransitionMap
LinearChart = _core.LinearChart
ParametricChart = _core.ParametricChart
MetricTensor = _core.MetricTensor
MetricStore = _core.MetricStore
GeodesicSolver = _core.GeodesicSolver
TangentSpaceIndex = _core.TangentSpaceIndex

# ---------------------------------------------------------------------------
# Array conversion helpers (used internally, also exported)
# ---------------------------------------------------------------------------

VectorLike = Union["np.ndarray", Sequence[float], "Any"]

Array2D = Union["Any", Sequence[Sequence[float]]]


def _ensure_numpy_float64(array):
    """Convert *array* to a contiguous float64 numpy array.

    Handles numpy arrays, PyTorch tensors, and plain Python sequences.
    """
    import numpy as np

    if hasattr(array, "detach"):
        # PyTorch tensor – move to CPU, convert to numpy
        array = array.detach().cpu()
    if hasattr(array, "numpy"):
        array = array.numpy()  # type: ignore[attr-defined]
    arr = np.asarray(array, dtype=np.float64)
    return np.ascontiguousarray(arr)


def torch_to_eigen(tensor):
    """Convert a :class:`torch.Tensor` to an Eigen-compatible numpy reference.

    Parameters
    ----------
    tensor : torch.Tensor
        Source tensor.  If on GPU it will be moved to CPU first.

    Returns
    -------
    numpy.ndarray
        A contiguous float64 numpy array that pybind11 can pass as an
        ``Eigen::VectorXd`` or ``Eigen::MatrixXd`` reference.
    """
    import numpy as np

    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "numpy"):
        tensor = tensor.numpy()
    arr = np.asarray(tensor, dtype=np.float64)
    return np.ascontiguousarray(arr)


def numpy_to_eigen(array):
    """Convert a numpy array to an Eigen-compatible numpy reference.

    Parameters
    ----------
    array : array_like
        Source array.  Will be cast to float64.

    Returns
    -------
    numpy.ndarray
        Contiguous float64 numpy array suitable for Eigen interop.
    """
    import numpy as np

    arr = np.asarray(array, dtype=np.float64)
    return np.ascontiguousarray(arr)


def eigen_to_torch(vector):
    """Convert an Eigen vector (numpy array from C++) to a :class:`torch.Tensor`.

    Parameters
    ----------
    vector : numpy.ndarray
        1-D or 2-D float64 numpy array.

    Returns
    -------
    torch.Tensor
        Float64 CPU tensor with the same data.
    """
    import numpy as np

    import torch

    arr = np.asarray(vector, dtype=np.float64)
    return torch.from_numpy(np.ascontiguousarray(arr)).clone()


def eigen_to_numpy(vector):
    """Convert an Eigen vector (returned from C++) to a numpy array.

    This is essentially a no-op copy that guarantees a contiguous float64 array.

    Parameters
    ----------
    vector : numpy.ndarray or array_like

    Returns
    -------
    numpy.ndarray
    """
    import numpy as np

    arr = np.asarray(vector, dtype=np.float64)
    return np.ascontiguousarray(arr.copy())


# ---------------------------------------------------------------------------
# ManifoldPoint  –  Pythonic wrapper
# ---------------------------------------------------------------------------


class ManifoldPoint:
    """A point on the Riemannian manifold with dual representation.

    Provides numpy-friendly access to ``local_coords`` and ``ambient_coords``
    without requiring direct interaction with the C++ extension.

    Parameters
    ----------
    chart_id : int
        Home chart identifier.
    local_coords : array-like, shape ``(d,)``
        Local chart coordinates x ∈ R^d.
    ambient_coords : array-like, shape ``(D,)``
        Ambient embedding coordinates y ∈ R^D.
    global_id : int
        Unique identifier across the database.
    timestamp : float
        Insertion / modification time.
    """

    __slots__ = ("_cpp", "_local", "_ambient")

    def __init__(
        self,
        local_coords: Optional[VectorLike] = None,
        ambient_coords: Optional[VectorLike] = None,
        chart_id: int = 0,
        global_id: int = 0,
        timestamp: float = 0.0,
    ) -> None:
        import numpy as np

        lc = (
            _ensure_numpy_float64(local_coords)
            if local_coords is not None
            else np.array([], dtype=np.float64)
        )
        ac = (
            _ensure_numpy_float64(ambient_coords)
            if ambient_coords is not None
            else np.array([], dtype=np.float64)
        )
        self._cpp = _core.ManifoldPoint(
            chart_id, lc, ac, global_id, timestamp
        )
        self._local = lc.copy()
        self._ambient = ac.copy()

    # -- Properties ----------------------------------------------------------

    @property
    def chart_id(self) -> int:
        """Home chart identifier."""
        return int(self._cpp.chart_id)

    @property
    def global_id(self) -> int:
        """Unique identifier across the database."""
        return int(self._cpp.global_id)

    @property
    def local_coords(self):
        """Local chart coordinates x ∈ R^d as a numpy array."""
        return self._local

    @local_coords.setter
    def local_coords(self, value: VectorLike) -> None:
        self._local = _ensure_numpy_float64(value)
        self._cpp.local_coords = self._local

    @property
    def ambient_coords(self):
        """Ambient embedding coordinates y ∈ R^D as a numpy array."""
        return self._ambient

    @ambient_coords.setter
    def ambient_coords(self, value: VectorLike) -> None:
        self._ambient = _ensure_numpy_float64(value)
        self._cpp.ambient_coords = self._ambient

    @property
    def timestamp(self) -> float:
        """Insertion / modification time."""
        return float(self._cpp.timestamp)

    @timestamp.setter
    def timestamp(self, value: float) -> None:
        self._cpp.timestamp = value

    # -- Methods ------------------------------------------------------------

    def local_norm(self) -> float:
        """Euclidean norm of the local coordinate vector: ||x||_2."""
        return float(self._cpp.local_norm())

    def ambient_norm(self) -> float:
        """Euclidean norm of the ambient coordinate vector: ||y||_2."""
        return float(self._cpp.ambient_norm())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary."""
        return {
            "id": self.global_id,
            "chart_id": self.chart_id,
            "timestamp": self.timestamp,
            "local_coords": self._local.copy(),
            "ambient_coords": self._ambient.copy(),
        }

    def __repr__(self) -> str:
        return (
            f"ManifoldPoint(id={self.global_id}, chart={self.chart_id}, "
            f"local_dim={self._local.size}, ambient_dim={self._ambient.size})"
        )


# ---------------------------------------------------------------------------
# GeodesicPath  –  Pythonic wrapper
# ---------------------------------------------------------------------------


class GeodesicPath:
    """Discrete approximation of a geodesic curve between two manifold points.

    Provides numpy-friendly access to the sampled ``points`` and cumulative
    ``arc_lengths``.

    Parameters
    ----------
    points : list of ManifoldPoint
        Sampled points along the geodesic.
    arc_lengths : list of float
        Cumulative arc lengths at each sample point.
    total_length : float
        Total geodesic arc length.
    converged : bool
        Whether the solver converged.
    num_steps : int
        Number of integration steps taken.
    """

    __slots__ = ("_total_length", "_converged", "_num_steps",
                 "_points", "_arc_lengths")

    def __init__(
        self,
        points: Optional[List[Dict[str, Any]]] = None,
        arc_lengths: Optional[List[float]] = None,
        total_length: float = 0.0,
        converged: bool = False,
        num_steps: int = 0,
    ) -> None:
        self._total_length = total_length
        self._converged = converged
        self._num_steps = num_steps
        self._points = points or []
        self._arc_lengths = list(arc_lengths) if arc_lengths else []

    @classmethod
    def from_cpp(cls, cpp_path: Any) -> "GeodesicPath":
        """Construct from a C++ ``GeodesicPath`` object."""
        import numpy as np

        points = []
        for p in cpp_path.points:
            points.append({
                "id": int(p.global_id),
                "chart_id": int(p.chart_id),
                "local_coords": eigen_to_numpy(p.local_coords),
                "ambient_coords": eigen_to_numpy(p.ambient_coords),
            })
        arc_lengths = [float(a) for a in cpp_path.arc_lengths]
        return cls(
            points=points,
            arc_lengths=arc_lengths,
            total_length=float(cpp_path.total_length),
            converged=bool(cpp_path.converged),
            num_steps=int(cpp_path.num_steps),
        )

    # -- Properties ----------------------------------------------------------

    @property
    def total_length(self) -> float:
        """Total geodesic arc length."""
        return self._total_length

    @property
    def converged(self) -> bool:
        """Whether the solver converged."""
        return self._converged

    @property
    def num_steps(self) -> int:
        """Number of integration steps taken."""
        return self._num_steps

    @property
    def points(self) -> List[Dict[str, Any]]:
        """Sampled points along the geodesic (list of dicts)."""
        return self._points

    @property
    def arc_lengths(self) -> List[float]:
        """Cumulative arc lengths at each sample point."""
        return self._arc_lengths

    @property
    def is_empty(self) -> bool:
        """True if the path contains no points."""
        return len(self._points) == 0

    @property
    def size(self) -> int:
        """Number of sample points along the path."""
        return len(self._points)

    # -- numpy accessors ------------------------------------------------------

    def points_array(self):
        """Return all ambient coordinates as a numpy array ``(N, D)``."""
        import numpy as np

        if not self._points:
            return np.empty((0,), dtype=np.float64)
        return np.stack([p["ambient_coords"] for p in self._points], axis=0)

    def arc_lengths_array(self):
        """Return arc lengths as a 1-D numpy float64 array."""
        import numpy as np

        return np.array(self._arc_lengths, dtype=np.float64)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary."""
        return {
            "total_length": self._total_length,
            "converged": self._converged,
            "num_steps": self._num_steps,
            "points": self._points,
            "arc_lengths": self._arc_lengths,
        }

    def __repr__(self) -> str:
        return (
            f"GeodesicPath(points={self.size}, "
            f"total_length={self._total_length:.6f}, "
            f"converged={self._converged})"
        )


# ---------------------------------------------------------------------------
# NeighborResult  –  Pythonic wrapper
# ---------------------------------------------------------------------------


class NeighborResult:
    """Result entry from a geodesic k-nearest-neighbour query.

    Stores the neighbour point, geodesic distance, and Euclidean residual
    for quality assessment.

    Parameters
    ----------
    point : ManifoldPoint or dict
        The neighbour point.
    distance : float
        True geodesic distance d_g(p, q).
    euclidean_residual : float
        |d_g(p,q) - ||y_p - y_q|||.
    """

    __slots__ = ("_point", "_distance", "_euclidean_residual")

    def __init__(
        self,
        point: Optional[Dict[str, Any]] = None,
        distance: float = math.inf,
        euclidean_residual: float = 0.0,
    ) -> None:
        self._point = point or {}
        self._distance = distance
        self._euclidean_residual = euclidean_residual

    @classmethod
    def from_cpp(cls, nr: Any) -> "NeighborResult":
        """Construct from a C++ ``NeighborResult``."""
        p = nr.point
        return cls(
            point={
                "id": int(p.global_id),
                "chart_id": int(p.chart_id),
                "timestamp": float(p.timestamp),
                "local_coords": eigen_to_numpy(p.local_coords),
                "ambient_coords": eigen_to_numpy(p.ambient_coords),
            },
            distance=float(nr.geodesic_distance),
            euclidean_residual=float(nr.euclidean_residual),
        )

    # -- Properties ----------------------------------------------------------

    @property
    def point(self) -> Dict[str, Any]:
        """The neighbour point as a dict."""
        return self._point

    @property
    def id(self) -> int:
        """Unique identifier of the neighbour point."""
        return int(self._point.get("id", -1))

    @property
    def chart_id(self) -> int:
        """Chart identifier of the neighbour point."""
        return int(self._point.get("chart_id", 0))

    @property
    def distance(self) -> float:
        """True geodesic distance d_g(p, q)."""
        return self._distance

    @property
    def euclidean_residual(self) -> float:
        """|d_g(p,q) - ||y_p - y_q|||"""
        return self._euclidean_residual

    @property
    def local_coords(self):
        """Local coordinates of the neighbour point (numpy array)."""
        return self._point.get("local_coords")

    @property
    def ambient_coords(self):
        """Ambient coordinates of the neighbour point (numpy array)."""
        return self._point.get("ambient_coords")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary."""
        return {
            "id": self.id,
            "chart_id": self.chart_id,
            "distance": self._distance,
            "euclidean_residual": self._euclidean_residual,
            "local_coords": self.local_coords,
            "ambient_coords": self.ambient_coords,
            "timestamp": self._point.get("timestamp", 0.0),
        }

    def __repr__(self) -> str:
        return (
            f"NeighborResult(id={self.id}, chart={self.chart_id}, "
            f"distance={self._distance:.6f})"
        )

    def __lt__(self, other: "NeighborResult") -> bool:
        if not isinstance(other, NeighborResult):
            return NotImplemented
        return self._distance < other._distance

    def __gt__(self, other: "NeighborResult") -> bool:
        if not isinstance(other, NeighborResult):
            return NotImplemented
        return self._distance > other._distance


# ---------------------------------------------------------------------------
# ManifoldDB  –  High-level Python wrapper
# ---------------------------------------------------------------------------


class ManifoldDB:
    """High-level Python wrapper around the C++ ``ManifoldDB`` engine.

    This class provides a user-friendly interface that accepts numpy arrays
    and PyTorch tensors, returning Python-native types (dicts, lists, wrappers)
    instead of raw C++ objects.

    Parameters
    ----------
    storage_path : str
        Filesystem path for persistent storage (metrics, indexes, etc.).
    intrinsic_dim : int
        Default intrinsic (manifold) dimension used during atlas construction.
    enable_cuda : bool
        Whether to enable CUDA-accelerated solvers (requires a CUDA build).
    geodesic_tolerance : float
        Tolerance passed to the geodesic solver.
    max_charts : int
        Maximum number of charts to discover during atlas construction.
    rbf_bandwidth : float
        Bandwidth parameter for RBF metric interpolation.
    """

    def __init__(
        self,
        storage_path: str = "./manifolddb_data",
        intrinsic_dim: int = 10,
        enable_cuda: bool = False,
        geodesic_tolerance: float = 1e-6,
        max_charts: int = 20,
        rbf_bandwidth: float = 1.0,
    ) -> None:
        if intrinsic_dim < 1:
            raise ValueError("intrinsic_dim must be >= 1")
        if geodesic_tolerance <= 0:
            raise ValueError("geodesic_tolerance must be > 0")
        if max_charts < 1:
            raise ValueError("max_charts must be >= 1")
        if rbf_bandwidth <= 0:
            raise ValueError("rbf_bandwidth must be > 0")

        cfg = Config()
        cfg.storage_path = storage_path
        cfg.default_intrinsic_dim = intrinsic_dim
        cfg.enable_cuda = enable_cuda
        cfg.geodesic_tolerance = geodesic_tolerance

        self._db = _core.ManifoldDB(cfg)
        self._intrinsic_dim = intrinsic_dim
        self._storage_path = storage_path
        self._max_charts = max_charts
        self._rbf_bandwidth = rbf_bandwidth

    # ── Low-level access ────────────────────────────────────────────────────

    @property
    def core(self):
        """Direct access to the underlying C++ ``ManifoldDB`` object.

        Use this when you need fine-grained control not exposed by the
        high-level wrapper.
        """
        return self._db

    @property
    def atlas(self):
        """Direct access to the underlying :class:`Atlas`."""
        return self._db.atlas()

    @property
    def metric_store(self):
        """Direct access to the underlying :class:`MetricStore`."""
        return self._db.metric_store()

    @property
    def solver(self):
        """Direct access to the underlying :class:`GeodesicSolver`."""
        return self._db.solver

    # ── Data ingestion ──────────────────────────────────────────────────────

    def insert(
        self,
        points: Array2D,
        modality_id: int = 0,
    ) -> None:
        """Insert data points into the database.

        Parameters
        ----------
        points : array-like, shape ``(N, D)``
            Ambient-space data.  Accepts a numpy array (float32 / float64),
            a PyTorch tensor, or a list-of-lists.  Each row is one D-dimensional
            point.
        modality_id : int
            Modality identifier (default 0).  Use different values for
            multi-modal data (e.g. text embeddings vs. image embeddings).

        Raises
        ------
        ValueError
            If *points* is not 1-D or 2-D, or is empty.
        """
        import numpy as np

        arr = _ensure_numpy_float64(points)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(
                f"Expected a 1-D or 2-D array, got {arr.ndim}-D"
            )
        if arr.shape[0] == 0:
            raise ValueError("Cannot insert zero points")
        if arr.shape[1] == 0:
            raise ValueError("Points must have at least one dimension")

        # The C++ binding accepts a column-major Eigen matrix (each column
        # is one point).  Transpose the (N, D) numpy array to get (D, N).
        matrix = np.ascontiguousarray(arr.T)
        self._db.insert(matrix, modality_id)

    # ── Atlas construction ─────────────────────────────────────────────────

    def build(self, method: str = "linear") -> None:
        """Build the atlas from inserted data.

        Parameters
        ----------
        method : str
            Atlas construction method.  Currently supported:

            - ``'linear'`` — PCA-based linear charts (default)
            - ``'pca'`` — Alias for ``'linear'``

        Raises
        ------
        ValueError
            If *method* is not recognised.
        RuntimeError
            If no data has been inserted yet.
        """
        method = method.lower()
        if method in ("linear", "pca"):
            self._db.build_atlas_linear(self._intrinsic_dim)
        else:
            raise ValueError(
                f"Unknown build method '{method}'. "
                f"Supported: 'linear', 'pca'"
            )

    def build_atlas_linear(
        self, intrinsic_dim: Optional[int] = None
    ) -> None:
        """Build the atlas using PCA-based linear charts.

        Parameters
        ----------
        intrinsic_dim : int or None
            Target intrinsic dimension.  If *None*, uses the value supplied at
            construction.
        """
        dim = intrinsic_dim if intrinsic_dim is not None else self._intrinsic_dim
        if dim < 1:
            raise ValueError("intrinsic_dim must be >= 1")
        self._db.build_atlas_linear(dim)

    # ── Geodesic queries ───────────────────────────────────────────────────

    def query_knn(
        self,
        query: Array2D,
        k: int = 10,
        max_distance: float = math.inf,
    ) -> List[Dict[str, Any]]:
        """K-nearest neighbours by geodesic distance.

        Parameters
        ----------
        query : array-like, shape ``(D,)`` or ``(1, D)``
            Query point in ambient space.
        k : int
            Number of nearest neighbours to retrieve.
        max_distance : float
            Exclude neighbours farther than this geodesic distance.

        Returns
        -------
        list[dict]
            Each dict has keys ``id``, ``chart_id``, ``distance``,
            ``euclidean_residual``, ``local_coords``, ``ambient_coords``.

        Raises
        ------
        ValueError
            If *k* < 1.
        """
        if k < 1:
            raise ValueError("k must be >= 1")

        q = _ensure_numpy_float64(query).ravel()
        results_cpp = self._db.query_geodesic_knn(q, k, max_distance)
        return [self._nr_to_dict(nr) for nr in results_cpp]

    def query_ball(
        self,
        center: Array2D,
        radius: float,
    ) -> List[Dict[str, Any]]:
        """All points within a geodesic ball.

        Parameters
        ----------
        center : array-like, shape ``(D,)`` or ``(1, D)``
            Centre of the ball in ambient space.
        radius : float
            Geodesic radius.

        Returns
        -------
        list[dict]
            Each dict has keys ``id``, ``chart_id``, ``local_coords``,
            ``ambient_coords``.

        Raises
        ------
        ValueError
            If *radius* < 0.
        """
        if radius < 0:
            raise ValueError("radius must be >= 0")

        c = _ensure_numpy_float64(center).ravel()
        points_cpp = self._db.query_geodesic_ball(c, radius)
        return [self._mp_to_dict(mp) for mp in points_cpp]

    def cross_modal_query(
        self,
        query: Array2D,
        source_modality: int,
        target_modality: int,
        k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search one modality using a query from another modality.

        Parameters
        ----------
        query : array-like, shape ``(D,)``
            Query point (from *source_modality*) in ambient space.
        source_modality : int
            Modality ID of the query.
        target_modality : int
            Modality ID to search in.
        k : int
            Number of results.

        Returns
        -------
        list[dict]
            Each dict has keys ``id``, ``chart_id``, ``distance``,
            ``euclidean_residual``, ``local_coords``, ``ambient_coords``.

        Raises
        ------
        ValueError
            If *k* < 1 or modalities are the same (use :meth:`query_knn` instead).
        """
        if k < 1:
            raise ValueError("k must be >= 1")

        q = _ensure_numpy_float64(query).ravel()
        results_cpp = self._db.query_cross_modal(
            q, source_modality, target_modality, k
        )
        return [self._nr_to_dict(nr) for nr in results_cpp]

    def geodesic_path(
        self,
        start: Array2D,
        end: Array2D,
        tolerance: float = 1e-6,
    ) -> Dict[str, Any]:
        """Compute the geodesic path between two points.

        Parameters
        ----------
        start : array-like, shape ``(D,)``
            Start point in ambient space.
        end : array-like, shape ``(D,)``
            End point in ambient space.
        tolerance : float
            Solver tolerance for the geodesic BVP.

        Returns
        -------
        dict
            ``total_length``, ``converged``, ``num_steps``, ``points``
            (list of dicts), ``arc_lengths`` (list of floats).
        """
        if tolerance <= 0:
            raise ValueError("tolerance must be > 0")

        s = _ensure_numpy_float64(start).ravel()
        e = _ensure_numpy_float64(end).ravel()

        # Temporarily override solver tolerance
        old_tol = self._db.solver.config().tolerance
        try:
            self._db.solver.config().tolerance = tolerance
            path = self._db.query_geodesic_path(s, e)
        finally:
            self._db.solver.config().tolerance = old_tol

        return {
            "total_length": float(path.total_length),
            "converged": bool(path.converged),
            "num_steps": int(path.num_steps),
            "points": [self._mp_to_dict(p) for p in path.points],
            "arc_lengths": [float(a) for a in path.arc_lengths],
        }

    # ── Schema evolution ───────────────────────────────────────────────────

    def evolve(self, new_data: Array2D) -> None:
        """Extend the manifold to accommodate new data points.

        Internally inserts the data as a new virtual modality and rebuilds
        the atlas.

        Parameters
        ----------
        new_data : array-like, shape ``(N, D)``
        """
        import numpy as np

        arr = _ensure_numpy_float64(new_data)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        matrix = np.ascontiguousarray(arr.T)

        # Build list of column vectors for evolve_schema
        vectors = [matrix[:, i] for i in range(matrix.shape[1])]
        self._db.evolve_schema(vectors)

    # ── Statistics ──────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Return database statistics.

        Returns
        -------
        dict
            ``num_charts``, ``total_points``, ``index_size``,
            ``build_time_ms``, ``avg_geodesic_time_ms``.
        """
        s = self._db.stats()
        return {
            "num_charts": int(s.num_charts),
            "total_points": int(s.total_points),
            "index_size": int(s.index_size),
            "build_time_ms": float(s.build_time_ms),
            "avg_geodesic_time_ms": float(s.avg_geodesic_time_ms),
        }

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def num_charts(self) -> int:
        """Number of charts in the atlas."""
        return int(self._db.atlas().num_charts())

    @property
    def total_points(self) -> int:
        """Total number of points across all modalities."""
        return int(self._db.stats().total_points)

    @property
    def storage_path(self) -> str:
        """Filesystem path used for persistent storage."""
        return self._storage_path

    # ── Dunder helpers ─────────────────────────────────────────────────────

    def __repr__(self) -> str:
        s = self._db.stats()
        return (
            f"ManifoldDB(storage_path={self._storage_path!r}, "
            f"intrinsic_dim={self._intrinsic_dim}, "
            f"num_charts={s.num_charts}, "
            f"total_points={s.total_points})"
        )

    def __len__(self) -> int:
        """Total number of points in the database."""
        return self.total_points

    # ── Internal conversion helpers ─────────────────────────────────────────

    @staticmethod
    def _mp_to_dict(mp: Any) -> Dict[str, Any]:
        """Convert a C++ ``ManifoldPoint`` to a plain dict."""
        return {
            "id": int(mp.global_id),
            "chart_id": int(mp.chart_id),
            "timestamp": float(mp.timestamp),
            "local_coords": eigen_to_numpy(mp.local_coords),
            "ambient_coords": eigen_to_numpy(mp.ambient_coords),
        }

    @staticmethod
    def _nr_to_dict(nr: Any) -> Dict[str, Any]:
        """Convert a C++ ``NeighborResult`` to a plain dict."""
        return {
            "id": int(nr.point.global_id),
            "chart_id": int(nr.point.chart_id),
            "timestamp": float(nr.point.timestamp),
            "distance": float(nr.geodesic_distance),
            "euclidean_residual": float(nr.euclidean_residual),
            "local_coords": eigen_to_numpy(nr.point.local_coords),
            "ambient_coords": eigen_to_numpy(nr.point.ambient_coords),
        }


# ---------------------------------------------------------------------------
# Public API summary
# ---------------------------------------------------------------------------
__all__ = [
    "__version__",
    # High-level wrapper
    "ManifoldDB",
    # Python wrappers
    "ManifoldPoint",
    "GeodesicPath",
    "NeighborResult",
    # Array helpers
    "torch_to_eigen",
    "numpy_to_eigen",
    "eigen_to_torch",
    "eigen_to_numpy",
    "_ensure_numpy_float64",
    # C++ classes (re-exports for advanced use)
    "LinearChart",
    "ParametricChart",
    "MetricTensor",
    "MetricStore",
    "GeodesicSolver",
    "TangentSpaceIndex",
    "Atlas",
    "TransitionMap",
    # Enums
    "SolverType",
    "DistanceType",
    "ChartType",
    # Config / stats
    "Config",
    "Stats",
    "SolverConfig",
    # Exceptions
    "ChartNotFoundError",
    "GeodesicSolverError",
    "IndexBuildError",
    "SerializationError",
    "DimensionMismatchError",
]
