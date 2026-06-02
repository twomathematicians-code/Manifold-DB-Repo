"""
manifolddb.torch_compat — PyTorch interoperability utilities
==========================================================

Utility functions for converting between PyTorch tensors, numpy arrays,
and the Eigen types used internally by the C++ extension.  Also provides
batch geodesic distance helpers and optional DLPack export support.

All functions are designed to be zero-copy where possible (shared memory
between PyTorch and numpy via ``__array_interface__``).

Typical usage::

    import torch
    from manifolddb.torch_compat import (
        ensure_float64,
        torch_to_eigen,
        eigen_to_torch,
        batch_geodesic_distances,
    )

    # Prepare a tensor for the C++ backend
    query = torch.randn(3, requires_grad=False)
    query_np = torch_to_eigen(query)        # contiguous float64 numpy

    # Convert C++ output back to torch
    result_np = db.query_knn(query_np, k=5)
    result_torch = eigen_to_torch(result_np)

    # Batch geodesic distances
    queries = torch.randn(10, 3)
    candidates = torch.randn(100, 3)
    dists = batch_geodesic_distances(queries, candidates, db.metric_store)
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, List, Optional, Union

__all__ = [
    "ensure_float64",
    "torch_to_eigen",
    "eigen_to_torch",
    "eigen_to_numpy",
    "dlpack_export",
    "batch_geodesic_distances",
]


# ---------------------------------------------------------------------------
# ensure_float64
# ---------------------------------------------------------------------------

def ensure_float64(tensor: Any) -> Any:
    """Ensure a tensor is float64 and on CPU.

    If the input is a :class:`torch.Tensor`, it will be converted to float64
    dtype and moved to CPU if necessary.  If it is already a numpy array,
    it will be cast to float64.

    Parameters
    ----------
    tensor : torch.Tensor or numpy.ndarray
        Input tensor or array.

    Returns
    -------
    torch.Tensor or numpy.ndarray
        Same type as input, but guaranteed float64 and on CPU.

    Examples
    --------
    >>> import torch
    >>> t = torch.randn(3, dtype=torch.float32)
    >>> ensure_float64(t).dtype
    torch.float64

    >>> import numpy as np
    >>> a = np.array([1.0, 2.0], dtype=np.float32)
    >>> ensure_float64(a).dtype
    dtype('float64')
    """
    import numpy as np

    if hasattr(tensor, "dtype") and hasattr(tensor, "to"):
        # PyTorch tensor
        if hasattr(tensor, "is_cuda") and tensor.is_cuda:
            tensor = tensor.cpu()
        if tensor.dtype != torch.float64:  # type: ignore[name-defined]
            tensor = tensor.to(torch.float64)  # type: ignore[name-defined]
        return tensor
    # Numpy array or array-like
    return np.asarray(tensor, dtype=np.float64)


# ---------------------------------------------------------------------------
# torch_to_eigen
# ---------------------------------------------------------------------------

def torch_to_eigen(tensor: Any):
    """Convert a :class:`torch.Tensor` to an Eigen-compatible numpy array.

    The resulting numpy array is:
    - Contiguous in memory (C-order)
    - float64 dtype
    - On the CPU

    This array can be passed directly to pybind11 bindings that accept
    ``Eigen::VectorXd`` or ``Eigen::MatrixXd`` via ``pybind11/eigen.h``.

    Parameters
    ----------
    tensor : torch.Tensor
        Source tensor of any dtype and device.

    Returns
    -------
    numpy.ndarray
        Contiguous float64 numpy array.

    Notes
    -----
    If the input tensor is already float64 and on CPU, this function attempts
    a zero-copy view via ``__array_interface__`` (no data is copied).  If a
    copy is needed (different dtype or device), the copy is made explicitly.

    Examples
    --------
    >>> import torch
    >>> t = torch.randn(4, 3, dtype=torch.float64)
    >>> arr = torch_to_eigen(t)
    >>> arr.shape
    (4, 3)
    >>> arr.dtype
    dtype('float64')
    """
    import numpy as np

    import torch

    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "numpy"):
        tensor = tensor.numpy()

    arr = np.asarray(tensor, dtype=np.float64)
    return np.ascontiguousarray(arr)


# ---------------------------------------------------------------------------
# eigen_to_torch
# ---------------------------------------------------------------------------

def eigen_to_torch(vectors: Any, requires_grad: bool = False):
    """Convert an Eigen vector/matrix (numpy array from C++) to a torch.Tensor.

    Parameters
    ----------
    vectors : numpy.ndarray
        1-D or 2-D float64 numpy array returned from the C++ extension.
    requires_grad : bool
        Whether the resulting tensor should track gradients.  Default False,
        since Eigen data does not support autograd.

    Returns
    -------
    torch.Tensor
        Float64 CPU tensor with the same shape and data.

    Notes
    -----
    The tensor is always cloned so that mutations to the original numpy array
    do not affect the tensor.

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    >>> t = eigen_to_torch(arr)
    >>> t
    tensor([1., 2., 3.], dtype=torch.float64)
    """
    import numpy as np

    import torch

    arr = np.asarray(vectors, dtype=np.float64)
    arr = np.ascontiguousarray(arr)
    return torch.from_numpy(arr).clone().requires_grad_(requires_grad)


# ---------------------------------------------------------------------------
# eigen_to_numpy
# ---------------------------------------------------------------------------

def eigen_to_numpy(vectors: Any):
    """Convert an Eigen vector (numpy array from C++) to a numpy array.

    Ensures the result is a contiguous, float64, owned copy (not a view
    into the C++ memory).

    Parameters
    ----------
    vectors : numpy.ndarray or array_like
        Data returned from the C++ extension.

    Returns
    -------
    numpy.ndarray
        Contiguous float64 numpy array.

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    >>> eigen_to_numpy(arr)
    array([1., 2., 3.])
    """
    import numpy as np

    arr = np.asarray(vectors, dtype=np.float64)
    return np.ascontiguousarray(arr.copy())


# ---------------------------------------------------------------------------
# dlpack_export  (optional)
# ---------------------------------------------------------------------------

def dlpack_export(db: Any, chart_id: int) -> Optional[Any]:
    """Export a chart's point data as a DLPack tensor (zero-copy).

    This function attempts to export the indexed points for a given chart
    as a DLPack capsule that can be consumed by any DLPack-compatible
    framework (JAX, TensorFlow, PyTorch, etc.).

    If DLPack support is not available (e.g., the C++ extension was built
    without DLPack headers), a warning is issued and ``None`` is returned.

    Parameters
    ----------
    db : ManifoldDB or _core.ManifoldDB
        The database instance.
    chart_id : int
        Chart identifier whose points should be exported.

    Returns
    -------
    torch.Tensor or None
        A PyTorch tensor wrapping the DLPack capsule, or ``None`` if
        DLPack export is not supported.

    Examples
    --------
    >>> from manifolddb import ManifoldDB
    >>> from manifolddb.torch_compat import dlpack_export
    >>> db = ManifoldDB("./data", intrinsic_dim=3)
    >>> # After building atlas and inserting data...
    >>> tensor = dlpack_export(db, chart_id=0)
    >>> if tensor is not None:
    ...     print(f"Exported {tensor.shape[0]} points")
    """
    # Check if the C++ extension has DLPack support
    core_db = db.core if hasattr(db, "core") else db

    # Attempt to use DLPack via PyTorch's __dlpack__ protocol
    try:
        atlas = core_db.atlas()
        chart = atlas.get_chart(chart_id)
        if chart is None:
            warnings.warn(
                f"Chart {chart_id} not found in atlas; cannot export.",
                stacklevel=2,
            )
            return None

        # The current C++ bindings do not expose DLPack directly,
        # so we fall back to numpy-based export.
        # This is a placeholder for a future DLPack-enabled build.
        warnings.warn(
            "DLPack direct export is not yet supported. "
            "Use torch_to_eigen() for manual conversion.",
            stacklevel=2,
        )
        return None

    except (AttributeError, RuntimeError) as exc:
        warnings.warn(
            f"DLPack export failed: {exc}. "
            "Falling back to None.",
            stacklevel=2,
        )
        return None


# ---------------------------------------------------------------------------
# batch_geodesic_distances
# ---------------------------------------------------------------------------

def batch_geodesic_distances(
    query: Any,
    candidates: Any,
    metric_store: Any,
    chart_id: Optional[int] = None,
) -> Any:
    """Compute batch geodesic distances between query and candidate points.

    Uses the C++ ``GeodesicSolver.batch_geodesic_distance`` under the hood
    for efficient computation when both query and candidates are on the same
    chart.

    Parameters
    ----------
    query : array-like, shape ``(Q,)`` or ``(Q, d)``
        Query point(s) in local chart coordinates.  If 1-D, a single query.
    candidates : array-like, shape ``(N, d)`` or ``(N,)``
        Candidate points in local chart coordinates.
        If 1-D, treated as a single candidate.
    metric_store : MetricStore or ManifoldDB
        The metric store to use, or a ManifoldDB instance (from which
        the metric store is extracted).
    chart_id : int or None
        Chart ID for metric evaluation.  If ``None`` and *metric_store* is
        a ManifoldDB, chart 0 is used.

    Returns
    -------
    numpy.ndarray, shape ``(Q, N)`` or ``(N,)``
        Pairwise geodesic distances.  If *query* is 1-D, returns a 1-D
        array of length N.  If *query* is 2-D with Q rows, returns a 2-D
        array of shape ``(Q, N)``.

    Raises
    ------
    ValueError
        If the dimensions of query and candidates are incompatible.

    Examples
    --------
    >>> import numpy as np
    >>> from manifolddb.torch_compat import batch_geodesic_distances
    >>> query = np.array([0.1, 0.2], dtype=np.float64)
    >>> candidates = np.random.randn(50, 2).astype(np.float64)
    >>> dists = batch_geodesic_distances(query, candidates, db.metric_store)
    >>> dists.shape
    (50,)
    """
    import numpy as np

    # Resolve metric_store
    if hasattr(metric_store, "metric_store"):
        ms = metric_store.metric_store()
    else:
        ms = metric_store

    # Convert inputs
    q = torch_to_eigen(query)
    c = torch_to_eigen(candidates)

    # Resolve chart_id
    if chart_id is None:
        chart_id = 0

    # Attempt to use C++ batch distance if available
    try:
        # The C++ GeodesicSolver has batch_geodesic_distance(chart_id,
        # query_local, candidates_local).  It returns a 1-D array of
        # distances from a single query to N candidates.
        from manifolddb import manifolddb_core as _core

        # Build a temporary solver from the metric store
        solver_cfg = _core.SolverConfig()
        solver = _core.GeodesicSolver(ms, solver_cfg)

        if q.ndim == 1:
            # Single query → 1-D result
            distances = solver.batch_geodesic_distance(
                chart_id, q, c
            )
            return eigen_to_numpy(distances)
        else:
            # Multiple queries → 2-D result
            all_dists = []
            for i in range(q.shape[0]):
                qi = q[i]
                dists_i = solver.batch_geodesic_distance(
                    chart_id, qi, c
                )
                all_dists.append(eigen_to_numpy(dists_i))
            return np.vstack(all_dists)

    except (AttributeError, RuntimeError) as exc:
        warnings.warn(
            f"Batch geodesic distance via C++ backend failed: {exc}. "
            "Falling back to sequential Euclidean approximation.",
            stacklevel=2,
        )
        # Fallback: Euclidean distance in local coordinates
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if c.ndim == 1:
            c = c.reshape(1, -1)
        # (Q, 1, d) - (1, N, d) → (Q, N)
        diff = q[:, np.newaxis, :] - c[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=2))
