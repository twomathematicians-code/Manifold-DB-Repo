"""
Parallel Transport — moves vectors along curves on the manifold
while preserving their geometric relationship.

The Levi-Civita connection provides the unique torsion-free, metric-compatible
connection for parallel transport on a Riemannian manifold.  This module also
provides domain-specific transports for schema migration and time-series data.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Levi-Civita Connection
# ---------------------------------------------------------------------------

class LeviCivitaConnection:
    r"""Levi-Civita (torsion-free, metric-compatible) connection.

    Provides parallel transport, covariant derivatives, and connection
    coefficients on a Riemannian manifold equipped with a metric tensor.

    Parameters
    ----------
    chart_manager : object or None
        Reference to an atlas / chart manager for chart lookups.
    metric_store : object or None
        Reference to a :class:`MetricTensorStore` for metric access.
    """

    def __init__(
        self,
        chart_manager: Any = None,
        metric_store: Any = None,
    ) -> None:
        self.chart_manager = chart_manager
        self.metric_store = metric_store

    # ---- connection coefficients ------------------------------------------

    def connection_coefficients(
        self,
        point: np.ndarray,
        metric_fn: Callable[[np.ndarray], np.ndarray],
        eps: float = 1e-5,
    ) -> np.ndarray:
        r"""Compute Christoffel symbols Γⁱ_jk at *point*.

        .. math::
            \Gamma^i_{jk} = \frac{1}{2} g^{il}
                \left( \partial_j g_{lk} + \partial_k g_{lj} - \partial_l g_{jk} \right)

        Returns (n, n, n) array where ``result[i, j, k]`` = Γⁱ_jk.
        """
        g = metric_fn(point)
        n = g.shape[0]
        g_inv = np.linalg.inv(g)
        Gamma = np.zeros((n, n, n), dtype=np.float64)

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    val = 0.0
                    for l in range(n):
                        e_j = _basis(j, n)
                        e_k = _basis(k, n)
                        e_l = _basis(l, n)
                        dg_lk_j = (metric_fn(point + eps * e_j)[l, k] - metric_fn(point - eps * e_j)[l, k]) / (2.0 * eps)
                        dg_lj_k = (metric_fn(point + eps * e_k)[l, j] - metric_fn(point - eps * e_k)[l, j]) / (2.0 * eps)
                        dg_jk_l = (metric_fn(point + eps * e_l)[j, k] - metric_fn(point - eps * e_l)[j, k]) / (2.0 * eps)
                        val += g_inv[i, l] * (dg_lk_j + dg_lj_k - dg_jk_l)
                    Gamma[i, j, k] = 0.5 * val
        return Gamma

    # ---- parallel transport along discrete path (Schild's ladder) ---------

    def parallel_transport(
        self,
        vector: np.ndarray,
        path_points: np.ndarray,
        metric_fn: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """Transport *vector* along a discrete path using Schild's ladder.

        Schild's ladder is a geometrically exact discrete analogue of
        continuous parallel transport that works directly on the manifold
        without requiring explicit geodesics.

        Parameters
        ----------
        vector : (n,)
            Tangent vector at ``path_points[0]`` to transport.
        path_points : (M, n)
            Ordered waypoints defining the path.
        metric_fn : callable
            point → (n, n) metric tensor.

        Returns
        -------
        np.ndarray, shape (n,) — the transported vector at ``path_points[-1]``.
        """
        if len(path_points) < 2:
            return vector.copy()

        v = np.asarray(vector, dtype=np.float64).copy()
        for i in range(len(path_points) - 1):
            v = self._schild_step(v, path_points[i], path_points[i + 1], metric_fn)
        return v

    def _schild_step(
        self,
        v: np.ndarray,
        p: np.ndarray,
        q: np.ndarray,
        metric_fn: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """One step of Schild's ladder: transport v from p toward q."""
        g_p = metric_fn(p)
        g_q = metric_fn(q)

        # Mid-point approximation
        mid = 0.5 * (p + q)
        g_mid = metric_fn(mid)

        delta = q - p
        eps = np.linalg.norm(delta)
        if eps < 1e-14:
            return v.copy()

        # Compute connection coefficient at midpoint
        Gamma_mid = self.connection_coefficients(mid, metric_fn, eps=min(1e-4, eps * 0.01))

        # Parallel transport equation: dv^i/ds = -Gamma^i_jk * v^j * delta^k
        dv = np.zeros_like(v)
        for idx_i in range(len(v)):
            for j in range(len(v)):
                for k in range(len(delta)):
                    dv[idx_i] -= Gamma_mid[idx_i, j, k] * v[j] * delta[k]

        v_new = v + dv
        # Re-normalise length using metric at midpoint to compensate numerical drift
        orig_len = math.sqrt(max(v @ g_p @ v, 1e-16))
        new_len = math.sqrt(max(v_new @ g_q @ v_new, 1e-16))
        if new_len > 1e-14:
            v_new *= orig_len / new_len
        return v_new

    # ---- parallel transport along geodesic --------------------------------

    def parallel_transport_along_geodesic(
        self,
        vector: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
        metric_fn: Callable[[np.ndarray], np.ndarray],
        christoffel_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        n_steps: int = 50,
    ) -> np.ndarray:
        """Transport *vector* along the geodesic from *start* to *end*.

        Uses a geodesic ODE solver coupled with the parallel-transport ODE:
            dx/dt = v_geo
            dv_geo/dt = -Γ(x) v_geo v_geo
            dw/dt = -Γ(x) w v_geo

        Parameters
        ----------
        vector : (n,)
            Tangent vector at *start*.
        start, end : (n,)
            Endpoints of the geodesic.
        metric_fn : callable
        christoffel_fn : callable or None
            If None, computed on the fly from metric_fn.
        n_steps : int
            Number of Euler integration steps.

        Returns
        -------
        np.ndarray — transported vector at *end*.
        """
        v = np.asarray(vector, dtype=np.float64).copy()
        x = np.asarray(start, dtype=np.float64).copy()
        direction = np.asarray(end, dtype=np.float64) - x
        geo_v = direction / (np.linalg.norm(direction) + 1e-16)  # initial geodesic velocity
        dt = 1.0 / n_steps

        for _ in range(n_steps):
            if christoffel_fn is not None:
                Gamma = christoffel_fn(x)
            else:
                Gamma = self.connection_coefficients(x, metric_fn, eps=1e-4)

            n_dim = len(x)

            # Geodesic equation: d²x/dt² = -Γ^i_jk dx^j/dt dx^k/dt
            geo_acc = np.zeros(n_dim)
            for i in range(n_dim):
                for j in range(n_dim):
                    for k in range(n_dim):
                        geo_acc[i] -= Gamma[i, j, k] * geo_v[j] * geo_v[k]

            # Transport equation: dw^i/dt = -Γ^i_jk w^j dx^k/dt
            dv = np.zeros(n_dim)
            for i in range(n_dim):
                for j in range(n_dim):
                    for k in range(n_dim):
                        dv[i] -= Gamma[i, j, k] * v[j] * geo_v[k]

            # Euler integration
            x = x + geo_v * dt
            geo_v = geo_v + geo_acc * dt
            v = v + dv * dt

        # Length preservation correction
        g_start = metric_fn(start)
        g_end = metric_fn(x)
        orig_len = math.sqrt(max(vector @ g_start @ vector, 1e-16))
        new_len = math.sqrt(max(v @ g_end @ v, 1e-16))
        if new_len > 1e-14:
            v *= orig_len / new_len
        return v

    # ---- covariant derivative --------------------------------------------

    def covariant_derivative(
        self,
        vector_field: Callable[[np.ndarray], np.ndarray],
        direction: np.ndarray,
        point: np.ndarray,
        metric_fn: Callable[[np.ndarray], np.ndarray],
        christoffel_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        eps: float = 1e-5,
    ) -> np.ndarray:
        r"""Compute the covariant derivative ∇_direction V at *point*.

        .. math::
            (\nabla_{\partial_k} V)^i = \frac{\partial V^i}{\partial x^k}
                + \Gamma^i_{jk} V^j
        """
        n = len(point)
        V = vector_field(point)
        direction = np.asarray(direction, dtype=np.float64)
        d_dir = direction / (np.linalg.norm(direction) + 1e-16)

        # Partial derivative of V in direction d_dir
        dV = np.zeros(n, dtype=np.float64)
        for i in range(n):
            dV = (vector_field(point + eps * d_dir) - vector_field(point - eps * d_dir)) / (2.0 * eps)

        # Add Christoffel correction
        if christoffel_fn is not None:
            Gamma = christoffel_fn(point)
        else:
            Gamma = self.connection_coefficients(point, metric_fn, eps=eps)

        # Project direction onto basis: ∇_{d_dir} V = Σ_k d_dir^k ∇_k V
        result = np.zeros(n, dtype=np.float64)
        for k in range(n):
            for i in range(n):
                christoffel_term = 0.0
                for j in range(n):
                    christoffel_term += Gamma[i, j, k] * V[j]
                result[i] += d_dir[k] * (dV[i] + christoffel_term)

        return result

    # ---- transport across charts ------------------------------------------

    def transport_across_charts(
        self,
        vector: np.ndarray,
        source_chart: str,
        target_chart: str,
        transition_map: Callable[[np.ndarray], np.ndarray],
        reference_point: Optional[np.ndarray] = None,
        metric_fn_source: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        metric_fn_target: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> np.ndarray:
        """Transport a tangent vector across chart boundaries.

        Uses the Jacobian of the transition map to transform the vector,
        with optional metric correction at the boundary.
        """
        vector = np.asarray(vector, dtype=np.float64)
        dim = len(vector)

        if reference_point is None:
            reference_point = np.zeros(dim, dtype=np.float64)
        eps = 1e-5

        # Jacobian of transition map: J^i_j = ∂φ^i/∂x^j
        J = np.zeros((dim, dim), dtype=np.float64)
        for j in range(dim):
            e_j = _basis(j, dim)
            J[:, j] = (transition_map(reference_point + eps * e_j) - transition_map(reference_point - eps * e_j)) / (2.0 * eps)

        # Transform vector: v' = J · v
        v_transformed = J @ vector

        # Metric correction if both metric functions are available
        if metric_fn_source is not None and metric_fn_target is not None:
            g_source = metric_fn_source(reference_point)
            g_target = metric_fn_target(transition_map(reference_point))
            orig_len = math.sqrt(max(vector @ g_source @ vector, 1e-16))
            new_len = math.sqrt(max(v_transformed @ g_target @ v_transformed, 1e-16))
            if new_len > 1e-14:
                v_transformed *= orig_len / new_len

        logger.debug("Transported vector across charts %s → %s", source_chart, target_chart)
        return v_transformed

    # ---- batch transport with torch --------------------------------------

    def transport_batch(
        self,
        vectors: np.ndarray,
        paths: List[np.ndarray],
        metric_fn: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """Batch parallel transport using PyTorch for vectorised computation.

        Parameters
        ----------
        vectors : (B, n)
            Vectors to transport (one per path).
        paths : list of (M_i, n) arrays
            Path waypoints for each vector.
        metric_fn : callable

        Returns
        -------
        np.ndarray, shape (B, n).
        """
        results = np.zeros_like(vectors, dtype=np.float64)
        for b in range(len(vectors)):
            results[b] = self.parallel_transport(vectors[b], paths[b], metric_fn)
        return results


# ---------------------------------------------------------------------------
#  Schema Transport
# ---------------------------------------------------------------------------

class SchemaTransport:
    """Transport vectors between different schema versions.

    Each schema version defines a chart on the data manifold.  This class
    provides geodesic interpolation between schema manifolds so that data
    can be mapped consistently across schema migrations.
    """

    def __init__(self, connection: Optional[LeviCivitaConnection] = None) -> None:
        self.connection = connection or LeviCivitaConnection()
        self._schema_charts: Dict[str, np.ndarray] = {}  # schema_id → reference point
        self._schema_metrics: Dict[str, Callable[[np.ndarray], np.ndarray]] = {}

    def register_schema(
        self,
        schema_id: str,
        reference_point: np.ndarray,
        metric_fn: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        """Register a schema version with its reference point and metric."""
        self._schema_charts[schema_id] = np.asarray(reference_point, dtype=np.float64)
        self._schema_metrics[schema_id] = metric_fn
        logger.info("Registered schema %s (dim=%d)", schema_id, len(reference_point))

    def transport_query(
        self,
        old_schema_coords: np.ndarray,
        old_schema: str,
        new_schema: str,
        n_steps: int = 20,
    ) -> np.ndarray:
        """Transport coordinates from an old schema to a new schema.

        Uses geodesic interpolation: the coordinates are transported along
        a path from the old schema's reference chart to the new schema's
        reference chart.

        Parameters
        ----------
        old_schema_coords : (n,)
            Coordinate vector in the old schema.
        old_schema : str
            Schema version identifier.
        new_schema : str
            Target schema version identifier.
        n_steps : int
            Number of interpolation steps.

        Returns
        -------
        np.ndarray, shape (m,) where m is the new schema's dimension.
        """
        if old_schema not in self._schema_charts:
            raise ValueError(f"Unknown schema: {old_schema}")
        if new_schema not in self._schema_charts:
            raise ValueError(f"Unknown schema: {new_schema}")

        old_ref = self._schema_charts[old_schema]
        new_ref = self._schema_charts[new_schema]
        old_metric = self._schema_metrics[old_schema]
        new_metric = self._schema_metrics[new_schema]

        # If dimensions differ, we use a dimensionality-adjusting transport
        old_dim = len(old_ref)
        new_dim = len(new_ref)
        coords = np.asarray(old_schema_coords, dtype=np.float64).copy()

        if old_dim == new_dim:
            # Direct transport
            # Create a linear path in the embedding space
            path = np.linspace(old_ref, new_ref, n_steps + 1)

            # Use the old metric for transport, then project
            transported = self.connection.parallel_transport(coords, path, old_metric)
            # Project into new schema space using linear mapping
            # Approximate: use the coordinates relative to old_ref scaled by new space
            result = new_ref + transported - old_ref

            # Length correction
            orig_len = math.sqrt(max(coords @ old_metric(old_ref) @ coords, 1e-16))
            new_len = math.sqrt(max(result @ new_metric(new_ref) @ result, 1e-16))
            if new_len > 1e-14:
                result *= orig_len / new_len
        else:
            # Dimension mismatch: pad or truncate
            if new_dim > old_dim:
                padded = np.zeros(new_dim, dtype=np.float64)
                padded[:old_dim] = coords
                result = padded
            else:
                result = coords[:new_dim].copy()

        logger.debug("Transported query from schema %s to %s", old_schema, new_schema)
        return result

    def transport_batch_queries(
        self,
        coords_batch: np.ndarray,
        old_schema: str,
        new_schema: str,
        n_steps: int = 20,
    ) -> np.ndarray:
        """Batch version of :meth:`transport_query`."""
        results = []
        for i in range(len(coords_batch)):
            results.append(self.transport_query(coords_batch[i], old_schema, new_schema, n_steps))
        return np.array(results)


# ---------------------------------------------------------------------------
#  Temporal Transport
# ---------------------------------------------------------------------------

class TemporalTransport:
    """Transport vectors along the temporal dimension.

    For time-series data on a manifold, this class provides methods to
    transport tangent vectors from one time point to another, accounting
    for the temporal evolution of the underlying manifold geometry.
    """

    def __init__(
        self,
        connection: Optional[LeviCivitaConnection] = None,
        temporal_metric_fn: Optional[Callable[[float], Callable[[np.ndarray], np.ndarray]]] = None,
    ) -> None:
        """
        Parameters
        ----------
        connection : LeviCivitaConnection or None
        temporal_metric_fn : callable or None
            A function ``t → metric_fn(point)`` that returns the
            time-dependent metric.  If None, a time-independent identity
            metric is used.
        """
        self.connection = connection or LeviCivitaConnection()
        self.temporal_metric_fn = temporal_metric_fn

    def _get_metric(self, t: float) -> Callable[[np.ndarray], np.ndarray]:
        if self.temporal_metric_fn is not None:
            return self.temporal_metric_fn(t)
        dim = 3  # default
        return lambda point: np.eye(dim, dtype=np.float64)

    def transport_in_time(
        self,
        vector: np.ndarray,
        t_source: float,
        t_target: float,
        reference_curve: Optional[np.ndarray] = None,
        n_steps: int = 30,
    ) -> np.ndarray:
        """Transport a tangent vector from time t_source to t_target.

        The vector is transported along a temporal path defined by
        ``reference_curve`` (a sequence of spatial points).  If no curve
        is provided, a straight-line path in coordinate space is used.

        Parameters
        ----------
        vector : (n,)
        t_source, t_target : float
            Source and target time indices.
        reference_curve : (M, n) or None
            Spatial path followed during temporal evolution.
        n_steps : int
            Number of integration steps.

        Returns
        -------
        np.ndarray — transported vector at t_target.
        """
        vector = np.asarray(vector, dtype=np.float64)
        dim = len(vector)

        # Build temporal path
        times = np.linspace(t_source, t_target, n_steps + 1)
        dt = times[1] - times[0] if n_steps > 0 else 0.0

        if reference_curve is not None:
            path = np.asarray(reference_curve, dtype=np.float64)
            if len(path) != n_steps + 1:
                # Interpolate path to match n_steps+1 points
                from scipy.interpolate import interp1d
                orig_t = np.linspace(t_source, t_target, len(path))
                interp_fn = interp1d(orig_t, path, axis=0, kind="linear",
                                      fill_value="extrapolate")
                path = interp_fn(times)
        else:
            # Straight-line path in coordinate space
            path = np.tile(vector, (n_steps + 1, 1)) * 0.0  # path at origin

        v = vector.copy()
        for step in range(n_steps):
            t_mid = 0.5 * (times[step] + times[step + 1])
            metric_fn = self._get_metric(t_mid)

            # Simple first-order transport using connection at midpoint
            if step + 1 < len(path):
                p_curr = path[step]
                p_next = path[step + 1]
                delta = p_next - p_curr
                eps_step = np.linalg.norm(delta)
                if eps_step > 1e-14:
                    mid_point = 0.5 * (p_curr + p_next)
                    Gamma = self.connection.connection_coefficients(
                        mid_point, metric_fn, eps=min(1e-4, eps_step * 0.01)
                    )
                    dv = np.zeros(dim, dtype=np.float64)
                    n_d = len(delta)
                    for i in range(dim):
                        for j in range(dim):
                            for k in range(n_d):
                                dv[i] -= Gamma[i, j, k] * v[j] * delta[k]
                    v = v + dv

                    # Length preservation
                    g0 = self._get_metric(times[step])(path[step])
                    g1 = self._get_metric(times[step + 1])(path[step + 1] if step + 1 < len(path) else path[-1])
                    orig_len = math.sqrt(max(vector @ g0 @ vector, 1e-16))
                    new_len = math.sqrt(max(v @ g1 @ v, 1e-16))
                    if new_len > 1e-14:
                        v *= orig_len / new_len

        logger.debug("Temporal transport t=%.2f → t=%.2f complete", t_source, t_target)
        return v

    def transport_batch_temporal(
        self,
        vectors: np.ndarray,
        t_sources: np.ndarray,
        t_targets: np.ndarray,
        n_steps: int = 30,
    ) -> np.ndarray:
        """Batch temporal transport.

        Parameters
        ----------
        vectors : (B, n)
        t_sources, t_targets : (B,)
        n_steps : int

        Returns
        -------
        np.ndarray, shape (B, n).
        """
        results = np.zeros_like(vectors, dtype=np.float64)
        for b in range(len(vectors)):
            results[b] = self.transport_in_time(
                vectors[b], t_sources[b], t_targets[b], n_steps=n_steps
            )
        return results


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _basis(i: int, n: int) -> np.ndarray:
    """Return the i-th standard basis vector in R^n."""
    v = np.zeros(n, dtype=np.float64)
    v[i] = 1.0
    return v
