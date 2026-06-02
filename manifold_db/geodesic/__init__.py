"""
Geodesic Query Engine for the Manifold Database.

This module provides tools for computing geodesic distances, solving geodesic
equations on Riemannian manifolds, and bridging tangent space ↔ manifold
via exponential / logarithmic maps.

Sub-modules
-----------
solver
    GeodesicSolver, compute_christoffel_symbols, IntegrationMethod, GeodesicResult
distance
    RiemannianDistance, WassersteinDistance, FisherRaoDistance, DistanceComputer
exponential_map
    ExponentialMap, RetractionMap, InverseRetraction
"""

from manifold_db.geodesic.solver import (
    GeodesicResult,
    GeodesicSolver,
    IntegrationMethod,
    compute_christoffel_symbols,
)

from manifold_db.geodesic.distance import (
    DistanceComputer,
    FisherRaoDistance,
    RiemannianDistance,
    WassersteinDistance,
)

from manifold_db.geodesic.exponential_map import (
    ExponentialMap,
    InverseRetraction,
    RetractionMap,
)

__all__ = [
    # solver
    "GeodesicSolver",
    "GeodesicResult",
    "IntegrationMethod",
    "compute_christoffel_symbols",
    # distance
    "RiemannianDistance",
    "WassersteinDistance",
    "FisherRaoDistance",
    "DistanceComputer",
    # exponential_map
    "ExponentialMap",
    "RetractionMap",
    "InverseRetraction",
]
