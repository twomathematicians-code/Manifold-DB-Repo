"""
manifold_db.atlas — Core Atlas module for the Manifold Database.

This package provides the foundational storage layer for a manifold-structured
database, implementing the mathematical concept of an **atlas** — a collection
of overlapping coordinate charts (local patches) connected by smooth transition
maps (diffeomorphisms).

Public API
----------
Charts:
    :class:`Chart` — local coordinate patch on the manifold.

Transition Maps:
    :class:`TransitionMap` — abstract base for diffeomorphisms between charts.
    :class:`LinearTransition` — linear (matrix-only) transition.
    :class:`AffineTransition` — affine (matrix + bias) transition.
    :class:`NeuralTransition` — invertible MLP coupling-layer transition.

Management:
    :class:`AtlasManager` — CRUD orchestration for charts and transitions.
    :class:`AtlasBuilder` — automatic manifold learning and chart discovery.

Factory:
    :func:`create_transition_map` — instantiate a transition from a serialised dict.
"""

from .atlas_builder import AtlasBuilder
from .atlas_manager import AtlasManager
from .chart import Chart
from .transition_map import (
    AffineTransition,
    LinearTransition,
    NeuralTransition,
    TransitionMap,
    create_transition_map,
)

__all__ = [
    # Charts
    "Chart",
    # Transition maps
    "TransitionMap",
    "LinearTransition",
    "AffineTransition",
    "NeuralTransition",
    "create_transition_map",
    # Management
    "AtlasManager",
    # Building
    "AtlasBuilder",
]
