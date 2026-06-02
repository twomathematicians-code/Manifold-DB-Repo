"""
tangent_index — Tangent Space Index module for the Manifold Database.

This module provides efficient approximate nearest-neighbour search on
manifold-valued data by projecting queries into local tangent spaces where
Euclidean search is valid.

Public API
----------
* :class:`TangentSpace`    — local linear approximation of the manifold.
* :class:`TangentBundle`   — collection of tangent spaces covering the manifold.
* :class:`TangentSpaceIndex` — main index structure for query acceleration.

Quick start
-----------
>>> from manifold_db.tangent_index import TangentSpaceIndex
>>> index = TangentSpaceIndex(intrinsic_dim=3)
>>> stats = index.build_from_data(ids, data, n_anchors=50)
>>> results = index.search(query, k=10)
"""

from .index import TangentSpaceIndex
from .tangent_bundle import TangentBundle
from .tangent_space import TangentSpace

__all__ = [
    "TangentSpace",
    "TangentBundle",
    "TangentSpaceIndex",
]
