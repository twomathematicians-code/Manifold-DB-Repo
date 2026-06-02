"""
Manifold Query Module - DSL parser, query AST, builder, and execution engine.

Public API:
    QueryParser          – parse SQL-like strings → ManifoldQuery
    QueryBuilder         – fluent builder → ManifoldQuery
    ManifoldQuery        – compiled query representation
    QueryEngine          – execute queries against the manifold database
    QueryResult          – query result container (list/dict/DataFrame)
    ExecutionPlan        – execution plan with cost estimates
    QueryType, MetricType, CostTier – enums
    SelectQuery, TangentQuery, CrossModalQuery, TransportQuery – AST nodes
    GeodesicWhereClause  – WHERE clause for geodesic predicates
"""

from manifold_db.query.dsl import (
    CostTier,
    GeodesicWhereClause,
    ManifoldQuery,
    MetricType,
    QueryBuilder,
    QueryParser,
    QueryType,
    SelectQuery,
    TangentQuery,
    CrossModalQuery,
    TransportQuery,
)
from manifold_db.query.engine import (
    ExecutionPlan,
    ExecutionStep,
    QueryEngine,
    QueryResult,
)

__all__ = [
    # Enums
    "CostTier",
    "MetricType",
    "QueryType",
    # AST nodes
    "GeodesicWhereClause",
    "SelectQuery",
    "TangentQuery",
    "CrossModalQuery",
    "TransportQuery",
    # Query construction
    "ManifoldQuery",
    "QueryParser",
    "QueryBuilder",
    # Execution
    "QueryEngine",
    "QueryResult",
    "ExecutionPlan",
    "ExecutionStep",
]
