"""
manifold_db.api — REST API server for the Manifold Database.

Provides a FastAPI-based HTTP interface for all database operations including
data insertion, querying (geodesic, tangent-space, cross-modal), atlas
management, persistence, and system monitoring.

Public API:
    ManifoldApp          – main FastAPI application factory
    create_app           – build and configure the application
    InsertRequest        – single data-point insertion request
    BatchInsertRequest   – multi-point insertion request
    QueryRequest         – general query request (ManifoldQuery JSON)
    GeodesicQueryRequest – geodesic ball query request
    CrossModalQueryRequest – cross-modal retrieval request
    ChartInfo            – chart summary response
    DatabaseStats        – database statistics response
    HealthResponse       – health-check response
    QueryResultResponse  – query execution result response
    ExplainResponse      – query plan explanation response
"""

from manifold_db.api.middleware import (
    ErrorHandlerMiddleware,
    RateLimitMiddleware,
    RequestLoggingMiddleware,
)
from manifold_db.api.routes import (
    router_atlas,
    router_charts,
    router_insert,
    router_query,
    router_system,
)
from manifold_db.api.server import (
    BatchInsertRequest,
    ChartInfo,
    CrossModalQueryRequest,
    DatabaseStats,
    ExplainResponse,
    GeodesicQueryRequest,
    HealthResponse,
    InsertRequest,
    QueryRequest,
    QueryResultResponse,
)

__all__ = [
    # Application
    "create_app",
    # Request/response models
    "InsertRequest",
    "BatchInsertRequest",
    "QueryRequest",
    "GeodesicQueryRequest",
    "CrossModalQueryRequest",
    "ChartInfo",
    "DatabaseStats",
    "HealthResponse",
    "QueryResultResponse",
    "ExplainResponse",
    # Middleware
    "RequestLoggingMiddleware",
    "RateLimitMiddleware",
    "ErrorHandlerMiddleware",
    # Route groups
    "router_insert",
    "router_query",
    "router_charts",
    "router_atlas",
    "router_system",
]
