"""
API middleware — authentication, rate limiting, request logging, error handling.

This module provides FastAPI/Starlette-compatible ASGI middleware classes:

- :class:`RequestLoggingMiddleware` — logs every request with method, path, status, timing
- :class:`RateLimitMiddleware` — simple in-memory rate limiting per client IP
- :class:`ErrorHandlerMiddleware` — converts unhandled exceptions to JSON error responses
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Tuple

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Request Logging Middleware
# ═══════════════════════════════════════════════════════════════


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every HTTP request with timing information.

    Emits structured log entries including:

    - HTTP method and path
    - Query string
    - Client IP
    - Response status code
    - Elapsed time in milliseconds

    Parameters
    ----------
    app : ASGI app
        The next ASGI application in the middleware chain.
    skip_paths : set of str, optional
        Paths to skip logging for (e.g. ``/health``, ``/docs``).
    """

    def __init__(
        self,
        app: Any,
        skip_paths: Optional[set] = None,
    ) -> None:
        super().__init__(app)
        self._skip_paths: set = skip_paths or {"/docs", "/redoc", "/openapi.json", "/health"}

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Process a single request, log it, and pass through."""
        path = request.url.path
        if path in self._skip_paths:
            return await call_next(request)

        start_time = time.perf_counter()
        response: Response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        client_ip = self._get_client_ip(request)
        logger.info(
            "%s %s %s %.1fms ip=%s",
            request.method,
            path,
            response.status_code,
            elapsed_ms,
            client_ip,
        )
        return response

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        """Extract the client IP from the request, respecting X-Forwarded-For."""
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        if request.client:
            return request.client.host
        return "unknown"


# ═══════════════════════════════════════════════════════════════
# Rate Limit Middleware
# ═══════════════════════════════════════════════════════════════


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiter using a sliding window.

    Tracks the number of requests per client IP within a time window.
    When the limit is exceeded, returns HTTP 429 Too Many Requests.

    Parameters
    ----------
    app : ASGI app
        The next ASGI application in the middleware chain.
    max_requests : int
        Maximum number of requests allowed per window.
    window_seconds : int
        Length of the sliding window in seconds.
    skip_paths : set of str, optional
        Paths to exempt from rate limiting.
    """

    def __init__(
        self,
        app: Any,
        max_requests: int = 100,
        window_seconds: int = 60,
        skip_paths: Optional[set] = None,
    ) -> None:
        super().__init__(app)
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._skip_paths: set = skip_paths or {"/docs", "/redoc", "/openapi.json", "/health"}
        # ip → list of request timestamps
        self._request_log: Dict[str, list] = defaultdict(list)

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Check rate limit and either forward or reject the request."""
        path = request.url.path
        if path in self._skip_paths:
            return await call_next(request)

        client_ip = self._get_client_ip(request)
        now = time.time()

        # Prune old entries outside the window
        if client_ip in self._request_log:
            cutoff = now - self._window_seconds
            self._request_log[client_ip] = [
                ts for ts in self._request_log[client_ip] if ts > cutoff
            ]

        # Check limit
        if len(self._request_log[client_ip]) >= self._max_requests:
            logger.warning(
                "Rate limit exceeded for ip=%s (limit=%d/%ds)",
                client_ip,
                self._max_requests,
                self._window_seconds,
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too Many Requests",
                    "detail": (
                        f"Rate limit of {self._max_requests} requests per "
                        f"{self._window_seconds}s exceeded."
                    ),
                    "retry_after": self._window_seconds,
                },
            )

        # Record the request
        self._request_log[client_ip].append(now)
        return await call_next(request)

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        """Extract the client IP from the request."""
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        if request.client:
            return request.client.host
        return "unknown"


# ═══════════════════════════════════════════════════════════════
# Error Handler Middleware
# ═══════════════════════════════════════════════════════════════


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Convert unhandled exceptions into structured JSON error responses.

    Without this middleware, unhandled exceptions produce a raw 500 Internal
    Server Error with an HTML body.  This middleware catches those and returns
    a clean JSON payload.

    Parameters
    ----------
    app : ASGI app
        The next ASGI application in the middleware chain.
    debug : bool
        If True, include the full traceback in the error response
        (use only in development).
    """

    def __init__(
        self,
        app: Any,
        debug: bool = False,
    ) -> None:
        super().__init__(app)
        self._debug = debug

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Wrap the request in a try/except and convert errors to JSON."""
        try:
            return await call_next(request)
        except Exception as exc:
            logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
            status_code = 500
            detail = str(exc)

            # Map known exception types to HTTP status codes
            if isinstance(exc, ValueError):
                status_code = 400
                detail = f"Bad request: {exc}"
            elif isinstance(exc, PermissionError):
                status_code = 403
                detail = f"Forbidden: {exc}"
            elif isinstance(exc, FileNotFoundError):
                status_code = 404
                detail = f"Not found: {exc}"
            elif isinstance(exc, NotImplementedError):
                status_code = 501
                detail = f"Not implemented: {exc}"

            error_body: Dict[str, Any] = {
                "error": type(exc).__name__,
                "detail": detail,
                "status": status_code,
            }

            if self._debug:
                import traceback
                error_body["traceback"] = traceback.format_exc()

            return JSONResponse(
                status_code=status_code,
                content=error_body,
            )
