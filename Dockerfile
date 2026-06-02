# =============================================================================
# Manifold DB — Multi-stage Dockerfile
# =============================================================================
# Build:  docker build -t manifold-db .
# Run:    docker run -p 8000:8000 manifold-db
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: builder — install all dependencies (Python + system)
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# Metadata labels
LABEL maintainer="Manifold DB Contributors"
LABEL description="A geometric inference engine for data on Riemannian manifolds"
LABEL org.opencontainers.image.source="https://github.com/manifold-db/manifold-db"
LABEL org.opencontainers.image.title="manifold-db"
LABEL org.opencontainers.image.description="Geometric inference engine for Riemannian manifold data"

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN groupadd -r manifold && useradd -r -g manifold -d /app -s /sbin/nologin manifold

WORKDIR /build

# Install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel setuptools-scm

# Copy dependency specification first (layer caching — rebuild only when deps change)
COPY pyproject.toml .

# Install Python dependencies into a prefix we can copy later
RUN pip install --no-cache-dir --prefix=/install .

# ---------------------------------------------------------------------------
# Stage 2: runtime — lean image with only what's needed to run
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Metadata labels (repeated so they appear on the final image)
LABEL maintainer="Manifold DB Contributors"
LABEL description="A geometric inference engine for data on Riemannian manifolds"
LABEL org.opencontainers.image.source="https://github.com/manifold-db/manifold-db"
LABEL org.opencontainers.image.title="manifold-db"
LABEL org.opencontainers.image.description="Geometric inference engine for Riemannian manifold data"

# Runtime system libraries (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Recreate non-root user
RUN groupadd -r manifold && useradd -r -g manifold -d /app -s /sbin/nologin manifold

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY manifold_db/ manifold_db/

# Create data directory
RUN mkdir -p /app/data && chown -R manifold:manifold /app

# Environment
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MANIFOLD_DB_HOST=0.0.0.0 \
    MANIFOLD_DB_PORT=8000 \
    MANIFOLD_DB_WORKERS=4 \
    MANIFOLD_DB_LOG_LEVEL=info

# Expose API port
EXPOSE 8000

# Health check — probe the /health endpoint every 30 s
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${MANIFOLD_DB_PORT:-8000}/health || exit 1

# Drop to non-root user
USER manifold

# Default command — start the API server via uvicorn
CMD ["uvicorn", "manifold_db.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
