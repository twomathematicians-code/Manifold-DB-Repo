#!/usr/bin/env bash
# =============================================================================
# Manifold DB — Test Runner Script
# =============================================================================
# Usage:
#   bash scripts/run_tests.sh                # run all non-slow, non-gpu tests
#   bash scripts/run_tests.sh --all          # run all tests (including slow)
#   bash scripts/run_tests.sh --cov          # run with coverage report
#   bash scripts/run_tests.sh --integration  # run integration tests only
#   bash scripts/run_tests.sh --benchmark    # run benchmarks only
#   bash scripts/run_tests.sh --parallel     # run with pytest-xdist parallelism
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
PYTEST_ARGS=()
TEST_MARKER="-m not slow and not gpu"
COVERAGE=false
PARALLEL=false
EXIT_CODE=0

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)
            TEST_MARKER=""
            shift
            ;;
        --cov|--coverage)
            COVERAGE=true
            shift
            ;;
        --integration)
            TEST_MARKER="-m integration"
            shift
            ;;
        --benchmark)
            TEST_MARKER="-m benchmark --benchmark-only --benchmark-sort=name"
            shift
            ;;
        --slow)
            TEST_MARKER="-m slow"
            shift
            ;;
        --parallel|-p)
            PARALLEL=true
            shift
            ;;
        --verbose|-v)
            PYTEST_ARGS+=("-v")
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --all          Run all tests (no marker filtering)"
            echo "  --cov          Enable coverage report (terminal + HTML)"
            echo "  --integration Run integration tests only"
            echo "  --benchmark    Run benchmarks only"
            echo "  --slow         Run slow tests only"
            echo "  --parallel, -p Run with pytest-xdist (auto-detect CPU count)"
            echo "  --verbose, -v  Verbose output"
            echo "  -h, --help     Show this help"
            exit 0
            ;;
        *)
            PYTEST_ARGS+=("$1")
            shift
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Check environment
# ---------------------------------------------------------------------------
if ! command -v python3 &>/dev/null; then
    error "python3 not found."
fi

if ! python3 -c "import pytest" 2>/dev/null; then
    error "pytest not found. Run: pip install -e '.[dev]'"
fi

# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------
CMD=(pytest tests/ --tb=short -v)

# Marker filter
if [[ -n "$TEST_MARKER" ]]; then
    CMD+=("$TEST_MARKER")
fi

# Coverage
if [[ "$COVERAGE" == true ]]; then
    CMD+=(
        --cov=manifold_db
        --cov-report=term-missing
        --cov-report=html:htmlcov
        --cov-report=xml:coverage.xml
    )
fi

# Parallel
if [[ "$PARALLEL" == true ]]; then
    if ! python3 -c "import xdist" 2>/dev/null; then
        warn "pytest-xdist not found. Running sequentially."
    else
        NPROC=$(python3 -c "import os; print(min(os.cpu_count() or 1, 8))")
        CMD+=("-n" "$NPROC")
        info "Running with $NPROC parallel workers."
    fi
fi

# Append any extra args
CMD+=("${PYTEST_ARGS[@]}")

# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------
echo ""
info "Running: ${CMD[*]}"
echo "================================================================"
START_TIME=$(date +%s)

python3 "${CMD[@]}" || EXIT_CODE=$?

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo "================================================================"
if [[ $EXIT_CODE -eq 0 ]]; then
    success "All tests passed in ${MINUTES}m ${SECONDS}s."
else
    error "Tests failed after ${MINUTES}m ${SECONDS}s (exit code: ${EXIT_CODE})."
fi

# ---------------------------------------------------------------------------
# Coverage summary
# ---------------------------------------------------------------------------
if [[ "$COVERAGE" == true ]] && [[ -f coverage.xml ]]; then
    info "Coverage report: htmlcov/index.html"
    python3 -c "
import xml.etree.ElementTree as ET
tree = ET.parse('coverage.xml')
root = tree.getroot()
line_rate = float(root.attrib.get('line-rate', 0))
pct = line_rate * 100
print(f'  Overall coverage: {pct:.1f}%')
if pct >= 80:
    print('  ✅ Coverage target met (≥80%)')
else:
    print('  ⚠️  Coverage below 80% target')
" || true
fi

exit $EXIT_CODE
