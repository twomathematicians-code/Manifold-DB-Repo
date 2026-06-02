#!/usr/bin/env bash
# =============================================================================
# Manifold DB — Installation Script
# =============================================================================
# Usage:
#   bash scripts/install.sh          # install with dev dependencies
#   bash scripts/install.sh --all    # install all optional dependencies
#   bash scripts/install.sh --venv   # create a venv first
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Colors for output
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'  # No Color

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
EXTRAS="dev"
CREATE_VENV=false
VENV_NAME=".venv"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)
            EXTRAS="all"
            shift
            ;;
        --venv)
            CREATE_VENV=true
            shift
            ;;
        --venv-name)
            VENV_NAME="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--all] [--venv [--venv-name NAME]]"
            echo ""
            echo "Options:"
            echo "  --all            Install all optional dependencies (dev + docs + benchmark)"
            echo "  --venv           Create a virtual environment before installing"
            echo "  --venv-name NAME Name for the venv directory (default: .venv)"
            echo "  -h, --help       Show this help"
            exit 0
            ;;
        *)
            error "Unknown option: $1 (use --help for usage)"
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Step 1: Check Python version
# ---------------------------------------------------------------------------
info "Checking Python version..."

if ! command -v python3 &>/dev/null; then
    error "python3 not found. Please install Python 3.10+."
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')

if [[ "$PYTHON_MAJOR" -lt 3 ]] || [[ "$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -lt 10 ]]; then
    error "Python 3.10+ required. Found: $PYTHON_VERSION"
fi

success "Python $PYTHON_VERSION found."

# ---------------------------------------------------------------------------
# Step 2: Optionally create virtual environment
# ---------------------------------------------------------------------------
if [[ "$CREATE_VENV" == true ]]; then
    info "Creating virtual environment: $VENV_NAME"

    if [[ -d "$VENV_NAME" ]]; then
        warn "Virtual environment '$VENV_NAME' already exists — skipping creation."
    else
        python3 -m venv "$VENV_NAME"
        success "Virtual environment created: $VENV_NAME"
    fi

    info "Activating virtual environment..."
    # shellcheck source=/dev/null
    source "$VENV_NAME/bin/activate"
    success "Virtual environment activated."
fi

# ---------------------------------------------------------------------------
# Step 3: Upgrade pip
# ---------------------------------------------------------------------------
info "Upgrading pip..."
pip install --upgrade pip setuptools wheel 2>&1 | tail -1
success "pip upgraded."

# ---------------------------------------------------------------------------
# Step 4: Install manifold-db
# ---------------------------------------------------------------------------
info "Installing manifold-db [$EXTRAS]..."
pip install -e ".[$EXTRAS]" 2>&1 | tail -3
success "manifold-db installed successfully."

# ---------------------------------------------------------------------------
# Step 5: Verify installation
# ---------------------------------------------------------------------------
info "Verifying installation..."
python3 -c "
import manifold_db
print(f'  manifold-db version: {manifold_db.__version__ if hasattr(manifold_db, \"__version__\") else \"0.1.0\"}')
print(f'  Package location:   {manifold_db.__file__}')
"
success "Verification complete."

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  Manifold DB installed successfully!  ${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
if [[ "$CREATE_VENV" == true ]]; then
    echo -e "  ${CYAN}Virtual env:${NC}  $VENV_NAME"
    echo -e "  ${CYAN}Activate:${NC}    source $VENV_NAME/bin/activate"
fi
echo -e "  ${CYAN}CLI:${NC}         manifold-db --help"
echo -e "  ${CYAN}Server:${NC}      manifold-db serve"
echo ""
