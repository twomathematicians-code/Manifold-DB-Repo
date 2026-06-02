# Contributing to ManifoldDB

Thank you for your interest in contributing to ManifoldDB! This document provides guidelines for setting up the development environment, coding standards, and the pull request process.

## Table of Contents

1. [Development Environment Setup](#development-environment-setup)
2. [Building from Source](#building-from-source)
3. [Code Style Guidelines](#code-style-guidelines)
4. [Testing](#testing)
5. [Pull Request Process](#pull-request-process)
6. [Issue Reporting](#issue-reporting)
7. [Commit Messages](#commit-messages)

---

## Development Environment Setup

### Prerequisites

| Tool | Minimum Version | Purpose |
|------|----------------|---------|
| Python | 3.9+ | Runtime |
| C++ Compiler | C++20 support | Building the extension |
| CMake | 3.18+ | Build system (alternative) |
| Eigen | 3.3+ | Linear algebra |
| pybind11 | 2.11+ | Python bindings |
| PyTorch | 1.12+ | Build tooling & optional CUDA |
| NumPy | 1.21+ | Array interop |
| pytest | 7.0+ | Testing |

### Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/manifolddb/manifolddb.git
cd manifolddb

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate     # Windows

# 3. Install development dependencies
pip install -e ".[dev]"

# 4. Verify the build
python -c "import manifolddb; print(manifolddb.__version__)"
```

### Platform-Specific Notes

**Ubuntu/Debian:**
```bash
sudo apt install libeigen3-dev pybind11-dev
```

**macOS (Homebrew):**
```bash
brew install eigen pybind11
```

**Windows (vcpkg):**
```bash
vcpkg install eigen3 pybind11
```

---

## Building from Source

### pip (Recommended)

```bash
pip install -e .
```

This uses `torch.utils.cpp_extension` to compile the C++ extension automatically.

### CMake

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### With CUDA

```bash
# Ensure CUDA toolkit is installed and in PATH
pip install -e .  # Auto-detects CUDA_HOME
```

---

## Code Style Guidelines

### C++

- **Standard:** C++20
- **Formatting:** Follow the existing style in the codebase (4-space indentation, braces on same line)
- **Naming:**
  - Classes: `PascalCase` (e.g., `MetricTensor`, `GeodesicSolver`)
  - Functions/methods: `snake_case` (e.g., `compute_local_metric`, `solve_ivp`)
  - Member variables: `snake_case_` with trailing underscore (e.g., `intrinsic_dim_`, `basis_`)
  - Constants: `kCamelCase` or `UPPER_SNAKE_CASE`
- **Includes:** Use `#pragma once` for headers
- **Documentation:** Use `///` for doc comments on public APIs
- **Exceptions:** Throw `DBException` subclasses for library errors
- **Memory:** Use `std::shared_ptr` and `std::unique_ptr`; avoid raw `new`/`delete`

### Python

- **Formatting:** [Black](https://github.com/psf/black) with default settings (88 char line length)
- **Linting:** [Ruff](https://github.com/astral-sh/ruff)
- **Type hints:** Use Python 3.9+ syntax (e.g., `list[int]` instead of `List[int]`)
- **Docstrings:** Google-style docstrings with NumPy conventions for parameters/returns
- **Imports:** Absolute imports; group as: stdlib → third-party → local

```python
"""Module docstring."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


def compute_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the Euclidean distance between two arrays.

    Args:
        a: First array, shape (N,).
        b: Second array, shape (N,).

    Returns:
        Euclidean distance as a float.

    Raises:
        ValueError: If shapes do not match.
    """
    ...
```

### Running Formatters

```bash
# Format C++ (if clang-format is configured)
find cpp -name '*.hpp' -o -name '*.cpp' | xargs clang-format -i

# Format Python
black python/ tests/
ruff check python/ tests/ --fix
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_chart.py -v

# Run a specific test
pytest tests/test_chart.py::TestLinearChart::test_linear_chart_embed_project -v

# Run with coverage
pytest tests/ --cov=manifolddb --cov-report=html
```

### Test Structure

Tests are located in the `tests/` directory and mirror the source structure:

| File | Coverage |
|------|----------|
| `test_chart.py` | `LinearChart`, `ParametricChart`, Christoffel symbols, exp/log maps |
| `test_metric_tensor.py` | `MetricTensor`, `MetricStore` |
| `test_atlas.py` | `Atlas`, `TransitionMap`, path finding, chart discovery |
| `test_geodesic_solver.py` | `GeodesicSolver` (IVP, BVP, parallel transport) |
| `test_manifold_db.py` | `ManifoldDB` integration tests |

### Skip Behavior

All tests are automatically skipped if the C++ extension `_manifolddb_core` is not built. To build it:

```bash
pip install -e .
```

### Writing New Tests

1. Place tests in the appropriate file under `tests/`
2. Use `@pytest.mark.requires_core` for tests needing the C++ extension
3. Use fixtures from `conftest.py` where possible
4. Include docstrings explaining what is being tested

```python
def test_my_new_feature(self, core, rng):
    """Verify that the new feature works correctly.

    This test creates a chart and verifies that the feature
    produces the expected result.
    """
    chart = core.LinearChart(id=0, basis=np.eye(3), origin=np.zeros(3))
    result = chart.some_new_method()
    assert result is not None
```

---

## Pull Request Process

1. **Fork** the repository and create a feature branch from `main`
2. **Make changes** following the code style guidelines
3. **Write tests** for new functionality
4. **Run the test suite** and ensure all tests pass:
   ```bash
   pytest tests/ -v
   ```
5. **Update documentation** if API changes were made
6. **Open a Pull Request** with:
   - Clear title and description
   - Reference to any related issues
   - List of changes made
7. **Address review feedback** promptly

### PR Checklist

- [ ] Code compiles without warnings
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Documentation updated (if applicable)
- [ ] Commit messages follow conventions

---

## Issue Reporting

### Bug Reports

When filing a bug report, please include:

1. **Environment:** OS, Python version, compiler, CUDA version (if applicable)
2. **Minimal reproducer:** The smallest code snippet that triggers the bug
3. **Expected behavior:** What you expected to happen
4. **Actual behavior:** What actually happened (with error messages)
5. **Steps to reproduce:** How to trigger the bug

### Feature Requests

For feature requests, please describe:

1. **Motivation:** Why this feature would be useful
2. **Proposed API:** How you envision the feature being used
3. **Alternatives considered:** Any other approaches you've thought of

---

## Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions/changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `build`: Build system changes
- `ci`: CI/CD changes
- `chore`: Maintenance tasks

**Examples:**

```
feat(geodesic): add symplectic Euler integrator
fix(atlas): handle empty data in discover_charts_linear
test(metric): add serialize/deserialize round-trip test
docs(readme): add multi-modal usage example
```

---

## Getting Help

- **GitHub Issues:** For bug reports and feature requests
- **Discussions:** For questions and community support
- **Email:** For security-related issues (please report privately)

Thank you for contributing to ManifoldDB!
