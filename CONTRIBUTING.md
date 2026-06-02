# Contributing to Manifold Database

Thank you for your interest in contributing to Manifold Database! This guide will help you get started with development and ensure your contributions align with our standards.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Commit Messages](#commit-messages)
- [Pull Requests](#pull-requests)
- [Issue Guidelines](#issue-guidelines)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you agree to abide by its terms. Be respectful, inclusive, and constructive in all interactions.

## Development Setup

### Prerequisites

- **Python 3.10+** (tested on 3.10, 3.11, 3.12)
- **Git** with SSH or HTTPS access
- **pip** or **conda** for package management

### Clone and Install

```bash
# Clone the repository
git clone https://github.com/manifold-db/manifold-db.git
cd manifold-db

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Verify installation
manifold-db version
python -c "import manifold_db; print(manifold_db.__version__)"
```

### Quick Start

```bash
# Install pre-commit hooks
pre-commit install

# Run linting and formatting
make lint
make format

# Run tests
make test

# Full check (lint + test)
make check
```

## Code Style

Manifold Database enforces consistent code style using the following tools:

| Tool | Purpose | Config |
|------|---------|--------|
| **Black** | Code formatting | `line-length = 88` |
| **Ruff** | Linting and import sorting | `select = ["E", "F", "W", "I"]` |
| **isort** | Import ordering | `profile = "black"` |
| **mypy** | Static type checking | `strict` mode |

### Formatting

```bash
# Auto-format all files
black manifold_db/ tests/

# Sort imports
isort manifold_db/ tests/

# Or use the Makefile target
make format
```

### Linting

```bash
# Run linter
ruff check manifold_db/ tests/

# Auto-fix lint issues
ruff check --fix manifold_db/ tests/

# Type checking
mypy manifold_db/

# Full lint check
make lint
```

### Pre-commit Hooks

We use pre-commit to run checks automatically on every commit:

```bash
pre-commit install          # Install hooks
pre-commit run --all-files  # Run all hooks manually
```

## Testing

### Running Tests

```bash
# Run all tests
make test

# Run with pytest directly
pytest tests/ -v

# Run a specific test file
pytest tests/unit/test_tangent_index.py -v

# Run a specific test function
pytest tests/unit/test_tangent_index.py::test_project_lift_roundtrip -v

# Run by marker
pytest -m "unit"           # Unit tests only
pytest -m "integration"   # Integration tests only
pytest -m "benchmark"      # Benchmark tests only
```

### Test Coverage

```bash
# Run tests with coverage
make test-cov

# Generate HTML coverage report
pytest --cov=manifold_db --cov-report=html tests/
open htmlcov/index.html
```

We maintain a minimum coverage of **85%** for all new code. Check the coverage badge on the README.

### Writing Tests

- Place unit tests in `tests/unit/` mirroring the source structure
- Place integration tests in `tests/integration/`
- Use `pytest` fixtures defined in `tests/conftest.py`
- Follow the Arrange-Act-Assert pattern
- Use descriptive test names: `test_<what>_<when>_<expected>`

```python
def test_tangent_space_project_lift_roundtrip():
    """Project and lift should recover the original point."""
    ts = TangentSpace(base_point=np.zeros(3), data=np.random.randn(50, 3))
    original = np.random.randn(3)
    tangent_coords = ts.project(original)
    recovered = ts.lift(tangent_coords)
    np.testing.assert_allclose(recovered, original, atol=1e-10)
```

## Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/) for all commit messages:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation changes |
| `style` | Code style (formatting, semicolons) |
| `refactor` | Code refactoring (no feature/fix) |
| `perf` | Performance improvement |
| `test` | Adding or updating tests |
| `chore` | Maintenance tasks |
| `ci` | CI/CD changes |

### Examples

```
feat(atlas): add automatic chart overlap detection
fix(geodesic): correct RK45 step size for high-curvature regions
docs(readme): add cross-modal retrieval example
test(tangent): add parallel transport batch tests
refactor(metric): simplify learned metric forward pass
perf(index): batch insert with vectorised anchor lookup
```

## Pull Requests

### Before Submitting

1. **Run the full check**: `make check` (lint + test)
2. **Update tests** for any new or changed functionality
3. **Update documentation** for any public API changes
4. **Rebase** on the latest `main` branch
5. **Squash** fixup commits into logical units

### PR Checklist

- [ ] All tests pass (`make check`)
- [ ] New code has test coverage
- [ ] Documentation updated (if applicable)
- [ ] Commit messages follow Conventional Commits
- [ ] No new linting or type errors
- [ ] PR description clearly explains the change

### PR Template

When opening a PR, please include:

- **Summary**: What does this PR do?
- **Motivation**: Why is this change needed?
- **Changes**: List of files modified
- **Testing**: How was this tested?
- **Breaking changes**: Any public API changes?

## Issue Guidelines

When opening an issue, please use the appropriate template:

- 🐛 **Bug Report**: Unexpected behaviour or crashes → [`bug_report.md`](.github/ISSUE_TEMPLATE/bug_report.md)
- ✨ **Feature Request**: New functionality or enhancements → [`feature_request.md`](.github/ISSUE_TEMPLATE/feature_request.md)
- ❓ **Question**: Usage questions or clarification → [`question.md`](.github/ISSUE_TEMPLATE/question.md)

### Good Issue Reports Include

- **Clear title** summarising the issue
- **Minimal reproducible example** when applicable
- **Expected vs actual behaviour**
- **Environment details** (Python version, OS, installation method)
- **Logs or stack traces** (formatted in code blocks)

---

Thank you for contributing to Manifold Database! 🙏
