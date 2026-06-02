"""
Setup script for ManifoldDB – Riemannian geometric inference engine.

Compiles the C++ library and PyBind11 bindings via
``torch.utils.cpp_extension.BuildExtension`` with optional CUDA support.

Usage::

    pip install .              # CPU-only build
    MANIFOLDDB_CUDA=1 pip install .   # Enable CUDA kernels
"""

import glob
import os
import sys

from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _cuda_available():
    """Check whether the CUDA toolkit is reachable."""
    try:
        from torch.utils.cpp_extension import CUDA_HOME
        return CUDA_HOME is not None and os.path.isdir(CUDA_HOME)
    except ImportError:
        return False


def _gather_sources(pattern="cpp/src/*.cpp"):
    """Return a sorted list of .cpp source files matching *pattern*."""
    files = sorted(glob.glob(os.path.join(ROOT, pattern)))
    if not files:
        raise FileNotFoundError(
            f"No source files found for pattern '{pattern}' in {ROOT}"
        )
    return files


# ═══════════════════════════════════════════════════════════════════════════════
#  Paths
# ═══════════════════════════════════════════════════════════════════════════════

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT, "cpp", "src")
INCLUDE_DIRS = [
    os.path.join(ROOT, "cpp", "include"),
]

# ═══════════════════════════════════════════════════════════════════════════════
#  Source files
# ═══════════════════════════════════════════════════════════════════════════════

# Collect ALL .cpp files from cpp/src/
SOURCES = _gather_sources("cpp/src/*.cpp")

# ═══════════════════════════════════════════════════════════════════════════════
#  Compiler flags
# ═══════════════════════════════════════════════════════════════════════════════

extra_compile_args = {
    "cxx": ["-O3", "-std=c++20", "-w", "-DEIGEN_NO_DEBUG"],
    "nvcc": ["-O3", "--use_fast_math", "-std=c++20"],
}

extra_link_args: list[str] = []

# ═══════════════════════════════════════════════════════════════════════════════
#  CUDA detection
# ═══════════════════════════════════════════════════════════════════════════════

USE_CUDA = os.environ.get("MANIFOLDDB_CUDA", "0") == "1" and _cuda_available()

# ═══════════════════════════════════════════════════════════════════════════════
#  Build the extension
# ═══════════════════════════════════════════════════════════════════════════════

if USE_CUDA:
    ext = CUDAExtension(
        name="manifolddb.manifolddb_core",
        sources=SOURCES,
        include_dirs=INCLUDE_DIRS,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
else:
    ext = CppExtension(
        name="manifolddb.manifolddb_core",
        sources=SOURCES,
        include_dirs=INCLUDE_DIRS,
        extra_compile_args=extra_compile_args["cxx"],
        extra_link_args=extra_link_args,
    )

# ═══════════════════════════════════════════════════════════════════════════════
#  Packages
# ═══════════════════════════════════════════════════════════════════════════════

packages = find_packages(
    where="python",
    exclude=("tests*", "docs*", "examples*"),
)

# ═══════════════════════════════════════════════════════════════════════════════
#  setup()
# ═══════════════════════════════════════════════════════════════════════════════

setup(
    name="manifolddb",
    version="0.1.0",
    description=(
        "Riemannian geometric inference engine with geodesic solvers "
        "and manifold-aware data structures"
    ),
    long_description=open(os.path.join(ROOT, "README.md")).read(),
    long_description_content_type="text/markdown",
    author="ManifoldDB Contributors",
    license="MIT",
    url="https://github.com/manifolddb/manifolddb",
    python_requires=">=3.9",
    packages=packages,
    package_dir={
        "manifolddb": "python/manifolddb",
    },
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
    install_requires=[
        "torch>=1.12",
        "numpy>=1.21",
        "scikit-learn>=1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "mypy>=1.0",
            "ruff>=0.1",
        ],
        "cuda": ["torch>=1.12"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
