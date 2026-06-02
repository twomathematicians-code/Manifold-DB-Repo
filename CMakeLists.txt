cmake_minimum_required(VERSION 3.18)
project(manifolddb
    VERSION 0.1.0
    DESCRIPTION "Riemannian geometric inference engine"
    LANGUAGES CXX
)

# ── C++ Standard ──────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Optional CUDA ─────────────────────────────────────────────────────────────
option(MANIFOLDDB_ENABLE_CUDA "Enable CUDA support" OFF)
if(MANIFOLDDB_ENABLE_CUDA)
    enable_language(CUDA)
    message(STATUS "CUDA support enabled")
endif()

# ── Dependencies ──────────────────────────────────────────────────────────────
find_package(Torch REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(pybind11 REQUIRED)

# ── Gather source files from cpp/src/ ─────────────────────────────────────────
file(GLOB MANIFOLDDB_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/src/*.cpp"
)

# ── Core library ───────────────────────────────────────────────────────────────
pybind11_add_module(manifolddb_core
    ${MANIFOLDDB_SOURCES}
)

target_include_directories(manifolddb_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)

# Torch provides its own transitive deps (CUDA, MKL, etc.)
target_link_libraries(manifolddb_core PRIVATE
    Torch::Torch
    Eigen3::Eigen
)

# Release optimisations
target_compile_options(manifolddb_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>: -O3>
)

# Suppress Eigen debug / assertion noise in release builds
target_compile_definitions(manifolddb_core PRIVATE
    EIGEN_NO_DEBUG
)

# ── Install ────────────────────────────────────────────────────────────────────
install(TARGETS manifolddb_core
    LIBRARY DESTINATION manifolddb
)
