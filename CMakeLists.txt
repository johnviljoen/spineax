cmake_minimum_required(VERSION 3.15...3.27)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

# Force all CUDA libraries to be linked dynamically.
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# 1. Find the Python interpreter from your active environment.
find_package(Python COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Using Python interpreter: ${Python_EXECUTABLE}")

# 2. Get the path to the CUDA libraries from the Python environment.
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "
import sys
import os
from importlib.util import find_spec
from pathlib import Path

def get_base_path(spec):
    if spec.submodule_search_locations: return Path(spec.submodule_search_locations[0])
    return None

try:
    # Find the base path for nvidia packages
    cudnn_spec = find_spec('nvidia.cudnn')
    if cudnn_spec is None: raise ImportError('A nvidia-* pip package must be installed.')
    base_path = get_base_path(cudnn_spec)
    if base_path is None: raise ImportError('Could not determine path for nvidia packages.')
    site_packages = base_path.parent.parent
    print(f'NVIDIA_PATH={site_packages}')

    # Find CUDA Runtime
    cudart_spec = find_spec('nvidia.cuda_runtime')
    if cudart_spec is None: raise ImportError('nvidia.cuda_runtime not found.')
    cudart_path = get_base_path(cudart_spec)
    cudart_libs = list((cudart_path / 'lib').glob('libcudart.so*'))
    if not cudart_libs: raise ImportError(f'libcudart.so not found in {cudart_path}')
    cudart_lib_file = sorted(cudart_libs, key=lambda p: len(str(p)))[-1]
    print(f'CUDART_LIBRARY_PATH={cudart_lib_file.resolve()}')

    # Find cuDSS
    cudss_spec = find_spec('nvidia.cu12')
    if cudss_spec is None: raise ImportError('nvidia.cudss not found.')
    cudss_path = get_base_path(cudss_spec)
    cudss_libs = list((cudss_path / 'lib').glob('libcudss.so*'))
    if not cudss_libs: raise ImportError(f'libcudss.so not found in {cudss_path}')
    cudss_lib_file = sorted(cudss_libs, key=lambda p: len(str(p)))[-1]
    print(f'CUDSS_LIBRARY_PATH={cudss_lib_file.resolve()}')

    # Find cuBLAS
    cublas_spec = find_spec('nvidia.cublas')
    if cublas_spec is None: raise ImportError('nvidia.cublas not found. Did you pip install nvidia-cublas-cu12?')
    cublas_path = get_base_path(cublas_spec)
    cublas_libs = list((cublas_path / 'lib').glob('libcublas.so*'))
    if not cublas_libs: raise ImportError(f'libcublas.so not found in {cublas_path}')
    cublas_lib_file = sorted(cublas_libs, key=lambda p: len(str(p)))[-1]
    print(f'CUBLAS_LIBRARY_PATH={cublas_lib_file.resolve()}')

except ImportError as e:
    sys.exit(f'Could not find required packages: {e}')
"
    OUTPUT_VARIABLE PYTHON_OUTPUTS
    RESULT_VARIABLE PYTHON_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(PYTHON_RESULT)
    message(FATAL_ERROR "Failed to find NVIDIA packages: ${PYTHON_RESULT}")
endif()

# Parse outputs
string(REGEX MATCH "NVIDIA_PATH=([^\n]*)" _ ${PYTHON_OUTPUTS})
set(ENV_NVIDIA_PATH ${CMAKE_MATCH_1})
string(REGEX MATCH "CUDART_LIBRARY_PATH=([^\n]*)" _ ${PYTHON_OUTPUTS})
set(ENV_CUDART_PATH ${CMAKE_MATCH_1})
string(REGEX MATCH "CUDSS_LIBRARY_PATH=([^\n]*)" _ ${PYTHON_OUTPUTS})
set(ENV_CUDSS_PATH ${CMAKE_MATCH_1})
string(REGEX MATCH "CUBLAS_LIBRARY_PATH=([^\n]*)" _ ${PYTHON_OUTPUTS})
set(ENV_CUBLAS_PATH ${CMAKE_MATCH_1})

message(STATUS "Found Python CUDA libs at: ${ENV_NVIDIA_PATH}")
message(STATUS "Found libcudart.so at: ${ENV_CUDART_PATH}")
message(STATUS "Found libcudss.so at: ${ENV_CUDSS_PATH}")
message(STATUS "Found libcublas.so at: ${ENV_CUBLAS_PATH}")

# CUDART
add_library(cudart_from_python SHARED IMPORTED)
set_target_properties(cudart_from_python PROPERTIES
    IMPORTED_LOCATION "${ENV_CUDART_PATH}"
    INTERFACE_INCLUDE_DIRECTORIES "${ENV_NVIDIA_PATH}/nvidia/cuda_runtime/include"
)

# CUDSS
add_library(cudss_from_python SHARED IMPORTED)
set_target_properties(cudss_from_python PROPERTIES
    IMPORTED_LOCATION "${ENV_CUDSS_PATH}"
    INTERFACE_INCLUDE_DIRECTORIES "${ENV_NVIDIA_PATH}/nvidia/cu12/include"
)

# cuBLAS
add_library(cublas_from_python SHARED IMPORTED)
set_target_properties(cublas_from_python PROPERTIES
    IMPORTED_LOCATION "${ENV_CUBLAS_PATH}"
    INTERFACE_INCLUDE_DIRECTORIES "${ENV_NVIDIA_PATH}/nvidia/cublas/include"
)

find_package(nanobind CONFIG REQUIRED)

# XLA
execute_process(
  COMMAND "${Python_EXECUTABLE}"
          "-c" "from jax import ffi; print(ffi.include_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR)
message(STATUS "XLA include directory: ${XLA_DIR}")

# O3 optimization build
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

enable_language(CUDA)

# cuDSS FFI files

# traditional batch
nanobind_add_module(batch_solve NOMINSIZE 
  src/spineax/cudss/batch_solve.cpp
)

# pseudo batch (enable features lacking in normal batch whilst also solving a batch)
nanobind_add_module(pbatch_solve NOMINSIZE 
  src/spineax/cudss/pseudo_batch_solve.cpp
)

# single solve and Return Everything (return all data that cuDSS has)
nanobind_add_module(single_solve_re NOMINSIZE 
  src/spineax/cudss/single_solve_re.cpp
)

# traditional single solve
nanobind_add_module(single_solve NOMINSIZE
  src/spineax/cudss/single_solve.cpp
)

foreach(TARGET 
  batch_solve
  pbatch_solve
  single_solve 
  single_solve_re
)
    target_include_directories(${TARGET} PRIVATE
        ${XLA_DIR}
        cudss_from_python
        cudart_from_python
        cublas_from_python
        "${ENV_NVIDIA_PATH}/nvidia/cuda_nvcc/include"
    )

    target_link_libraries(${TARGET} PRIVATE
        cudss_from_python
        cudart_from_python
        cublas_from_python
    )
    
    # Add RPATH so the .so can find its dependencies
    set_target_properties(${TARGET} PROPERTIES
        BUILD_RPATH "${ENV_NVIDIA_PATH}/nvidia/cu12/lib:${ENV_NVIDIA_PATH}/nvidia/cublas/lib:${ENV_NVIDIA_PATH}/nvidia/cuda_runtime/lib"
        INSTALL_RPATH "${ENV_NVIDIA_PATH}/nvidia/cu12/lib:${ENV_NVIDIA_PATH}/nvidia/cublas/lib:${ENV_NVIDIA_PATH}/nvidia/cuda_runtime/lib"
    )
endforeach()

install(
  TARGETS 
    batch_solve
    pbatch_solve
    single_solve 
    single_solve_re
  LIBRARY 
  DESTINATION spineax
)