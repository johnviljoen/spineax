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
from importlib.util import find_spec
from pathlib import Path

def get_base_path(spec):
    if spec.submodule_search_locations: return Path(spec.submodule_search_locations[0])
    return None

def find_lib(base_path, pattern):
    '''Find library matching pattern in lib/ directory'''
    libs = list((base_path / 'lib').glob(pattern))
    if not libs: return None
    # Return the one with longest name (has version suffix)
    return sorted(libs, key=lambda p: len(str(p)))[-1]

try:
    # CUDA 13 packages use nvidia.cu13 for everything
    # CUDA 12 packages use nvidia.cuda_runtime, nvidia.cublas, nvidia.cu12, etc.

    # Try to find the main CUDA module (cu13 or cu12)
    cu13_spec = find_spec('nvidia.cu13')
    cu12_spec = find_spec('nvidia.cu12')
    cuda_runtime_spec = find_spec('nvidia.cuda_runtime')

    if cu13_spec:
        # CUDA 13 mode - everything is in nvidia.cu13
        base_path = get_base_path(cu13_spec)
        site_packages = base_path.parent.parent
        print(f'NVIDIA_PATH={site_packages}')
        print(f'CUDA_VERSION=13')

        # All libs are in nvidia/cu13/lib
        cudart_lib = find_lib(base_path, 'libcudart.so*')
        if not cudart_lib: raise ImportError('libcudart.so not found in nvidia.cu13')
        print(f'CUDART_LIBRARY_PATH={cudart_lib.resolve()}')
        print(f'CUDART_INCLUDE_PATH={base_path}/include')

        cudss_lib = find_lib(base_path, 'libcudss.so*')
        if not cudss_lib: raise ImportError('libcudss.so not found in nvidia.cu13')
        print(f'CUDSS_LIBRARY_PATH={cudss_lib.resolve()}')
        print(f'CUDSS_INCLUDE_PATH={base_path}/include')

        cublas_lib = find_lib(base_path, 'libcublas.so*')
        if not cublas_lib: raise ImportError('libcublas.so not found in nvidia.cu13')
        print(f'CUBLAS_LIBRARY_PATH={cublas_lib.resolve()}')
        print(f'CUBLAS_INCLUDE_PATH={base_path}/include')

        # nvcc include is also in cu13
        print(f'NVCC_INCLUDE_PATH={base_path}/include')

    elif cuda_runtime_spec or cu12_spec:
        # CUDA 12 mode - separate packages
        if cuda_runtime_spec:
            cudart_path = get_base_path(cuda_runtime_spec)
        else:
            cudart_path = get_base_path(cu12_spec)
        site_packages = cudart_path.parent.parent
        print(f'NVIDIA_PATH={site_packages}')
        print(f'CUDA_VERSION=12')

        cudart_lib = find_lib(cudart_path, 'libcudart.so*')
        if not cudart_lib: raise ImportError('libcudart.so not found')
        print(f'CUDART_LIBRARY_PATH={cudart_lib.resolve()}')
        print(f'CUDART_INCLUDE_PATH={cudart_path}/include')

        # cuDSS is in nvidia.cu12
        cudss_spec = find_spec('nvidia.cu12')
        if not cudss_spec: raise ImportError('nvidia.cu12 not found for cuDSS')
        cudss_path = get_base_path(cudss_spec)
        cudss_lib = find_lib(cudss_path, 'libcudss.so*')
        if not cudss_lib: raise ImportError('libcudss.so not found')
        print(f'CUDSS_LIBRARY_PATH={cudss_lib.resolve()}')
        print(f'CUDSS_INCLUDE_PATH={cudss_path}/include')

        # cuBLAS
        cublas_spec = find_spec('nvidia.cublas')
        if not cublas_spec: raise ImportError('nvidia.cublas not found')
        cublas_path = get_base_path(cublas_spec)
        cublas_lib = find_lib(cublas_path, 'libcublas.so*')
        if not cublas_lib: raise ImportError('libcublas.so not found')
        print(f'CUBLAS_LIBRARY_PATH={cublas_lib.resolve()}')
        print(f'CUBLAS_INCLUDE_PATH={cublas_path}/include')

        # nvcc include
        nvcc_spec = find_spec('nvidia.cuda_nvcc')
        if nvcc_spec:
            nvcc_path = get_base_path(nvcc_spec)
            print(f'NVCC_INCLUDE_PATH={nvcc_path}/include')
    else:
        raise ImportError('No CUDA packages found (tried nvidia.cu13, nvidia.cu12, nvidia.cuda_runtime)')

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
string(REGEX MATCH "CUDART_INCLUDE_PATH=([^\n]*)" _ ${PYTHON_OUTPUTS})
set(ENV_CUDART_INCLUDE ${CMAKE_MATCH_1})
string(REGEX MATCH "CUDSS_LIBRARY_PATH=([^\n]*)" _ ${PYTHON_OUTPUTS})
set(ENV_CUDSS_PATH ${CMAKE_MATCH_1})
string(REGEX MATCH "CUDSS_INCLUDE_PATH=([^\n]*)" _ ${PYTHON_OUTPUTS})
set(ENV_CUDSS_INCLUDE ${CMAKE_MATCH_1})
string(REGEX MATCH "CUBLAS_LIBRARY_PATH=([^\n]*)" _ ${PYTHON_OUTPUTS})
set(ENV_CUBLAS_PATH ${CMAKE_MATCH_1})
string(REGEX MATCH "CUBLAS_INCLUDE_PATH=([^\n]*)" _ ${PYTHON_OUTPUTS})
set(ENV_CUBLAS_INCLUDE ${CMAKE_MATCH_1})
string(REGEX MATCH "NVCC_INCLUDE_PATH=([^\n]*)" _ ${PYTHON_OUTPUTS})
set(ENV_NVCC_INCLUDE ${CMAKE_MATCH_1})
string(REGEX MATCH "CUDA_VERSION=([^\n]*)" _ ${PYTHON_OUTPUTS})
set(CUDA_VERSION ${CMAKE_MATCH_1})

message(STATUS "Detected CUDA version: ${CUDA_VERSION}")
message(STATUS "Found Python CUDA libs at: ${ENV_NVIDIA_PATH}")
message(STATUS "Found libcudart.so at: ${ENV_CUDART_PATH}")
message(STATUS "Found libcudss.so at: ${ENV_CUDSS_PATH}")
message(STATUS "Found libcublas.so at: ${ENV_CUBLAS_PATH}")
message(STATUS "NVCC include path: ${ENV_NVCC_INCLUDE}")

# CUDART
add_library(cudart_from_python SHARED IMPORTED)
set_target_properties(cudart_from_python PROPERTIES
    IMPORTED_LOCATION "${ENV_CUDART_PATH}"
    INTERFACE_INCLUDE_DIRECTORIES "${ENV_CUDART_INCLUDE}"
)

# CUDSS
add_library(cudss_from_python SHARED IMPORTED)
set_target_properties(cudss_from_python PROPERTIES
    IMPORTED_LOCATION "${ENV_CUDSS_PATH}"
    INTERFACE_INCLUDE_DIRECTORIES "${ENV_CUDSS_INCLUDE}"
)

# cuBLAS
add_library(cublas_from_python SHARED IMPORTED)
set_target_properties(cublas_from_python PROPERTIES
    IMPORTED_LOCATION "${ENV_CUBLAS_PATH}"
    INTERFACE_INCLUDE_DIRECTORIES "${ENV_CUBLAS_INCLUDE}"
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
  src/spineax/cudss/pseudo_batch_solve.cu
)

# single solve and Return Everything (return all data that cuDSS has)
nanobind_add_module(single_solve_re NOMINSIZE 
  src/spineax/cudss/single_solve_re.cpp
)

# traditional single solve
nanobind_add_module(single_solve NOMINSIZE
  src/spineax/cudss/single_solve.cpp
)

# Get library directories from library paths
get_filename_component(ENV_CUDART_LIB_DIR "${ENV_CUDART_PATH}" DIRECTORY)
get_filename_component(ENV_CUDSS_LIB_DIR "${ENV_CUDSS_PATH}" DIRECTORY)
get_filename_component(ENV_CUBLAS_LIB_DIR "${ENV_CUBLAS_PATH}" DIRECTORY)

# Set INSTALL_RPATH based on CUDA version
# CUDA 13: all libs in nvidia/cu13/lib
# CUDA 12: libs in nvidia/cuda_runtime/lib, nvidia/cublas/lib, nvidia/cu12/lib
if(CUDA_VERSION STREQUAL "13")
    set(INSTALL_RPATH_VALUE "$ORIGIN/../nvidia/cu13/lib")
else()
    set(INSTALL_RPATH_VALUE "$ORIGIN/../nvidia/cu12/lib:$ORIGIN/../nvidia/cublas/lib:$ORIGIN/../nvidia/cuda_runtime/lib")
endif()

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
        "${ENV_NVCC_INCLUDE}"
    )

    target_link_libraries(${TARGET} PRIVATE
        cudss_from_python
        cudart_from_python
        cublas_from_python
    )

    # Suppress warnings from XLA FFI headers that have control flow issues
    # These are in jaxlib's XLA headers, not in spineax code
    target_compile_options(${TARGET} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-return-type -Wno-attributes>
    )
    
    # Add RPATH so the .so can find its dependencies
    # Use $ORIGIN to make paths relative to the installed module location
    set_target_properties(${TARGET} PROPERTIES
        BUILD_RPATH "${ENV_CUDSS_LIB_DIR}:${ENV_CUBLAS_LIB_DIR}:${ENV_CUDART_LIB_DIR}"
        INSTALL_RPATH "${INSTALL_RPATH_VALUE}"
        INSTALL_RPATH_USE_LINK_PATH FALSE
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
