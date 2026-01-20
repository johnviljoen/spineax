cmake_minimum_required(VERSION 3.15...3.27)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

# Force all CUDA libraries to be linked dynamically.
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# 1. Find the Python interpreter from your active environment.
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
message(STATUS "Using Python interpreter: ${Python_EXECUTABLE}")

# 2. Use NVIDIA libs from Python site-packages/nvidia/*/ (directories there contain include/ and lib/).
# --- Derive NVIDIA prefix from CMAKE_PREFIX_PATH ---
# We look for a prefix containing "site-packages", then scan nvidia/* for include/lib.

set(_site_prefix "")
foreach(_p IN LISTS CMAKE_PREFIX_PATH)
  message(STATUS "CMAKE_PREFIX_PATH entry: ${_p}")
  if(_p MATCHES "site-packages")
    set(_site_prefix "${_p}")
  endif()
endforeach()

if(NOT _site_prefix)
  message(FATAL_ERROR
    "Could not find a 'site-packages' entry in CMAKE_PREFIX_PATH.\n"
    "CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}\n")
endif()

set(_nvidia_root "${_site_prefix}/nvidia")

set(ENV_NVIDIA_LIB_DIRS "")
set(ENV_NVIDIA_INCLUDE_DIRS "")
set(ENV_NVIDIA_LEAFS "")

if(EXISTS "${_nvidia_root}")
  file(GLOB _nvidia_candidates LIST_DIRECTORIES true "${_nvidia_root}/*")
  foreach(_cand IN LISTS _nvidia_candidates)
    if(IS_DIRECTORY "${_cand}")
      if(EXISTS "${_cand}/lib")
        list(APPEND ENV_NVIDIA_LIB_DIRS "${_cand}/lib")
        get_filename_component(_leaf "${_cand}" NAME)
        list(APPEND ENV_NVIDIA_LEAFS "${_leaf}")
      endif()
      if(EXISTS "${_cand}/include")
        list(APPEND ENV_NVIDIA_INCLUDE_DIRS "${_cand}/include")
      endif()
    endif()
  endforeach()
endif()

list(REMOVE_DUPLICATES ENV_NVIDIA_LIB_DIRS)
list(REMOVE_DUPLICATES ENV_NVIDIA_INCLUDE_DIRS)
list(REMOVE_DUPLICATES ENV_NVIDIA_LEAFS)

if(NOT ENV_NVIDIA_LIB_DIRS)
  message(FATAL_ERROR
    "Found site-packages prefix: ${_site_prefix}\n"
    "But could not find any nvidia/* with lib/ and include/ under:\n"
    "  ${_nvidia_root}")
endif()

set(ENV_NVIDIA_BUILD_RPATH "")
foreach(_libdir IN LISTS ENV_NVIDIA_LIB_DIRS)
  if(ENV_NVIDIA_BUILD_RPATH)
    set(ENV_NVIDIA_BUILD_RPATH "${ENV_NVIDIA_BUILD_RPATH}:${_libdir}")
  else()
    set(ENV_NVIDIA_BUILD_RPATH "${_libdir}")
  endif()
endforeach()

set(ENV_NVIDIA_INSTALL_RPATH "")
foreach(_leaf IN LISTS ENV_NVIDIA_LEAFS)
  if(ENV_NVIDIA_INSTALL_RPATH)
    set(ENV_NVIDIA_INSTALL_RPATH "${ENV_NVIDIA_INSTALL_RPATH}:$ORIGIN/../nvidia/${_leaf}/lib")
  else()
    set(ENV_NVIDIA_INSTALL_RPATH "$ORIGIN/../nvidia/${_leaf}/lib")
  endif()
endforeach()

message(STATUS "Using NVIDIA lib dirs: ${ENV_NVIDIA_LIB_DIRS}")
foreach(_libdir IN LISTS ENV_NVIDIA_LIB_DIRS)
  message(STATUS "  ${_libdir}")
endforeach()
message(STATUS "Using NVIDIA include dirs:")
foreach(_incdir IN LISTS ENV_NVIDIA_INCLUDE_DIRS)
  message(STATUS "  ${_incdir}")
endforeach()

find_library(ENV_CUDART_PATH
  NAMES cudart libcudart libcudart.so libcudart.so.13 libcudart.so.12
  HINTS ${ENV_NVIDIA_LIB_DIRS}
  NO_DEFAULT_PATH
  REQUIRED
)

find_library(ENV_CUBLAS_PATH
  NAMES cublas libcublas libcublas.so libcublas.so.13 libcublas.so.12
  HINTS ${ENV_NVIDIA_LIB_DIRS}
  NO_DEFAULT_PATH
  REQUIRED
)

find_library(ENV_CUDSS_PATH
  NAMES cudss libcudss libcudss.so libcudss.so.0 libcudss.so.13 libcudss.so.12
  HINTS ${ENV_NVIDIA_LIB_DIRS}
  NO_DEFAULT_PATH
  REQUIRED
)

message(STATUS "Found libcudart at: ${ENV_CUDART_PATH}")
message(STATUS "Found libcudss  at: ${ENV_CUDSS_PATH}")
message(STATUS "Found libcublas at: ${ENV_CUBLAS_PATH}")

# --- Imported targets (clean linking) ---
add_library(nvidia::cudart SHARED IMPORTED GLOBAL)
set_target_properties(nvidia::cudart PROPERTIES
  IMPORTED_LOCATION "${ENV_CUDART_PATH}"
  INTERFACE_INCLUDE_DIRECTORIES "${ENV_NVIDIA_INCLUDE_DIRS}"
)

add_library(nvidia::cudss SHARED IMPORTED GLOBAL)
set_target_properties(nvidia::cudss PROPERTIES
  IMPORTED_LOCATION "${ENV_CUDSS_PATH}"
  INTERFACE_INCLUDE_DIRECTORIES "${ENV_NVIDIA_INCLUDE_DIRS}"
)

add_library(nvidia::cublas SHARED IMPORTED GLOBAL)
set_target_properties(nvidia::cublas PROPERTIES
  IMPORTED_LOCATION "${ENV_CUBLAS_PATH}"
  INTERFACE_INCLUDE_DIRECTORIES "${ENV_NVIDIA_INCLUDE_DIRS}"
)

find_package(nanobind CONFIG REQUIRED)

# XLA
execute_process(
  COMMAND "${Python_EXECUTABLE}"
  "-c" "from jax import ffi; print(ffi.include_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR)
message(STATUS "XLA include directory: ${XLA_DIR}")

# 3 optimization build
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
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

foreach(TARGET
  batch_solve
  pbatch_solve
  single_solve
  single_solve_re
)
  target_include_directories(${TARGET} PRIVATE
    ${XLA_DIR}
    nvidia::cudss
    nvidia::cudart
    nvidia::cublas
    ${ENV_NVIDIA_INCLUDE_DIRS}
  )

  target_link_libraries(${TARGET} PRIVATE
    nvidia::cudss
    nvidia::cudart
    nvidia::cublas
  )

  # Suppress warnings from XLA FFI headers that have control flow issues
  # These are in jaxlib's XLA headers, not in spineax code
  target_compile_options(${TARGET} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-return-type -Wno-attributes>
  )

  # Add RPATH so the .so can find its dependencies
  # Use $ORIGIN to make paths relative to the installed module location
  # Modules install to site-packages/spineax/, CUDA libs are in site-packages/nvidia/*/lib/
  set_target_properties(${TARGET} PROPERTIES
    BUILD_RPATH "${ENV_NVIDIA_BUILD_RPATH}"
    INSTALL_RPATH "${ENV_NVIDIA_INSTALL_RPATH}"
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
