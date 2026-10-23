# 3.26 is the floor for find_package(Python COMPONENTS Development.SABIModule),
# which nanobind's STABLE_ABI (abi3 wheel) build requires.
cmake_minimum_required(VERSION 3.26...3.27)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

# Force all CUDA libraries to be linked dynamically.
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# 1. Find the Python interpreter from your active environment.
# Development.SABIModule is required so nanobind's STABLE_ABI actually engages and
# emits a limited-API pbatch_solve.abi3.so; without it STABLE_ABI is silently
# ignored and the module is built version-specific (e.g. .cpython-313.so).
find_package(Python COMPONENTS Interpreter Development.Module Development.SABIModule REQUIRED)
message(STATUS "Using Python interpreter: ${Python_EXECUTABLE}")

# 2. Use NVIDIA libs from Python site-packages/nvidia/*/ (directories there contain include/ and lib/).
# --- Derive NVIDIA prefix from CMAKE_PREFIX_PATH ---
# We look for a prefix containing "site-packages", then scan nvidia/* for include/lib.

set(_site_prefix "")
foreach(_p IN LISTS CMAKE_PREFIX_PATH)
  message(STATUS "CMAKE_PREFIX_PATH entry: ${_p}")
  # Support both site-packages (pip/venv) and dist-packages (Debian/Ubuntu system Python)
  if(_p MATCHES "site-packages" OR _p MATCHES "dist-packages")
    # Under pip build isolation the ephemeral /tmp/pip-build-env-*/overlay
    # site-packages also appears in CMAKE_PREFIX_PATH but contains no nvidia
    # packages — only accept a prefix that actually has nvidia/ in it, and
    # otherwise keep the first plain match as a fallback for the error message.
    if(EXISTS "${_p}/nvidia")
      set(_site_prefix "${_p}")
      break()
    elseif(NOT _site_prefix)
      set(_site_prefix "${_p}")
    endif()
  endif()
endforeach()

if(NOT _site_prefix)
  message(FATAL_ERROR
    "Could not find a 'site-packages' or 'dist-packages' entry in CMAKE_PREFIX_PATH.\n"
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
  NAMES cudss libcudss libcudss.so libcudss.so.1
  HINTS ${ENV_NVIDIA_LIB_DIRS}
  NO_DEFAULT_PATH
  REQUIRED
)

find_library(ENV_CUSPARSE_PATH
  NAMES cusparse libcusparse libcusparse.so libcusparse.so.13 libcusparse.so.12
  HINTS ${ENV_NVIDIA_LIB_DIRS}
  NO_DEFAULT_PATH
  REQUIRED
)

message(STATUS "Found libcudart at: ${ENV_CUDART_PATH}")
message(STATUS "Found libcudss  at: ${ENV_CUDSS_PATH}")
message(STATUS "Found libcublas at: ${ENV_CUBLAS_PATH}")
message(STATUS "Found libcusparse at: ${ENV_CUSPARSE_PATH}")

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

add_library(nvidia::cusparse SHARED IMPORTED GLOBAL)
set_target_properties(nvidia::cusparse PROPERTIES
  IMPORTED_LOCATION "${ENV_CUSPARSE_PATH}"
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

# cuDSS FFI files

# The token API (docs/token_design.md): every solve is the batch_size>=1 case
# of the block-diagonal construction, so this one module is the whole backend.
# Plain C++ since the zero-copy redesign (design doc step 10): no device code,
# only CUDA runtime API calls — nvcc is NOT required to build spineax.
#
# STABLE_ABI: compile against CPython's limited API (Py_LIMITED_API) so a single
# abi3 wheel works on every CPython >= 3.12. nanobind requires >= 3.12 for stable
# ABI, which is why pyproject sets requires-python = ">=3.12". Paired with
# [tool.scikit-build.wheel] py-api = "cp312" so the wheel is tagged cp312-abi3.
nanobind_add_module(pbatch_solve STABLE_ABI NOMINSIZE
  spineax/cudss/solver.cpp
  spineax/cudss/csr_transpose.cpp
)

# Build list of targets
set(SPINEAX_TARGETS pbatch_solve)

foreach(TARGET ${SPINEAX_TARGETS})
  target_include_directories(${TARGET} PRIVATE
    ${XLA_DIR}
    nvidia::cudss
    nvidia::cudart
    nvidia::cublas
    nvidia::cusparse
    ${ENV_NVIDIA_INCLUDE_DIRS}
  )

  target_link_libraries(${TARGET} PRIVATE
    nvidia::cudss
    nvidia::cudart
    nvidia::cublas
    nvidia::cusparse
  )

  # Suppress warnings from XLA FFI headers that have control flow issues
  # These are in jaxlib's XLA headers, not in spineax code
  target_compile_options(${TARGET} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-return-type -Wno-attributes>
  )

  # Add RPATH so the .so can find its dependencies
  # Use $ORIGIN to make paths relative to the installed module location
  # Modules install to site-packages/spineax/, CUDA libs are in site-packages/nvidia/*/lib/
  # INSTALL_RPATH: the portable $ORIGIN entries resolve in a real site-packages
  # install (spineax/ and nvidia/ are siblings). Editable installs load the .so
  # from the build tree where no sibling nvidia/ exists, so the absolute env lib
  # dirs are appended as a fallback (ignored where they don't exist).
  set_target_properties(${TARGET} PROPERTIES
    BUILD_RPATH "${ENV_NVIDIA_BUILD_RPATH}"
    INSTALL_RPATH "${ENV_NVIDIA_INSTALL_RPATH}:${ENV_NVIDIA_BUILD_RPATH}"
    INSTALL_RPATH_USE_LINK_PATH FALSE
  )
endforeach()

install(
  TARGETS ${SPINEAX_TARGETS}
  LIBRARY
  DESTINATION spineax
)
