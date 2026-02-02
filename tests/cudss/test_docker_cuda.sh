#!/bin/bash
#
# Test script to validate spineax builds and runs on both CUDA 12 and CUDA 13
# Usage: ./tests/cudss/test_docker_cuda.sh [--cu12-only | --cu13-only] [--no-build]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default settings
RUN_CU12=true
RUN_CU13=true
BUILD_IMAGES=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cu12-only)
            RUN_CU13=false
            shift
            ;;
        --cu13-only)
            RUN_CU12=false
            shift
            ;;
        --no-build)
            BUILD_IMAGES=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--cu12-only | --cu13-only] [--no-build]"
            echo ""
            echo "Options:"
            echo "  --cu12-only   Only test CUDA 12"
            echo "  --cu13-only   Only test CUDA 13"
            echo "  --no-build    Skip building Docker images (use existing)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Examples to test
EXAMPLES=(
    "examples/cudss/batched_rhs.py"
    "examples/cudss/composability.py"
    "examples/cudss/datatypes.py"
    "examples/cudss/outputs.py"
    "examples/cudss/solver_types.py"
)

# Track results
declare -A RESULTS
FAILED=0

log_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

build_image() {
    local cuda_version=$1
    local dockerfile="${SCRIPT_DIR}/Dockerfile.cu${cuda_version}"
    local image_name="spineax-cu${cuda_version}"

    log_info "Building Docker image for CUDA ${cuda_version}..."

    if docker build -f "${dockerfile}" -t "${image_name}" "${PROJECT_ROOT}"; then
        log_success "Built ${image_name}"
        return 0
    else
        log_error "Failed to build ${image_name}"
        return 1
    fi
}

run_example() {
    local cuda_version=$1
    local example=$2
    local image_name="spineax-cu${cuda_version}"

    log_info "Running ${example} on CUDA ${cuda_version}..."

    if docker run --rm --gpus all "${image_name}" python "${example}" 2>&1; then
        log_success "CUDA ${cuda_version}: ${example}"
        RESULTS["cu${cuda_version}:${example}"]="PASS"
        return 0
    else
        log_error "CUDA ${cuda_version}: ${example}"
        RESULTS["cu${cuda_version}:${example}"]="FAIL"
        FAILED=$((FAILED + 1))
        return 1
    fi
}

test_cuda_version() {
    local cuda_version=$1

    echo ""
    echo "========================================"
    echo " Testing CUDA ${cuda_version}"
    echo "========================================"
    echo ""

    if [ "$BUILD_IMAGES" = true ]; then
        if ! build_image "${cuda_version}"; then
            log_error "Skipping CUDA ${cuda_version} tests due to build failure"
            for example in "${EXAMPLES[@]}"; do
                RESULTS["cu${cuda_version}:${example}"]="SKIP"
            done
            FAILED=$((FAILED + ${#EXAMPLES[@]}))
            return 1
        fi
    fi

    for example in "${EXAMPLES[@]}"; do
        run_example "${cuda_version}" "${example}" || true
    done
}

build_image_no_pbatch() {
    local cuda_version=$1
    local dockerfile="${SCRIPT_DIR}/Dockerfile.cu${cuda_version}.no-pbatch"
    local image_name="spineax-cu${cuda_version}-no-pbatch"

    log_info "Building Docker image for CUDA ${cuda_version} (no pbatch)..."

    if docker build -f "${dockerfile}" -t "${image_name}" "${PROJECT_ROOT}"; then
        log_success "Built ${image_name}"
        return 0
    else
        log_error "Failed to build ${image_name}"
        return 1
    fi
}

test_no_pbatch_fallback() {
    local cuda_version=$1
    local image_name="spineax-cu${cuda_version}-no-pbatch"

    echo ""
    echo "========================================"
    echo " Testing CUDA ${cuda_version} (no pbatch - fallback)"
    echo "========================================"
    echo ""

    if [ "$BUILD_IMAGES" = true ]; then
        if ! build_image_no_pbatch "${cuda_version}"; then
            log_error "Skipping no-pbatch tests due to build failure"
            RESULTS["cu${cuda_version}:no-pbatch:fallback"]="SKIP"
            FAILED=$((FAILED + 1))
            return 1
        fi
    fi

    # Test that PBATCH_AVAILABLE is False
    log_info "Testing PBATCH_AVAILABLE=False on CUDA ${cuda_version} (no pbatch)..."
    if docker run --rm --gpus all "${image_name}" python -c "
from spineax.cudss import solver
assert solver.PBATCH_AVAILABLE == False, 'PBATCH_AVAILABLE should be False'
assert solver.vmap_using_pseudo_batch == False, 'vmap_using_pseudo_batch should be False'
print('PBATCH_AVAILABLE:', solver.PBATCH_AVAILABLE)
print('vmap_using_pseudo_batch:', solver.vmap_using_pseudo_batch)
print('Fallback mode verified!')
" 2>&1; then
        log_success "CUDA ${cuda_version}: no-pbatch fallback verified"
        RESULTS["cu${cuda_version}:no-pbatch:fallback"]="PASS"
    else
        log_error "CUDA ${cuda_version}: no-pbatch fallback test failed"
        RESULTS["cu${cuda_version}:no-pbatch:fallback"]="FAIL"
        FAILED=$((FAILED + 1))
        return 1
    fi

    # Test that solve still works in fallback mode
    log_info "Testing solve works in fallback mode on CUDA ${cuda_version}..."
    if docker run --rm --gpus all "${image_name}" python -c "
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
from spineax.cudss.solver import CuDSSSolver

A = jnp.array([[4., 0., 1.], [0., 3., 2.], [0., 0., 5.]], dtype=jnp.float32)
b = jnp.array([5., 11., 14.], dtype=jnp.float32)
LHS = jsparse.BCSR.fromdense(A)
solver = CuDSSSolver(LHS.indptr, LHS.indices, 0, 1, 1)
x, inertia = solver(b, LHS.data)
print('Solution:', x)
print('Solve works in fallback mode!')
" 2>&1; then
        log_success "CUDA ${cuda_version}: no-pbatch solve works"
        RESULTS["cu${cuda_version}:no-pbatch:solve"]="PASS"
    else
        log_error "CUDA ${cuda_version}: no-pbatch solve failed"
        RESULTS["cu${cuda_version}:no-pbatch:solve"]="FAIL"
        FAILED=$((FAILED + 1))
        return 1
    fi
}

print_summary() {
    echo ""
    echo "========================================"
    echo " Test Summary"
    echo "========================================"
    echo ""

    for key in "${!RESULTS[@]}"; do
        local result="${RESULTS[$key]}"
        case $result in
            PASS)
                log_success "$key"
                ;;
            FAIL)
                log_error "$key"
                ;;
            SKIP)
                echo -e "${YELLOW}[SKIP]${NC} $key"
                ;;
        esac
    done

    echo ""
    if [ $FAILED -eq 0 ]; then
        log_success "All tests passed!"
    else
        log_error "${FAILED} test(s) failed"
    fi
}

# Main execution
cd "${PROJECT_ROOT}"

echo "========================================"
echo " spineax Docker CUDA Test Suite"
echo "========================================"
echo ""
echo "Project root: ${PROJECT_ROOT}"
echo "Testing CUDA 12: ${RUN_CU12}"
echo "Testing CUDA 13: ${RUN_CU13}"
echo "Build images: ${BUILD_IMAGES}"

# Check for GPU access
if ! docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
    log_error "Cannot access GPU via Docker. Make sure:"
    echo "  1. NVIDIA Container Toolkit is installed"
    echo "  2. Docker daemon is configured for GPU access"
    echo "  3. You have a compatible NVIDIA GPU"
    exit 1
fi

log_success "GPU access verified"

# Run tests
if [ "$RUN_CU12" = true ]; then
    test_cuda_version 12
    test_no_pbatch_fallback 12
fi

if [ "$RUN_CU13" = true ]; then
    test_cuda_version 13
    test_no_pbatch_fallback 13
fi

print_summary

exit $FAILED
