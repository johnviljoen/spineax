/*
This is a different method for solving batch systems with cuDSS without actually
using cuDSS's batch API. This is done as it allows us to retrieve some data which
is not currently made available when using the batch API. Therefore in subsequent
releases of cuDSS this file will become redundant and we should only use 
- single_solve.cpp
- batch_solve.cpp
- ragged_solve.cpp

But for now (cuDSS version 0.6.0), this allows us to get batch inertias out before
they officially support it

(I previously made this file .cu just so cmakelists sees that it needs nvcc for the summation kernel)
*/

#include <cstdint>
#include <memory>
#include <vector>
#include <complex>
#include <type_traits>
// #include <cuComplex.h> // For device-side complex number operations

#include "cuda_runtime_api.h"
#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"
#include "cudss.h"

namespace ffi = xla::ffi;
namespace nb = nanobind;

// verification ================================================================
#define CUDSS_CALL_AND_CHECK(call, status, msg) \
    do { \
        status = call; \
        if (status != CUDSS_STATUS_SUCCESS) { \
            printf("Example FAILED: CUDSS call ended unsuccessfully with status = %d, details: " #msg "\n", status); \
            return ffi::Error::Success(); \
        } \
    } while(0);

// debugging ===================================================================
template <typename T>
void print_device_data(
    const char* label,
    void* device_ptr,
    size_t n_batch,
    size_t n_elements_per_batch)
{
    // Ensure we have a valid pointer and something to print
    if (!device_ptr || n_batch == 0 || n_elements_per_batch == 0) return;

    std::cout << "\n--- Debug Print: " << label << " ---" << std::endl;

    // Calculate total size and create a host-side vector
    size_t total_elements = n_batch * n_elements_per_batch;
    std::vector<T> host_data(total_elements);

    // Copy all data from GPU to CPU in one go
    cudaMemcpy(
        host_data.data(),
        device_ptr,
        total_elements * sizeof(T),
        cudaMemcpyDeviceToHost
    );

    // Loop through each batch and print its contents
    for (size_t i = 0; i < n_batch; ++i) {
        std::cout << "Batch " << i << ": [";
        size_t batch_start_index = i * n_elements_per_batch;
        for (size_t j = 0; j < n_elements_per_batch; ++j) {
            std::cout << host_data[batch_start_index + j];
            if (j < n_elements_per_batch - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "------------------------------------" << std::endl;
}


// Helper function for data types ==============================================
template <ffi::DataType T> cudaDataType get_cuda_data_type();
template<> cudaDataType get_cuda_data_type<ffi::F32>() { return CUDA_R_32F; }
template<> cudaDataType get_cuda_data_type<ffi::F64>() { return CUDA_R_64F; }
template<> cudaDataType get_cuda_data_type<ffi::C64>() { return CUDA_C_32F; }
template<> cudaDataType get_cuda_data_type<ffi::C128>() { return CUDA_C_64F; }

template <ffi::DataType T>
struct get_native_data_type;
template<> struct get_native_data_type<ffi::F32> { using type = float; };
template<> struct get_native_data_type<ffi::F64> { using type = double; };
template<> struct get_native_data_type<ffi::C64> { using type = std::complex<float>; };
template<> struct get_native_data_type<ffi::C128> { using type = std::complex<double>; };

// Structure definitions =======================================================
template <ffi::DataType T>
struct CudssBatchState {
    static xla::ffi::TypeId id;
    cudssHandle_t handle = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t data = nullptr;
    cudssMatrix_t A = nullptr;
    cudssMatrix_t x = nullptr;
    cudssMatrix_t b = nullptr;
    cudssMatrixType_t mtype = CUDSS_MTYPE_SYMMETRIC;
    cudssMatrixViewType_t mview = CUDSS_MVIEW_UPPER;
    cudssIndexBase_t base = CUDSS_BASE_ZERO;
    cudssStatus_t status = CUDSS_STATUS_SUCCESS;
    int32_t* batched_offsets_ptr = nullptr; // the pseudo batch must form these manually in C++
    int32_t* batched_columns_ptr = nullptr; // the pseudo batch must form these manually in C++
    int64_t n;
    int64_t nnz;
    int64_t nrhs;
    int64_t call_count = 0; // necessary for detecting if we need further instantiation in execution stage
    size_t sizeWritten;
    int32_t do_refactorize; // host-side flag read from traced GPU signal
    cudaDataType cuda_dtype = get_cuda_data_type<T>();

    // Cache pointer addresses to detect if sparsity pattern has changed
    int32_t* cached_offsets_ptr = nullptr;
    int32_t* cached_columns_ptr = nullptr;

    // this is literally only for debugging
    using native_dtype = typename get_native_data_type<T>::type;

    ~CudssBatchState() {
        // TODO: Fix destructor - currently causes double free on CUDA 13 / cuDSS 0.7
        // The issue is likely related to cudaMallocAsync/cudaFree mismatch or
        // destruction order of cuDSS objects that reference batched_*_ptr memory.
        // For now, we leak memory to avoid the crash.

        // cudaDeviceSynchronize();
        //
        // if (handle) {
        //     cudssMatrixDestroy(A);
        //     cudssMatrixDestroy(b);
        //     cudssMatrixDestroy(x);
        //     cudssDataDestroy(handle, data);
        //     cudssConfigDestroy(config);
        //     cudssDestroy(handle);
        // }
        // if (batched_offsets_ptr) cudaFree(batched_offsets_ptr);
        // if (batched_columns_ptr) cudaFree(batched_columns_ptr);
    }
};

template <> ffi::TypeId CudssBatchState<ffi::F32>::id = {};
template <> ffi::TypeId CudssBatchState<ffi::F64>::id = {};
template <> ffi::TypeId CudssBatchState<ffi::C64>::id = {};
template <> ffi::TypeId CudssBatchState<ffi::C128>::id = {};

// instantiate =================================================================
// create everything that is not a function of the context (cudaStream_t)
template <ffi::DataType T>
static ffi::ErrorOr<std::unique_ptr<CudssBatchState<T>>> CudssInstantiate(
    const int64_t batch_size_64,               // need to know without other structural data
    const int64_t device_id,                // the device to run this on
    const int64_t mtype_id,                 // {0: gen, 1: sym, 2: herm, 3: spd, 4: hpd}
    const int64_t mview_id                  // {0: full, 1: triu, 2: tril}
) {

    // printf("in pseudo batch instantiate\n");

    // printf("in instantiate\n");
    // make a new state which will manage CuDSS's state
    auto state = std::make_unique<CudssBatchState<T>>();

    // check on the type of matrix being solved
    if (mtype_id == 0) {
        // printf("general matrix chosen\n");
        state->mtype = CUDSS_MTYPE_GENERAL;
    } else if (mtype_id == 1) {
        // printf("symmetric matrix chosen\n");
        state->mtype = CUDSS_MTYPE_SYMMETRIC;
    } else if (mtype_id == 2) {
        // printf("hermitian matrix chosen\n");
        state->mtype = CUDSS_MTYPE_HERMITIAN;
    } else if (mtype_id == 3) {
        // printf("symmetric PD matrix chosen\n");
        state->mtype = CUDSS_MTYPE_SPD;
    } else if (mtype_id == 4) {
        // printf("hermitian PD matrix chosen\n");
        state->mtype = CUDSS_MTYPE_HPD;
    } else {
        throw std::invalid_argument("Invalid mtype_id. Valid options: 0: general, 1: symmetric, 2: hermitian, 3: spd, 4: hpd");
    }

    // check on the view of the matrix provided
    if (mview_id == 0) {
        // printf("full view provided\n");
        state->mview = CUDSS_MVIEW_FULL;
    } else if (mview_id == 1) {
        // printf("upper view provided\n");
        state->mview = CUDSS_MVIEW_UPPER;
    } else if (mview_id == 2) {
        // printf("lower view provided\n");
        state->mview = CUDSS_MVIEW_LOWER;
    } else {
        throw std::invalid_argument("Invalid mview_id. Valid options: 0: full, 1: upper, 2: lower");
    }

    // Store uniform dimensions and batch size

    state->nrhs = 1;

    // CUDA setup can happen here before any cudaMallocs
    cudaSetDevice(device_id);

    // Note: diag_temp and perm_temp are not used - output goes directly to FFI buffers
    // The allocations were removed as they caused double-free due to state->n being uninitialized here

    return ffi::ErrorOr<std::unique_ptr<CudssBatchState<T>>>(std::move(state));
    // return state; // simply return the created CudssBatchState
}

// execution ===================================================================

// GPU kernel to create batched column indices in parallel
__global__ void create_batched_columns_kernel(
    const int32_t* __restrict__ single_columns,
    int32_t* __restrict__ batched_columns,
    int64_t nnz, int64_t n, int64_t batch_size)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = batch_size * nnz;

    if (idx < total_elements) {
        int64_t batch_idx = idx / nnz;          // Which batch
        int64_t col_idx = idx % nnz;            // Which element within batch
        int32_t column_offset = batch_idx * n;  // Offset for this batch

        batched_columns[idx] = single_columns[col_idx] + column_offset;
    }
}

// GPU kernel to create batched row offsets in parallel
__global__ void create_batched_offsets_kernel(
    const int32_t* __restrict__ single_offsets,
    int32_t* __restrict__ batched_offsets,
    int64_t n, int64_t nnz, int64_t batch_size)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = batch_size * n + 1;

    if (idx < total_elements) {
        if (idx == batch_size * n) {
            // Last element: total nnz
            batched_offsets[idx] = batch_size * nnz;
        } else {
            int64_t batch_idx = idx / n;        // Which batch
            int64_t row_idx = idx % n;          // Which row within batch
            int32_t nnz_offset = batch_idx * nnz; // Offset for this batch

            batched_offsets[idx] = single_offsets[row_idx] + nnz_offset;
        }
    }
}

// GPU-accelerated function to populate batched CSR structure
// Note: Memory must be pre-allocated before calling this function
static void create_batched_csr_structure(
    int32_t* single_offsets_ptr, int32_t* single_columns_ptr,
    int64_t n, int64_t nnz, int64_t batch_size,
    int32_t** batched_offsets_ptr, int32_t** batched_columns_ptr,
    cudaStream_t stream = 0)
{
    // Launch kernels to populate batched arrays from single pattern
    const int threads = 256;

    // Launch kernel to create batched columns
    int64_t total_columns = batch_size * nnz;
    int blocks_columns = (total_columns + threads - 1) / threads;
    create_batched_columns_kernel<<<blocks_columns, threads, 0, stream>>>(
        single_columns_ptr, *batched_columns_ptr, nnz, n, batch_size);

    // Launch kernel to create batched offsets
    int64_t total_offsets = batch_size * n + 1;
    int blocks_offsets = (total_offsets + threads - 1) / threads;
    create_batched_offsets_kernel<<<blocks_offsets, threads, 0, stream>>>(
        single_offsets_ptr, *batched_offsets_ptr, n, nnz, batch_size);

    // No synchronization needed here - stream will handle dependencies
}

template <ffi::DataType T>
static ffi::Error CudssExecute(
    cudaStream_t stream,                    // JAXs stream given to this context (jit)
    CudssBatchState<T>* state,                      // the state we instantiated in CudssInstantiate
    ffi::Buffer<T> b_values_buf,            // the real input data that varies per solution
    ffi::Buffer<T> csr_values_buf,          // the real input data that varies per solution
    ffi::Buffer<ffi::S32> offsets_buf,      // sparsity pattern row offsets
    ffi::Buffer<ffi::S32> columns_buf,      // sparsity pattern column indices
    ffi::Buffer<ffi::S32> refactorize_signal,      // whether we should refactorize within jit
    ffi::ResultBuffer<T> out_values_buf,    // the output buffer we write the answer to
    ffi::ResultBuffer<T> diag_buf, // the output buffer for inertia [batch_size, 3]
    ffi::ResultBuffer<ffi::S32> perm_buf, // the output buffer for inertia [batch_size, 3]
    const int64_t batch_size_64,               // need to know without other structural data
    const int64_t device_id,                    // the device to run this on
    const int64_t mtype_id,                     // {0: gen, 1: sym, 2: herm, 3: spd, 4: hpd}
    const int64_t mview_id                      // {0: full, 1: triu, 2: tril}
) {
    // printf("in execute \n");
    // cudaStreamSynchronize(stream);
    if (state->call_count == 0) {
        
        // figure this out on first call
        state->n = offsets_buf.element_count() - 1;
        state->nnz = columns_buf.element_count();

        // Allocate device memory for batched CSR structure (done once)
        cudaMallocAsync(&state->batched_columns_ptr, batch_size_64 * state->nnz * sizeof(int32_t), stream);
        cudaMallocAsync(&state->batched_offsets_ptr, (batch_size_64 * state->n + 1) * sizeof(int32_t), stream);

        // form the new batched offsets and ptrs here!
        create_batched_csr_structure(
            offsets_buf.typed_data(), columns_buf.typed_data(),
            state->n, state->nnz, batch_size_64,
            &state->batched_offsets_ptr, &state->batched_columns_ptr,
            stream
        );

        // Cache the input pointers to detect pattern changes on subsequent calls
        state->cached_offsets_ptr = offsets_buf.typed_data();
        state->cached_columns_ptr = columns_buf.typed_data();

        // CuDSS setup
        CUDSS_CALL_AND_CHECK(cudssCreate(&state->handle), state->status, "cudssCreate");
        CUDSS_CALL_AND_CHECK(cudssSetStream(state->handle, stream), state->status, "cudssSetStream");
        CUDSS_CALL_AND_CHECK(cudssConfigCreate(&state->config), state->status, "cudssConfigCreate");
        CUDSS_CALL_AND_CHECK(cudssDataCreate(state->handle, &state->data), state->status, "cudssDataCreate");
        
        // CuDSS structures creation
        int64_t batched_n = state->n * batch_size_64;
        CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&state->b, batched_n, state->nrhs, batched_n,
            b_values_buf.typed_data(), state->cuda_dtype, CUDSS_LAYOUT_COL_MAJOR), state->status, "cudssMatrixCreateDn for b");

        CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&state->x, batched_n, state->nrhs, batched_n,
            out_values_buf->typed_data(), state->cuda_dtype, CUDSS_LAYOUT_COL_MAJOR), state->status, "cudssMatrixCreateDn for x");

        // Use singular matrix creation APIs
        int64_t batched_nnz = state->nnz * batch_size_64;
        CUDSS_CALL_AND_CHECK(cudssMatrixCreateCsr(&state->A, batched_n, batched_n, batched_nnz,
            state->batched_offsets_ptr, NULL,
            state->batched_columns_ptr,
            csr_values_buf.typed_data(),
            CUDA_R_32I, state->cuda_dtype,
            state->mtype, state->mview, state->base), state->status, "cudssMatrixCreateCsr");

        // CuDSS config
        // iterative refinement of the soln is pretty n i f t y
        int iter_ref_nsteps = 5;
        CUDSS_CALL_AND_CHECK(cudssConfigSet(state->config, CUDSS_CONFIG_IR_N_STEPS,
                            &iter_ref_nsteps, sizeof(iter_ref_nsteps)), state->status, "cudssConfigSet ir_nsteps");

        // cold solve - analyze, factorize, solve
        CUDSS_CALL_AND_CHECK(cudssExecute(state->handle, CUDSS_PHASE_ANALYSIS, 
            state->config, state->data, state->A, state->x, state->b), state->status, "cudssExecute analysis");

        CUDSS_CALL_AND_CHECK(cudssExecute(state->handle, CUDSS_PHASE_FACTORIZATION, 
            state->config, state->data, state->A, state->x, state->b), state->status, "cudssExecute factorization");

        CUDSS_CALL_AND_CHECK(cudssExecute(state->handle, CUDSS_PHASE_SOLVE, 
            state->config, state->data, state->A, state->x, state->b), state->status, "cudssExecute solve");

        // so we dont init again...
        state->call_count++;

    } else {
        // stream can change between calls!!!
        CUDSS_CALL_AND_CHECK(cudssSetStream(state->handle, stream), state->status, "cudssSetStream");

        // Check if sparsity pattern pointers have changed
        int32_t* current_offsets_ptr = offsets_buf.typed_data();
        int32_t* current_columns_ptr = columns_buf.typed_data();

        if (current_offsets_ptr != state->cached_offsets_ptr ||
            current_columns_ptr != state->cached_columns_ptr) {
            // Pattern pointers changed - recompute batched structure
            create_batched_csr_structure(
                current_offsets_ptr, current_columns_ptr,
                state->n, state->nnz, batch_size_64,
                &state->batched_offsets_ptr, &state->batched_columns_ptr,
                stream
            );

            // Update cached pointers
            state->cached_offsets_ptr = current_offsets_ptr;
            state->cached_columns_ptr = current_columns_ptr;
        }
        // else: Pointers unchanged - batched structure is still valid, skip kernel!

        // Read refactorize signal from GPU to host (cudaMemcpy is synchronous)
        cudaMemcpy(&state->do_refactorize, refactorize_signal.typed_data(),
                   sizeof(int32_t), cudaMemcpyDeviceToHost);

        // Update the values pointer which changes between calls
        CUDSS_CALL_AND_CHECK(cudssMatrixSetValues(state->b, b_values_buf.typed_data()), state->status, "update_pointers b");
        CUDSS_CALL_AND_CHECK(cudssMatrixSetValues(state->x, out_values_buf->typed_data()), state->status, "update_pointers x");

        // Conditionally refactorize based on traced signal
        if (state->do_refactorize) {
            // printf("we ARE refactorizing in batch solver\n");
            // Only update the LHS matrix IF we are refactorizing - otherwise it stays the same - along with the factorization
            CUDSS_CALL_AND_CHECK(cudssMatrixSetValues(state->A, csr_values_buf.typed_data()), state->status, "update_pointers A");
            CUDSS_CALL_AND_CHECK(cudssExecute(state->handle, CUDSS_PHASE_REFACTORIZATION,
                state->config, state->data, state->A, state->x, state->b), state->status, "cudssExecute refactorization");
        }

        // Always solve
        CUDSS_CALL_AND_CHECK(cudssExecute(state->handle, CUDSS_PHASE_SOLVE,
            state->config, state->data, state->A, state->x, state->b), state->status, "cudssExecute solve");

    }

    cudssDataGet(state->handle, state->data, CUDSS_DATA_DIAG, diag_buf->typed_data(),
                    batch_size_64 * state->n * sizeof(typename get_native_data_type<T>::type), &state->sizeWritten);
    cudssDataGet(state->handle, state->data, CUDSS_DATA_PERM_REORDER_ROW, perm_buf->typed_data(),
                    batch_size_64 * state->n * sizeof(int32_t), &state->sizeWritten);

    // Synchronize to ensure all async operations complete before returning control to JAX
    // This prevents race conditions where JAX may free buffers before cuDSS finishes writing
    cudaStreamSynchronize(stream);

    return ffi::Error::Success();
}

// XLA/nanobind boilerplate ====================================================

// XLA ffi handler definitions macro
#define DEFINE_CUDSS_FFI_HANDLERS(TypeName, DataType) \
    XLA_FFI_DEFINE_HANDLER(kCudssInstantiate##TypeName, CudssInstantiate<DataType>, \
        ffi::Ffi::BindInstantiate() \
            .Attr<int64_t>("batch_size") \
            .Attr<int64_t>("device_id") \
            .Attr<int64_t>("mtype_id") \
            .Attr<int64_t>("mview_id")); \
    \
    XLA_FFI_DEFINE_HANDLER(kCudssExecute##TypeName, CudssExecute<DataType>, \
        ffi::Ffi::Bind() \
            .Ctx<ffi::PlatformStream<cudaStream_t>>() \
            .Ctx<ffi::State<CudssBatchState<DataType>>>() \
            .Arg<ffi::Buffer<DataType>>() \
            .Arg<ffi::Buffer<DataType>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Ret<ffi::Buffer<DataType>>() \
            .Ret<ffi::Buffer<DataType>>() \
            .Ret<ffi::Buffer<ffi::S32>>() \
            /* Attributes must also be passed to execute */ \
            .Attr<int64_t>("batch_size") \
            .Attr<int64_t>("device_id") \
            .Attr<int64_t>("mtype_id") \
            .Attr<int64_t>("mview_id"));

// Generate all the FFI handlers using the macro
DEFINE_CUDSS_FFI_HANDLERS(f32, ffi::F32);
DEFINE_CUDSS_FFI_HANDLERS(f64, ffi::F64);
DEFINE_CUDSS_FFI_HANDLERS(c64, ffi::C64);
DEFINE_CUDSS_FFI_HANDLERS(c128, ffi::C128);

// nanobind module exporting macro
#define EXPORT_CUDSS_HANDLERS(m, TypeName, DataType) \
    m.def("type_id_" #TypeName, []() { \
        return nb::capsule(reinterpret_cast<void*>(&CudssBatchState<DataType>::id)); \
    }); \
    m.def("handler_" #TypeName, []() { \
        nb::dict d; \
        d["instantiate"] = nb::capsule(reinterpret_cast<void*>(kCudssInstantiate##TypeName)); \
        d["execute"] = nb::capsule(reinterpret_cast<void*>(kCudssExecute##TypeName)); \
        return d; \
    });

// generate all nanobind modules! :)
NB_MODULE(pbatch_solve, m) {
    EXPORT_CUDSS_HANDLERS(m, f32, ffi::F32);
    EXPORT_CUDSS_HANDLERS(m, f64, ffi::F64);
    EXPORT_CUDSS_HANDLERS(m, c64, ffi::C64);
    EXPORT_CUDSS_HANDLERS(m, c128, ffi::C128);
}

