/*This code is currently unused until cuDSS updates some of their
batched outputs to support inertia retrieval*/

#include <cstdint>
#include <memory>
#include <vector>
#include <complex>

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
    cudaStream_t last_stream = nullptr; // track stream for synchronization
    int64_t n = 0;
    int64_t nnz = 0;
    int64_t nrhs = 0;
    int32_t ubatch_size = 0; // this must be int32_t or cuDSS will error
    int64_t call_count = 0; // necessary for detecting if we need further instantiation in execution stage
    size_t sizeWritten = 0;
    cudaDataType cuda_dtype = get_cuda_data_type<T>();

    // this is literally only for debugging
    using native_dtype = typename get_native_data_type<T>::type;

    ~CudssBatchState() {
        if (handle) {
            // Synchronize with the last stream before destroying resources
            if (last_stream) {
                cudaStreamSynchronize(last_stream);
            }
            // CuDSS destruction
            cudssMatrixDestroy(A);
            cudssMatrixDestroy(b);
            cudssMatrixDestroy(x);
            cudssDataDestroy(handle, data);
            cudssConfigDestroy(config);
            cudssDestroy(handle);
        }
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
    int32_t* offsets_ptr,                   // pointers and sizes of csr structure defn per batch element
    const int64_t offsets_size,             // pointers and sizes of csr structure defn per batch element
    int32_t* columns_ptr,                   // pointers and sizes of csr structure defn per batch element
    const int64_t columns_size,             // pointers and sizes of csr structure defn per batch element
    const int64_t device_id,                // the device to run this on
    const int64_t mtype_id,                 // {0: gen, 1: sym, 2: herm, 3: spd, 4: hpd}
    const int64_t mview_id                  // {0: full, 1: triu, 2: tril}
) {

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
    state->n = offsets_size - 1;
    state->nnz = columns_size;
    state->nrhs = 1;

    // Python ints are being passed as int64_t's
    state->ubatch_size = static_cast<int32_t>(batch_size_64);

    // CUDA setup can happen here before any cudaMallocs
    cudaSetDevice(device_id);

    return state; // simply return the created CudssBatchState
}

// execution ===================================================================
template <ffi::DataType T>
static ffi::Error CudssExecute(
    cudaStream_t stream,                    // JAXs stream given to this context (jit)
    CudssBatchState<T>* state,                      // the state we instantiated in CudssInstantiate
    ffi::Buffer<T> b_values_buf,            // the real input data that varies per solution
    ffi::Buffer<T> csr_values_buf,          // the real input data that varies per solution
    ffi::ResultBuffer<T> out_values_buf,    // the output buffer we write the answer to
    ffi::ResultBuffer<ffi::S32> inertia_buf,  // the output buffer we write the answer to
    const int64_t batch_size_64,               // need to know without other structural data
    int32_t* offsets_ptr,             // pointers and sizes of csr structure defn
    const int64_t offsets_size,             // pointers and sizes of csr structure defn
    int32_t* columns_ptr,             // pointers and sizes of csr structure defn
    const int64_t columns_size,             // pointers and sizes of csr structure defn
    const int64_t device_id,                    // the device to run this on
    const int64_t mtype_id,                     // {0: gen, 1: sym, 2: herm, 3: spd, 4: hpd}
    const int64_t mview_id                      // {0: full, 1: triu, 2: tril}
) {
    // printf("in execute \n");
    // Track stream for cleanup synchronization
    state->last_stream = stream;

    if (state->call_count == 0) {

        // CuDSS setup
        CUDSS_CALL_AND_CHECK(cudssCreate(&state->handle), state->status, "cudssCreate");
        CUDSS_CALL_AND_CHECK(cudssSetStream(state->handle, stream), state->status, "cudssSetStream");
        CUDSS_CALL_AND_CHECK(cudssConfigCreate(&state->config), state->status, "cudssConfigCreate");
        CUDSS_CALL_AND_CHECK(cudssDataCreate(state->handle, &state->data), state->status, "cudssDataCreate");
        
        // CuDSS structures creation
        CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&state->b, state->n, state->nrhs, state->n,
            b_values_buf.typed_data(), state->cuda_dtype, CUDSS_LAYOUT_COL_MAJOR), state->status, "cudssMatrixCreateDn for b");

        CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&state->x, state->n, state->nrhs, state->n,
            out_values_buf->typed_data(), state->cuda_dtype, CUDSS_LAYOUT_COL_MAJOR), state->status, "cudssMatrixCreateDn for x");

        // Use singular matrix creation APIs
        CUDSS_CALL_AND_CHECK(cudssMatrixCreateCsr(&state->A, state->n, state->n, state->nnz,
            offsets_ptr, NULL,
            columns_ptr,
            csr_values_buf.typed_data(),
            CUDA_R_32I, state->cuda_dtype,
            state->mtype, state->mview, state->base), state->status, "cudssMatrixCreateCsr");

        // CuDSS config
        // uniform batch of size state->ubatch_size
        CUDSS_CALL_AND_CHECK(cudssConfigSet(state->config, CUDSS_CONFIG_UBATCH_SIZE,
                       &state->ubatch_size, sizeof(state->ubatch_size)), state->status, "cudssConfigSet ubatch_size");

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

        // debug
        // cudaStreamSynchronize(stream);
        // using DType = typename CudssBatchState<T>::native_dtype;
        // printf("state->n: %ld\n", state->n);
        // printf("state->nnz: %ld\n", state->nnz);
        // printf("state->nnz: %d\n", state->ubatch_size);
        // print_device_data<DType>("csr_values", csr_values_buf.typed_data(), state->ubatch_size, state->nnz);
        // print_device_data<DType>("b_values", b_values_buf.typed_data(), state->ubatch_size, state->n);
        // print_device_data<DType>("x_values", out_values_buf->typed_data(), state->ubatch_size, state->n);
        // print_device_data<int32_t>("csr_offsets", offsets_ptr, 1, state->n+1);
        // print_device_data<int32_t>("csr_columns", columns_ptr, 1, state->nnz);

        // so we dont init again...
        state->call_count++;

    } else {
        // stream can change between calls!!!
        CUDSS_CALL_AND_CHECK(cudssSetStream(state->handle, stream), state->status, "cudssSetStream");

        // update the csr_values data on device - just overwrite prior pointers
        CUDSS_CALL_AND_CHECK(cudssMatrixSetValues(state->A, csr_values_buf.typed_data()), state->status, "update_pointers A");
        CUDSS_CALL_AND_CHECK(cudssMatrixSetValues(state->b, b_values_buf.typed_data()), state->status, "update_pointers b");
        CUDSS_CALL_AND_CHECK(cudssMatrixSetValues(state->x, out_values_buf->typed_data()), state->status, "update_pointers x");

        // warm solve - refactorize, solve
        CUDSS_CALL_AND_CHECK(cudssExecute(state->handle, CUDSS_PHASE_REFACTORIZATION, 
            state->config, state->data, state->A, state->x, state->b), state->status, "cudssExecute refactorization");

        CUDSS_CALL_AND_CHECK(cudssExecute(state->handle, CUDSS_PHASE_SOLVE, 
            state->config, state->data, state->A, state->x, state->b), state->status, "cudssExecute solve");

    }

    // we dont call CUDSS_CALL_AND_CHECK here as these features are not technically supported in batch
    // analysis phase - by default get everything - then decide on Python side what to discard
    // cudssDataGet(state->handle, state->data, CUDSS_DATA_DIAG, diag_buf->typed_data(),
    //                 diag_buf->size_bytes(), &state->sizeWritten);
    // cudssDataGet(state->handle, state->data, CUDSS_DATA_PERM_REORDER_ROW, perm_reorder_row_buf->typed_data(),
    //                 perm_reorder_row_buf->size_bytes(), &state->sizeWritten);

    // cudaStreamSynchronize(stream);

    return ffi::Error::Success();
}


// XLA/nanobind boilerplate ====================================================

// XLA ffi handler definitions macro
#define DEFINE_CUDSS_FFI_HANDLERS(TypeName, DataType) \
    XLA_FFI_DEFINE_HANDLER(kCudssInstantiate##TypeName, CudssInstantiate<DataType>, \
        ffi::Ffi::BindInstantiate() \
            .Attr<int64_t>("batch_size") \
            .Attr<ffi::Pointer<int32_t>>("offsets_ptr") \
            .Attr<int64_t>("offsets_size") \
            .Attr<ffi::Pointer<int32_t>>("columns_ptr") \
            .Attr<int64_t>("columns_size") \
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
            .Ret<ffi::Buffer<DataType>>() \
            .Ret<ffi::Buffer<ffi::S32>>() \
            /* Attributes must also be passed to execute */ \
            .Attr<int64_t>("batch_size") \
            .Attr<ffi::Pointer<int32_t>>("offsets_ptr") \
            .Attr<int64_t>("offsets_size") \
            .Attr<ffi::Pointer<int32_t>>("columns_ptr") \
            .Attr<int64_t>("columns_size") \
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
    m.def("state_type_" #TypeName, []() { \
        static auto kTypeInfo = ffi::MakeTypeInfo<CudssBatchState<DataType>>(); \
        nb::dict d; \
        d["type_id"] = nb::capsule(reinterpret_cast<void*>(&CudssBatchState<DataType>::id)); \
        d["type_info"] = nb::capsule(reinterpret_cast<void*>(&kTypeInfo)); \
        return d; \
    }); \
    m.def("handler_" #TypeName, []() { \
        nb::dict d; \
        d["instantiate"] = nb::capsule(reinterpret_cast<void*>(kCudssInstantiate##TypeName)); \
        d["execute"] = nb::capsule(reinterpret_cast<void*>(kCudssExecute##TypeName)); \
        return d; \
    });

// generate all nanobind modules! :)
NB_MODULE(batch_solve, m) {
    EXPORT_CUDSS_HANDLERS(m, f32, ffi::F32);
    EXPORT_CUDSS_HANDLERS(m, f64, ffi::F64);
    EXPORT_CUDSS_HANDLERS(m, c64, ffi::C64);
    EXPORT_CUDSS_HANDLERS(m, c128, ffi::C128);
}
