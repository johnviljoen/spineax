/* Persistent factorization: factor once, solve many (different RHS).

   Two FFI handlers backed by a process-global LRU registry of cuDSS
   factorizations:

     factorize(csr_values, offsets, columns) -> token (int32[1])
         analysis + factorization; the structure, a private copy of the values,
         and the cuDSS factors are kept alive in the registry under a fresh id,
         which is written to the token output buffer.

     solve(token, b_values) -> x
         look the factorization up by token, run the SOLVE phase only, write x.

   The token is an ordinary JAX array, so threading it from a forward solve into
   a custom_vjp backward solve (a) creates the data dependency XLA needs to order
   solve after factorize, and (b) lets the backward reuse the forward's
   factorization. For a SYMMETRIC operator J = Jᵀ, the adjoint λ = J⁻ᵀ v = J⁻¹ v,
   so the same token solves both the forward (x = J⁻¹ b) and the adjoint with no
   refactorization.

   Memory is bounded by an LRU cache (capacity from SPINEAX_FACTOR_CACHE, default
   8); evicting an entry frees its cuDSS objects and the owned device arrays.
*/

#include <cstdint>
#include <cstdlib>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "cuda_runtime_api.h"
#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"
#include "cudss.h"

namespace ffi = xla::ffi;
namespace nb = nanobind;

#define CUDSS_CALL_AND_CHECK(call, status, msg) \
    do { \
        status = call; \
        if (status != CUDSS_STATUS_SUCCESS) { \
            printf("FAILED: CUDSS call ended unsuccessfully with status = %d, details: " #msg "\n", status); \
            return ffi::Error::Internal("cudss call failed: " #msg); \
        } \
    } while(0);

#define CUDA_CHECK(call)                                       \
  do {                                                         \
    cudaError_t err = call;                                    \
    if (err != cudaSuccess) {                                  \
      printf("CUDA Error at %s %d: %s\n", __FILE__, __LINE__,   \
             cudaGetErrorString(err));                         \
      return ffi::Error::Internal("A CUDA call failed.");      \
    }                                                          \
  } while (0)

// dtype helpers ===============================================================
template <ffi::DataType T> cudssDataType_t get_cudss_data_type();
template<> cudssDataType_t get_cudss_data_type<ffi::F32>() { return CUDSS_R_32F; }
template<> cudssDataType_t get_cudss_data_type<ffi::F64>() { return CUDSS_R_64F; }

template <ffi::DataType T> struct native_t;
template<> struct native_t<ffi::F32> { using type = float; };
template<> struct native_t<ffi::F64> { using type = double; };

// A persistent factorization ==================================================
template <ffi::DataType T>
struct FactorEntry {
    using nat = typename native_t<T>::type;
    cudssHandle_t handle = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t   data   = nullptr;
    cudssMatrix_t A = nullptr, x = nullptr, b = nullptr;
    // device memory OWNED by this entry (the JAX input buffers are transient, but
    // the SOLVE phase — with iterative refinement — still reads A's csr arrays).
    int32_t* d_offsets = nullptr;
    int32_t* d_columns = nullptr;
    nat*     d_values  = nullptr;
    int64_t n = 0, nnz = 0;
    cudaStream_t last_stream = nullptr;

    ~FactorEntry() {
        if (last_stream) cudaStreamSynchronize(last_stream);
        if (A) cudssMatrixDestroy(A);
        if (b) cudssMatrixDestroy(b);
        if (x) cudssMatrixDestroy(x);
        if (data && handle) cudssDataDestroy(handle, data);
        if (config) cudssConfigDestroy(config);
        if (handle) cudssDestroy(handle);
        if (d_offsets) cudaFree(d_offsets);
        if (d_columns) cudaFree(d_columns);
        if (d_values)  cudaFree(d_values);
    }
};

// process-global LRU registry (per dtype) =====================================
template <ffi::DataType T>
struct Registry {
    std::mutex mu;
    std::unordered_map<int32_t, std::shared_ptr<FactorEntry<T>>> map;
    std::list<int32_t> lru;                       // front = most recently used
    int32_t next_id = 1;

    static size_t capacity() {
        const char* e = std::getenv("SPINEAX_FACTOR_CACHE");
        if (e) { try { return std::stoul(e); } catch (...) {} }
        return 8;
    }
    void touch(int32_t id) {
        lru.remove(id);
        lru.push_front(id);
    }
    int32_t insert(std::shared_ptr<FactorEntry<T>> entry) {
        std::lock_guard<std::mutex> lk(mu);
        while (map.size() >= capacity() && !lru.empty()) {
            int32_t old = lru.back();
            lru.pop_back();
            map.erase(old);                       // shared_ptr -> ~FactorEntry frees GPU
        }
        int32_t id = next_id++;
        map[id] = std::move(entry);
        lru.push_front(id);
        return id;
    }
    std::shared_ptr<FactorEntry<T>> get(int32_t id) {
        std::lock_guard<std::mutex> lk(mu);
        auto it = map.find(id);
        if (it == map.end()) return nullptr;
        touch(id);
        return it->second;
    }
};

template <ffi::DataType T> Registry<T>& registry() { static Registry<T> r; return r; }

// factorize ===================================================================
template <ffi::DataType T>
static ffi::Error Factorize(
    cudaStream_t stream,
    ffi::Buffer<T>        csr_values_buf,
    ffi::Buffer<ffi::S32> offsets_buf,
    ffi::Buffer<ffi::S32> columns_buf,
    ffi::ResultBuffer<ffi::S32> token_buf,        // int32[1]
    const int64_t device_id,
    const int64_t mtype_id,
    const int64_t mview_id
) {
    using nat = typename native_t<T>::type;
    cudaSetDevice(device_id);

    auto e = std::make_shared<FactorEntry<T>>();
    e->last_stream = stream;
    e->n   = offsets_buf.element_count() - 1;
    e->nnz = columns_buf.element_count();

    cudssMatrixType_t mtype =
        mtype_id == 0 ? CUDSS_MTYPE_GENERAL :
        mtype_id == 1 ? CUDSS_MTYPE_SYMMETRIC :
        mtype_id == 2 ? CUDSS_MTYPE_HERMITIAN :
        mtype_id == 3 ? CUDSS_MTYPE_SPD : CUDSS_MTYPE_HPD;
    cudssMatrixViewType_t mview =
        mview_id == 0 ? CUDSS_MVIEW_FULL :
        mview_id == 1 ? CUDSS_MVIEW_UPPER : CUDSS_MVIEW_LOWER;

    // Own private device copies of the structure + values (the JAX buffers are
    // transient; SOLVE with iterative refinement still reads A).
    CUDA_CHECK(cudaMallocAsync(&e->d_offsets, (e->n + 1) * sizeof(int32_t), stream));
    CUDA_CHECK(cudaMallocAsync(&e->d_columns, e->nnz * sizeof(int32_t), stream));
    CUDA_CHECK(cudaMallocAsync(&e->d_values,  e->nnz * sizeof(nat), stream));
    CUDA_CHECK(cudaMemcpyAsync(e->d_offsets, offsets_buf.typed_data(), (e->n + 1) * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(e->d_columns, columns_buf.typed_data(), e->nnz * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(e->d_values,  csr_values_buf.typed_data(), e->nnz * sizeof(nat), cudaMemcpyDeviceToDevice, stream));

    cudssStatus_t status = CUDSS_STATUS_SUCCESS;
    CUDSS_CALL_AND_CHECK(cudssCreate(&e->handle), status, "cudssCreate");
    CUDSS_CALL_AND_CHECK(cudssSetStream(e->handle, stream), status, "cudssSetStream");
    CUDSS_CALL_AND_CHECK(cudssConfigCreate(&e->config), status, "cudssConfigCreate");
    CUDSS_CALL_AND_CHECK(cudssDataCreate(e->handle, &e->data), status, "cudssDataCreate");

    int iter_ref = 5;
    CUDSS_CALL_AND_CHECK(cudssConfigSet(e->config, CUDSS_CONFIG_IR_N_STEPS, &iter_ref, sizeof(iter_ref)), status, "cfg ir");

    cudssDataType_t dt = get_cudss_data_type<T>();
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateCsr(&e->A, e->n, e->n, e->nnz,
        e->d_offsets, NULL, e->d_columns, e->d_values,
        CUDSS_R_32I, CUDSS_R_32I, dt, mtype, mview, CUDSS_BASE_ZERO), status, "createCsr A");

    // Placeholder dense matrices for analysis/factorization (1 rhs). cuDSS needs
    // x and b objects; their data pointers are reset per solve.
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&e->b, e->n, 1, e->n,
        e->d_values, dt, CUDSS_LAYOUT_COL_MAJOR), status, "createDn b");  // dummy ptr
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&e->x, e->n, 1, e->n,
        e->d_values, dt, CUDSS_LAYOUT_COL_MAJOR), status, "createDn x");  // dummy ptr

    CUDSS_CALL_AND_CHECK(cudssExecute(e->handle, CUDSS_PHASE_ANALYSIS,
        e->config, e->data, e->A, e->x, e->b), status, "analysis");
    CUDSS_CALL_AND_CHECK(cudssExecute(e->handle, CUDSS_PHASE_FACTORIZATION,
        e->config, e->data, e->A, e->x, e->b), status, "factorization");

    int32_t id = registry<T>().insert(e);
    CUDA_CHECK(cudaMemcpyAsync(token_buf->typed_data(), &id, sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return ffi::Error::Success();
}

// solve =======================================================================
template <ffi::DataType T>
static ffi::Error SolveWith(
    cudaStream_t stream,
    ffi::Buffer<ffi::S32> token_buf,              // int32[1]
    ffi::Buffer<T>        b_values_buf,
    ffi::ResultBuffer<T>  out_values_buf,
    const int64_t device_id
) {
    cudaSetDevice(device_id);

    int32_t id = 0;
    CUDA_CHECK(cudaMemcpyAsync(&id, token_buf.typed_data(), sizeof(int32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto e = registry<T>().get(id);
    if (!e) return ffi::Error::Internal("spineax factor_solve: unknown/evicted factorization token");

    // Multi-RHS: b is (nrhs, n) row-major in JAX, which is bit-identical to a
    // cuDSS column-major (n, nrhs) dense matrix with leading dim n. nrhs=1 is the
    // ordinary single-RHS case. We build fresh dense descriptors for the actual
    // nrhs each solve (cheap) rather than the 1-column placeholders from factorize.
    int64_t nrhs = e->n > 0 ? (b_values_buf.element_count() / e->n) : 1;
    if (nrhs < 1) nrhs = 1;

    cudssStatus_t status = CUDSS_STATUS_SUCCESS;
    e->last_stream = stream;
    CUDSS_CALL_AND_CHECK(cudssSetStream(e->handle, stream), status, "setStream");

    cudssDataType_t dt = get_cudss_data_type<T>();
    cudssMatrix_t bmat = nullptr, xmat = nullptr;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&bmat, e->n, nrhs, e->n,
        const_cast<typename native_t<T>::type*>(b_values_buf.typed_data()),
        dt, CUDSS_LAYOUT_COL_MAJOR), status, "createDn b (solve)");
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&xmat, e->n, nrhs, e->n,
        out_values_buf->typed_data(), dt, CUDSS_LAYOUT_COL_MAJOR), status, "createDn x (solve)");

    cudssStatus_t solve_status = cudssExecute(e->handle, CUDSS_PHASE_SOLVE,
        e->config, e->data, e->A, xmat, bmat);
    cudssMatrixDestroy(bmat);
    cudssMatrixDestroy(xmat);
    if (solve_status != CUDSS_STATUS_SUCCESS)
        return ffi::Error::Internal("cudss call failed: multi-rhs solve");
    return ffi::Error::Success();
}

// FFI handler + nanobind boilerplate ==========================================
#define DEFINE_FS_HANDLERS(TypeName, DataType) \
    XLA_FFI_DEFINE_HANDLER(kFactorize##TypeName, Factorize<DataType>, \
        ffi::Ffi::Bind() \
            .Ctx<ffi::PlatformStream<cudaStream_t>>() \
            .Arg<ffi::Buffer<DataType>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Ret<ffi::Buffer<ffi::S32>>() \
            .Attr<int64_t>("device_id") \
            .Attr<int64_t>("mtype_id") \
            .Attr<int64_t>("mview_id")); \
    XLA_FFI_DEFINE_HANDLER(kSolveWith##TypeName, SolveWith<DataType>, \
        ffi::Ffi::Bind() \
            .Ctx<ffi::PlatformStream<cudaStream_t>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Arg<ffi::Buffer<DataType>>() \
            .Ret<ffi::Buffer<DataType>>() \
            .Attr<int64_t>("device_id"));

DEFINE_FS_HANDLERS(f32, ffi::F32);
DEFINE_FS_HANDLERS(f64, ffi::F64);

#define EXPORT_FS(m, TypeName) \
    m.def("factorize_handler_" #TypeName, []() { \
        return nb::capsule(reinterpret_cast<void*>(kFactorize##TypeName)); }); \
    m.def("solve_handler_" #TypeName, []() { \
        return nb::capsule(reinterpret_cast<void*>(kSolveWith##TypeName)); });

NB_MODULE(factor_solve, m) {
    EXPORT_FS(m, f32);
    EXPORT_FS(m, f64);
}
