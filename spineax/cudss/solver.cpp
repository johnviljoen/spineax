#include <cstdint>
#include <memory>
#include <vector>
#include <complex>
#include <type_traits>
#include "cuda_runtime_api.h"
#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"
#include "cudss.h"
#include <mutex>
#include <map>
#include <atomic>
#include <cstdlib>
#include <list>
#include <string>

namespace ffi = xla::ffi;
namespace nb = nanobind;

// Helper function for data types ==============================================
template <ffi::DataType T> cudssDataType_t get_cudss_data_type();
template<> cudssDataType_t get_cudss_data_type<ffi::F32>() { return CUDSS_R_32F; }
template<> cudssDataType_t get_cudss_data_type<ffi::F64>() { return CUDSS_R_64F; }
template<> cudssDataType_t get_cudss_data_type<ffi::C64>() { return CUDSS_C_32F; }
template<> cudssDataType_t get_cudss_data_type<ffi::C128>() { return CUDSS_C_64F; }

template <ffi::DataType T>
struct get_native_data_type;
template<> struct get_native_data_type<ffi::F32> { using type = float; };
template<> struct get_native_data_type<ffi::F64> { using type = double; };
template<> struct get_native_data_type<ffi::C64> { using type = std::complex<float>; };
template<> struct get_native_data_type<ffi::C128> { using type = std::complex<double>; };

#define CUDSS_TOKEN_CHECK(call, msg) \
    do { \
        cudssStatus_t s_ = (call); \
        if (s_ != CUDSS_STATUS_SUCCESS) { \
            return ffi::Error::Internal( \
                std::string("spineax pbatch token: cuDSS call failed (status ") + \
                std::to_string(static_cast<int>(s_)) + "): " msg); \
        } \
    } while (0)

#define CUDA_TOKEN_CHECK(call)                                  \
  do {                                                          \
    cudaError_t err_ = (call);                                  \
    if (err_ != cudaSuccess) {                                  \
      return ffi::Error::Internal(                              \
          std::string("spineax pbatch token: CUDA call failed: ") + \
          cudaGetErrorString(err_));                            \
    }                                                           \
  } while (0)

static ffi::Error validate_ffi_device(const int64_t device_id) {
    int current_device = -1;
    cudaError_t status = cudaGetDevice(&current_device);
    if (status != cudaSuccess) {
        return ffi::Error::Internal(
            std::string("spineax token: cudaGetDevice failed: ") +
            cudaGetErrorString(status));
    }
    if (current_device != device_id) {
        return ffi::Error::Internal(
            "spineax token: configured device does not match the FFI stream device");
    }
    return ffi::Error::Success();
}

struct BatchFactorEntry {
    std::mutex operation_mu;
    cudssHandle_t handle = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t   data   = nullptr;
    cudssMatrix_t A = nullptr;
    cudssMatrix_t x_dummy = nullptr;
    cudssMatrix_t b_dummy = nullptr;
    cudaEvent_t done = nullptr;    // completion of this entry's last phase - keeps streams alive
    uint32_t fp[2] = {0, 0};       // structure fingerprint from analysis
    int64_t block_n = 0, block_nnz = 0, batch = 0;  // system N = batch*block_n
    size_t elem_size = 0;
    cudssDataType_t dtype = CUDSS_R_64F;
    cudssMatrixType_t mtype = CUDSS_MTYPE_SYMMETRIC;
    cudssMatrixViewType_t mview = CUDSS_MVIEW_UPPER;
    int64_t device_id = 0;
    int32_t origin_id = 0;  // first id of this entry's lineage (set at insert)

    ~BatchFactorEntry() {
        int previous_device = -1;
        cudaGetDevice(&previous_device);
        cudaSetDevice(static_cast<int>(device_id));
        if (done) {
            cudaEventSynchronize(done);  // an in-progress phase must finish
            cudaEventDestroy(done);      // before the factors are freed below
        }
        if (A) cudssMatrixDestroy(A);
        if (b_dummy) cudssMatrixDestroy(b_dummy);
        if (x_dummy) cudssMatrixDestroy(x_dummy);
        if (data && handle) cudssDataDestroy(handle, data);
        if (config) cudssConfigDestroy(config);
        if (handle) cudssDestroy(handle);
        if (previous_device >= 0) cudaSetDevice(previous_device);
    }
};

// An id names ONE immutable numeric state, forever: analyze mints an id,
// every numeric phase renames its entry to a fresh id (rekey), and a call
// arriving with an id that is no longer resident — renamed away or LRU-
// evicted, the same thing from here — rebuilds the state it describes from
// its own operands (every phase call carries the full CSR data, zero-copy).
struct BatchTokenRegistry {
    std::mutex mu;
    std::map<int32_t, std::shared_ptr<BatchFactorEntry>> entries;
    std::list<int32_t> lru;  // front = most recently used
    // Lineage (issue #27): every rekey of an entry keeps its origin_id, and
    // these maps let a factorize arriving with a consumed id find a LIVE
    // relative whose cuDSS analysis it can reuse (branch) instead of paying
    // a full re-analysis rebuild. Purely an optimization: any miss falls
    // back to the rebuild path, which is always correct.
    std::map<int32_t, int32_t> lineage;        // origin -> current resident id
    std::map<int32_t, int32_t> retired;        // consumed id -> its origin
    std::list<int32_t> retired_order;          // front = newest retired id
    static constexpr size_t kRetiredCap = 4096;
    std::atomic<int32_t> next_id{1};
    std::atomic<int64_t> rebuilds{0};
    std::atomic<int64_t> branches{0};

    static BatchTokenRegistry& instance() {
        static BatchTokenRegistry r;
        return r;
    }
    static size_t capacity() {
        const char* e = std::getenv("SPINEAX_FACTOR_CACHE");
        if (e) { try { return std::stoul(e); } catch (...) {} }
        return 8;
    }
    // caller holds mu. The counter wraps (int32, ~2^31 mints); 0 and live
    // ids are skipped, so an id stays unique for as long as any plausible
    // token citing it survives.
    int32_t fresh_id_locked() {
        int32_t id;
        do { id = next_id.fetch_add(1); } while (id == 0 || entries.count(id));
        return id;
    }
    // caller holds mu. Drop the origin->id lineage pointer if it names `id`.
    void unlink_lineage_locked(int32_t id,
                               const std::shared_ptr<BatchFactorEntry>& e) {
        auto it = lineage.find(e->origin_id);
        if (it != lineage.end() && it->second == id) lineage.erase(it);
    }
    // caller holds mu. Record `dead_id` -> origin so stale tokens can still
    // find the lineage; capped FIFO so long rekey chains stay bounded (an
    // aged-out id just loses the branch shortcut, never correctness).
    void retire_locked(int32_t dead_id, int32_t origin) {
        if (retired.emplace(dead_id, origin).second) {
            retired_order.push_front(dead_id);
            while (retired.size() > kRetiredCap) {
                retired.erase(retired_order.back());
                retired_order.pop_back();
            }
        } else {
            retired[dead_id] = origin;
        }
    }
    // Register under `id` (0 = mint fresh), evicting LRU overflow.
    int32_t insert(std::shared_ptr<BatchFactorEntry> entry, int32_t id = 0) {
        std::lock_guard<std::mutex> lk(mu);
        while (entries.size() >= capacity() && !lru.empty()) {
            int32_t old = lru.back();
            lru.pop_back();
            auto it = entries.find(old);
            if (it != entries.end()) {
                unlink_lineage_locked(old, it->second);
                entries.erase(it);
            }
        }
        if (id == 0) id = fresh_id_locked();
        if (entry->origin_id == 0) entry->origin_id = id;
        lineage[entry->origin_id] = id;
        entries[id] = std::move(entry);
        lru.push_front(id);
        return id;
    }
    // Rename a live entry: a numeric phase consumes its input state's name.
    int32_t rekey(int32_t old_id, const std::shared_ptr<BatchFactorEntry>& e) {
        std::lock_guard<std::mutex> lk(mu);
        entries.erase(old_id);
        lru.remove(old_id);
        int32_t id = fresh_id_locked();
        if (e->origin_id == 0) e->origin_id = old_id;
        retire_locked(old_id, e->origin_id);
        lineage[e->origin_id] = id;
        entries[id] = e;
        lru.push_front(id);
        return id;
    }
    bool release(int32_t id) {
        std::lock_guard<std::mutex> lk(mu);
        lru.remove(id);
        auto it = entries.find(id);
        if (it == entries.end()) return false;
        unlink_lineage_locked(id, it->second);
        entries.erase(it);
        return true;
    }
    // Live relative of a consumed id's lineage, or nullptr (issue #27).
    std::shared_ptr<BatchFactorEntry> lineage_lookup(int32_t stale_id,
                                                     int32_t* out_id) {
        std::lock_guard<std::mutex> lk(mu);
        int32_t origin = 0;
        if (lineage.count(stale_id)) {
            origin = stale_id;
        } else {
            auto rit = retired.find(stale_id);
            if (rit != retired.end()) origin = rit->second;
        }
        if (origin == 0) return nullptr;
        auto lit = lineage.find(origin);
        if (lit == lineage.end()) return nullptr;
        auto eit = entries.find(lit->second);
        if (eit == entries.end()) return nullptr;
        *out_id = lit->second;
        lru.remove(lit->second);
        lru.push_front(lit->second);
        return eit->second;
    }
    bool maps_to(int32_t id, const std::shared_ptr<BatchFactorEntry>& entry) {
        std::lock_guard<std::mutex> lk(mu);
        auto found = entries.find(id);
        return found != entries.end() && found->second == entry;
    }
    size_t size() {
        std::lock_guard<std::mutex> lk(mu);
        return entries.size();
    }
};

struct PhaseLease {
    std::shared_ptr<BatchFactorEntry> entry;
    std::unique_lock<std::mutex> operation_lk;
    int32_t token_id = 0;
};

static bool mtype_from_id(int64_t mtype_id, cudssMatrixType_t* out) {
    switch (mtype_id) {
        case 0: *out = CUDSS_MTYPE_GENERAL; return true;
        case 1: *out = CUDSS_MTYPE_SYMMETRIC; return true;
        case 2: *out = CUDSS_MTYPE_HERMITIAN; return true;
        case 3: *out = CUDSS_MTYPE_SPD; return true;
        case 4: *out = CUDSS_MTYPE_HPD; return true;
    }
    return false;
}

static bool mview_from_id(int64_t mview_id, cudssMatrixViewType_t* out) {
    switch (mview_id) {
        case 0: *out = CUDSS_MVIEW_FULL; return true;
        case 1: *out = CUDSS_MVIEW_UPPER; return true;
        case 2: *out = CUDSS_MVIEW_LOWER; return true;
    }
    return false;
}

template <ffi::DataType T>
static ffi::Error token_begin_phase(cudaStream_t stream,
                                    ffi::Buffer<ffi::S32>& token_buf,
                                    const int64_t device_id,
                                    PhaseLease* out) {
    if (auto err = validate_ffi_device(device_id); err.failure()) return err;
    int64_t count = token_buf.element_count();
    std::vector<int32_t> ids(count);
    CUDA_TOKEN_CHECK(cudaMemcpyAsync(ids.data(), token_buf.typed_data(),
                                     count * sizeof(int32_t),
                                     cudaMemcpyDeviceToHost, stream));
    CUDA_TOKEN_CHECK(cudaStreamSynchronize(stream));
    for (int64_t i = 1; i < count; ++i) {
        if (ids[i] != ids[0]) {
            return ffi::Error::Internal(
                "spineax pbatch token: batched token ids differ (" +
                std::to_string(ids[0]) + " vs " + std::to_string(ids[i]) +
                ") — stacked distinct single-system tokens cannot be batch-"
                "operated. Batch them at analysis time (vmap(analyze) or "
                "batch-shaped values) so they share one block-diagonal entry.");
        }
    }
    out->token_id = ids[0];
    while (true) {
        std::shared_ptr<BatchFactorEntry> e;
        auto& r = BatchTokenRegistry::instance();
        {
            std::lock_guard<std::mutex> lk(r.mu);
            auto it = r.entries.find(ids[0]);
            if (it != r.entries.end()) {
                r.lru.remove(ids[0]);
                r.lru.push_front(ids[0]);
                e = it->second;
            }
        }
        // Not resident (renamed away or evicted) is not an error: the caller
        // rebuilds from its own operands.
        if (!e) return ffi::Error::Success();

        std::unique_lock<std::mutex> operation_lk(e->operation_mu);
        // A numeric phase may have rekeyed this entry while this thread was
        // waiting. Retry the lookup so a diverged token rebuilds its own state.
        if (!r.maps_to(ids[0], e)) continue;
        if (e->device_id != device_id) {
            return ffi::Error::Internal(
                "spineax token: payload device differs from its factorization");
        }
        if (e->dtype != get_cudss_data_type<T>()) {
            return ffi::Error::Internal(
                "spineax pbatch token: dtype mismatch for token " +
                std::to_string(ids[0]));
        }
        if (count != 1 && count != e->batch) {
            return ffi::Error::Internal(
                "spineax pbatch token: got " + std::to_string(count) +
                " token ids for an entry with batch size " +
                std::to_string(e->batch));
        }
        CUDA_TOKEN_CHECK(cudaStreamWaitEvent(stream, e->done, 0));
        CUDSS_TOKEN_CHECK(cudssSetStream(e->handle, stream), "cudssSetStream");
        out->entry = std::move(e);
        out->operation_lk = std::move(operation_lk);
        return ffi::Error::Success();
    }
}

// issue #27: a factorize whose input id was consumed (star/tree branching
// from one analyzed scope) may run its FACTORIZATION on a LIVE entry of the
// same lineage — cuDSS reuses that entry's analysis implicitly, so the
// branch costs a factorization instead of a full re-analysis. The relative
// is then rekeyed like any consumed state; if its previous token is still
// held somewhere, its next use self-heals (existing machinery). Guards
// mirror token_begin_phase; any failure leaves *out empty and the caller
// takes the rebuild path, which is always correct.
template <ffi::DataType T>
static void try_lineage_steal(cudaStream_t stream, int32_t stale_id,
                              int64_t token_count, const int64_t device_id,
                              const int64_t mtype_id, const int64_t mview_id,
                              PhaseLease* out) {
    auto& r = BatchTokenRegistry::instance();
    int32_t d_id = 0;
    auto d = r.lineage_lookup(stale_id, &d_id);
    if (!d) return;
    std::unique_lock<std::mutex> operation_lk(d->operation_mu);
    // Raced: the relative was consumed or evicted while this thread waited.
    // No retry — stealing the NEW state would clobber factors the winner
    // just produced; rebuilding costs the same analysis either way.
    if (!r.maps_to(d_id, d)) return;
    cudssMatrixType_t mtype;
    cudssMatrixViewType_t mview;
    if (!mtype_from_id(mtype_id, &mtype) || !mview_from_id(mview_id, &mview)) return;
    if (d->device_id != device_id || d->dtype != get_cudss_data_type<T>() ||
        d->mtype != mtype || d->mview != mview) return;
    if (token_count != 1 && token_count != d->batch) return;
    if (cudaStreamWaitEvent(stream, d->done, 0) != cudaSuccess) return;
    if (cudssSetStream(d->handle, stream) != CUDSS_STATUS_SUCCESS) return;
    out->entry = std::move(d);
    out->operation_lk = std::move(operation_lk);
    out->token_id = d_id;
}

template <ffi::DataType T>
static ffi::Error batch_token_repoint(
    BatchFactorEntry* e, cudaStream_t stream,
    ffi::Buffer<ffi::S32>& offsets_buf,     // (B*n + 1,) expanded
    ffi::Buffer<ffi::S32>& columns_buf,     // (B*nnz,) expanded
    ffi::Buffer<ffi::U32>& fingerprint_buf, // uint32[2] structure checksum
    ffi::Buffer<T>& values_buf              // (B*nnz,) block values
) {
    const int64_t N = e->batch * e->block_n;
    const int64_t NNZ = e->batch * e->block_nnz;
    if ((int64_t)offsets_buf.element_count() != N + 1 ||
        (int64_t)columns_buf.element_count() != NNZ) {
        return ffi::Error::Internal(
            "spineax pbatch token: structure size (" +
            std::to_string(offsets_buf.element_count()) + ", " +
            std::to_string(columns_buf.element_count()) +
            ") != expanded block system (" + std::to_string(N + 1) + ", " +
            std::to_string(NNZ) + ")");
    }
    if ((int64_t)values_buf.element_count() != NNZ) {
        return ffi::Error::Internal(
            "spineax pbatch token: values size " +
            std::to_string(values_buf.element_count()) + " != batch*nnz = " +
            std::to_string(NNZ));
    }
    uint32_t fp[2] = {0, 0};
    CUDA_TOKEN_CHECK(cudaMemcpyAsync(fp, fingerprint_buf.typed_data(),
                                     sizeof(fp), cudaMemcpyDeviceToHost,
                                     stream));
    CUDA_TOKEN_CHECK(cudaStreamSynchronize(stream));
    if (fp[0] != e->fp[0] || fp[1] != e->fp[1]) {
        return ffi::Error::Internal(
            "spineax token: CSR structure fingerprint mismatch — the "
            "offsets/columns behind this token differ from the pattern that "
            "was analyzed. A FactorToken's structure is immutable (cuDSS's "
            "analysis and pivot order are tied to it); analyze the new "
            "pattern instead of editing the token's leaves.");
    }
    CUDSS_TOKEN_CHECK(cudssMatrixSetCsrPointers(e->A,
        offsets_buf.typed_data(), NULL, columns_buf.typed_data(),
        values_buf.typed_data()), "cudssMatrixSetCsrPointers");
    return ffi::Error::Success();
}

// entry construction: cuDSS objects + block ANALYSIS ==========================
// The Python wrappers hand over the ALREADY-EXPANDED block-diagonal structure.
// Shared by analyze and by every phase's rebuild-on-miss path.
template <ffi::DataType T>
static ffi::Error create_entry(
    cudaStream_t stream,
    ffi::Buffer<T>& csr_values_buf,          // (B*nnz,) contiguous == block values
    ffi::Buffer<ffi::S32>& offsets_buf,      // (B*n + 1,) expanded
    ffi::Buffer<ffi::S32>& columns_buf,      // (B*nnz,) expanded
    ffi::Buffer<ffi::U32>& fingerprint_buf,  // uint32[2] structure checksum
    const int64_t batch_size,
    const int64_t device_id,
    const int64_t mtype_id,
    const int64_t mview_id,
    const int64_t reordering_id,
    const int64_t memory_id,
    std::shared_ptr<BatchFactorEntry>* out
) {
    using nat = typename get_native_data_type<T>::type;
    if (auto err = validate_ffi_device(device_id); err.failure()) return err;

    auto e = std::make_shared<BatchFactorEntry>();
    CUDA_TOKEN_CHECK(cudaEventCreateWithFlags(&e->done, cudaEventDisableTiming));
    // The analyzed pattern's fingerprint: every later phase verifies its
    // structure content against this (see batch_token_repoint).
    CUDA_TOKEN_CHECK(cudaMemcpyAsync(e->fp, fingerprint_buf.typed_data(),
                                     sizeof(e->fp), cudaMemcpyDeviceToHost,
                                     stream));
    CUDA_TOKEN_CHECK(cudaStreamSynchronize(stream));
    e->batch = batch_size;
    const int64_t N = offsets_buf.element_count() - 1;
    const int64_t NNZ = columns_buf.element_count();
    if (batch_size < 1 || N % batch_size != 0 || NNZ % batch_size != 0) {
        return ffi::Error::Internal(
            "spineax pbatch token: expanded structure (" + std::to_string(N) +
            ", " + std::to_string(NNZ) + ") is not divisible by batch_size " +
            std::to_string(batch_size));
    }
    e->block_n = N / batch_size;
    e->block_nnz = NNZ / batch_size;
    if ((int64_t)csr_values_buf.element_count() != NNZ) {
        return ffi::Error::Internal(
            "spineax pbatch token: values size " +
            std::to_string(csr_values_buf.element_count()) + " != batch*nnz = " +
            std::to_string(NNZ));
    }
    e->elem_size = sizeof(nat);
    e->dtype = get_cudss_data_type<T>();
    e->device_id = device_id;

    if (!mtype_from_id(mtype_id, &e->mtype)) {
        return ffi::Error::Internal(
            "spineax pbatch token: invalid mtype_id (0 general, 1 symmetric, 2 hermitian, 3 spd, 4 hpd)");
    }
    if (!mview_from_id(mview_id, &e->mview)) {
        return ffi::Error::Internal(
            "spineax pbatch token: invalid mview_id (0 full, 1 upper, 2 lower)");
    }

    CUDSS_TOKEN_CHECK(cudssCreate(&e->handle), "cudssCreate");
    CUDSS_TOKEN_CHECK(cudssSetStream(e->handle, stream), "cudssSetStream");
    CUDSS_TOKEN_CHECK(cudssConfigCreate(&e->config), "cudssConfigCreate");
    if (reordering_id < 0 || reordering_id > CUDSS_REORDERING_ALG_NONE) {
        return ffi::Error::Internal(
            "spineax pbatch token: invalid reordering_id (cudssReorderingAlg_t: "
            "0 default, 1 btf_colamd, 2 colamd, 3 amd, 4 nested_dissection, 5 none)");
    }
    if (reordering_id) {
        auto alg = static_cast<cudssReorderingAlg_t>(reordering_id);
        CUDSS_TOKEN_CHECK(cudssConfigSet(e->config, CUDSS_CONFIG_REORDERING_ALG,
                                         &alg, sizeof(alg)), "cudssConfigSet reordering");
    }
    if (memory_id != 0 && memory_id != 1) {
        return ffi::Error::Internal(
            "spineax pbatch token: invalid memory_id (0 device, 1 hybrid host+device)");
    }
    if (memory_id) {
        int hybrid = 1;
        CUDSS_TOKEN_CHECK(cudssConfigSet(e->config, CUDSS_CONFIG_HYBRID_MEMORY_MODE,
                                         &hybrid, sizeof(hybrid)), "cudssConfigSet hybrid memory");
    }
    CUDSS_TOKEN_CHECK(cudssDataCreate(e->handle, &e->data), "cudssDataCreate");

    // Point the descriptor at THIS call's buffers; every later phase repoints
    // at its own call's buffers before executing (zero-copy discipline).
    CUDSS_TOKEN_CHECK(cudssMatrixCreateCsr(&e->A, N, N, NNZ,
        const_cast<int32_t*>(offsets_buf.typed_data()), NULL,
        const_cast<int32_t*>(columns_buf.typed_data()),
        const_cast<nat*>(csr_values_buf.typed_data()),
        CUDSS_R_32I, CUDSS_R_32I, e->dtype,
        e->mtype, e->mview, CUDSS_BASE_ZERO), "cudssMatrixCreateCsr");

    // Placeholder dense descriptors: ANALYSIS/FACTORIZATION never dereference
    // x/b data, but the API requires the objects. The (stale after this call)
    // pointer is never read.
    CUDSS_TOKEN_CHECK(cudssMatrixCreateDn(&e->b_dummy, N, 1, N,
        const_cast<nat*>(csr_values_buf.typed_data()),
        e->dtype, CUDSS_LAYOUT_COL_MAJOR), "cudssMatrixCreateDn b (dummy)");
    CUDSS_TOKEN_CHECK(cudssMatrixCreateDn(&e->x_dummy, N, 1, N,
        const_cast<nat*>(csr_values_buf.typed_data()),
        e->dtype, CUDSS_LAYOUT_COL_MAJOR), "cudssMatrixCreateDn x (dummy)");

    CUDSS_TOKEN_CHECK(cudssExecute(e->handle, CUDSS_PHASE_ANALYSIS,
        e->config, e->data, e->A, e->x_dummy, e->b_dummy), "cudssExecute analysis");
    *out = std::move(e);
    return ffi::Error::Success();
}

// Write `id` to every slot of the token result buffer (1 or B equal copies).
static ffi::Error token_write_id(cudaStream_t stream,
                                 ffi::ResultBuffer<ffi::S32>& token_buf,
                                 int32_t id) {
    std::vector<int32_t> ids(token_buf->element_count(), id);
    CUDA_TOKEN_CHECK(cudaMemcpyAsync(token_buf->typed_data(), ids.data(),
                                     ids.size() * sizeof(int32_t),
                                     cudaMemcpyHostToDevice, stream));
    CUDA_TOKEN_CHECK(cudaStreamSynchronize(stream));  // ids is a stack local
    return ffi::Error::Success();
}

// analyze: fresh entry, ANALYSIS only =========================================
template <ffi::DataType T>
static ffi::Error PbatchTokenAnalyze(
    cudaStream_t stream,
    ffi::Buffer<T> csr_values_buf,
    ffi::Buffer<ffi::S32> offsets_buf,
    ffi::Buffer<ffi::S32> columns_buf,
    ffi::Buffer<ffi::U32> fingerprint_buf,
    ffi::ResultBuffer<ffi::S32> token_buf,  // int32[1]
    const int64_t batch_size,
    const int64_t device_id,
    const int64_t mtype_id,
    const int64_t mview_id,
    const int64_t reordering_id,
    const int64_t memory_id
) {
    std::shared_ptr<BatchFactorEntry> e;
    if (auto err = create_entry<T>(stream, csr_values_buf, offsets_buf,
                                   columns_buf, fingerprint_buf, batch_size,
                                   device_id, mtype_id, mview_id,
                                   reordering_id, memory_id, &e);
        err.failure()) return err;
    CUDA_TOKEN_CHECK(cudaEventRecord(e->done, stream));
    int32_t id = BatchTokenRegistry::instance().insert(std::move(e));
    return token_write_id(stream, token_buf, id);
}

// factorize / refactorize: block numeric phase ================================
template <ffi::DataType T, bool kRefactorize>
static ffi::Error PbatchTokenNumeric(
    cudaStream_t stream,
    ffi::Buffer<ffi::S32> token_in,         // 1 or B equal ids
    ffi::Buffer<ffi::S32> offsets_buf,      // (B*n + 1,) expanded
    ffi::Buffer<ffi::S32> columns_buf,      // (B*nnz,) expanded
    ffi::Buffer<ffi::U32> fingerprint_buf,  // uint32[2] structure checksum
    ffi::Buffer<T> csr_values_buf,          // (B*nnz,)
    ffi::ResultBuffer<ffi::S32> token_out,  // same count, FRESH id
    const int64_t device_id,
    const int64_t mtype_id,
    const int64_t mview_id,
    const int64_t reordering_id,
    const int64_t memory_id
) {
    auto& r = BatchTokenRegistry::instance();
    PhaseLease lease;
    if (auto err = token_begin_phase<T>(stream, token_in, device_id, &lease);
        err.failure()) return err;
    auto& e = lease.entry;

    bool rebuilt = !e;
    bool stole = false;
    if (rebuilt && !kRefactorize) {
        // Branch from a live relative's analysis before paying a rebuild
        // (issue #27). The repoint doubles as the size + structure-
        // fingerprint gate: on mismatch fall through to the rebuild path.
        // Refactorize is excluded — its heal contract is fresh pivots from
        // a fresh factorization, and a relative's pivot order came from
        // DIFFERENT values.
        PhaseLease steal;
        try_lineage_steal<T>(stream, lease.token_id, token_in.element_count(),
                             device_id, mtype_id, mview_id, &steal);
        if (steal.entry &&
            !batch_token_repoint<T>(steal.entry.get(), stream, offsets_buf,
                                    columns_buf, fingerprint_buf,
                                    csr_values_buf).failure()) {
            lease = std::move(steal);  // e (reference) now sees the relative
            rebuilt = false;
            stole = true;
        }
    }
    if (rebuilt) {
        // Input state not resident: rebuild from this call's own buffers.
        // The old pivot order went with the old entry, so a refactorize
        // rebuild is a fresh full factorization too. The expanded structure
        // IS the system here, hence batch_size 1.
        if (auto err = create_entry<T>(stream, csr_values_buf, offsets_buf,
                                       columns_buf, fingerprint_buf, 1,
                                       device_id, mtype_id, mview_id,
                                       reordering_id, memory_id, &e);
            err.failure()) return err;
        r.rebuilds.fetch_add(1);
    } else if (!stole) {
        if (auto err = batch_token_repoint<T>(e.get(), stream, offsets_buf,
                                              columns_buf, fingerprint_buf,
                                              csr_values_buf); err.failure()) return err;
    }
    CUDSS_TOKEN_CHECK(cudssExecute(e->handle,
        (kRefactorize && !rebuilt) ? CUDSS_PHASE_REFACTORIZATION
                                   : CUDSS_PHASE_FACTORIZATION,
        e->config, e->data, e->A, e->x_dummy, e->b_dummy),
        "cudssExecute factorization");
    CUDA_TOKEN_CHECK(cudaEventRecord(e->done, stream));

    // A numeric phase consumes its input state's name: the entry moves to a
    // fresh id, so every id ever handed out names one immutable state. A
    // stolen relative is consumed the same way (lease.token_id is ITS id).
    int32_t new_id = rebuilt ? r.insert(e) : r.rekey(lease.token_id, e);
    if (stole) r.branches.fetch_add(1);
    return token_write_id(stream, token_out, new_id);
}

// solve: block SOLVE (multi-RHS via the layout identity on the block system)
template <ffi::DataType T>
static ffi::Error PbatchTokenSolve(
    cudaStream_t stream,
    ffi::Buffer<ffi::S32> token_in,         // 1 or B equal ids
    ffi::Buffer<ffi::S32> offsets_buf,      // (B*n + 1,) expanded
    ffi::Buffer<ffi::S32> columns_buf,      // (B*nnz,) expanded
    ffi::Buffer<ffi::U32> fingerprint_buf,  // uint32[2] structure checksum
    ffi::Buffer<T> csr_values_buf,          // (B*nnz,) — last-factorized values
    ffi::Buffer<T> b_values_buf,            // (B*n,) or (R, B*n) row-major
    ffi::ResultBuffer<T> out_values_buf,
    const int64_t device_id,
    const int64_t mtype_id,
    const int64_t mview_id,
    const int64_t reordering_id,
    const int64_t memory_id
) {
    PhaseLease lease;
    if (auto err = token_begin_phase<T>(stream, token_in, device_id, &lease);
        err.failure()) return err;
    auto& e = lease.entry;

    if (!e) {
        // Not resident: rebuild the token's state — its values buffer is by
        // contract the values that produced its factors — and heal in place
        // under the caller's id, so later solves of this token hit.
        if (auto err = create_entry<T>(stream, csr_values_buf, offsets_buf,
                                       columns_buf, fingerprint_buf, 1,
                                       device_id, mtype_id, mview_id,
                                       reordering_id, memory_id, &e);
            err.failure()) return err;
        CUDSS_TOKEN_CHECK(cudssExecute(e->handle, CUDSS_PHASE_FACTORIZATION,
            e->config, e->data, e->A, e->x_dummy, e->b_dummy),
            "cudssExecute factorization (rebuild)");
        auto& r = BatchTokenRegistry::instance();
        r.rebuilds.fetch_add(1);
        r.insert(e, lease.token_id);
    } else {
        // SOLVE never reads A's values (spineax runs cuDSS-internal IR
        // permanently OFF; refinement is JAX-side in _refined_solve), but
        // repointing at this call's live buffers keeps every path
        // zero-copy-safe and fingerprint-checked.
        if (auto err = batch_token_repoint<T>(e.get(), stream, offsets_buf,
                                              columns_buf, fingerprint_buf,
                                              csr_values_buf); err.failure()) return err;
    }
    const int64_t N = e->batch * e->block_n;
    if (N == 0 || (int64_t)b_values_buf.element_count() % N != 0) {
        return ffi::Error::Internal(
            "spineax pbatch token: rhs size " +
            std::to_string(b_values_buf.element_count()) +
            " is not a multiple of batch*n = " + std::to_string(N));
    }
    int64_t nrhs = b_values_buf.element_count() / N;

    cudssMatrix_t bmat = nullptr, xmat = nullptr;
    CUDSS_TOKEN_CHECK(cudssMatrixCreateDn(&bmat, N, nrhs, N,
        const_cast<typename get_native_data_type<T>::type*>(b_values_buf.typed_data()),
        e->dtype, CUDSS_LAYOUT_COL_MAJOR), "cudssMatrixCreateDn b (solve)");
    CUDSS_TOKEN_CHECK(cudssMatrixCreateDn(&xmat, N, nrhs, N,
        out_values_buf->typed_data(), e->dtype, CUDSS_LAYOUT_COL_MAJOR),
        "cudssMatrixCreateDn x (solve)");

    cudssStatus_t solve_status = cudssExecute(e->handle, CUDSS_PHASE_SOLVE,
        e->config, e->data, e->A, xmat, bmat);
    cudssMatrixDestroy(bmat);
    cudssMatrixDestroy(xmat);
    if (solve_status != CUDSS_STATUS_SUCCESS) {
        return ffi::Error::Internal(
            "spineax pbatch token: cuDSS solve failed (status " +
            std::to_string(static_cast<int>(solve_status)) + ")");
    }
    CUDA_TOKEN_CHECK(cudaEventRecord(e->done, stream));
    return ffi::Error::Success();
}

// query: read every cuDSS data item from a factorized token ===================
static constexpr int64_t kNdPartitionTreeSize = (1 << 10) - 1;  // nd_nlevels=10 default

template <ffi::DataType T>
static ffi::Error PbatchTokenQuery(
    cudaStream_t stream,
    ffi::Buffer<ffi::S32> token_in,                    // 1 or B equal ids
    ffi::Buffer<ffi::S32> offsets_buf,                 // (B*n + 1,) expanded
    ffi::Buffer<ffi::S32> columns_buf,                 // (B*nnz,) expanded
    ffi::Buffer<ffi::U32> fingerprint_buf,             // uint32[2] structure checksum
    ffi::Buffer<T> csr_values_buf,                     // (B*nnz,) — last-factorized values
    ffi::ResultBuffer<ffi::S64> lu_nnz_buf,            // [1]
    ffi::ResultBuffer<ffi::S32> npivots_buf,           // [1]
    ffi::ResultBuffer<ffi::S32> inertia_buf,           // [2] cuDSS native (block-global)
    ffi::ResultBuffer<ffi::S32> perm_reorder_row_buf,  // [N]
    ffi::ResultBuffer<ffi::S32> perm_reorder_col_buf,  // [N]
    ffi::ResultBuffer<ffi::S32> perm_row_buf,          // [N] (reordering alg 1/2 only)
    ffi::ResultBuffer<ffi::S32> perm_col_buf,          // [N] (reordering alg 1/2 only)
    ffi::ResultBuffer<ffi::S32> perm_matching_buf,     // [N]
    ffi::ResultBuffer<T> diag_buf,                     // [N]
    ffi::ResultBuffer<ffi::F32> scale_row_buf,         // [N]
    ffi::ResultBuffer<ffi::F32> scale_col_buf,         // [N]
    ffi::ResultBuffer<ffi::S32> nd_partition_tree_buf, // [kNdPartitionTreeSize]
    ffi::ResultBuffer<ffi::S32> nsuperpanels_buf,      // [1]
    ffi::ResultBuffer<ffi::S64> schur_shape_buf,       // [2]
    const int64_t device_id,
    const int64_t mtype_id,
    const int64_t mview_id,
    const int64_t reordering_id,
    const int64_t memory_id
) {
    PhaseLease lease;
    if (auto err = token_begin_phase<T>(stream, token_in, device_id, &lease);
        err.failure()) return err;
    auto& e = lease.entry;

    if (!e) {
        // Same heal as solve: rebuild the token's factorized state from its
        // own buffers, in place under the caller's id.
        if (auto err = create_entry<T>(stream, csr_values_buf, offsets_buf,
                                       columns_buf, fingerprint_buf, 1,
                                       device_id, mtype_id, mview_id,
                                       reordering_id, memory_id, &e);
            err.failure()) return err;
        CUDSS_TOKEN_CHECK(cudssExecute(e->handle, CUDSS_PHASE_FACTORIZATION,
            e->config, e->data, e->A, e->x_dummy, e->b_dummy),
            "cudssExecute factorization (rebuild)");
        auto& r = BatchTokenRegistry::instance();
        r.rebuilds.fetch_add(1);
        r.insert(e, lease.token_id);
    } else {
        // repoint = size + structure-fingerprint validation (query reads no
        // buffers itself, but should reject tampered tokens like every phase)
        if (auto err = batch_token_repoint<T>(e.get(), stream, offsets_buf,
                                              columns_buf, fingerprint_buf,
                                              csr_values_buf); err.failure()) return err;
    }

    const int64_t N = e->batch * e->block_n;
    // The output buffers are sized by the caller's static token metadata; a
    // mismatch (e.g. query of a vmap-minted batch token from inside vmap)
    // must fail loudly rather than overrun the buffers.
    if ((int64_t)diag_buf->element_count() != N) {
        return ffi::Error::Internal(
            "spineax token: query output size " +
            std::to_string(diag_buf->element_count()) +
            " != block system dimension " + std::to_string(N) +
            " (query is an eager/outer-level operation — call it outside "
            "vmap with batch-shaped token metadata)");
    }
    size_t written = 0;

    // host-side scalars: dataGet to host, then H2D into the result buffer;
    // zero on failure so Python always gets well-defined values
    #define QUERY_HOST_SCALAR(PARAM, TYPE, COUNT, BUF) \
        do { \
            TYPE tmp_[COUNT] = {}; \
            if (cudssDataGet(e->handle, e->data, PARAM, tmp_, sizeof(tmp_), \
                             &written) != CUDSS_STATUS_SUCCESS) { \
                for (int i_ = 0; i_ < (COUNT); ++i_) tmp_[i_] = 0; \
            } \
            CUDA_TOKEN_CHECK(cudaMemcpy((BUF)->typed_data(), tmp_, sizeof(tmp_), \
                                        cudaMemcpyHostToDevice)); \
        } while (0)

    // device-side arrays: dataGet writes the device buffer directly
    #define QUERY_DEVICE_ARRAY(PARAM, BUF, BYTES) \
        do { \
            if (cudssDataGet(e->handle, e->data, PARAM, (BUF)->typed_data(), \
                             (BYTES), &written) != CUDSS_STATUS_SUCCESS) { \
                CUDA_TOKEN_CHECK(cudaMemset((BUF)->typed_data(), 0, (BYTES))); \
            } \
        } while (0)

    QUERY_HOST_SCALAR(CUDSS_DATA_LU_NNZ, int64_t, 1, lu_nnz_buf);
    QUERY_HOST_SCALAR(CUDSS_DATA_NPIVOTS, int32_t, 1, npivots_buf);
    QUERY_HOST_SCALAR(CUDSS_DATA_INERTIA, int32_t, 2, inertia_buf);
    QUERY_HOST_SCALAR(CUDSS_DATA_NSUPERPANELS, int32_t, 1, nsuperpanels_buf);
    QUERY_HOST_SCALAR(CUDSS_DATA_SCHUR_SHAPE, int64_t, 2, schur_shape_buf);

    QUERY_DEVICE_ARRAY(CUDSS_DATA_PERM_REORDER_ROW, perm_reorder_row_buf, N * sizeof(int32_t));
    QUERY_DEVICE_ARRAY(CUDSS_DATA_PERM_REORDER_COL, perm_reorder_col_buf, N * sizeof(int32_t));
    QUERY_DEVICE_ARRAY(CUDSS_DATA_PERM_ROW, perm_row_buf, N * sizeof(int32_t));
    QUERY_DEVICE_ARRAY(CUDSS_DATA_PERM_COL, perm_col_buf, N * sizeof(int32_t));
    QUERY_DEVICE_ARRAY(CUDSS_DATA_PERM_MATCHING, perm_matching_buf, N * sizeof(int32_t));
    QUERY_DEVICE_ARRAY(CUDSS_DATA_DIAG, diag_buf, N * (int64_t)e->elem_size);
    QUERY_DEVICE_ARRAY(CUDSS_DATA_SCALE_ROW, scale_row_buf, N * sizeof(float));
    QUERY_DEVICE_ARRAY(CUDSS_DATA_SCALE_COL, scale_col_buf, N * sizeof(float));
    // cuDSS >= 0.8 removed CUDSS_DATA_ELIMINATION_TREE; the nested-dissection
    // partition tree is its successor and exposes the same reordering structure.
    QUERY_DEVICE_ARRAY(CUDSS_DATA_ND_PARTITION_TREE, nd_partition_tree_buf,
                       kNdPartitionTreeSize * sizeof(int32_t));

    #undef QUERY_HOST_SCALAR
    #undef QUERY_DEVICE_ARRAY
    CUDA_TOKEN_CHECK(cudaEventRecord(e->done, stream));
    return ffi::Error::Success();
}

// token FFI handler definitions ===============================================
#define DEFINE_PBATCH_TOKEN_FFI_HANDLERS(TypeName, DataType) \
    XLA_FFI_DEFINE_HANDLER(kPbatchTokenAnalyze##TypeName, PbatchTokenAnalyze<DataType>, \
        ffi::Ffi::Bind() \
            .Ctx<ffi::PlatformStream<cudaStream_t>>() \
            .Arg<ffi::Buffer<DataType>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Arg<ffi::Buffer<ffi::U32>>() \
            .Ret<ffi::Buffer<ffi::S32>>() \
            .Attr<int64_t>("batch_size") \
            .Attr<int64_t>("device_id") \
            .Attr<int64_t>("mtype_id") \
            .Attr<int64_t>("mview_id") \
            .Attr<int64_t>("reordering_id") \
            .Attr<int64_t>("memory_id")); \
    \
    XLA_FFI_DEFINE_HANDLER(kPbatchTokenFactorize##TypeName, (PbatchTokenNumeric<DataType, false>), \
        ffi::Ffi::Bind() \
            .Ctx<ffi::PlatformStream<cudaStream_t>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Arg<ffi::Buffer<ffi::U32>>() \
            .Arg<ffi::Buffer<DataType>>() \
            .Ret<ffi::Buffer<ffi::S32>>() \
            .Attr<int64_t>("device_id") \
            .Attr<int64_t>("mtype_id") \
            .Attr<int64_t>("mview_id") \
            .Attr<int64_t>("reordering_id") \
            .Attr<int64_t>("memory_id")); \
    \
    XLA_FFI_DEFINE_HANDLER(kPbatchTokenRefactorize##TypeName, (PbatchTokenNumeric<DataType, true>), \
        ffi::Ffi::Bind() \
            .Ctx<ffi::PlatformStream<cudaStream_t>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Arg<ffi::Buffer<ffi::U32>>() \
            .Arg<ffi::Buffer<DataType>>() \
            .Ret<ffi::Buffer<ffi::S32>>() \
            .Attr<int64_t>("device_id") \
            .Attr<int64_t>("mtype_id") \
            .Attr<int64_t>("mview_id") \
            .Attr<int64_t>("reordering_id") \
            .Attr<int64_t>("memory_id")); \
    \
    XLA_FFI_DEFINE_HANDLER(kPbatchTokenSolve##TypeName, PbatchTokenSolve<DataType>, \
        ffi::Ffi::Bind() \
            .Ctx<ffi::PlatformStream<cudaStream_t>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Arg<ffi::Buffer<ffi::U32>>() \
            .Arg<ffi::Buffer<DataType>>() \
            .Arg<ffi::Buffer<DataType>>() \
            .Ret<ffi::Buffer<DataType>>() \
            .Attr<int64_t>("device_id") \
            .Attr<int64_t>("mtype_id") \
            .Attr<int64_t>("mview_id") \
            .Attr<int64_t>("reordering_id") \
            .Attr<int64_t>("memory_id")); \
    \
    XLA_FFI_DEFINE_HANDLER(kPbatchTokenQuery##TypeName, PbatchTokenQuery<DataType>, \
        ffi::Ffi::Bind() \
            .Ctx<ffi::PlatformStream<cudaStream_t>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Arg<ffi::Buffer<ffi::S32>>() \
            .Arg<ffi::Buffer<ffi::U32>>() \
            .Arg<ffi::Buffer<DataType>>() \
            .Ret<ffi::Buffer<ffi::S64>>() \
            .Ret<ffi::Buffer<ffi::S32>>() \
            .Ret<ffi::Buffer<ffi::S32>>() \
            .Ret<ffi::Buffer<ffi::S32>>() \
            .Ret<ffi::Buffer<ffi::S32>>() \
            .Ret<ffi::Buffer<ffi::S32>>() \
            .Ret<ffi::Buffer<ffi::S32>>() \
            .Ret<ffi::Buffer<ffi::S32>>() \
            .Ret<ffi::Buffer<DataType>>() \
            .Ret<ffi::Buffer<ffi::F32>>() \
            .Ret<ffi::Buffer<ffi::F32>>() \
            .Ret<ffi::Buffer<ffi::S32>>() \
            .Ret<ffi::Buffer<ffi::S32>>() \
            .Ret<ffi::Buffer<ffi::S64>>() \
            .Attr<int64_t>("device_id") \
            .Attr<int64_t>("mtype_id") \
            .Attr<int64_t>("mview_id") \
            .Attr<int64_t>("reordering_id") \
            .Attr<int64_t>("memory_id"));

DEFINE_PBATCH_TOKEN_FFI_HANDLERS(f32, ffi::F32);
DEFINE_PBATCH_TOKEN_FFI_HANDLERS(f64, ffi::F64);
DEFINE_PBATCH_TOKEN_FFI_HANDLERS(c64, ffi::C64);
DEFINE_PBATCH_TOKEN_FFI_HANDLERS(c128, ffi::C128);

#define EXPORT_PBATCH_TOKEN_HANDLERS(m, TypeName) \
    m.def("token_handlers_" #TypeName, []() { \
        nb::dict d; \
        d["analyze"] = nb::capsule(reinterpret_cast<void*>(kPbatchTokenAnalyze##TypeName)); \
        d["factorize"] = nb::capsule(reinterpret_cast<void*>(kPbatchTokenFactorize##TypeName)); \
        d["refactorize"] = nb::capsule(reinterpret_cast<void*>(kPbatchTokenRefactorize##TypeName)); \
        d["solve"] = nb::capsule(reinterpret_cast<void*>(kPbatchTokenSolve##TypeName)); \
        d["query"] = nb::capsule(reinterpret_cast<void*>(kPbatchTokenQuery##TypeName)); \
        return d; \
    });

void register_csr_transpose_handlers(nb::module_& m);

// generate all nanobind modules! :)
NB_MODULE(pbatch_solve, m) {

    EXPORT_PBATCH_TOKEN_HANDLERS(m, f32);
    EXPORT_PBATCH_TOKEN_HANDLERS(m, f64);
    EXPORT_PBATCH_TOKEN_HANDLERS(m, c64);
    EXPORT_PBATCH_TOKEN_HANDLERS(m, c128);
    register_csr_transpose_handlers(m);

    m.def("token_release", [](int32_t id) {
        return BatchTokenRegistry::instance().release(id);
    });
    m.def("token_registry_size", []() {
        return BatchTokenRegistry::instance().size();
    });
    m.def("token_cache_capacity", []() {
        return BatchTokenRegistry::instance().capacity();
    });
    m.def("token_rebuild_count", []() {
        return BatchTokenRegistry::instance().rebuilds.load();
    });
    m.def("token_branch_count", []() {
        return BatchTokenRegistry::instance().branches.load();
    });
    m.def("token_lineage_stats", []() {
        auto& r = BatchTokenRegistry::instance();
        std::lock_guard<std::mutex> lk(r.mu);
        nb::dict d;
        d["lineage"] = r.lineage.size();
        d["retired"] = r.retired.size();
        return d;
    });
    m.def("nd_partition_tree_size", []() {
        return kNdPartitionTreeSize;
    });
}
