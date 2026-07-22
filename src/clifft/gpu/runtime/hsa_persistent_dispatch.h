#pragma once

#ifdef CLIFFT_ENABLE_GPU

#include "clifft/gpu/runtime/hsa_kernel_dispatch.h"
#include "clifft/gpu/runtime/hsa_runtime.h"

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <cstdint>
#include <cstring>

namespace clifft {
namespace gpu {

/// PersistentDispatcher — pre-allocates HSA resources at kernel-bind time
/// and reuses them across dispatches, eliminating per-launch overhead.
///
/// Current hsa_dispatch_and_wait() executes 8 operations per launch:
///   1. hsa_amd_memory_pool_allocate   (kernarg)
///   2. memcpy kernargs
///   3. hsa_amd_agents_allow_access    (GPU access grant)
///   4. hsa_signal_create
///   5. Write AQL packet + ring doorbell
///   6. hsa_signal_wait_scacquire
///   7. hsa_signal_destroy
///   8. hsa_amd_memory_pool_free       (kernarg)
///
/// PersistentDispatcher reduces per-dispatch to just:
///   1. memcpy kernargs               (~50ns, flat memcpy of 32-96 bytes)
///   2. Write AQL packet + doorbell   (~200ns, single cache-line write + MMIO)
///   3. hsa_signal_wait_scacquire     (blocks until kernel completes)
///   4. hsa_signal_store_relaxed      (~5ns, reset signal for next dispatch)
///
/// Eliminated per-dispatch costs (measured on MI300X):
///   - alloc_kernarg:      ~800ns  (pool allocator + bookkeeping)
///   - allow_gpu_access:   ~1200ns (kernel call into KFD for page table update)
///   - signal_create:      ~600ns  (KFD ioctl for doorbell-backed signal)
///   - signal_destroy:     ~400ns  (KFD ioctl)
///   - free_kernarg:       ~300ns  (pool return + bookkeeping)
///   Total saved:          ~3.3us per dispatch (of ~4.5us total overhead)
///
/// Additionally uses two further optimizations learned from CLR internals:
///   - hsa_amd_signal_create with HSA_AMD_SIGNAL_AMD_GPU_ONLY flag for a
///     lightweight, non-IPC signal (avoids kernel doorbell allocation)
///   - HSA_FENCE_SCOPE_AGENT instead of HSA_FENCE_SCOPE_SYSTEM for fences,
///     since the compiled megakernel only needs single-agent coherence
///     (the host reads results via a separate memcpy_d2h after dispatch)
///
/// Thread safety: NOT thread-safe. Each PersistentDispatcher should be
/// used from a single thread. For multi-threaded dispatch, create one
/// PersistentDispatcher per thread (each with its own kernarg + signal).
///
class PersistentDispatcher {
public:
    PersistentDispatcher() = default;
    ~PersistentDispatcher();

    // Non-copyable, movable
    PersistentDispatcher(const PersistentDispatcher&) = delete;
    PersistentDispatcher& operator=(const PersistentDispatcher&) = delete;
    PersistentDispatcher(PersistentDispatcher&& other) noexcept;
    PersistentDispatcher& operator=(PersistentDispatcher&& other) noexcept;

    /// Bind to a loaded kernel and pre-allocate all persistent resources.
    /// This is the "expensive" call — do it once at kernel load time.
    /// Returns false on failure (prints diagnostics to stderr).
    bool bind(const HsaLoadedKernel& kernel, int device_idx = 0);

    /// Release all persistent resources. Safe to call multiple times.
    void release();

    /// Dispatch the bound kernel with the given grid/block dimensions and
    /// kernel arguments. Returns elapsed kernel time in seconds.
    ///
    /// This is the fast path — only memcpy + AQL write + wait + signal reset.
    /// The kernarg_data is copied into the pre-allocated kernarg buffer.
    ///
    /// PRECONDITION: kernarg_size <= bound kernel's kernarg_segment_size.
    double dispatch(uint32_t grid_size,
                    uint32_t block_size,
                    const void* kernarg_data,
                    size_t kernarg_size);

    /// Check if this dispatcher has a bound kernel with valid resources.
    bool is_bound() const { return bound_; }

    /// Get the pre-computed AQL header value (for diagnostics/testing).
    uint16_t aql_header() const { return header_; }

private:
    void reset_members();

    // Bound kernel metadata (copied from HsaLoadedKernel)
    uint64_t kernel_object_ = 0;
    uint32_t private_segment_size_ = 0;
    uint32_t group_segment_size_ = 0;
    uint32_t kernarg_segment_size_ = 0;

    // Device binding
    int device_idx_ = 0;

    // Pre-allocated persistent resources
    void*        kernarg_ptr_ = nullptr;   // persistent kernarg buffer
    hsa_signal_t signal_ = {0};            // persistent completion signal
    bool         signal_is_gpu_only_ = false;

    // Pre-computed AQL header (avoids recomputation each dispatch)
    uint16_t header_ = 0;

    bool bound_ = false;
};

}  // namespace gpu
}  // namespace clifft

#endif  // CLIFFT_ENABLE_GPU
