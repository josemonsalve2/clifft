// hsa_persistent_dispatch.cc — Zero-overhead HSA kernel dispatch
//
// Pre-allocates kernarg buffer, completion signal, and GPU access grant at
// kernel bind time.  Per-dispatch cost is reduced to:
//   memcpy kernargs + write AQL packet + ring doorbell + wait + reset signal
//
// Design informed by:
//   - CLR/rocclr (HIP runtime): ManagedBuffer for kernarg pooling,
//     HwQueueTracker signal pool with signal reuse via store_relaxed
//   - Kog Labs monokernel: eliminates ~4.5us dispatch overhead on MI300X
//     by removing kernel boundary costs; we adopt the same principle for
//     resource reuse across repeated dispatches of the same kernel
//   - ROCR-Runtime samples: AQL packet construction patterns
//
// Key optimizations over hsa_dispatch_and_wait():
//   1. Kernarg buffer allocated once, reused via memcpy (no alloc/free per dispatch)
//   2. Signal created once, reset via hsa_signal_store_relaxed (no create/destroy)
//   3. GPU access granted once at bind time (no per-dispatch KFD ioctl)
//   4. GPU-only signal (HSA_AMD_SIGNAL_AMD_GPU_ONLY) when available — avoids
//      interrupt doorbell allocation, uses polling which is faster for
//      short-running kernels dispatched in tight loops
//   5. Agent-scope fences instead of system-scope — sufficient when host
//      reads results via explicit memcpy_d2h after the dispatch loop

#ifdef CLIFFT_ENABLE_GPU

#include "clifft/gpu/hsa_persistent_dispatch.h"

#include <atomic>
#include <iostream>

namespace clifft {
namespace gpu {

// -----------------------------------------------------------------------
// Error helper
// -----------------------------------------------------------------------
static bool check_pd(hsa_status_t status, const char* msg) {
    if (status == HSA_STATUS_SUCCESS) return true;
    const char* err = nullptr;
    hsa_status_string(status, &err);
    std::cerr << "[clifft-persistent-dispatch] " << msg << ": "
              << (err ? err : "unknown") << "\n";
    return false;
}

// -----------------------------------------------------------------------
// Construction / destruction / move
// -----------------------------------------------------------------------

PersistentDispatcher::~PersistentDispatcher() {
    release();
}

PersistentDispatcher::PersistentDispatcher(PersistentDispatcher&& other) noexcept {
    *this = std::move(other);
}

PersistentDispatcher& PersistentDispatcher::operator=(PersistentDispatcher&& other) noexcept {
    if (this != &other) {
        release();
        kernel_object_ = other.kernel_object_;
        private_segment_size_ = other.private_segment_size_;
        group_segment_size_ = other.group_segment_size_;
        kernarg_segment_size_ = other.kernarg_segment_size_;
        device_idx_ = other.device_idx_;
        kernarg_ptr_ = other.kernarg_ptr_;
        signal_ = other.signal_;
        signal_is_gpu_only_ = other.signal_is_gpu_only_;
        header_ = other.header_;
        bound_ = other.bound_;
        other.reset_members();
    }
    return *this;
}

void PersistentDispatcher::reset_members() {
    kernel_object_ = 0;
    private_segment_size_ = 0;
    group_segment_size_ = 0;
    kernarg_segment_size_ = 0;
    device_idx_ = 0;
    kernarg_ptr_ = nullptr;
    signal_ = {0};
    signal_is_gpu_only_ = false;
    header_ = 0;
    bound_ = false;
}

// -----------------------------------------------------------------------
// bind() — one-time expensive setup
// -----------------------------------------------------------------------
bool PersistentDispatcher::bind(const HsaLoadedKernel& kernel, int device_idx) {
    if (bound_) {
        std::cerr << "[clifft-persistent-dispatch] already bound, release first\n";
        return false;
    }
    if (!kernel.valid) {
        std::cerr << "[clifft-persistent-dispatch] kernel not valid\n";
        return false;
    }

    auto& rt = hsa_runtime();
    if (!rt.initialized) {
        std::cerr << "[clifft-persistent-dispatch] runtime not initialized\n";
        return false;
    }

    device_idx_ = device_idx;
    auto& dev = rt.device(device_idx_);

    // Copy kernel metadata
    kernel_object_ = kernel.kernel_object;
    private_segment_size_ = kernel.private_segment_size;
    group_segment_size_ = kernel.group_segment_size;
    kernarg_segment_size_ = kernel.kernarg_segment_size;

    // --- Step 1: Pre-allocate persistent kernarg buffer ---
    // Allocate from the kernarg pool. This buffer will be reused for every
    // dispatch by overwriting with memcpy. The kernarg pool is fine-grained
    // (coherent) memory, so no explicit cache flush is needed.
    kernarg_ptr_ = rt.alloc_kernarg(kernarg_segment_size_, device_idx_);
    if (!kernarg_ptr_) {
        std::cerr << "[clifft-persistent-dispatch] failed to allocate "
                  << kernarg_segment_size_ << " bytes from kernarg pool\n";
        return false;
    }

    // --- Step 2: Grant GPU access once (not per dispatch) ---
    // On MI300X with kernarg pool memory, this may be a no-op if the pool
    // is already GPU-accessible, but we call it defensively. The important
    // thing is we do NOT call it on every dispatch.
    rt.allow_gpu_access(kernarg_ptr_, kernarg_segment_size_, device_idx_);

    // --- Step 3: Create persistent completion signal ---
    // Try GPU-only signal first (cheaper: no interrupt doorbell, pure polling).
    // This is what CLR uses for direct-dispatch active-wait mode.
    // Fall back to standard signal if the AMD extension is not available.
    hsa_status_t sig_status = hsa_amd_signal_create(
        1, 0, nullptr, HSA_AMD_SIGNAL_AMD_GPU_ONLY, &signal_);
    if (sig_status == HSA_STATUS_SUCCESS) {
        signal_is_gpu_only_ = true;
    } else {
        // Fallback: standard IPC-capable signal
        sig_status = hsa_signal_create(1, 0, nullptr, &signal_);
        if (!check_pd(sig_status, "signal_create")) {
            rt.free_kernarg(kernarg_ptr_);
            kernarg_ptr_ = nullptr;
            return false;
        }
        signal_is_gpu_only_ = false;
    }

    // --- Step 4: Pre-compute AQL header ---
    // Use agent-scope fences instead of system-scope. The compiled megakernel
    // only accesses device memory that was set up before dispatch and read
    // back via explicit memcpy_d2h afterward. Agent scope avoids the cost
    // of system-scope cache flushes across the Infinity Fabric.
    //
    // CLR uses agent scope by default when fenceScopeAgent_ is set:
    //   header = KERNEL_DISPATCH | (AGENT << SCACQUIRE) | (AGENT << SCRELEASE)
    header_ = HSA_PACKET_TYPE_KERNEL_DISPATCH;
    header_ |= (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE);
    header_ |= (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);

    bound_ = true;

    std::cerr << "[clifft-persistent-dispatch] bound: kernarg="
              << kernarg_segment_size_ << "B"
              << " signal=" << (signal_is_gpu_only_ ? "gpu-only" : "standard")
              << " fence=agent-scope\n";
    return true;
}

// -----------------------------------------------------------------------
// release() — cleanup persistent resources
// -----------------------------------------------------------------------
void PersistentDispatcher::release() {
    if (!bound_) return;

    // Destroy signal
    if (signal_.handle != 0) {
        hsa_signal_destroy(signal_);
    }

    // Free kernarg buffer
    if (kernarg_ptr_) {
        auto& rt = hsa_runtime();
        rt.free_kernarg(kernarg_ptr_);
    }

    reset_members();
}

// -----------------------------------------------------------------------
// dispatch() — the fast path
// -----------------------------------------------------------------------
double PersistentDispatcher::dispatch(uint32_t grid_size,
                                      uint32_t block_size,
                                      const void* kernarg_data,
                                      size_t kernarg_size) {
    auto& rt = hsa_runtime();
    auto& dev = rt.device(device_idx_);

    // --- Fast-path step 1: memcpy kernargs into persistent buffer ---
    // This is a flat CPU memcpy of typically 32-96 bytes into fine-grained
    // (coherent) memory. No allocation, no GPU access grant needed.
    std::memcpy(kernarg_ptr_, kernarg_data, kernarg_size);

    // --- Fast-path step 2: Reset signal for this dispatch ---
    // The signal was created with initial value 1, and the GPU decrements
    // it to 0 on completion. We reset it to 1 for the next dispatch.
    // hsa_signal_store_relaxed is ~5ns (single atomic store).
    hsa_signal_store_relaxed(signal_, 1);

    // --- Fast-path step 3: Write AQL dispatch packet ---
    // Acquire a write slot in the hardware queue (lock-free atomic add).
    uint64_t write_index = hsa_queue_add_write_index_relaxed(dev.queue, 1);
    uint32_t queue_mask = dev.queue->size - 1;

    hsa_kernel_dispatch_packet_t* packet =
        reinterpret_cast<hsa_kernel_dispatch_packet_t*>(dev.queue->base_address)
        + (write_index & queue_mask);

    // Write packet fields with INVALID header first to prevent premature
    // execution by the command processor (standard AQL protocol).
    packet->header = HSA_PACKET_TYPE_INVALID;

    packet->setup = 1;  // 1D dispatch
    packet->workgroup_size_x = static_cast<uint16_t>(block_size);
    packet->workgroup_size_y = 1;
    packet->workgroup_size_z = 1;
    packet->grid_size_x = grid_size;
    packet->grid_size_y = 1;
    packet->grid_size_z = 1;
    packet->kernel_object = kernel_object_;
    packet->kernarg_address = kernarg_ptr_;
    packet->private_segment_size = private_segment_size_;
    packet->group_segment_size = group_segment_size_;
    packet->completion_signal = signal_;

    // Record start time
    uint64_t t0 = rt.timestamp_now();

    // Atomic release fence + header store makes the packet valid.
    // The command processor polls the header and begins execution.
    std::atomic_thread_fence(std::memory_order_release);
    packet->header = header_;

    // --- Fast-path step 4: Ring the doorbell ---
    // MMIO write to the queue doorbell signal notifies the command processor.
    hsa_signal_store_relaxed(dev.queue->doorbell_signal,
                              static_cast<hsa_signal_value_t>(write_index));

    // --- Fast-path step 5: Wait for completion ---
    // For GPU-only signals, this is a pure user-space polling loop (no
    // kernel/KFD involvement). For standard signals, this may use
    // interrupts depending on the wait state.
    //
    // HSA_WAIT_STATE_ACTIVE: pure spin-wait, lowest latency but burns CPU.
    // Use this for short-running compiled kernels in tight dispatch loops.
    hsa_signal_wait_scacquire(signal_, HSA_SIGNAL_CONDITION_LT, 1,
                               UINT64_MAX, HSA_WAIT_STATE_ACTIVE);

    uint64_t t1 = rt.timestamp_now();

    // No cleanup needed — kernarg buffer and signal are persistent.
    // Signal will be reset to 1 at the top of the next dispatch() call.

    return rt.elapsed_seconds(t0, t1);
}

}  // namespace gpu
}  // namespace clifft

#endif  // CLIFFT_ENABLE_GPU
