#pragma once

#ifdef CLIFFT_ENABLE_GPU

#include "clifft/gpu/runtime/hsa_runtime.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace clifft {
namespace gpu {

constexpr size_t kHsaDispatchKernargSlots = 64;

/// A loaded kernel ready for dispatch via HSA AQL queue.
/// Loaded once from .hsaco, cached for the process lifetime.
///
/// Persistent dispatch resources (kernarg buffer, completion signal) are
/// pre-allocated at load time so the hot dispatch path only needs:
///   memcpy args -> write AQL packet -> doorbell -> wait -> signal reset
///
/// Thread safety: one HsaLoadedKernel may have only one host producer because
/// its kernarg slots, packet template, and completion signal are reused.
struct HsaLoadedKernel {
    uint64_t kernel_object = 0;        // from HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT
    uint32_t private_segment_size = 0; // scratch memory per thread
    uint32_t group_segment_size = 0;   // shared memory (LDS) per workgroup
    uint32_t kernarg_segment_size = 0; // kernel argument buffer size
    void*    executable_handle = nullptr;  // opaque hsa_executable_t
    void*    reader_handle = nullptr;      // opaque hsa_code_object_reader_t

    // --- persistent dispatch resources (allocated once at load time) ---
    void*    persistent_kernarg = nullptr; // pre-allocated kernarg slot array
    size_t   persistent_kernarg_stride = 0;
    size_t   persistent_kernarg_bytes = 0;
    uint64_t persistent_signal = 0;       // opaque hsa_signal_t (created once, reset per dispatch)
    int      device_idx = 0;              // device this kernel was loaded for

    // Optimization 5: invariant fields are populated once. X dimensions are
    // cached on use because the public dispatch API permits them to vary.
    mutable hsa_kernel_dispatch_packet_t packet_template{};
    mutable uint32_t cached_grid_size = 0;
    mutable uint32_t cached_block_size = 0;

    bool     valid = false;
};

struct HsaDispatchRequest {
    uint32_t grid_size = 0;
    uint32_t block_size = 0;
    const void* kernarg_data = nullptr;
    size_t kernarg_size = 0;
};

/// Load a .hsaco code object and extract a kernel function by name.
/// The executable is frozen and ready for dispatch.
HsaLoadedKernel hsa_load_kernel(const std::string& hsaco_path,
                                 const std::string& kernel_name,
                                 int device_idx = 0);

/// Free resources associated with a loaded kernel.
void hsa_free_kernel(HsaLoadedKernel& kernel);

/// Dispatch a kernel via HSA AQL queue and wait for completion.
/// Returns the elapsed kernel time in seconds (measured via HSA timestamps).
///
/// kernarg_data: pointer to packed kernel arguments (will be copied to
///               a kernarg-capable buffer internally).
/// kernarg_size: size in bytes of the argument buffer.
/// grid_size:    total number of threads (= num_blocks * block_size).
/// block_size:   threads per workgroup.
double hsa_dispatch_and_wait(const HsaLoadedKernel& kernel,
                              int device_idx,
                              uint32_t grid_size,
                              uint32_t block_size,
                              const void* kernarg_data,
                              size_t kernarg_size);

/// Optimizations 2, 3, and 6: publish several dispatch packets, ring the
/// doorbell once, and wait once on a final barrier-and packet.
/// All requests use the same loaded kernel; use sequential=true when request
/// i consumes GPU memory written by request i-1.
double hsa_dispatch_batch_and_wait(const HsaLoadedKernel& kernel,
                                   int device_idx,
                                   const HsaDispatchRequest* requests,
                                   size_t request_count,
                                   bool sequential);

}  // namespace gpu
}  // namespace clifft

#endif  // CLIFFT_ENABLE_GPU
