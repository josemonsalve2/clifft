#pragma once

#ifdef CLIFFT_ENABLE_GPU

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace clifft {
namespace gpu {

struct GpuDevice {
    hsa_agent_t agent;
    hsa_queue_t* queue = nullptr;
    uint32_t cu_count = 0;
    uint32_t wavefront_size = 64;
    uint32_t queue_size = 0;
    char arch_name[64] = {};
    hsa_amd_memory_pool_t device_pool;      // device-local VRAM (coarse-grained)
    hsa_amd_memory_pool_t host_fine_pool;   // host fine-grained (CPU-visible, coherent)
    hsa_amd_memory_pool_t kernarg_pool;     // kernarg-capable pool
    bool valid = false;
};

struct HsaRuntime {
    std::vector<GpuDevice> devices;
    hsa_agent_t cpu_agent;
    hsa_amd_memory_pool_t cpu_fine_pool;
    hsa_amd_memory_pool_t cpu_coarse_pool;
    bool initialized = false;

    bool init();
    void shutdown();
    GpuDevice& device(int idx = 0);
    int device_count() const;
    std::string info() const;

    // Total size of the device-local (coarse-grained VRAM) pool, in bytes.
    // Queried from HSA rather than hardcoded so the global-tier resident pool
    // scales with the device instead of a constant that silently undersizes
    // itself on larger parts (see v2_kernel.cc's budget derivation).
    uint64_t device_pool_bytes(int device_idx = 0) const;

    // Memory management using HSA memory pools
    void* device_malloc(size_t bytes, int device_idx = 0);
    void  device_free(void* ptr);
    void* host_malloc(size_t bytes, bool fine_grained = true);
    void  host_free(void* ptr);

    // Transfers
    void memcpy_h2d(void* dst, const void* src, size_t bytes, int device_idx = 0);
    void memcpy_d2h(void* dst, const void* src, size_t bytes, int device_idx = 0);
    void memset_device(void* dst, uint32_t value, size_t count);

    // Allow GPU agent to access host memory (needed for kernarg)
    void allow_gpu_access(void* ptr, size_t bytes, int device_idx = 0);

    // Kernarg allocation (from the specific kernarg pool)
    void* alloc_kernarg(size_t bytes, int device_idx = 0);
    void  free_kernarg(void* ptr);

    // Timing via HSA system timestamp
    uint64_t timestamp_frequency() const;
    uint64_t timestamp_now() const;
    double elapsed_seconds(uint64_t start, uint64_t end) const;

private:
    hsa_signal_t copy_signal_ = {0};
    bool copy_signal_valid_ = false;
    void ensure_copy_signal();
};

// Global runtime instance (initialized once, used throughout process lifetime)
HsaRuntime& hsa_runtime();

}  // namespace gpu
}  // namespace clifft

#endif  // CLIFFT_ENABLE_GPU
