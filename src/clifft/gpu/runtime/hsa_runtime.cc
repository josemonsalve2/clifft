// hsa_runtime.cc — HSA direct runtime layer for clifft GPU backend
//
// Provides low-level GPU control through the HSA (Heterogeneous System
// Architecture) API, bypassing HIP runtime overhead for:
//   - Agent (GPU) discovery and queue creation
//   - Memory pool management (device VRAM, host fine-grained, kernarg)
//   - Memory allocation, transfer, and access control
//   - High-resolution timing via HSA system timestamps
//
// This replaces HIP runtime calls (hipMalloc, hipFree, hipMemcpy, etc.)
// for the compiled megakernel path, where dispatch overhead matters.
// The SVM interpreter path continues to use HIP <<<>>> syntax for AOT kernels.
//
// API patterns verified against:
//   /shared/jmonsalv/installers/rocm/repos/ROCR-Runtime/samples/common/hsa_test.cpp

#ifdef CLIFFT_ENABLE_GPU

#include "clifft/gpu/runtime/hsa_runtime.h"

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <atomic>
#include <cstring>
#include <iostream>
#include <sstream>

namespace clifft {
namespace gpu {

// -----------------------------------------------------------------------
// Error checking helper
// -----------------------------------------------------------------------
static bool check_hsa(hsa_status_t status, const char* msg) {
    if (status == HSA_STATUS_SUCCESS) return true;
    const char* err = nullptr;
    hsa_status_string(status, &err);
    std::cerr << "[clifft-hsa] " << msg << ": "
              << (err ? err : "unknown error") << " (status=" << status << ")\n";
    return false;
}

// -----------------------------------------------------------------------
// Agent iteration callback
// -----------------------------------------------------------------------
struct AgentIterData {
    std::vector<hsa_agent_t> cpus;
    std::vector<hsa_agent_t> gpus;
};

static hsa_status_t agent_callback(hsa_agent_t agent, void* data) {
    auto* d = static_cast<AgentIterData*>(data);
    hsa_device_type_t type;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    if (type == HSA_DEVICE_TYPE_CPU)
        d->cpus.push_back(agent);
    else if (type == HSA_DEVICE_TYPE_GPU)
        d->gpus.push_back(agent);
    return HSA_STATUS_SUCCESS;
}

// -----------------------------------------------------------------------
// Memory pool iteration callback
// -----------------------------------------------------------------------
struct PoolIterData {
    hsa_amd_memory_pool_t fine_grained = {0};
    hsa_amd_memory_pool_t coarse_grained = {0};
    hsa_amd_memory_pool_t kernarg = {0};
    hsa_amd_memory_pool_t group = {0};
};

static hsa_status_t pool_callback(hsa_amd_memory_pool_t pool, void* data) {
    auto* d = static_cast<PoolIterData*>(data);

    hsa_amd_segment_t segment;
    hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);

    if (segment == HSA_AMD_SEGMENT_GLOBAL) {
        uint32_t flags = 0;
        hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);

        bool alloc_allowed = false;
        hsa_amd_memory_pool_get_info(pool,
            HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc_allowed);
        if (!alloc_allowed) return HSA_STATUS_SUCCESS;

        if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) {
            d->kernarg = pool;
        }
        if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) {
            d->fine_grained = pool;
        }
        if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) {
            d->coarse_grained = pool;
        }
    } else if (segment == HSA_AMD_SEGMENT_GROUP) {
        d->group = pool;
    }
    return HSA_STATUS_SUCCESS;
}

// -----------------------------------------------------------------------
// HsaRuntime implementation
// -----------------------------------------------------------------------

bool HsaRuntime::init() {
    if (initialized) return true;

    if (!check_hsa(hsa_init(), "hsa_init")) return false;

    // Discover agents
    AgentIterData agents;
    if (!check_hsa(hsa_iterate_agents(agent_callback, &agents), "iterate_agents"))
        return false;

    if (agents.gpus.empty()) {
        std::cerr << "[clifft-hsa] no GPU agents found\n";
        return false;
    }
    if (agents.cpus.empty()) {
        std::cerr << "[clifft-hsa] no CPU agents found\n";
        return false;
    }

    cpu_agent = agents.cpus[0];

    // Discover CPU memory pools
    PoolIterData cpu_pools;
    hsa_amd_agent_iterate_memory_pools(cpu_agent, pool_callback, &cpu_pools);
    cpu_fine_pool = cpu_pools.fine_grained;
    cpu_coarse_pool = cpu_pools.coarse_grained;

    // Set up each GPU device
    for (auto& gpu_agent : agents.gpus) {
        GpuDevice dev;
        dev.agent = gpu_agent;

        // Query device properties
        hsa_agent_get_info(gpu_agent, HSA_AGENT_INFO_NAME, dev.arch_name);
        hsa_agent_get_info(gpu_agent, HSA_AGENT_INFO_WAVEFRONT_SIZE, &dev.wavefront_size);

        // CU count via AMD-specific info
        hsa_agent_get_info(gpu_agent,
            static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT),
            &dev.cu_count);

        // Discover GPU memory pools
        PoolIterData gpu_pools;
        hsa_amd_agent_iterate_memory_pools(gpu_agent, pool_callback, &gpu_pools);
        dev.device_pool = gpu_pools.coarse_grained;
        dev.host_fine_pool = cpu_pools.fine_grained;  // CPU fine-grained is host-visible
        dev.kernarg_pool = gpu_pools.kernarg.handle ? gpu_pools.kernarg : cpu_pools.fine_grained;

        // Create a dispatch queue
        uint32_t queue_size = 0;
        hsa_agent_get_info(gpu_agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size);
        if (queue_size > 4096) queue_size = 4096;
        dev.queue_size = queue_size;

        hsa_status_t qs = hsa_queue_create(gpu_agent, queue_size,
            HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr,
            UINT32_MAX, UINT32_MAX, &dev.queue);
        if (qs != HSA_STATUS_SUCCESS) {
            check_hsa(qs, "hsa_queue_create");
            continue;
        }

        dev.valid = true;
        devices.push_back(dev);
    }

    if (devices.empty()) {
        std::cerr << "[clifft-hsa] no valid GPU devices after queue creation\n";
        return false;
    }

    initialized = true;
    std::cerr << "[clifft-hsa] initialized: " << devices.size() << " GPU(s), "
              << "device[0]=" << devices[0].arch_name
              << " CUs=" << devices[0].cu_count
              << " wavefront=" << devices[0].wavefront_size << "\n";
    return true;
}

void HsaRuntime::ensure_copy_signal() {
    if (copy_signal_valid_) return;
    hsa_status_t s = hsa_amd_signal_create(1, 0, nullptr,
                                            HSA_AMD_SIGNAL_AMD_GPU_ONLY,
                                            &copy_signal_);
    if (s != HSA_STATUS_SUCCESS) {
        s = hsa_signal_create(1, 0, nullptr, &copy_signal_);
    }
    check_hsa(s, "create persistent copy signal");
    copy_signal_valid_ = true;
}

void HsaRuntime::shutdown() {
    if (!initialized) return;
    if (copy_signal_valid_) {
        hsa_signal_destroy(copy_signal_);
        copy_signal_valid_ = false;
    }
    for (auto& dev : devices) {
        if (dev.queue) {
            hsa_queue_destroy(dev.queue);
            dev.queue = nullptr;
        }
    }
    devices.clear();
    hsa_shut_down();
    initialized = false;
}

GpuDevice& HsaRuntime::device(int idx) {
    return devices.at(idx);
}

int HsaRuntime::device_count() const {
    return static_cast<int>(devices.size());
}

std::string HsaRuntime::info() const {
    std::ostringstream ss;
    ss << "hsa devices: " << devices.size();
    for (size_t i = 0; i < devices.size(); ++i) {
        ss << "\n[" << i << "] " << devices[i].arch_name
           << " CUs=" << devices[i].cu_count
           << " wavefront=" << devices[i].wavefront_size;
    }
    return ss.str();
}

// -----------------------------------------------------------------------
// Memory management
// -----------------------------------------------------------------------

uint64_t HsaRuntime::device_pool_bytes(int device_idx) const {
    if (device_idx < 0 || device_idx >= static_cast<int>(devices.size())) return 0;
    size_t sz = 0;
    if (hsa_amd_memory_pool_get_info(devices[device_idx].device_pool,
                                     HSA_AMD_MEMORY_POOL_INFO_SIZE, &sz) != HSA_STATUS_SUCCESS)
        return 0;
    return static_cast<uint64_t>(sz);
}

void* HsaRuntime::device_malloc(size_t bytes, int device_idx) {
    void* ptr = nullptr;
    auto& dev = devices.at(device_idx);
    if (!check_hsa(hsa_amd_memory_pool_allocate(dev.device_pool, bytes, 0, &ptr),
                   "device_malloc"))
        return nullptr;
    // Grant both GPU and CPU access at allocation time so subsequent
    // memcpy_h2d / memcpy_d2h can use direct std::memcpy without
    // per-call hsa_amd_agents_allow_access (which is a KFD ioctl, ~1ms).
    hsa_agent_t agents[] = {dev.agent, cpu_agent};
    hsa_amd_agents_allow_access(2, agents, nullptr, ptr);
    return ptr;
}

void HsaRuntime::device_free(void* ptr) {
    if (ptr) hsa_amd_memory_pool_free(ptr);
}

void* HsaRuntime::host_malloc(size_t bytes, bool fine_grained) {
    void* ptr = nullptr;
    auto pool = fine_grained ? cpu_fine_pool : cpu_coarse_pool;
    if (!check_hsa(hsa_amd_memory_pool_allocate(pool, bytes, 0, &ptr),
                   "host_malloc"))
        return nullptr;
    return ptr;
}

void HsaRuntime::host_free(void* ptr) {
    if (ptr) hsa_amd_memory_pool_free(ptr);
}

void HsaRuntime::memcpy_h2d(void* dst, const void* src, size_t bytes, int device_idx) {
    // CPU writes directly to device memory. allow_gpu_access must have
    // been called at allocation time (hsa_copy_to_device does this).
    std::memcpy(dst, src, bytes);
}

void HsaRuntime::memcpy_d2h(void* dst, const void* src, size_t bytes, int device_idx) {
    // CPU reads directly from device memory. allow_gpu_access must have
    // been called at allocation time (hsa_copy_to_device does this).
    std::memcpy(dst, src, bytes);
}

void HsaRuntime::memset_device(void* dst, uint32_t value, size_t count) {
    check_hsa(hsa_amd_memory_fill(dst, value, count), "memset_device");
}

void HsaRuntime::allow_gpu_access(void* ptr, size_t bytes, int device_idx) {
    auto& dev = devices.at(device_idx);
    hsa_agent_t agents[] = {dev.agent};
    hsa_status_t s = hsa_amd_agents_allow_access(1, agents, nullptr, ptr);
    if (s != HSA_STATUS_SUCCESS) {
        std::cerr << "[clifft-hsa] allow_gpu_access FAILED for " << ptr
                  << " (" << bytes << " bytes)\n";
    }
    check_hsa(s, "allow_gpu_access");
}

void* HsaRuntime::alloc_kernarg(size_t bytes, int device_idx) {
    auto& dev = devices.at(device_idx);
    void* ptr = nullptr;
    if (!check_hsa(hsa_amd_memory_pool_allocate(dev.kernarg_pool, bytes, 0, &ptr),
                   "alloc_kernarg"))
        return nullptr;
    return ptr;
}

void HsaRuntime::free_kernarg(void* ptr) {
    if (ptr) hsa_amd_memory_pool_free(ptr);
}

// -----------------------------------------------------------------------
// Timing
// -----------------------------------------------------------------------

uint64_t HsaRuntime::timestamp_frequency() const {
    uint64_t freq = 0;
    hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &freq);
    return freq;
}

uint64_t HsaRuntime::timestamp_now() const {
    uint64_t ts = 0;
    hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP, &ts);
    return ts;
}

double HsaRuntime::elapsed_seconds(uint64_t start, uint64_t end) const {
    uint64_t freq = timestamp_frequency();
    if (freq == 0) return 0.0;
    return static_cast<double>(end - start) / static_cast<double>(freq);
}

// -----------------------------------------------------------------------
// Global runtime singleton
// -----------------------------------------------------------------------
static HsaRuntime g_hsa_runtime;

HsaRuntime& hsa_runtime() {
    if (!g_hsa_runtime.initialized) {
        g_hsa_runtime.init();
    }
    return g_hsa_runtime;
}

}  // namespace gpu
}  // namespace clifft

#endif  // CLIFFT_ENABLE_GPU
