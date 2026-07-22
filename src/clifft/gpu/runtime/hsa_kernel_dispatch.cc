// hsa_kernel_dispatch.cc — HSA code object loading and AQL kernel dispatch
//
// Replaces hipModuleLoad / hipModuleLaunchKernel with direct HSA equivalents
// for lower dispatch overhead.
//
// Dispatch flow:
//   1. Load .hsaco code object → hsa_executable → extract kernel_object
//   2. Pack kernargs into kernarg-pool buffer
//   3. Write AQL dispatch packet to queue
//   4. Ring doorbell, wait on completion signal
//   5. Measure elapsed time via HSA timestamps
//
// API patterns verified against ROCR-Runtime samples/common/hsa_test.cpp

#ifdef CLIFFT_ENABLE_GPU

#include "clifft/gpu/runtime/hsa_kernel_dispatch.h"

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <atomic>
#include <cstring>
#include <iostream>
#include <thread>

namespace clifft {
namespace gpu {

namespace {

constexpr size_t kKernargAlignment = 16;

size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

uint16_t dispatch_header(bool barrier) {
    uint16_t header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
    if (barrier) {
        header |= static_cast<uint16_t>(1u << HSA_PACKET_HEADER_BARRIER);
    }
    header |= static_cast<uint16_t>(
        HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE);
    header |= static_cast<uint16_t>(
        HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    return header;
}

uint16_t barrier_header() {
    uint16_t header = HSA_PACKET_TYPE_BARRIER_AND;
    header |= static_cast<uint16_t>(1u << HSA_PACKET_HEADER_BARRIER);
    header |= static_cast<uint16_t>(
        HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE);
    header |= static_cast<uint16_t>(
        HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    return header;
}

// Queue-full handling: reserve consecutive slots only when the producer will
// remain at most queue->size packets ahead of the consumer. A bare
// hsa_queue_add_write_index_relaxed can over-reserve and then block progress.
uint64_t reserve_queue_slots(hsa_queue_t* queue, uint64_t count) {
    for (;;) {
        uint64_t write_index = hsa_queue_load_write_index_relaxed(queue);
        uint64_t read_index = hsa_queue_load_read_index_scacquire(queue);
        if (write_index - read_index + count > queue->size) {
            std::this_thread::yield();
            continue;
        }
        uint64_t observed = hsa_queue_cas_write_index_relaxed(
            queue, write_index, write_index + count);
        if (observed == write_index) {
            return write_index;
        }
    }
}

void publish_header(uint16_t* packet_header, uint16_t header) {
    // Optimization 2: publish the header last with release ordering so every
    // packet body store is visible before the command processor sees its type.
    std::atomic_ref<uint16_t>(*packet_header).store(
        header, std::memory_order_release);
}

void update_packet_template(const HsaLoadedKernel& kernel,
                            uint32_t grid_size,
                            uint32_t block_size) {
    if (kernel.cached_grid_size == grid_size &&
        kernel.cached_block_size == block_size) {
        return;
    }
    kernel.packet_template.workgroup_size_x =
        static_cast<uint16_t>(block_size);
    kernel.packet_template.grid_size_x = grid_size;
    kernel.cached_grid_size = grid_size;
    kernel.cached_block_size = block_size;
}

}  // namespace

// -----------------------------------------------------------------------
// Error helper (same pattern as hsa_runtime.cc)
// -----------------------------------------------------------------------
static bool check_hsa_d(hsa_status_t status, const char* msg) {
    if (status == HSA_STATUS_SUCCESS) return true;
    const char* err = nullptr;
    hsa_status_string(status, &err);
    std::cerr << "[clifft-hsa-dispatch] " << msg << ": "
              << (err ? err : "unknown") << "\n";
    return false;
}

// -----------------------------------------------------------------------
// Load kernel from .hsaco
// -----------------------------------------------------------------------
HsaLoadedKernel hsa_load_kernel(const std::string& hsaco_path,
                                 const std::string& kernel_name,
                                 int device_idx) {
    HsaLoadedKernel lk;
    auto& rt = hsa_runtime();
    if (!rt.initialized) {
        std::cerr << "[clifft-hsa-dispatch] runtime not initialized\n";
        return lk;
    }
    auto& dev = rt.device(device_idx);

    // Create code object reader from file
    // hsa_code_object_reader_create_from_file takes a file descriptor
    // Use the path-based overload via hsa_code_object_reader_create_from_memory
    // after reading the file. But the simpler approach: use hsa_executable_load_code_object
    // with hsa_code_object_deserialize.
    //
    // Actually, the modern ROCm HSA API provides:
    //   hsa_code_object_reader_create_from_file() — needs a file descriptor
    // Let's use the file descriptor approach.

    FILE* f = fopen(hsaco_path.c_str(), "rb");
    if (!f) {
        std::cerr << "[clifft-hsa-dispatch] cannot open " << hsaco_path << "\n";
        return lk;
    }
    fseek(f, 0, SEEK_END);
    size_t file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<char> code_data(file_size);
    fread(code_data.data(), 1, file_size, f);
    fclose(f);

    // Create code object reader from memory
    hsa_code_object_reader_t reader;
    hsa_status_t status = hsa_code_object_reader_create_from_memory(
        code_data.data(), file_size, &reader);
    if (!check_hsa_d(status, "code_object_reader_create_from_memory")) return lk;

    // Create executable
    hsa_executable_t executable;
    status = hsa_executable_create_alt(
        HSA_PROFILE_FULL,
        HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
        nullptr, &executable);
    if (!check_hsa_d(status, "executable_create_alt")) {
        hsa_code_object_reader_destroy(reader);
        return lk;
    }

    // Load code object into executable for this agent
    status = hsa_executable_load_agent_code_object(
        executable, dev.agent, reader, nullptr, nullptr);
    if (!check_hsa_d(status, "executable_load_agent_code_object")) {
        hsa_executable_destroy(executable);
        hsa_code_object_reader_destroy(reader);
        return lk;
    }

    // Freeze the executable
    status = hsa_executable_freeze(executable, nullptr);
    if (!check_hsa_d(status, "executable_freeze")) {
        hsa_executable_destroy(executable);
        hsa_code_object_reader_destroy(reader);
        return lk;
    }

    // Look up the kernel symbol
    // For HIP-compiled .hsaco, the mangled name includes a suffix.
    // Try the exact name first, then with ".kd" suffix.
    hsa_executable_symbol_t symbol;
    status = hsa_executable_get_symbol_by_name(
        executable, kernel_name.c_str(), &dev.agent, &symbol);
    if (status != HSA_STATUS_SUCCESS) {
        // Try with ".kd" suffix (AMDGPU kernel descriptor convention)
        std::string kd_name = kernel_name + ".kd";
        status = hsa_executable_get_symbol_by_name(
            executable, kd_name.c_str(), &dev.agent, &symbol);
        if (!check_hsa_d(status, "get_symbol_by_name")) {
            hsa_executable_destroy(executable);
            hsa_code_object_reader_destroy(reader);
            return lk;
        }
    }

    // Extract kernel metadata
    status = hsa_executable_symbol_get_info(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &lk.kernel_object);
    if (!check_hsa_d(status, "get KERNEL_OBJECT")) {
        hsa_executable_destroy(executable);
        hsa_code_object_reader_destroy(reader);
        return lk;
    }

    hsa_executable_symbol_get_info(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
        &lk.private_segment_size);
    hsa_executable_symbol_get_info(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
        &lk.group_segment_size);
    hsa_executable_symbol_get_info(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
        &lk.kernarg_segment_size);

    lk.executable_handle = reinterpret_cast<void*>(executable.handle);
    lk.reader_handle = reinterpret_cast<void*>(reader.handle);
    lk.device_idx = device_idx;

    // --- Pre-allocate persistent dispatch resources ---
    // Kernarg buffer: allocated once, memcpy'd per dispatch
    size_t ka_size = lk.kernarg_segment_size > 0 ? lk.kernarg_segment_size : 256;
    lk.persistent_kernarg_stride = align_up(ka_size, kKernargAlignment);
    lk.persistent_kernarg_bytes =
        lk.persistent_kernarg_stride * kHsaDispatchKernargSlots;
    lk.persistent_kernarg =
        rt.alloc_kernarg(lk.persistent_kernarg_bytes, device_idx);
    if (!lk.persistent_kernarg) {
        std::cerr << "[clifft-hsa-dispatch] failed to pre-allocate kernarg\n";
        hsa_executable_destroy(executable);
        hsa_code_object_reader_destroy(reader);
        return lk;
    }
    // Grant GPU access once (persists for the buffer lifetime)
    rt.allow_gpu_access(
        lk.persistent_kernarg, lk.persistent_kernarg_bytes, device_idx);

    // Completion signal: created once, reset to 1 before each dispatch
    hsa_signal_t sig;
    status = hsa_signal_create(1, 0, nullptr, &sig);
    if (!check_hsa_d(status, "pre-allocate signal")) {
        rt.free_kernarg(lk.persistent_kernarg);
        lk.persistent_kernarg = nullptr;
        hsa_executable_destroy(executable);
        hsa_code_object_reader_destroy(reader);
        return lk;
    }
    lk.persistent_signal = sig.handle;

    // Optimization 5: pre-populate all fields constant for this kernel.
    // X dimensions are cached at dispatch time; kernarg_address and
    // completion_signal are always packet-specific.
    std::memset(&lk.packet_template, 0, sizeof(lk.packet_template));
    lk.packet_template.header = HSA_PACKET_TYPE_INVALID;
    lk.packet_template.setup = 1;
    lk.packet_template.workgroup_size_y = 1;
    lk.packet_template.workgroup_size_z = 1;
    lk.packet_template.grid_size_y = 1;
    lk.packet_template.grid_size_z = 1;
    lk.packet_template.kernel_object = lk.kernel_object;
    lk.packet_template.private_segment_size = lk.private_segment_size;
    lk.packet_template.group_segment_size = lk.group_segment_size;

    lk.valid = true;

    std::cerr << "[clifft-hsa-dispatch] loaded kernel '" << kernel_name
              << "' from " << hsaco_path
              << " (private=" << lk.private_segment_size
              << " group=" << lk.group_segment_size
              << " kernarg=" << lk.kernarg_segment_size
              << " persistent_kernarg=" << lk.persistent_kernarg
              << " slots=" << kHsaDispatchKernargSlots << ")\n";

    return lk;
}

void hsa_free_kernel(HsaLoadedKernel& lk) {
    // Free persistent dispatch resources
    if (lk.persistent_signal) {
        hsa_signal_t sig;
        sig.handle = lk.persistent_signal;
        hsa_signal_destroy(sig);
        lk.persistent_signal = 0;
    }
    if (lk.persistent_kernarg) {
        auto& rt = hsa_runtime();
        rt.free_kernarg(lk.persistent_kernarg);
        lk.persistent_kernarg = nullptr;
        lk.persistent_kernarg_stride = 0;
        lk.persistent_kernarg_bytes = 0;
    }
    if (lk.executable_handle) {
        hsa_executable_t exec;
        exec.handle = reinterpret_cast<uint64_t>(lk.executable_handle);
        hsa_executable_destroy(exec);
        lk.executable_handle = nullptr;
    }
    if (lk.reader_handle) {
        hsa_code_object_reader_t reader;
        reader.handle = reinterpret_cast<uint64_t>(lk.reader_handle);
        hsa_code_object_reader_destroy(reader);
        lk.reader_handle = nullptr;
    }
    lk.valid = false;
}

// -----------------------------------------------------------------------
// Dispatch kernel via AQL queue  (persistent dispatch — zero alloc/free)
// -----------------------------------------------------------------------
//
// Hot-path cost (per dispatch):
//   1. memcpy kernargs into pre-allocated buffer
//   2. write AQL packet (non-atomic fields + atomic header)
//   3. ring doorbell
//   4. wait on pre-allocated signal
//   5. reset signal for next dispatch
//
// Eliminated from hot path (now done once at load time):
//   - hsa_amd_memory_pool_allocate   (kernarg)
//   - hsa_amd_agents_allow_access    (GPU access grant)
//   - hsa_signal_create / hsa_signal_destroy
// -----------------------------------------------------------------------
double hsa_dispatch_and_wait(const HsaLoadedKernel& kernel,
                              int device_idx,
                              uint32_t grid_size,
                              uint32_t block_size,
                              const void* kernarg_data,
                              size_t kernarg_size) {
    auto& rt = hsa_runtime();
    auto& dev = rt.device(device_idx);

    const size_t kernarg_capacity = kernel.kernarg_segment_size > 0
        ? kernel.kernarg_segment_size
        : kernel.persistent_kernarg_stride;
    if (!kernel.valid || device_idx != kernel.device_idx ||
        !kernarg_data || kernarg_size > kernarg_capacity ||
        block_size == 0 || block_size > UINT16_MAX || grid_size == 0) {
        std::cerr << "[clifft-hsa-dispatch] invalid dispatch request\n";
        return 0.0;
    }

    std::memcpy(kernel.persistent_kernarg, kernarg_data, kernarg_size);

    // Signal reuse ordering: the previous hsa_signal_wait_scacquire must have
    // observed completion before this store. Resetting sooner races the GPU's
    // completion decrement and can lose or falsely observe a completion.
    hsa_signal_t signal;
    signal.handle = kernel.persistent_signal;
    hsa_signal_store_relaxed(signal, 1);

    uint64_t write_index = reserve_queue_slots(dev.queue, 1);
    hsa_kernel_dispatch_packet_t* packet =
        reinterpret_cast<hsa_kernel_dispatch_packet_t*>(dev.queue->base_address)
        + (write_index & (dev.queue->size - 1));

    // Optimization 5: copy a pre-populated packet and overwrite only the
    // launch-specific address and signal. Repeated dimensions stay cached.
    update_packet_template(kernel, grid_size, block_size);
    *packet = kernel.packet_template;
    packet->kernarg_address = kernel.persistent_kernarg;
    packet->completion_signal = signal;

    uint64_t t0 = rt.timestamp_now();
    publish_header(&packet->header, dispatch_header(false));
    hsa_signal_store_screlease(dev.queue->doorbell_signal,
                               static_cast<hsa_signal_value_t>(write_index));

    // Optimization 1: ACTIVE busy-polls instead of entering the blocked
    // interrupt path. It burns one CPU core while waiting, but avoids scheduler
    // and wake-up latency and is appropriate for these short kernels.
    hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1,
                               UINT64_MAX, HSA_WAIT_STATE_ACTIVE);

    uint64_t t1 = rt.timestamp_now();
    return rt.elapsed_seconds(t0, t1);
}

double hsa_dispatch_batch_and_wait(const HsaLoadedKernel& kernel,
                                   int device_idx,
                                   const HsaDispatchRequest* requests,
                                   size_t request_count,
                                   bool sequential) {
    auto& rt = hsa_runtime();
    auto& dev = rt.device(device_idx);

    if (!kernel.valid || device_idx != kernel.device_idx || !requests ||
        request_count == 0 ||
        request_count > kHsaDispatchKernargSlots ||
        request_count + 1 > dev.queue->size) {
        std::cerr << "[clifft-hsa-dispatch] invalid HSA dispatch batch\n";
        return 0.0;
    }
    for (size_t i = 0; i < request_count; ++i) {
        const auto& request = requests[i];
        const size_t kernarg_capacity = kernel.kernarg_segment_size > 0
            ? kernel.kernarg_segment_size
            : kernel.persistent_kernarg_stride;
        if (!request.kernarg_data ||
            request.kernarg_size > kernarg_capacity ||
            request.block_size == 0 || request.block_size > UINT16_MAX ||
            request.grid_size == 0) {
            std::cerr << "[clifft-hsa-dispatch] invalid batch entry\n";
            return 0.0;
        }
    }

    hsa_signal_t signal;
    signal.handle = kernel.persistent_signal;
    // Safe only after the prior acquire wait; this object is single-producer.
    hsa_signal_store_relaxed(signal, 1);

    // Optimization 2: reserve N dispatch packets plus one barrier in one CAS.
    // Advance packet i as first_index + i and ring exactly once with the final
    // barrier index after all packet headers have been release-published.
    const uint64_t packet_count = request_count + 1;
    const uint64_t first_index =
        reserve_queue_slots(dev.queue, packet_count);
    const uint32_t queue_mask = dev.queue->size - 1;
    auto* queue_packets =
        reinterpret_cast<hsa_kernel_dispatch_packet_t*>(
            dev.queue->base_address);

    for (size_t i = 0; i < request_count; ++i) {
        const auto& request = requests[i];
        auto* kernarg_ptr = static_cast<unsigned char*>(
            kernel.persistent_kernarg) +
            i * kernel.persistent_kernarg_stride;
        std::memcpy(kernarg_ptr, request.kernarg_data, request.kernarg_size);

        update_packet_template(
            kernel, request.grid_size, request.block_size);
        auto* packet = queue_packets + ((first_index + i) & queue_mask);
        *packet = kernel.packet_template;
        packet->kernarg_address = kernarg_ptr;
        packet->completion_signal.handle = 0;

        // Optimization 6: later dispatches carry the barrier bit when their
        // inputs are produced by earlier GPU kernels. Independent batches
        // leave it clear and may overlap.
        publish_header(
            &packet->header, dispatch_header(sequential && i != 0));
    }

    // Optimization 3: only this final BARRIER_AND packet owns a completion
    // signal. Its barrier bit waits for every older queue packet; all five
    // dep_signal entries remain null because queue age supplies the dependency.
    const uint64_t barrier_index = first_index + request_count;
    auto* barrier = reinterpret_cast<hsa_barrier_and_packet_t*>(
        queue_packets + (barrier_index & queue_mask));
    std::memset(barrier, 0, sizeof(*barrier));
    barrier->completion_signal = signal;
    publish_header(&barrier->header, barrier_header());

    uint64_t t0 = rt.timestamp_now();
    hsa_signal_store_screlease(
        dev.queue->doorbell_signal,
        static_cast<hsa_signal_value_t>(barrier_index));
    hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1,
                              UINT64_MAX, HSA_WAIT_STATE_ACTIVE);
    uint64_t t1 = rt.timestamp_now();
    return rt.elapsed_seconds(t0, t1);
}

// Optimization 4: HIP's <<<>>> launch is asynchronous. The runtime uses pooled
// command/kernarg/signal resources, enqueues stream work, and normally returns
// without a host signal wait. Completion is deferred to an event, stream sync,
// or blocking transfer. HIP does not promise to combine several launches into
// one AQL doorbell; its main latency advantage here is deferred host signaling
// plus resource pooling. This file must wait when its caller immediately reads
// block_counts. Use the batch API when intermediate results remain GPU-resident
// so only the final barrier signals the host.

}  // namespace gpu
}  // namespace clifft

#endif  // CLIFFT_ENABLE_GPU
