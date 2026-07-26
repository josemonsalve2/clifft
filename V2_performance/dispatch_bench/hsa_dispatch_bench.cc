// hsa_dispatch_bench.cc — measure raw HSA launch-to-completion latency.
//
// Standalone: links only -lhsa-runtime64, does NOT touch the clifft build.
// Loads bench_empty.hsaco (built by build.sh from empty_kernel.c with the
// V2 flags) and dispatches it in three modes:
//
//   naive       alloc kernarg + allow_access + signal_create per dispatch,
//               destroy + free after      -> what hsa_dispatch_and_wait did
//               before PersistentDispatcher
//   persistent  kernarg + signal allocated once, packet written per dispatch,
//               signal reset with store_relaxed -> the V2 hot path
//   batched     N packets published back-to-back, doorbell rung once, wait
//               once on the last -> amortized floor for the AQL path
//
// Reported number is launch-to-completion wall latency per dispatch, i.e.
// the quantity that a HIP <<<>>> + hipStreamSynchronize pair also measures,
// so the two benchmarks are directly comparable.

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(expr, what)                                                  \
    do {                                                                   \
        hsa_status_t s_ = (expr);                                          \
        if (s_ != HSA_STATUS_SUCCESS) {                                    \
            const char* e_ = nullptr;                                      \
            hsa_status_string(s_, &e_);                                    \
            std::fprintf(stderr, "FATAL %s: %s\n", what, e_ ? e_ : "?");   \
            return 1;                                                      \
        }                                                                  \
    } while (0)

namespace {

struct Pools {
    hsa_amd_memory_pool_t fine{0};
    hsa_amd_memory_pool_t coarse{0};
    hsa_amd_memory_pool_t kernarg{0};
};

hsa_status_t pool_cb(hsa_amd_memory_pool_t pool, void* data) {
    auto* p = static_cast<Pools*>(data);
    hsa_amd_segment_t seg;
    hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg);
    if (seg != HSA_AMD_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;
    uint32_t flags = 0;
    hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
    bool alloc = false;
    hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc);
    if (!alloc) return HSA_STATUS_SUCCESS;
    if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) p->kernarg = pool;
    if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) p->fine = pool;
    if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) p->coarse = pool;
    return HSA_STATUS_SUCCESS;
}

hsa_status_t agent_cb(hsa_agent_t agent, void* data) {
    hsa_device_type_t t;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &t);
    auto* out = static_cast<std::vector<hsa_agent_t>*>(data);
    if (t == HSA_DEVICE_TYPE_GPU) out->push_back(agent);
    return HSA_STATUS_SUCCESS;
}

hsa_status_t cpu_cb(hsa_agent_t agent, void* data) {
    hsa_device_type_t t;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &t);
    auto* out = static_cast<std::vector<hsa_agent_t>*>(data);
    if (t == HSA_DEVICE_TYPE_CPU) out->push_back(agent);
    return HSA_STATUS_SUCCESS;
}

// Packet header for a 1D dispatch with the given fence scopes.
uint16_t make_header(hsa_fence_scope_t acq, hsa_fence_scope_t rel) {
    return (uint16_t)((HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
                      (1u << HSA_PACKET_HEADER_BARRIER) |
                      ((uint16_t)acq << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
                      ((uint16_t)rel << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE));
}

struct Args {
    void* out;
    uint64_t a, b, c, d;
};

}  // namespace

int main(int argc, char** argv) {
    const std::string hsaco = argc > 1 ? argv[1] : "bench_empty.hsaco";
    const int iters = argc > 2 ? std::atoi(argv[2]) : 2000;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 5;

    CHECK(hsa_init(), "hsa_init");

    std::vector<hsa_agent_t> gpus, cpus;
    hsa_iterate_agents(agent_cb, &gpus);
    hsa_iterate_agents(cpu_cb, &cpus);
    if (gpus.empty()) { std::fprintf(stderr, "no GPU agent\n"); return 1; }
    hsa_agent_t gpu = gpus[0];

    char arch[64] = {0};
    hsa_agent_get_info(gpu, HSA_AGENT_INFO_NAME, arch);

    Pools gp, cp;
    hsa_amd_agent_iterate_memory_pools(gpu, pool_cb, &gp);
    if (!cpus.empty()) hsa_amd_agent_iterate_memory_pools(cpus[0], pool_cb, &cp);
    hsa_amd_memory_pool_t kernarg_pool = gp.kernarg.handle ? gp.kernarg : cp.fine;
    if (!kernarg_pool.handle) { std::fprintf(stderr, "no kernarg pool\n"); return 1; }

    uint32_t qsize = 0;
    hsa_agent_get_info(gpu, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &qsize);
    qsize = std::min<uint32_t>(qsize, 4096);
    hsa_queue_t* queue = nullptr;
    CHECK(hsa_queue_create(gpu, qsize, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr,
                           UINT32_MAX, UINT32_MAX, &queue), "hsa_queue_create");

    // --- load the code object ---
    // hsa_code_object_reader_create_from_file() takes an hsa_file_t (an int fd),
    // NOT a path. Read the file and use the _from_memory form, which is exactly
    // what the production loader does (hsa_kernel_dispatch.cc:138-154) -- keep
    // this identical so the benchmark measures the same load path we ship.
    hsa_code_object_reader_t reader;
    std::vector<char> code_data;
    {
        FILE* f = std::fopen(hsaco.c_str(), "rb");
        if (!f) { std::fprintf(stderr, "cannot open %s\n", hsaco.c_str()); return 1; }
        std::fseek(f, 0, SEEK_END);
        long sz = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        code_data.resize(static_cast<size_t>(sz));
        size_t got = std::fread(code_data.data(), 1, code_data.size(), f);
        std::fclose(f);
        if (got != code_data.size()) { std::fprintf(stderr, "short read %s\n", hsaco.c_str()); return 1; }
    }
    CHECK(hsa_code_object_reader_create_from_memory(code_data.data(), code_data.size(), &reader),
          "code_object_reader_create_from_memory");
    hsa_executable_t exec;
    CHECK(hsa_executable_create_alt(HSA_PROFILE_FULL,
                                    HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                    nullptr, &exec), "executable_create_alt");
    CHECK(hsa_executable_load_agent_code_object(exec, gpu, reader, nullptr, nullptr),
          "executable_load_agent_code_object");
    CHECK(hsa_executable_freeze(exec, nullptr), "executable_freeze");

    // Symbol lookup: try the bare name, then "<name>.kd". Which one the runtime
    // answers to depends on the code-object version -- the ELF carries BOTH a
    // FUNC "bench_empty" and an OBJECT "bench_empty.kd" (the kernel descriptor),
    // and ROCm 7.2.3 resolves the descriptor. hsa_kernel_dispatch.cc:186-195 in
    // the production loader already does exactly this two-step; job 50501 lost
    // the entire HSA arm to INVALID_SYMBOL_NAME because this copy did not.
    hsa_executable_symbol_t sym;
    hsa_status_t sym_st = hsa_executable_get_symbol_by_name(exec, "bench_empty", &gpu, &sym);
    if (sym_st != HSA_STATUS_SUCCESS)
        sym_st = hsa_executable_get_symbol_by_name(exec, "bench_empty.kd", &gpu, &sym);
    CHECK(sym_st, "get_symbol_by_name (tried \"bench_empty\" and \"bench_empty.kd\")");
    uint64_t kobj = 0;
    uint32_t priv = 0, group = 0, karg_size = 0;
    hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kobj);
    hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &priv);
    hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &group);
    hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &karg_size);

    // --- output buffer (device coarse-grained, host-visible) ---
    void* out = nullptr;
    CHECK(hsa_amd_memory_pool_allocate(gp.coarse.handle ? gp.coarse : cp.fine, 4096, 0, &out),
          "alloc out");
    hsa_agent_t access[2] = {gpu, cpus.empty() ? gpu : cpus[0]};
    hsa_amd_agents_allow_access(cpus.empty() ? 1 : 2, access, nullptr, out);

    Args args{out, 1, 2, 3, 4};

    std::printf("# arch=%s queue_size=%u kernarg_seg=%u priv=%u group=%u iters=%d reps=%d\n",
                arch, qsize, karg_size, priv, group, iters, reps);
    std::printf("mode,rep,ns_per_dispatch\n");

    const uint16_t hdr_sys = make_header(HSA_FENCE_SCOPE_SYSTEM, HSA_FENCE_SCOPE_SYSTEM);
    const uint16_t hdr_agent = make_header(HSA_FENCE_SCOPE_AGENT, HSA_FENCE_SCOPE_AGENT);

    auto write_packet = [&](uint64_t idx, void* karg, hsa_signal_t sig, uint16_t hdr) {
        auto* base = static_cast<hsa_kernel_dispatch_packet_t*>(queue->base_address);
        hsa_kernel_dispatch_packet_t* p = &base[idx % queue->size];
        std::memset(p, 0, sizeof(*p));
        p->setup = 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
        p->workgroup_size_x = 64;
        p->workgroup_size_y = 1;
        p->workgroup_size_z = 1;
        p->grid_size_x = 64;
        p->grid_size_y = 1;
        p->grid_size_z = 1;
        p->private_segment_size = priv;
        p->group_segment_size = group;
        p->kernel_object = kobj;
        p->kernarg_address = karg;
        p->completion_signal = sig;
        __atomic_store_n((uint16_t*)p, hdr, __ATOMIC_RELEASE);
    };

    // ================= mode: naive =================
    // Per dispatch: alloc kernarg, allow_access, signal_create, dispatch,
    // wait, signal_destroy, free kernarg. This is the pre-PersistentDispatcher
    // path and the closest structural analogue to what a generic runtime does.
    for (int r = 0; r < reps; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            void* karg = nullptr;
            hsa_amd_memory_pool_allocate(kernarg_pool, karg_size ? karg_size : 64, 0, &karg);
            std::memcpy(karg, &args, sizeof(args));
            hsa_amd_agents_allow_access(1, &gpu, nullptr, karg);
            hsa_signal_t sig;
            hsa_signal_create(1, 0, nullptr, &sig);
            uint64_t idx = hsa_queue_add_write_index_relaxed(queue, 1);
            write_packet(idx, karg, sig, hdr_sys);
            hsa_signal_store_relaxed(queue->doorbell_signal, (hsa_signal_value_t)idx);
            hsa_signal_wait_scacquire(sig, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
                                      HSA_WAIT_STATE_BLOCKED);
            hsa_signal_destroy(sig);
            hsa_amd_memory_pool_free(karg);
        }
        auto t1 = std::chrono::steady_clock::now();
        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;
        std::printf("naive,%d,%.1f\n", r, ns);
    }

    // ================= mode: persistent =================
    // Kernarg + signal allocated once. Per dispatch: memcpy args, write packet,
    // doorbell, wait, reset signal. This is what V2 actually runs.
    {
        void* karg = nullptr;
        hsa_amd_memory_pool_allocate(kernarg_pool, karg_size ? karg_size : 64, 0, &karg);
        hsa_amd_agents_allow_access(1, &gpu, nullptr, karg);
        hsa_signal_t sig;
        // GPU-only signal: no interrupt doorbell, host polls. Falls back to a
        // regular signal if the flag is unsupported.
        if (hsa_amd_signal_create(1, 0, nullptr, HSA_AMD_SIGNAL_AMD_GPU_ONLY, &sig) !=
            HSA_STATUS_SUCCESS) {
            hsa_signal_create(1, 0, nullptr, &sig);
        }
        for (int r = 0; r < reps; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) {
                std::memcpy(karg, &args, sizeof(args));
                hsa_signal_store_relaxed(sig, 1);
                uint64_t idx = hsa_queue_add_write_index_relaxed(queue, 1);
                write_packet(idx, karg, sig, hdr_agent);
                hsa_signal_store_relaxed(queue->doorbell_signal, (hsa_signal_value_t)idx);
                while (hsa_signal_load_scacquire(sig) != 0) { /* spin */ }
            }
            auto t1 = std::chrono::steady_clock::now();
            double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;
            std::printf("persistent,%d,%.1f\n", r, ns);
        }
        hsa_signal_destroy(sig);
        hsa_amd_memory_pool_free(karg);
    }

    // ================= mode: batched =================
    // Publish BATCH packets, ring the doorbell once, wait once. Isolates the
    // per-packet cost from the launch-to-completion round trip.
    {
        const int BATCH = 16;
        void* karg = nullptr;
        hsa_amd_memory_pool_allocate(kernarg_pool, karg_size ? karg_size : 64, 0, &karg);
        hsa_amd_agents_allow_access(1, &gpu, nullptr, karg);
        std::memcpy(karg, &args, sizeof(args));
        hsa_signal_t sig;
        if (hsa_amd_signal_create(1, 0, nullptr, HSA_AMD_SIGNAL_AMD_GPU_ONLY, &sig) !=
            HSA_STATUS_SUCCESS) {
            hsa_signal_create(1, 0, nullptr, &sig);
        }
        for (int r = 0; r < reps; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters / BATCH; ++i) {
                hsa_signal_store_relaxed(sig, 1);
                uint64_t base_idx = hsa_queue_add_write_index_relaxed(queue, BATCH);
                for (int j = 0; j < BATCH; ++j) {
                    // only the last packet carries the completion signal
                    hsa_signal_t s = (j == BATCH - 1) ? sig : hsa_signal_t{0};
                    write_packet(base_idx + j, karg, s, hdr_agent);
                }
                hsa_signal_store_relaxed(queue->doorbell_signal,
                                         (hsa_signal_value_t)(base_idx + BATCH - 1));
                while (hsa_signal_load_scacquire(sig) != 0) { /* spin */ }
            }
            auto t1 = std::chrono::steady_clock::now();
            double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() /
                        ((iters / BATCH) * BATCH);
            std::printf("batched%d,%d,%.1f\n", BATCH, r, ns);
        }
        hsa_signal_destroy(sig);
        hsa_amd_memory_pool_free(karg);
    }

    hsa_amd_memory_pool_free(out);
    hsa_queue_destroy(queue);
    hsa_executable_destroy(exec);
    hsa_code_object_reader_destroy(reader);
    hsa_shut_down();
    return 0;
}
