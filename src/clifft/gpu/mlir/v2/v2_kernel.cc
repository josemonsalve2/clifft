// v2_kernel.cc — MLIR-V2 host driver (HIP-free). Uses ONLY the HSA runtime.
#include "clifft/gpu/mlir/v2/v2_kernel.h"

#include "clifft/gpu/runtime/hsa_runtime.h"
#include "clifft/gpu/runtime/hsa_kernel_dispatch.h"
#include "clifft/gpu/device_program.h"   // flatten_program, FlattenedProgram
#include "clifft/gpu/gpu_types.h"        // GpuInstr, BlockCounts, kMaxObs
#include "clifft/gpu/mlir/v2/v2_specializer.h"
#include "clifft/gpu/mlir/v2/v2_compile_cache.h"

#include <cstring>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#ifndef CLIFFT_V2_PROBE_HSACO
#define CLIFFT_V2_PROBE_HSACO ""
#endif
#ifndef CLIFFT_V2_COOP_HSACO
#define CLIFFT_V2_COOP_HSACO ""
#endif
#ifndef CLIFFT_V2_REGISTER_HSACO
#define CLIFFT_V2_REGISTER_HSACO ""
#endif

namespace clifft::gpu::v2 {

namespace {
// Upload a host vector to device; returns device ptr (or nullptr if empty).
template <typename T>
uint64_t upload(HsaRuntime& rt, const std::vector<T>& v) {
    if (v.empty()) return 0;
    void* d = rt.device_malloc(v.size() * sizeof(T));
    rt.memcpy_h2d(d, v.data(), v.size() * sizeof(T));
    return reinterpret_cast<uint64_t>(d);
}
}  // namespace

clifft::SurvivorResult v2_sample(const clifft::CompiledModule& program,
                                 uint32_t shots, uint64_t seed,
                                 double* kernel_seconds, std::string hsaco_path) {
    using namespace clifft::gpu;
    clifft::SurvivorResult result;
    result.total_shots = shots;
    result.observable_ones.assign(program.num_observables, 0);

    if (hsaco_path.empty()) hsaco_path = CLIFFT_V2_COOP_HSACO;
    if (hsaco_path.empty()) throw std::runtime_error("v2_sample: no coop .hsaco path");

    FlattenedProgram flat = flatten_program(program);

    auto& rt = hsa_runtime();
    if (!rt.init()) throw std::runtime_error("v2_sample: HSA init failed");

    // Tier selection by peak_rank (all three share the SAME execute_shot body):
    //   register (<=4): 1 shot/thread, statevector in registers, no LDS/barriers.
    //   coop (5-10): 256 threads/shot, amplitudes in LDS.
    //   global (11-19): amplitudes in HBM + work-stealing.
    // P0: the register tier fixes the 15-28x low-rank catastrophe (256 threads
    // were cooperating on a 1-16 amplitude statevector).
    constexpr uint32_t kRegMaxRank = 4;
    constexpr uint32_t kCoopMaxRank = 10;
    constexpr uint32_t kGlobalMaxRank = 19;
    constexpr uint32_t kNumXCDs = 8;
    enum Tier { REG, COOP, GLOBAL };
    const Tier tier = flat.peak_rank <= kRegMaxRank ? REG
                    : flat.peak_rank <= kCoopMaxRank ? COOP : GLOBAL;
    const bool use_global = (tier == GLOBAL);
    if (flat.peak_rank > kGlobalMaxRank)
        throw std::runtime_error("v2_sample: peak_rank " + std::to_string(flat.peak_rank) +
                                 " exceeds global tier max (" + std::to_string(kGlobalMaxRank) + ")");
    // Allow disabling the register tier (fall back to coop) for A/B measurement.
    const bool reg_disabled = getenv("V2_NO_REGISTER") != nullptr;
    const Tier eff_tier = (tier == REG && !reg_disabled) ? REG
                        : (tier == GLOBAL ? GLOBAL : COOP);

    const char* sym = eff_tier == GLOBAL ? "clifft_v2_global"
                    : eff_tier == REG ? "clifft_v2_register" : "clifft_v2_coop";
    // The register kernel lives in its own .hsaco (separate -DV2_REGISTER compile).
    std::string load_path = hsaco_path;
    if (eff_tier == REG) {
        load_path = CLIFFT_V2_REGISTER_HSACO;
        if (load_path.empty()) throw std::runtime_error("v2_sample: no register .hsaco path");
    }

    // SPECIALIZER (opt-in via V2_SPECIALIZE): emit a per-circuit kernel that
    // straight-lines the bytecode with constant operands, compile+cache it at
    // runtime, and dispatch that instead of the runtime interpreter. Byte-exact
    // by construction (calls the same v2_op_*). Register tier is the first slice;
    // other tiers/opcodes fall back to the interpreter. Compile time is never on
    // the sampling path (cached; warm before benchmarking).
    std::string spec_sym;
    if (getenv("V2_SPECIALIZE") && eff_tier == REG && specializer_toolchain_available()) {
        try {
            SpecTier st = SpecTier::Register;
            spec_sym = "clifft_v2_spec";
            std::string csrc = emit_specialized_kernel(flat, spec_sym, st);
            // Cache key: peak_rank + instr count + a cheap content hash of the ops.
            std::string key = "reg_r" + std::to_string(flat.peak_rank) + "_n" +
                              std::to_string(flat.instrs.size());
            std::string spath = compile_specialized(csrc, key);
            load_path = spath;
            sym = spec_sym.c_str();
        } catch (const std::exception& e) {
            spec_sym.clear();  // fall back to the interpreter kernel
            if (getenv("V2_SPECIALIZE_VERBOSE"))
                std::fprintf(stderr, "[v2-spec] fallback: %s\n", e.what());
        }
    }

    HsaLoadedKernel kernel = hsa_load_kernel(load_path, sym, 0);
    if (!kernel.valid) throw std::runtime_error(std::string("v2_sample: kernel load failed: ") + sym);

    // Upload device buffers (reuse flatten_program output — no second lowering).
    uint64_t d_instrs = upload(rt, flat.instrs);
    uint64_t d_fused_u2 = upload(rt, flat.fused_u2);
    uint64_t d_fused_u4 = upload(rt, flat.fused_u4);
    uint64_t d_obs_off = upload(rt, flat.observable_offsets);
    uint64_t d_obs_tgt = upload(rt, flat.observable_targets);
    uint64_t d_noise_sites = upload(rt, flat.noise_sites);
    uint64_t d_noise_channels = upload(rt, flat.noise_channels);
    uint64_t d_noise_hazards = upload(rt, flat.noise_hazards);
    uint64_t d_pauli_masks = upload(rt, flat.pauli_masks);
    uint64_t d_readout_noise = upload(rt, flat.readout_noise);
    uint64_t d_det_off = upload(rt, flat.detector_offsets);
    uint64_t d_det_tgt = upload(rt, flat.detector_targets);

    // One BlockCounts, zeroed; kernel atomic-adds into it.
    BlockCounts* d_counts = static_cast<BlockCounts*>(rt.device_malloc(sizeof(BlockCounts)));
    rt.memset_device(d_counts, 0, sizeof(BlockCounts) / sizeof(uint32_t));

    // Global tier: one HBM amplitude slice per resident workgroup + per-XCD work
    // counters. Grid = fixed pool of persistent workgroups (not one-per-shot).
    uint64_t d_global_v = 0, d_global_scratch = 0, d_work_counter = 0;
    uint32_t global_grid_wgs = 0;
    if (use_global) {
        // Resident pool sized to a fixed HBM budget: each workgroup owns one
        // amplitude slice (1<<peak_rank) + a half-size scratch. At rank 19 a
        // slice is 2^19*8 = 4MB, so cap total amplitude memory ~8GB.
        const uint64_t amp = 1ull << flat.peak_rank;
        const uint64_t bytes_per_wg = amp * sizeof(GpuComplex) + (amp / 2) * sizeof(GpuComplex);
        const uint64_t budget = 8ull << 30;  // 8 GB
        global_grid_wgs = static_cast<uint32_t>(budget / bytes_per_wg);
        if (global_grid_wgs < kNumXCDs) global_grid_wgs = kNumXCDs;
        if (global_grid_wgs > 2048) global_grid_wgs = 2048;
        if (const char* e = getenv("V2_GLOBAL_WGS")) global_grid_wgs = std::atoi(e);
        d_global_v = reinterpret_cast<uint64_t>(
            rt.device_malloc((size_t)global_grid_wgs * amp * sizeof(GpuComplex)));
        d_global_scratch = reinterpret_cast<uint64_t>(
            rt.device_malloc((size_t)global_grid_wgs * (amp / 2) * sizeof(GpuComplex)));
        d_work_counter = reinterpret_cast<uint64_t>(rt.device_malloc(kNumXCDs * sizeof(uint64_t)));
        if (!d_global_v || !d_global_scratch || !d_work_counter)
            throw std::runtime_error("v2_sample: global-tier HBM alloc failed");
        rt.memset_device(reinterpret_cast<void*>(d_work_counter), 0, kNumXCDs * (sizeof(uint64_t)/sizeof(uint32_t)));
    }

    // Kernarg layout MUST match clifft_v2_coop's signature (device_abi.h order).
    struct __attribute__((packed)) {
        uint64_t instrs; uint32_t num_instrs; uint32_t peak_rank;
        uint32_t total_meas_slots; uint32_t num_observables;
        uint64_t seed; uint64_t shot_offset; uint64_t shots;
        uint64_t block_counts; uint64_t fused_u2; uint64_t fused_u4;
        uint64_t obs_off; uint64_t obs_tgt;
        uint64_t noise_sites; uint64_t noise_channels; uint64_t noise_hazards;
        uint32_t num_noise_sites; uint32_t _pad_align;
        uint64_t pauli_masks;
        uint64_t readout_noise; uint64_t detector_offsets; uint64_t detector_targets;
        uint64_t global_v; uint64_t global_scratch; uint64_t work_counter;
    } kargs;
    kargs.instrs = d_instrs;
    kargs.num_instrs = flat.instrs.size();
    kargs.peak_rank = flat.peak_rank;
    kargs.total_meas_slots = flat.total_meas_slots;
    kargs.num_observables = flat.num_observables;
    kargs.seed = seed;
    kargs.shot_offset = 0;
    kargs.shots = shots;
    kargs.block_counts = reinterpret_cast<uint64_t>(d_counts);
    kargs.fused_u2 = d_fused_u2;
    kargs.fused_u4 = d_fused_u4;
    kargs.obs_off = d_obs_off;
    kargs.obs_tgt = d_obs_tgt;
    kargs.noise_sites = d_noise_sites;
    kargs.noise_channels = d_noise_channels;
    kargs.noise_hazards = d_noise_hazards;
    kargs.num_noise_sites = static_cast<uint32_t>(flat.noise_sites.size());
    kargs._pad_align = 0;
    kargs.pauli_masks = d_pauli_masks;
    kargs.readout_noise = d_readout_noise;
    kargs.detector_offsets = d_det_off;
    kargs.detector_targets = d_det_tgt;
    kargs.global_v = d_global_v;
    kargs.global_scratch = d_global_scratch;
    kargs.work_counter = d_work_counter;

    const uint32_t block = 256;
    // Register: 1 shot/thread -> grid = ceil(shots/256)*256 threads.
    // Coop: 1 shot/workgroup -> grid = shots*256 threads.
    // Global: fixed resident pool, each drains many shots via work-stealing.
    // kargs size differs by tier (global adds 3 pointers); the register/coop
    // kernels read only the prefix (offsetof global_v).
    uint32_t grid;
    if (eff_tier == GLOBAL)      grid = global_grid_wgs * block;
    else if (eff_tier == REG)    grid = ((shots + block - 1u) / block) * block;
    else                         grid = shots * block;
    const size_t karg_bytes = use_global ? sizeof(kargs) : offsetof(decltype(kargs), global_v);
    double ks = hsa_dispatch_and_wait(kernel, 0, grid, block, &kargs, karg_bytes);
    if (kernel_seconds) *kernel_seconds = ks;

    BlockCounts h_counts;
    rt.memcpy_d2h(&h_counts, d_counts, sizeof(BlockCounts));
    result.passed_shots = static_cast<uint32_t>(h_counts.passed);
    for (uint32_t i = 0; i < program.num_observables && i < kMaxObs; ++i) {
        if (i < result.observable_ones.size())
            result.observable_ones[i] = h_counts.observable_ones[i];
    }
    result.logical_errors = static_cast<uint32_t>(h_counts.logical_errors);

    rt.device_free(d_counts);
    if (d_instrs) rt.device_free(reinterpret_cast<void*>(d_instrs));
    if (d_fused_u2) rt.device_free(reinterpret_cast<void*>(d_fused_u2));
    if (d_fused_u4) rt.device_free(reinterpret_cast<void*>(d_fused_u4));
    if (d_obs_off) rt.device_free(reinterpret_cast<void*>(d_obs_off));
    if (d_obs_tgt) rt.device_free(reinterpret_cast<void*>(d_obs_tgt));
    if (d_noise_sites) rt.device_free(reinterpret_cast<void*>(d_noise_sites));
    if (d_noise_channels) rt.device_free(reinterpret_cast<void*>(d_noise_channels));
    if (d_noise_hazards) rt.device_free(reinterpret_cast<void*>(d_noise_hazards));
    if (d_pauli_masks) rt.device_free(reinterpret_cast<void*>(d_pauli_masks));
    if (d_readout_noise) rt.device_free(reinterpret_cast<void*>(d_readout_noise));
    if (d_global_v) rt.device_free(reinterpret_cast<void*>(d_global_v));
    if (d_global_scratch) rt.device_free(reinterpret_cast<void*>(d_global_scratch));
    if (d_work_counter) rt.device_free(reinterpret_cast<void*>(d_work_counter));
    if (d_det_off) rt.device_free(reinterpret_cast<void*>(d_det_off));
    if (d_det_tgt) rt.device_free(reinterpret_cast<void*>(d_det_tgt));
    hsa_free_kernel(kernel);
    return result;
}

ProbeResult run_probe_detailed(uint32_t n, std::string hsaco_path) {
    ProbeResult r;
    if (hsaco_path.empty()) hsaco_path = CLIFFT_V2_PROBE_HSACO;
    if (hsaco_path.empty()) {
        r.error = "no probe .hsaco path (build with CLIFFT_ENABLE_MLIR_V2)";
        return r;
    }

    auto& rt = hsa_runtime();
    if (!rt.init()) { r.error = "HSA init failed"; return r; }

    HsaLoadedKernel kernel;
    try {
        kernel = hsa_load_kernel(hsaco_path, "clifft_v2_probe", 0);
    } catch (const std::exception& e) {
        r.error = std::string("load failed: ") + e.what();
        return r;
    }
    if (!kernel.valid) { r.error = "kernel not valid after load"; return r; }

    // Device output buffer.
    uint32_t* d_out = static_cast<uint32_t*>(rt.device_malloc(sizeof(uint32_t) * n));
    if (!d_out) { r.error = "device_malloc failed"; hsa_free_kernel(kernel); return r; }

    // Pack kernel args: {u32* out; u32 n;} — natural layout for amdgpu_kernel.
    struct __attribute__((packed)) { uint64_t out; uint32_t n; } kargs;
    kargs.out = reinterpret_cast<uint64_t>(d_out);
    kargs.n = n;

    const uint32_t block = 256;
    const uint32_t grid = ((n + block - 1) / block) * block;
    try {
        r.kernel_seconds = hsa_dispatch_and_wait(kernel, 0, grid, block, &kargs, sizeof(kargs));
    } catch (const std::exception& e) {
        r.error = std::string("dispatch failed: ") + e.what();
        rt.device_free(d_out);
        hsa_free_kernel(kernel);
        return r;
    }

    r.out.resize(n);
    rt.memcpy_d2h(r.out.data(), d_out, sizeof(uint32_t) * n);
    rt.device_free(d_out);
    hsa_free_kernel(kernel);

    r.ok = true;
    for (uint32_t i = 0; i < n; ++i) {
        if (r.out[i] != i * 2u) { r.ok = false; r.error = "value mismatch at " + std::to_string(i); break; }
    }
    return r;
}

bool run_probe(uint32_t n, std::string hsaco_path) {
    return run_probe_detailed(n, std::move(hsaco_path)).ok;
}

}  // namespace clifft::gpu::v2
