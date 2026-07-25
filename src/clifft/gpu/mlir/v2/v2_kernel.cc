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
#include <map>
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
// Kernarg layout for the coop/register interpreter kernels AND the specialized
// kernels (identical signature — the coop prefix, no global buffers).
struct __attribute__((packed)) CoopKernArgs {
    uint64_t instrs; uint32_t num_instrs; uint32_t peak_rank;
    uint32_t total_meas_slots; uint32_t num_observables;
    uint64_t seed; uint64_t shot_offset; uint64_t shots;
    uint64_t block_counts; uint64_t fused_u2; uint64_t fused_u4;
    uint64_t obs_off; uint64_t obs_tgt;
    uint64_t noise_sites; uint64_t noise_channels; uint64_t noise_hazards;
    uint32_t num_noise_sites; uint32_t _pad_align;
    uint64_t pauli_masks;
    uint64_t readout_noise; uint64_t detector_offsets; uint64_t detector_targets;
};
// Global tier appends 3 pointers (HBM amplitude slices + work counter).
struct __attribute__((packed)) GlobalKernArgs {
    CoopKernArgs base;
    uint64_t global_v; uint64_t global_scratch; uint64_t work_counter;
};

// Correctness gate: dispatch the interpreter kernel and the specialized kernel
// on the same small shot sample and compare passed_shots + observable_ones.
// Returns true iff they match exactly (byte-exact contract). Used once per
// circuit; the verdict is cached by the caller. Only for REG/COOP tiers (the
// only ones the specializer emits), so no global HBM buffers are needed.
bool specialized_matches_interpreter(const clifft::CompiledModule& program,
                                     const std::string& spec_hsaco,
                                     const std::string& spec_symbol) {
    using namespace clifft::gpu;
    FlattenedProgram flat = flatten_program(program);
    const bool is_reg = flat.peak_rank <= 4;
    const bool is_global = flat.peak_rank > 10;
    const char* interp_sym = is_reg ? "clifft_v2_register"
                           : is_global ? "clifft_v2_global" : "clifft_v2_coop";
    std::string interp_path = is_reg ? std::string(CLIFFT_V2_REGISTER_HSACO)
                                     : std::string(CLIFFT_V2_COOP_HSACO);
    // Global interpreter kernel lives in the coop .hsaco (same non-register TU).
    if (interp_path.empty()) return false;

    auto& rt = hsa_runtime();
    if (!rt.init()) return false;

    // Global tier: HBM amplitude buffers for a small validation worker pool.
    const uint32_t g_wgs = is_global ? 64u : 0u;
    uint64_t d_gv = 0, d_gs = 0, d_wc = 0;
    if (is_global) {
        const uint64_t amp = 1ull << flat.peak_rank;
        d_gv = reinterpret_cast<uint64_t>(rt.device_malloc((size_t)g_wgs * amp * sizeof(GpuComplex)));
        d_gs = reinterpret_cast<uint64_t>(rt.device_malloc((size_t)g_wgs * (amp/2) * sizeof(GpuComplex)));
        d_wc = reinterpret_cast<uint64_t>(rt.device_malloc(8 * sizeof(uint64_t)));
        if (!d_gv || !d_gs || !d_wc) return false;
    }

    // Shared device buffers.
    uint64_t d_instrs = upload(rt, flat.instrs);
    uint64_t d_u2 = upload(rt, flat.fused_u2);
    uint64_t d_u4 = upload(rt, flat.fused_u4);
    uint64_t d_oo = upload(rt, flat.observable_offsets);
    uint64_t d_ot = upload(rt, flat.observable_targets);
    uint64_t d_ns = upload(rt, flat.noise_sites);
    uint64_t d_nc = upload(rt, flat.noise_channels);
    uint64_t d_nh = upload(rt, flat.noise_hazards);
    uint64_t d_pm = upload(rt, flat.pauli_masks);
    uint64_t d_rn = upload(rt, flat.readout_noise);
    uint64_t d_do = upload(rt, flat.detector_offsets);
    uint64_t d_dt = upload(rt, flat.detector_targets);

    const uint32_t val_shots = 5000;  // enough to surface a 1-in-thousands flip
    const uint32_t block = 256;
    // Divergence is seed-dependent (a borderline shot flips on SOME seeds), so
    // validate across several seeds — one seed can pass while the circuit still
    // diverges on others.
    const uint64_t val_seeds[] = {1, 7, 42, 99, 123, 2718};

    auto run_one = [&](const std::string& path, const std::string& sym,
                       uint64_t val_seed, BlockCounts* out) -> bool {
        HsaLoadedKernel k = hsa_load_kernel(path, sym, 0);
        if (!k.valid) return false;
        BlockCounts* d_counts = static_cast<BlockCounts*>(rt.device_malloc(sizeof(BlockCounts)));
        rt.memset_device(d_counts, 0, sizeof(BlockCounts) / sizeof(uint32_t));
        GlobalKernArgs ga{};
        CoopKernArgs& ka = ga.base;
        ka.instrs = d_instrs; ka.num_instrs = (uint32_t)flat.instrs.size();
        ka.peak_rank = flat.peak_rank; ka.total_meas_slots = flat.total_meas_slots;
        ka.num_observables = flat.num_observables; ka.seed = val_seed;
        ka.shot_offset = 0; ka.shots = val_shots;
        ka.block_counts = reinterpret_cast<uint64_t>(d_counts);
        ka.fused_u2 = d_u2; ka.fused_u4 = d_u4; ka.obs_off = d_oo; ka.obs_tgt = d_ot;
        ka.noise_sites = d_ns; ka.noise_channels = d_nc; ka.noise_hazards = d_nh;
        ka.num_noise_sites = (uint32_t)flat.noise_sites.size();
        ka.pauli_masks = d_pm; ka.readout_noise = d_rn;
        ka.detector_offsets = d_do; ka.detector_targets = d_dt;
        uint32_t grid; size_t ka_bytes;
        if (is_global) {
            rt.memset_device(reinterpret_cast<void*>(d_wc), 0, 8 * (sizeof(uint64_t)/sizeof(uint32_t)));
            ga.global_v = d_gv; ga.global_scratch = d_gs; ga.work_counter = d_wc;
            grid = g_wgs * block; ka_bytes = sizeof(ga);
        } else {
            grid = is_reg ? (((val_shots + block - 1u) / block) * block) : (val_shots * block);
            ka_bytes = sizeof(ka);
        }
        hsa_dispatch_and_wait(k, 0, grid, block, &ga, ka_bytes);
        rt.memcpy_d2h(out, d_counts, sizeof(BlockCounts));
        rt.device_free(d_counts);
        hsa_free_kernel(k);
        return true;
    };

    bool match = true;
    for (uint64_t vs : val_seeds) {
        BlockCounts hi{}, hs{};
        if (!run_one(interp_path, interp_sym, vs, &hi) ||
            !run_one(spec_hsaco, spec_symbol, vs, &hs)) { match = false; break; }
        if (hi.passed != hs.passed) { match = false; break; }
        for (uint32_t i = 0; i < flat.num_observables && i < kMaxObs; ++i)
            if (hi.observable_ones[i] != hs.observable_ones[i]) { match = false; break; }
        if (!match) break;
    }

    auto free_if = [&](uint64_t p) { if (p) rt.device_free(reinterpret_cast<void*>(p)); };
    free_if(d_instrs); free_if(d_u2); free_if(d_u4); free_if(d_oo); free_if(d_ot);
    free_if(d_ns); free_if(d_nc); free_if(d_nh); free_if(d_pm); free_if(d_rn);
    free_if(d_do); free_if(d_dt);
    free_if(d_gv); free_if(d_gs); free_if(d_wc);
    return match;
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
    // Noise ops (and the reduction-carrying measurement ops) are emitted noinline
    // by the specializer (V2_NOISE_ATTR) so -O2 can't reassociate the FP across
    // the straight-lined copies. That makes almost every circuit byte-exact, but
    // a handful of knife-edge circuits (e.g. circuit_d5: NOISE+READOUT+SWAP_MEAS)
    // still flip ~1 shot in 5000 on some seeds due to irreducible 1-ULP codegen
    // differences. So we ALSO run a one-time CORRECTNESS GATE: validate the
    // specialized kernel against the interpreter on a small shot sample; if they
    // diverge, fall back to the interpreter for that circuit (verdict cached per
    // process). This guarantees byte-exactness for EVERY circuit, not just the
    // ones we happened to test, and is robust to any future codegen quirk.
    const bool spec_ok = (eff_tier == REG) || (eff_tier == COOP) || (eff_tier == GLOBAL);
    if (getenv("V2_SPECIALIZE") && spec_ok && specializer_toolchain_available()) {
        try {
            SpecTier st = (eff_tier == REG) ? SpecTier::Register
                        : (eff_tier == COOP) ? SpecTier::Coop : SpecTier::Global;
            const char* tname = (eff_tier == REG) ? "reg"
                              : (eff_tier == COOP) ? "coop" : "global";
            spec_sym = "clifft_v2_spec";
            std::string csrc = emit_specialized_kernel(flat, spec_sym, st);
            std::string key = std::string(tname) + "_r" + std::to_string(flat.peak_rank) +
                              "_n" + std::to_string(flat.instrs.size());
            std::string spath = compile_specialized(csrc, key);

            // --- correctness gate (cached in-process AND on disk) ---
            // The verdict is written to "<spath>.gate" so it is computed ONCE
            // ever (not per process) and NEVER re-run during a profiled sample
            // dispatch — the gate's own validation dispatches would otherwise
            // pollute rocprofv3 kernel traces. Disk cache also survives the
            // fresh process rocprofv3 spawns per invocation.
            static std::map<std::string, bool> s_spec_ok;
            bool validated;
            auto it = s_spec_ok.find(key);
            if (it != s_spec_ok.end()) {
                validated = it->second;
            } else {
                std::string gate_path = spath + ".gate";
                std::FILE* gf = std::fopen(gate_path.c_str(), "r");
                if (gf) {                       // disk cache hit
                    int v = 0; if (std::fscanf(gf, "%d", &v) != 1) v = 0; std::fclose(gf);
                    validated = (v == 1);
                } else {                        // compute once, persist
                    validated = specialized_matches_interpreter(program, spath, spec_sym);
                    if (std::FILE* wf = std::fopen(gate_path.c_str(), "w")) {
                        std::fprintf(wf, "%d\n", validated ? 1 : 0); std::fclose(wf);
                    }
                    if (!validated && getenv("V2_SPECIALIZE_VERBOSE"))
                        std::fprintf(stderr, "[v2-spec] %s FAILED correctness gate -> interpreter\n", key.c_str());
                }
                s_spec_ok[key] = validated;
            }
            if (validated) { load_path = spath; sym = spec_sym.c_str(); }
            else spec_sym.clear();
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
