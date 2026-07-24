// v2_kernel.cc — MLIR-V2 host driver (HIP-free). Uses ONLY the HSA runtime.
#include "clifft/gpu/mlir/v2/v2_kernel.h"

#include "clifft/gpu/runtime/hsa_runtime.h"
#include "clifft/gpu/runtime/hsa_kernel_dispatch.h"
#include "clifft/gpu/device_program.h"   // flatten_program, FlattenedProgram
#include "clifft/gpu/gpu_types.h"        // GpuInstr, BlockCounts, kMaxObs

#include <cstring>
#include <stdexcept>

#ifndef CLIFFT_V2_PROBE_HSACO
#define CLIFFT_V2_PROBE_HSACO ""
#endif
#ifndef CLIFFT_V2_COOP_HSACO
#define CLIFFT_V2_COOP_HSACO ""
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

    HsaLoadedKernel kernel = hsa_load_kernel(hsaco_path, "clifft_v2_coop", 0);
    if (!kernel.valid) throw std::runtime_error("v2_sample: coop kernel load failed");

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

    const uint32_t block = 256;
    const uint32_t grid = shots * block;  // one workgroup per shot
    double ks = hsa_dispatch_and_wait(kernel, 0, grid, block, &kargs, sizeof(kargs));
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
