// v2_kernel.cc — MLIR-V2 host driver (HIP-free). Uses ONLY the HSA runtime.
#include "clifft/gpu/mlir/v2/v2_kernel.h"

#include "clifft/gpu/runtime/hsa_runtime.h"
#include "clifft/gpu/runtime/hsa_kernel_dispatch.h"

#include <cstring>

#ifndef CLIFFT_V2_PROBE_HSACO
#define CLIFFT_V2_PROBE_HSACO ""
#endif

namespace clifft::gpu::v2 {

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
