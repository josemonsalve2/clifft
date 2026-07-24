// v2_kernel.h — MLIR-V2 host driver (HIP-free). Loads amdgcn .hsaco code
// objects and dispatches them via the HSA runtime. See docs/v2/V2.md.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "clifft/backend/backend.h"   // CompiledModule
#include "clifft/svm/svm.h"            // SurvivorResult

namespace clifft::gpu::v2 {

// Run a circuit through the V2 coop interpreter (HIP-free, HSA-dispatched).
// One workgroup (256 threads) per shot. Returns a SurvivorResult comparable to
// clifft::sample_survivors (CPU) and the GPU-SVM path. `hsaco_path` defaults to
// the build-configured coop kernel. `kernel_seconds` (if non-null) receives the
// measured kernel time.
clifft::SurvivorResult v2_sample(const clifft::CompiledModule& program,
                                 uint32_t shots, uint64_t seed,
                                 double* kernel_seconds = nullptr,
                                 std::string hsaco_path = "");

// P0(a) toolchain proof: dispatch the plain-C probe kernel that writes
// out[i]=i*2 for i in [0,n), read it back, and verify. Returns true on match.
// `hsaco_path` defaults to the build-time-configured probe path.
bool run_probe(uint32_t n, std::string hsaco_path = "");

// Result of the probe (for tests): the read-back buffer + kernel timing.
struct ProbeResult {
    bool ok = false;
    std::vector<uint32_t> out;
    double kernel_seconds = 0.0;
    std::string error;
};
ProbeResult run_probe_detailed(uint32_t n, std::string hsaco_path = "");

}  // namespace clifft::gpu::v2
