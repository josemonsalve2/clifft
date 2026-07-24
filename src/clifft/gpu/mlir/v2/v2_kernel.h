// v2_kernel.h — MLIR-V2 host driver (HIP-free). Loads amdgcn .hsaco code
// objects and dispatches them via the HSA runtime. See docs/v2/V2.md.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace clifft::gpu::v2 {

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
