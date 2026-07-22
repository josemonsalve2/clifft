#include "clifft/gpu/gpu_sampler.h"

#include <stdexcept>

namespace clifft {
namespace gpu {

SurvivorResult gpu_sample_survivors(const CompiledModule&, uint64_t, const GpuSamplerOptions&) {
    throw std::runtime_error(
        "GPU sampler not available: clifft was built without CLIFFT_ENABLE_HIP");
}

std::string gpu_backend_info() { return "GPU backend not available (built without HIP)"; }

}  // namespace gpu
}  // namespace clifft
