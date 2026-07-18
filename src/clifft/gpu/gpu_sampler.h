#pragma once

#include "clifft/svm/svm.h"

#include <cstdint>
#include <optional>
#include <string>

namespace clifft {
namespace gpu {

struct GpuSamplerOptions {
    uint64_t seed = 42;
    uint32_t block_size = 256;
    bool keep_records = false;
};

SurvivorResult gpu_sample_survivors(const CompiledModule& program, uint64_t shots,
                                    const GpuSamplerOptions& options = {});

std::string gpu_backend_info();

}  // namespace gpu
}  // namespace clifft
