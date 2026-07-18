#pragma once

#include "clifft/backend/backend.h"
#include "clifft/gpu/gpu_types.h"

#include <vector>

namespace clifft {
namespace gpu {

struct FlattenedProgram {
    std::vector<GpuInstr> instrs;
    std::vector<GpuMask> pauli_masks;
    std::vector<GpuNoiseSite> noise_sites;
    std::vector<GpuChannel> noise_channels;
    std::vector<double> noise_hazards;
    std::vector<GpuReadoutNoise> readout_noise;
    std::vector<uint32_t> detector_offsets;
    std::vector<uint32_t> detector_targets;
    std::vector<uint32_t> observable_offsets;
    std::vector<uint32_t> observable_targets;
    std::vector<uint8_t> expected_observables;
    std::vector<GpuFusedU2Entry> fused_u2;
    std::vector<GpuFusedU4Entry> fused_u4;
    std::vector<GpuExpValMask> exp_val_masks;

    uint32_t peak_rank = 0;
    uint32_t total_meas_slots = 0;
    uint32_t num_observables = 0;
    uint32_t num_exp_vals = 0;
};

void validate_program(const CompiledModule& program);

FlattenedProgram flatten_program(const CompiledModule& program);

}  // namespace gpu
}  // namespace clifft
