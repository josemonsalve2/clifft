#pragma once

#include "clifft/backend/backend.h"
#include "clifft/gpu/gpu_types.h"

#include <vector>

namespace clifft {
namespace gpu {

/// Segment metadata for the split/persistent hybrid kernel.
/// Each segment represents a contiguous range of bytecode instructions
/// with a uniform tier assignment (per-thread, shared-coop, or global-coop).
/// Segments are computed at compile time from the active_k profile.
struct GpuSegment {
    uint32_t bc_start;       ///< First instruction index (inclusive)
    uint32_t bc_end;         ///< Last instruction index (exclusive)
    uint32_t local_peak_k;   ///< Maximum active_k within this segment
    uint8_t tier;            ///< 0 = per-thread, 1 = shared-coop, 2 = global-coop
};

/// Assign a tier based on local peak k.
inline uint8_t tier_for_peak_k(uint32_t local_peak_k) {
    if (local_peak_k <= kThreadMaxPeakRank) return 0;   // per-thread
    if (local_peak_k <= kSharedMaxPeakRank) return 1;   // shared-coop
    return 2;                                            // global-coop
}

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

    /// K-profile segments for the hybrid kernel.
    /// Empty when the source map has no active_k history (legacy path).
    std::vector<GpuSegment> segments;

    uint32_t peak_rank = 0;
    uint32_t total_meas_slots = 0;
    uint32_t num_observables = 0;
    uint32_t num_exp_vals = 0;
    bool has_extended_opcodes = false;

    /// True when segments span multiple tiers and the hybrid kernel
    /// is profitable. Computed by flatten_program().
    bool use_hybrid_kernel = false;
};

void validate_program(const CompiledModule& program);

FlattenedProgram flatten_program(const CompiledModule& program);

}  // namespace gpu
}  // namespace clifft
