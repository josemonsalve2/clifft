#pragma once

#include <cstdint>

namespace clifft {
namespace gpu {

constexpr uint32_t kThreadMaxPeakRank = 4;
constexpr uint32_t kThreadMaxAmplitudes = 1u << kThreadMaxPeakRank;
constexpr uint32_t kSharedMaxPeakRank = 10;
constexpr uint32_t kSharedMaxAmplitudes = 1u << kSharedMaxPeakRank;
constexpr uint32_t kGlobalMaxPeakRank = 19;
constexpr uint32_t kGlobalMaxAmplitudes = 1u << kGlobalMaxPeakRank;
constexpr uint32_t kMaxPeakRank = kGlobalMaxPeakRank;
constexpr uint32_t kMaxMeas = 1024;
constexpr uint32_t kMaxObs = 8;
constexpr uint32_t kMaxExpVals = 8;
constexpr uint64_t kMaxBatchShots = 100000000ULL;
constexpr uint8_t kFlagSign = 1u << 0;
constexpr uint8_t kFlagIdentity = 1u << 2;
constexpr uint8_t kFlagExpectedOne = 1u << 3;
constexpr double kInvSqrt2 = 0.70710678118654752440084436210484903928;
constexpr double kDustEpsilon = 1e-18;
constexpr uint32_t kNumXCDs = 8;  // MI300X has 8 XCDs

struct __attribute__((aligned(8))) GpuComplex {
    float re;
    float im;
};

struct GpuMask {
    uint64_t x[2];
    uint64_t z[2];
    uint8_t sign;
};

struct GpuChannel {
    uint64_t x[2];
    uint64_t z[2];
    double prob;
};

struct GpuNoiseSite {
    uint32_t offset;
    uint32_t count;
    double prob_sum;
};

struct GpuReadoutNoise {
    uint32_t meas_idx;
    double prob;
};

struct GpuFusedU2Entry {
    GpuComplex matrices[4][4];
    GpuComplex gamma_multipliers[4];
    uint8_t out_states[4];
};

struct GpuFusedU4Entry {
    struct Entry {
        GpuComplex matrix[4][4];
        GpuComplex gamma_multiplier;
        uint8_t out_state;
    };
    Entry entries[16];
};

struct GpuExpValMask {
    uint64_t x[2];
    uint64_t z[2];
    uint8_t sign;
};

struct GpuInstr {
    uint8_t opcode;
    uint8_t flags;
    uint16_t axis_1;
    uint16_t axis_2;
    uint32_t a;
    uint32_t b;
    uint64_t mask;
    double weight_re;
    double weight_im;
};

struct GpuProgram {
    const GpuInstr* instrs;
    uint32_t num_instrs;
    uint32_t peak_rank;
    uint32_t total_meas_slots;
    uint32_t num_observables;
    uint32_t num_exp_vals;
    const GpuMask* pauli_masks;
    const GpuNoiseSite* noise_sites;
    const GpuChannel* noise_channels;
    uint32_t num_noise_sites;
    const double* noise_hazards;
    const GpuReadoutNoise* readout_noise;
    const uint32_t* detector_offsets;
    const uint32_t* detector_targets;
    const uint32_t* observable_offsets;
    const uint32_t* observable_targets;
    const uint8_t* expected_observables;
    const GpuFusedU2Entry* fused_u2;
    const GpuFusedU4Entry* fused_u4;
    const GpuExpValMask* exp_val_masks;
    bool has_extended_opcodes;
};

struct BlockCounts {
    uint64_t passed;
    uint64_t logical_errors;
    uint64_t observable_ones[kMaxObs];
    double exp_val_sums[kMaxExpVals];
    uint64_t exp_val_count;
};

struct DeviceBuffers {
    GpuInstr* instrs = nullptr;
    GpuMask* pauli_masks = nullptr;
    GpuNoiseSite* noise_sites = nullptr;
    GpuChannel* noise_channels = nullptr;
    double* noise_hazards = nullptr;
    GpuReadoutNoise* readout_noise = nullptr;
    uint32_t* detector_offsets = nullptr;
    uint32_t* detector_targets = nullptr;
    uint32_t* observable_offsets = nullptr;
    uint32_t* observable_targets = nullptr;
    uint8_t* expected_observables = nullptr;
    BlockCounts* block_counts = nullptr;
    GpuComplex* global_v = nullptr;
    GpuComplex* global_scratch = nullptr;
    uint64_t* work_counter = nullptr;
    GpuFusedU2Entry* fused_u2 = nullptr;
    GpuFusedU4Entry* fused_u4 = nullptr;
    GpuExpValMask* exp_val_masks = nullptr;
};

}  // namespace gpu
}  // namespace clifft
