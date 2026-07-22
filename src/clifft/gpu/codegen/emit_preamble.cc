#include "clifft/gpu/codegen/codegen_types.h"

#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <string>

namespace clifft {
namespace gpu {
namespace codegen {

// -----------------------------------------------------------------------
// Formatting helpers
// -----------------------------------------------------------------------

std::string hex_double(double v) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%.17e", v);
    return buf;
}

std::string hex_float_pair(float re, float im) {
    char buf[128];
    snprintf(buf, sizeof(buf), "{%.9ef, %.9ef}", re, im);
    return buf;
}

// -----------------------------------------------------------------------
// Preamble sections: split into conditional parts
// -----------------------------------------------------------------------

// Always included: basic types, constants, ShotState, bit helpers, BlockCounts
// Compiled via: clang++ -x hip --offload-device-only --offload-arch=gfx942
// Uses AMDGCN intrinsics for thread/block IDs, barriers, shuffles.
// All functions are __device__ (callable from __global__ kernel).
static const char* kPreambleTypes = R"HIP(
#include <stdint.h>
#include <math.h>

// All helper functions are device-side.
// Under -x hip --offload-device-only, __device__ is required.
#define DEV __device__

// AMDGCN intrinsics for kernel launch parameters
#define TIDX  ((uint32_t)__builtin_amdgcn_workitem_id_x())
#define BIDX  ((uint32_t)__builtin_amdgcn_workgroup_id_x())
#define BSIZE 256u
// Full barrier with LDS fences (ensures LDS stores are visible across wavefronts)
DEV static inline __attribute__((always_inline, convergent))
void __barrier_lds() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}
#define BARRIER() __barrier_lds()
#define POPCOUNT64(x) __builtin_popcountll(x)

// LDS (shared memory)
#define LDS __shared__

// Warp-shuffle via ds_bpermute (AMDGCN wavefront=64)
// ds_bpermute reads lane (byte_offset/4) and returns its value.
DEV static inline __attribute__((always_inline)) int shfl_xor_i32(int val, int lane_mask) {
    int self_lane = TIDX & 63u;
    int target_lane = self_lane ^ lane_mask;
    // ds_bpermute takes byte offset = lane * 4
    return __builtin_amdgcn_ds_bpermute(target_lane << 2, val);
}
DEV static inline __attribute__((always_inline)) double shfl_xor_f64(double val, int lane_mask) {
    union { double d; int i[2]; } u;
    u.d = val;
    u.i[0] = shfl_xor_i32(u.i[0], lane_mask);
    u.i[1] = shfl_xor_i32(u.i[1], lane_mask);
    return u.d;
}

#define kThreadMaxAmplitudes 16u
#define kMaxMeas 1024u
#define kMaxObs 8u
#define kMaxExpVals 8u
#define kFlagSign     (1u << 0)
#define kFlagIdentity (1u << 2)
#define kFlagExpectedOne (1u << 3)
#define kInvSqrt2 0.70710678118654752440084436210484903928
#define kDustEpsilon 1e-18

struct GpuComplex { float re; float im; };

struct ShotState {
    uint64_t px[2];
    uint64_t pz[2];
    uint32_t active_k;
    uint32_t next_noise_idx;
    bool discarded;
    uint8_t meas[kMaxMeas];
    uint8_t obs[kMaxObs];
    GpuComplex v[kThreadMaxAmplitudes];
    double exp_vals[kMaxExpVals];
};

struct BlockCounts {
    uint64_t passed;
    uint64_t logical_errors;
    uint64_t observable_ones[8];
    double exp_val_sums[8];
    uint64_t exp_val_count;
};

DEV static inline bool bit_get(const uint64_t* words, uint32_t idx) {
    return ((words[idx >> 6] >> (idx & 63u)) & 1ULL) != 0;
}
DEV static inline void bit_set(uint64_t* words, uint32_t idx, bool value) {
    uint64_t mask = 1ULL << (idx & 63u);
    uint32_t word = idx >> 6;
    if (value) words[word] |= mask; else words[word] &= ~mask;
}
DEV static inline void bit_xor(uint64_t* words, uint32_t idx, bool value) {
    if (value) words[idx >> 6] ^= 1ULL << (idx & 63u);
}
DEV static inline void bit_swap(uint64_t* a, uint32_t ia, uint64_t* b, uint32_t ib) {
    bool va = bit_get(a, ia), vb = bit_get(b, ib);
    if (va != vb) {
        a[ia >> 6] ^= 1ULL << (ia & 63u);
        b[ib >> 6] ^= 1ULL << (ib & 63u);
    }
}
)HIP";

// RNG: only needed if measurements or noise use randomness
static const char* kPreambleRng = R"HIP(
DEV static inline __attribute__((always_inline)) uint64_t rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}
DEV static inline __attribute__((always_inline)) uint64_t splitmix64_next(uint64_t& state) {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
struct Rng {
    uint64_t s[4];
    DEV void seed(uint64_t seed_value, uint64_t shot_id) {
        uint64_t z = seed_value ^ (0x9e3779b97f4a7c15ULL * (shot_id + 1));
        s[0] = splitmix64_next(z);
        s[1] = splitmix64_next(z);
        s[2] = splitmix64_next(z);
        s[3] = splitmix64_next(z);
    }
    DEV uint64_t next() {
        const uint64_t result = rotl64(s[0] + s[3], 23) + s[0];
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t; s[3] = rotl64(s[3], 45);
        return result;
    }
    DEV double uniform() {
        return (double)(next() >> 11) * 0x1.0p-53;
    }
};
)HIP";

// Complex arithmetic: only needed when array sweeps transform amplitudes
static const char* kPreambleComplex = R"HIP(
DEV static inline __attribute__((always_inline)) GpuComplex cadd(GpuComplex a, GpuComplex b) { return {a.re+b.re, a.im+b.im}; }
DEV static inline __attribute__((always_inline)) GpuComplex csub(GpuComplex a, GpuComplex b) { return {a.re-b.re, a.im-b.im}; }
DEV static inline __attribute__((always_inline)) GpuComplex cscale(GpuComplex a, double s) {
    return {(float)((double)a.re*s), (float)((double)a.im*s)};
}
DEV static inline __attribute__((always_inline)) GpuComplex cmul(GpuComplex a, GpuComplex b) {
    return {(float)(a.re*b.re - a.im*b.im), (float)(a.re*b.im + a.im*b.re)};
}
DEV static inline __attribute__((always_inline)) double cnorm(GpuComplex a) {
    return (double)a.re*(double)a.re + (double)a.im*(double)a.im;
}
)HIP";

// Scatter/insert-zero-bit: needed for array sweep index calculations
static const char* kPreambleArraySweep = R"HIP(
DEV static inline __attribute__((always_inline)) uint64_t insert_zero_bit(uint64_t val, uint32_t pos) {
    uint64_t mask = (1ULL << pos) - 1ULL;
    return (val & mask) | ((val & ~mask) << 1);
}
DEV static inline __attribute__((always_inline)) uint64_t scatter_bits_1(uint64_t val, uint32_t bit_pos) {
    return insert_zero_bit(val, bit_pos);
}
DEV static inline __attribute__((always_inline)) uint64_t scatter_bits_2(uint64_t val, uint32_t bit1, uint32_t bit2) {
    uint32_t lo = bit1 < bit2 ? bit1 : bit2;
    uint32_t hi = bit1 < bit2 ? bit2 : bit1;
    val = insert_zero_bit(val, lo);
    return insert_zero_bit(val, hi);
}
)HIP";

// -----------------------------------------------------------------------
// Include device function string literals from ops/ .inc files
// -----------------------------------------------------------------------
#include "ops/frame_ops.inc"
#include "ops/array_ops.inc"
#include "ops/multi_qubit_ops.inc"
#include "ops/expand_ops.inc"
#include "ops/measurement_ops.inc"
#include "ops/noise_ops.inc"
#include "ops/unitary_ops.inc"
#include "ops/exp_val_ops.inc"
#include "ops/reduction_ops.inc"

// -----------------------------------------------------------------------
// Constant pool emission helpers
// -----------------------------------------------------------------------

static std::string emit_gpu_mask(const GpuMask& m) {
    char buf[256];
    snprintf(buf, sizeof(buf),
             "{{0x%016" PRIx64 "ULL, 0x%016" PRIx64 "ULL},"
             " {0x%016" PRIx64 "ULL, 0x%016" PRIx64 "ULL},"
             " %uu}",
             m.x[0], m.x[1], m.z[0], m.z[1], (unsigned)m.sign);
    return buf;
}

static std::string emit_gpu_channel(const GpuChannel& c) {
    char buf[256];
    snprintf(buf, sizeof(buf),
             "{{0x%016" PRIx64 "ULL, 0x%016" PRIx64 "ULL},"
             " {0x%016" PRIx64 "ULL, 0x%016" PRIx64 "ULL},"
             " %s}",
             c.x[0], c.x[1], c.z[0], c.z[1], hex_double(c.prob).c_str());
    return buf;
}

static std::string emit_fused_u2(const GpuFusedU2Entry& e) {
    std::ostringstream s;
    s << "{\n            { // matrices\n";
    for (int fs = 0; fs < 4; ++fs) {
        s << "              {";
        for (int j = 0; j < 4; ++j) {
            if (j) s << ", ";
            s << hex_float_pair(e.matrices[fs][j].re, e.matrices[fs][j].im);
        }
        s << "},\n";
    }
    s << "            },\n            { // gamma_multipliers\n";
    for (int fs = 0; fs < 4; ++fs) {
        if (fs) s << ", ";
        s << hex_float_pair(e.gamma_multipliers[fs].re, e.gamma_multipliers[fs].im);
    }
    s << "\n            },\n";
    s << "            {" << (unsigned)e.out_states[0] << "u, " << (unsigned)e.out_states[1]
      << "u, " << (unsigned)e.out_states[2] << "u, " << (unsigned)e.out_states[3] << "u}\n";
    s << "          }";
    return s.str();
}

static std::string emit_fused_u4(const GpuFusedU4Entry& e) {
    std::ostringstream s;
    s << "{\n            { // entries[16]\n";
    for (int fs = 0; fs < 16; ++fs) {
        const auto& en = e.entries[fs];
        s << "              { { // entry " << fs << " matrix\n";
        for (int r = 0; r < 4; ++r) {
            s << "                  {";
            for (int c = 0; c < 4; ++c) {
                if (c) s << ", ";
                s << hex_float_pair(en.matrix[r][c].re, en.matrix[r][c].im);
            }
            s << "},\n";
        }
        s << "                },\n";
        s << "                " << hex_float_pair(en.gamma_multiplier.re, en.gamma_multiplier.im)
          << ",\n";
        s << "                " << (unsigned)en.out_state << "u\n";
        s << "              },\n";
    }
    s << "            }\n          }";
    return s.str();
}

void emit_constant_pool(std::ostringstream& out, const FlattenedProgram& flat,
                                const UsedFunctions& uf) {
    if (uf.apply_pauli && !flat.pauli_masks.empty()) {
        out << "DEV static const GpuMask kPauliMasks["
            << flat.pauli_masks.size() << "] = {\n";
        for (size_t i = 0; i < flat.pauli_masks.size(); ++i) {
            out << "    " << emit_gpu_mask(flat.pauli_masks[i]);
            if (i + 1 < flat.pauli_masks.size()) out << ",";
            out << "\n";
        }
        out << "};\n";
    }

    if ((uf.noise || uf.noise_block) && !flat.noise_sites.empty()) {
        out << "DEV static const GpuNoiseSite kNoiseSites["
            << flat.noise_sites.size() << "] = {\n";
        for (size_t i = 0; i < flat.noise_sites.size(); ++i) {
            const auto& ns = flat.noise_sites[i];
            char buf[128];
            snprintf(buf, sizeof(buf), "    {%uu, %uu, %s}",
                     ns.offset, ns.count, hex_double(ns.prob_sum).c_str());
            out << buf;
            if (i + 1 < flat.noise_sites.size()) out << ",";
            out << "\n";
        }
        out << "};\n";

        out << "DEV static const GpuChannel kNoiseChannels["
            << flat.noise_channels.size() << "] = {\n";
        for (size_t i = 0; i < flat.noise_channels.size(); ++i) {
            out << "    " << emit_gpu_channel(flat.noise_channels[i]);
            if (i + 1 < flat.noise_channels.size()) out << ",";
            out << "\n";
        }
        out << "};\n";

        out << "DEV static const double kNoiseHazards["
            << flat.noise_hazards.size() << "] = {\n";
        for (size_t i = 0; i < flat.noise_hazards.size(); ++i) {
            out << "    " << hex_double(flat.noise_hazards[i]);
            if (i + 1 < flat.noise_hazards.size()) out << ",";
            out << "\n";
        }
        out << "};\n";
    }

    if (uf.readout_noise && !flat.readout_noise.empty()) {
        out << "DEV static const GpuReadoutNoise kReadoutNoise["
            << flat.readout_noise.size() << "] = {\n";
        for (size_t i = 0; i < flat.readout_noise.size(); ++i) {
            const auto& r = flat.readout_noise[i];
            out << "    {" << r.meas_idx << "u, " << hex_double(r.prob) << "}";
            if (i + 1 < flat.readout_noise.size()) out << ",";
            out << "\n";
        }
        out << "};\n";
    }

    if (uf.postselect && !flat.detector_offsets.empty()) {
        out << "DEV static const uint32_t kDetectorOffsets["
            << flat.detector_offsets.size() << "] = {";
        for (size_t i = 0; i < flat.detector_offsets.size(); ++i) {
            if (i) out << ", ";
            out << flat.detector_offsets[i] << "u";
        }
        out << "};\n";
        out << "DEV static const uint32_t kDetectorTargets["
            << flat.detector_targets.size() << "] = {";
        for (size_t i = 0; i < flat.detector_targets.size(); ++i) {
            if (i) out << ", ";
            out << flat.detector_targets[i] << "u";
        }
        out << "};\n";
    }

    if (uf.observable && !flat.observable_offsets.empty()) {
        out << "DEV static const uint32_t kObservableOffsets["
            << flat.observable_offsets.size() << "] = {";
        for (size_t i = 0; i < flat.observable_offsets.size(); ++i) {
            if (i) out << ", ";
            out << flat.observable_offsets[i] << "u";
        }
        out << "};\n";
        out << "DEV static const uint32_t kObservableTargets["
            << flat.observable_targets.size() << "] = {";
        for (size_t i = 0; i < flat.observable_targets.size(); ++i) {
            if (i) out << ", ";
            out << flat.observable_targets[i] << "u";
        }
        out << "};\n";
        out << "DEV static const uint8_t kExpectedObservables["
            << flat.expected_observables.size() << "] = {";
        for (size_t i = 0; i < flat.expected_observables.size(); ++i) {
            if (i) out << ", ";
            out << (unsigned)flat.expected_observables[i] << "u";
        }
        out << "};\n";
    }

    if (uf.array_u2 && !flat.fused_u2.empty()) {
        out << "DEV static const GpuFusedU2Entry kFusedU2["
            << flat.fused_u2.size() << "] = {\n";
        for (size_t i = 0; i < flat.fused_u2.size(); ++i) {
            out << "          " << emit_fused_u2(flat.fused_u2[i]);
            if (i + 1 < flat.fused_u2.size()) out << ",";
            out << "\n";
        }
        out << "};\n";
    }

    if (uf.array_u4 && !flat.fused_u4.empty()) {
        out << "DEV static const GpuFusedU4Entry kFusedU4["
            << flat.fused_u4.size() << "] = {\n";
        for (size_t i = 0; i < flat.fused_u4.size(); ++i) {
            out << "          " << emit_fused_u4(flat.fused_u4[i]);
            if (i + 1 < flat.fused_u4.size()) out << ",";
            out << "\n";
        }
        out << "};\n";
    }

    if (uf.exp_val && !flat.exp_val_masks.empty()) {
        out << "DEV static const GpuExpValMask kExpValMasks["
            << flat.exp_val_masks.size() << "] = {\n";
        for (size_t i = 0; i < flat.exp_val_masks.size(); ++i) {
            const auto& m = flat.exp_val_masks[i];
            char buf[256];
            snprintf(buf, sizeof(buf),
                     "    {{0x%016" PRIx64 "ULL, 0x%016" PRIx64 "ULL},"
                     " {0x%016" PRIx64 "ULL, 0x%016" PRIx64 "ULL},"
                     " %uu}",
                     m.x[0], m.x[1], m.z[0], m.z[1], (unsigned)m.sign);
            out << buf;
            if (i + 1 < flat.exp_val_masks.size()) out << ",";
            out << "\n";
        }
        out << "};\n";
    }

    out << "\n";
}

// -----------------------------------------------------------------------
// Emit preamble: conditional sections based on UsedFunctions
// -----------------------------------------------------------------------
void emit_preamble(std::ostringstream& out, const UsedFunctions& uf) {
    // Always: types, constants, bit helpers, BlockCounts
    out << kPreambleTypes;

    // Conditional type declarations for constant pool types
    if (uf.apply_pauli) out << kPauliMaskType;
    if (uf.noise || uf.noise_block) out << kNoiseTypes;
    if (uf.readout_noise) out << kReadoutNoiseType;
    if (uf.array_u2) out << kFusedU2Type;
    if (uf.array_u4) out << kFusedU4Type;
    if (uf.exp_val) out << kExpValMaskType;

    // RNG: only if measurements or noise need randomness
    if (uf.needs_rng) out << kPreambleRng;

    // Complex arithmetic: only if array sweeps need it
    if (uf.needs_complex_ops) out << kPreambleComplex;

    // Scatter bits: only if array sweep index calculations needed
    if (uf.needs_scatter) out << kPreambleArraySweep;

    // sample_branch: only if active measurements
    if (uf.needs_sample_branch) out << kPreambleSampleBranch;
}

// -----------------------------------------------------------------------
// Emit device functions: only what the circuit needs
// -----------------------------------------------------------------------
void emit_needed_functions(std::ostringstream& out, const UsedFunctions& uf) {
    // Frame ops: needed by frame opcodes and by array ops that call them
    if (uf.needs_frame_ops) out << kFrameOps;

    // apply_pauli_to_frame: needed by APPLY_PAULI and NOISE/NOISE_BLOCK
    if (uf.apply_pauli || uf.noise || uf.noise_block) out << kApplyPauliToFrame;
    if (uf.apply_pauli) out << kApplyPauli;

    // Noise helpers
    if (uf.noise || uf.noise_block) out << kNoiseHelper;

    // apply_phase: needed by array_s, array_t, array_rot
    if (uf.needs_apply_phase) out << kApplyPhase;

    // Individual array operations
    if (uf.array_cnot) out << kArrayCnot;
    if (uf.array_cz) out << kArrayCz;
    if (uf.array_swap) out << kArraySwap;
    if (uf.array_multi_cnot) out << kArrayMultiCnot;
    if (uf.array_multi_cz) out << kArrayMultiCz;
    if (uf.array_h) out << kArrayH;
    if (uf.array_s) out << kArrayS;
    if (uf.array_t) out << kArrayT;
    if (uf.array_rot) out << kArrayRot;
    if (uf.array_u2) out << kArrayU2;
    if (uf.array_u4) out << kArrayU4;

    // Expand operations
    if (uf.expand_plain) out << kExpandPlain;
    if (uf.expand_t) out << kExpandT;
    if (uf.expand_rot) out << kExpandRot;

    // Measurement helpers
    if (uf.meas_dormant_random) out << kMeasDormantRandom;
    if (uf.meas_active_diagonal) out << kMeasActiveDiag;
    if (uf.meas_active_interfere || uf.swap_meas_interfere) out << kMeasActiveInterfere;
    if (uf.swap_meas_interfere) out << kSwapMeasInterfere;
    if (uf.exp_val) out << kExecExpVal;
}

}  // namespace codegen
}  // namespace gpu
}  // namespace clifft
