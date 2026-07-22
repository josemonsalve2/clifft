#include "clifft/gpu/kernel_codegen.h"
#include "clifft/gpu/gpu_types.h"

#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace clifft {
namespace gpu {

using Op = clifft::Opcode;

// -----------------------------------------------------------------------
// UsedFunctions::compute_derived()
// -----------------------------------------------------------------------
void UsedFunctions::compute_derived() {
    needs_rng = meas_dormant_random || meas_active_diagonal ||
                meas_active_interfere || swap_meas_interfere ||
                noise || noise_block || readout_noise;

    needs_array_sweep = array_cnot || array_cz || array_swap ||
                        array_multi_cnot || array_multi_cz ||
                        array_h || array_s || array_t ||
                        array_rot || array_u2 || array_u4 ||
                        expand_plain || expand_t || expand_rot;

    // Complex arithmetic (cmul, cadd, csub, cscale, cnorm) is needed by
    // any array gate that transforms amplitudes (not just frame updates),
    // and by active measurements that compute probabilities.
    needs_complex_ops = array_h || array_s || array_t || array_rot ||
                        array_u2 || array_u4 ||
                        expand_t || expand_rot ||
                        meas_active_diagonal || meas_active_interfere ||
                        swap_meas_interfere || exp_val;

    needs_apply_phase = array_s || array_t || array_rot;

    // Frame ops are needed directly, and also called implicitly by array
    // 2-qubit ops (array_cnot calls frame_cnot, etc.)
    needs_frame_ops = frame_cnot || frame_cz || frame_h || frame_s || frame_swap ||
                      array_cnot || array_cz || array_swap ||
                      array_multi_cnot || array_multi_cz ||
                      array_h || array_s || array_t ||
                      swap_meas_interfere;

    // Scatter bits needed for 2-qubit array ops, 1-qubit array sweeps, expands
    needs_scatter = array_cnot || array_cz || array_swap ||
                    array_multi_cnot || array_multi_cz ||
                    array_h || array_s || array_t || array_rot ||
                    array_u2 || array_u4 ||
                    expand_plain || expand_t || expand_rot;

    needs_sample_branch = meas_active_diagonal || meas_active_interfere ||
                          swap_meas_interfere;
}

// -----------------------------------------------------------------------
// analyze_used_functions
// -----------------------------------------------------------------------
UsedFunctions analyze_used_functions(const FlattenedProgram& flat) {
    UsedFunctions uf{};
    for (const auto& instr : flat.instrs) {
        auto op = static_cast<Op>(instr.opcode);
        switch (op) {
            case Op::OP_FRAME_CNOT:      uf.frame_cnot = true; break;
            case Op::OP_FRAME_CZ:        uf.frame_cz = true; break;
            case Op::OP_FRAME_H:         uf.frame_h = true; break;
            case Op::OP_FRAME_S:
            case Op::OP_FRAME_S_DAG:     uf.frame_s = true; break;
            case Op::OP_FRAME_SWAP:      uf.frame_swap = true; break;
            case Op::OP_ARRAY_CNOT:      uf.array_cnot = true; break;
            case Op::OP_ARRAY_CZ:        uf.array_cz = true; break;
            case Op::OP_ARRAY_SWAP:      uf.array_swap = true; break;
            case Op::OP_ARRAY_MULTI_CNOT:uf.array_multi_cnot = true; break;
            case Op::OP_ARRAY_MULTI_CZ:  uf.array_multi_cz = true; break;
            case Op::OP_ARRAY_H:         uf.array_h = true; break;
            case Op::OP_ARRAY_S:
            case Op::OP_ARRAY_S_DAG:     uf.array_s = true; break;
            case Op::OP_ARRAY_T:
            case Op::OP_ARRAY_T_DAG:     uf.array_t = true; break;
            case Op::OP_ARRAY_ROT:       uf.array_rot = true; break;
            case Op::OP_ARRAY_U2:        uf.array_u2 = true; break;
            case Op::OP_ARRAY_U4:        uf.array_u4 = true; break;
            case Op::OP_EXPAND:          uf.expand_plain = true; break;
            case Op::OP_EXPAND_T:
            case Op::OP_EXPAND_T_DAG:    uf.expand_t = true; break;
            case Op::OP_EXPAND_ROT:      uf.expand_rot = true; break;
            case Op::OP_MEAS_DORMANT_STATIC:  uf.meas_dormant_static = true; break;
            case Op::OP_MEAS_DORMANT_RANDOM:  uf.meas_dormant_random = true; break;
            case Op::OP_MEAS_ACTIVE_DIAGONAL: uf.meas_active_diagonal = true; break;
            case Op::OP_MEAS_ACTIVE_INTERFERE:uf.meas_active_interfere = true; break;
            case Op::OP_SWAP_MEAS_INTERFERE:  uf.swap_meas_interfere = true; break;
            case Op::OP_APPLY_PAULI:     uf.apply_pauli = true; break;
            case Op::OP_NOISE:           uf.noise = true; break;
            case Op::OP_NOISE_BLOCK:     uf.noise_block = true; break;
            case Op::OP_READOUT_NOISE:   uf.readout_noise = true; break;
            case Op::OP_POSTSELECT:      uf.postselect = true; break;
            case Op::OP_OBSERVABLE:       uf.observable = true; break;
            case Op::OP_EXP_VAL:         uf.exp_val = true; break;
            default: break;
        }
    }
    uf.compute_derived();
    return uf;
}

// -----------------------------------------------------------------------
// Formatting helpers
// -----------------------------------------------------------------------
namespace {

static std::string hex_double(double v) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%.17e", v);
    return buf;
}

static std::string hex_float_pair(float re, float im) {
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

// sample_branch: needed for active measurements
static const char* kPreambleSampleBranch = R"HIP(
DEV static inline __attribute__((always_inline)) uint8_t sample_branch(Rng& rng, double prob0, double prob1, double total) {
    double eps = kDustEpsilon * total;
    if (prob1 <= eps) return 0;
    if (prob0 <= eps) return 1;
    return (rng.uniform() * total < prob0) ? 0 : 1;
}
)HIP";

// -----------------------------------------------------------------------
// Noise type declarations: only needed when noise ops are present
// -----------------------------------------------------------------------
static const char* kNoiseTypes = R"HIP(
struct GpuChannel { uint64_t x[2]; uint64_t z[2]; double prob; };
struct GpuNoiseSite  { uint32_t offset; uint32_t count; double prob_sum; };
)HIP";

static const char* kReadoutNoiseType = R"HIP(
struct GpuReadoutNoise { uint32_t meas_idx; double prob; };
)HIP";

static const char* kPauliMaskType = R"HIP(
struct GpuMask { uint64_t x[2]; uint64_t z[2]; uint8_t sign; };
)HIP";

static const char* kFusedU2Type = R"HIP(
struct GpuFusedU2Entry {
    GpuComplex matrices[4][4];
    GpuComplex gamma_multipliers[4];
    uint8_t out_states[4];
};
)HIP";

static const char* kFusedU4Type = R"HIP(
struct GpuFusedU4Entry {
    struct Entry {
        GpuComplex matrix[4][4];
        GpuComplex gamma_multiplier;
        uint8_t out_state;
    };
    Entry entries[16];
};
)HIP";

static const char* kExpValMaskType = R"HIP(
struct GpuExpValMask { uint64_t x[2]; uint64_t z[2]; uint8_t sign; };
)HIP";

// -----------------------------------------------------------------------
// Individual device function strings
// -----------------------------------------------------------------------

static const char* kFrameOps = R"HIP(
DEV static inline __attribute__((always_inline)) void frame_cnot(ShotState& st, uint32_t c, uint32_t t) {
    bool px_c = bit_get(st.px, c), pz_t = bit_get(st.pz, t);
    bit_xor(st.px, t, px_c); bit_xor(st.pz, c, pz_t);
}
DEV static inline __attribute__((always_inline)) void frame_cz(ShotState& st, uint32_t c, uint32_t t) {
    bool px_c = bit_get(st.px, c), px_t = bit_get(st.px, t);
    bit_xor(st.pz, t, px_c); bit_xor(st.pz, c, px_t);
}
DEV static inline __attribute__((always_inline)) void frame_h(ShotState& st, uint32_t axis) {
    bool px = bit_get(st.px, axis), pz = bit_get(st.pz, axis);
    bit_set(st.px, axis, pz); bit_set(st.pz, axis, px);
}
DEV static inline __attribute__((always_inline)) void frame_s(ShotState& st, uint32_t axis) {
    bit_xor(st.pz, axis, bit_get(st.px, axis));
}
DEV static inline __attribute__((always_inline)) void frame_swap(ShotState& st, uint32_t a, uint32_t b) {
    bit_swap(st.px, a, st.px, b); bit_swap(st.pz, a, st.pz, b);
}
)HIP";

static const char* kApplyPauliToFrame = R"HIP(
DEV static void apply_pauli_to_frame(ShotState& st, const uint64_t* x, const uint64_t* z) {
    st.px[0] ^= x[0]; st.px[1] ^= x[1];
    st.pz[0] ^= z[0]; st.pz[1] ^= z[1];
}
)HIP";

static const char* kApplyPauli = R"HIP(
DEV static void apply_pauli(ShotState& st, const GpuMask* masks, uint32_t mask_idx,
                            uint32_t condition_idx) {
    if (st.meas[condition_idx] == 0) return;
    const GpuMask& mask = masks[mask_idx];
    apply_pauli_to_frame(st, mask.x, mask.z);
}
)HIP";

static const char* kNoiseHelper = R"HIP(
DEV static void draw_next_noise_compiled(ShotState& st, Rng& rng,
                                         const double* noise_hazards,
                                         uint32_t num_noise_sites) {
    if (num_noise_sites == 0 || st.next_noise_idx >= num_noise_sites) {
        st.next_noise_idx = 0xffffffffu;
        return;
    }
    double current_hazard = (st.next_noise_idx == 0) ? 0.0
                            : noise_hazards[st.next_noise_idx - 1];
    double u = rng.uniform();
    double gap = -log(1.0 - u);
    double target = current_hazard + gap;
    uint32_t lo = 0, hi = num_noise_sites;
    while (lo < hi) {
        uint32_t mid = lo + ((hi - lo) >> 1);
        if (noise_hazards[mid] <= target) lo = mid + 1; else hi = mid;
    }
    st.next_noise_idx = (lo >= num_noise_sites) ? 0xffffffffu : lo;
}
)HIP";

static const char* kApplyPhase = R"HIP(
DEV static void apply_phase(ShotState& st, uint32_t axis, GpuComplex phase) {
    uint64_t axis_bit = 1ULL << axis;
    uint64_t iters = 1ULL << (st.active_k - 1);
    for (uint64_t i = 0; i < iters; ++i) {
        uint64_t idx = scatter_bits_1(i, axis) | axis_bit;
        st.v[idx] = cmul(st.v[idx], phase);
    }
}
)HIP";

static const char* kArrayCnot = R"HIP(
DEV static void array_cnot(ShotState& st, uint32_t c, uint32_t t) {
    uint64_t c_bit = 1ULL << c, t_bit = 1ULL << t;
    uint64_t iters = 1ULL << (st.active_k - 2);
    for (uint64_t i = 0; i < iters; ++i) {
        uint64_t base = scatter_bits_2(i, c, t) | c_bit;
        GpuComplex tmp = st.v[base];
        st.v[base] = st.v[base | t_bit];
        st.v[base | t_bit] = tmp;
    }
    frame_cnot(st, c, t);
}
)HIP";

static const char* kArrayCz = R"HIP(
DEV static void array_cz(ShotState& st, uint32_t a, uint32_t b) {
    uint64_t both_bits = (1ULL << a) | (1ULL << b);
    uint64_t iters = 1ULL << (st.active_k - 2);
    for (uint64_t i = 0; i < iters; ++i) {
        uint64_t idx = scatter_bits_2(i, a, b) | both_bits;
        st.v[idx].re = -st.v[idx].re; st.v[idx].im = -st.v[idx].im;
    }
    frame_cz(st, a, b);
}
)HIP";

static const char* kArraySwap = R"HIP(
DEV static void array_swap(ShotState& st, uint32_t a, uint32_t b) {
    uint64_t a_bit = 1ULL << a, b_bit = 1ULL << b;
    uint64_t iters = 1ULL << (st.active_k - 2);
    for (uint64_t i = 0; i < iters; ++i) {
        uint64_t base = scatter_bits_2(i, a, b);
        GpuComplex tmp = st.v[base | a_bit];
        st.v[base | a_bit] = st.v[base | b_bit];
        st.v[base | b_bit] = tmp;
    }
    frame_swap(st, a, b);
}
)HIP";

static const char* kArrayMultiCnot = R"HIP(
DEV static void array_multi_cnot(ShotState& st, uint32_t target, uint64_t ctrl_mask) {
    uint64_t t_bit = 1ULL << target;
    uint64_t half = 1ULL << (st.active_k - 1);
    for (uint64_t idx = 0; idx < half; ++idx) {
        uint64_t actual = scatter_bits_1(idx, target);
        if ((POPCOUNT64(actual & ctrl_mask) & 1) != 0) {
            GpuComplex tmp = st.v[actual];
            st.v[actual] = st.v[actual | t_bit];
            st.v[actual | t_bit] = tmp;
        }
    }
    for (uint32_t c = 0; c < st.active_k; ++c) {
        if ((ctrl_mask >> c) & 1ULL) frame_cnot(st, c, target);
    }
}
)HIP";

static const char* kArrayMultiCz = R"HIP(
DEV static void array_multi_cz(ShotState& st, uint32_t control, uint64_t target_mask) {
    uint64_t c_bit = 1ULL << control;
    uint64_t half = 1ULL << (st.active_k - 1);
    for (uint64_t idx = 0; idx < half; ++idx) {
        uint64_t actual = scatter_bits_1(idx, control) | c_bit;
        if ((POPCOUNT64(actual & target_mask) & 1) != 0) {
            st.v[actual].re = -st.v[actual].re; st.v[actual].im = -st.v[actual].im;
        }
    }
    for (uint32_t t = 0; t < st.active_k; ++t) {
        if ((target_mask >> t) & 1ULL) frame_cz(st, control, t);
    }
}
)HIP";

static const char* kArrayH = R"HIP(
DEV static void array_h(ShotState& st, uint32_t axis) {
    uint64_t axis_bit = 1ULL << axis;
    uint64_t iters = 1ULL << (st.active_k - 1);
    for (uint64_t i = 0; i < iters; ++i) {
        uint64_t idx0 = scatter_bits_1(i, axis), idx1 = idx0 | axis_bit;
        GpuComplex a = st.v[idx0], b = st.v[idx1];
        st.v[idx0] = cscale(cadd(a, b), kInvSqrt2);
        st.v[idx1] = cscale(csub(a, b), kInvSqrt2);
    }
    frame_h(st, axis);
}
)HIP";

static const char* kArrayS = R"HIP(
DEV static void array_s(ShotState& st, uint32_t axis, bool dagger) {
    apply_phase(st, axis, dagger ? (GpuComplex){0.0f, -1.0f} : (GpuComplex){0.0f, 1.0f});
    frame_s(st, axis);
}
)HIP";

static const char* kArrayT = R"HIP(
DEV static void array_t(ShotState& st, uint32_t axis, bool dagger) {
    bool px = bit_get(st.px, axis);
    if (axis >= st.active_k) return;
    double imag = dagger ? -kInvSqrt2 : kInvSqrt2;
    if (px) imag = -imag;
    apply_phase(st, axis, (GpuComplex){(float)kInvSqrt2, (float)imag});
}
)HIP";

static const char* kArrayRot = R"HIP(
DEV static void array_rot(ShotState& st, uint32_t axis, double z_re, double z_im) {
    if (axis < st.active_k) {
        bool px = bit_get(st.px, axis);
        double im = px ? -z_im : z_im;
        apply_phase(st, axis, (GpuComplex){(float)z_re, (float)im});
    }
}
)HIP";

static const char* kArrayU2 = R"HIP(
DEV static void array_u2_compiled(ShotState& st, uint32_t axis,
                                  const GpuFusedU2Entry* table, uint32_t cp_idx) {
    const GpuFusedU2Entry& entry = table[cp_idx];
    uint8_t in_state = (bit_get(st.pz, axis) ? 2 : 0) | (bit_get(st.px, axis) ? 1 : 0);
    uint8_t out = entry.out_states[in_state];
    bit_set(st.px, axis, (out & 1) != 0);
    bit_set(st.pz, axis, (out & 2) != 0);
    if (axis >= st.active_k) return;
    const GpuComplex* mat = entry.matrices[in_state];
    uint64_t axis_bit = 1ULL << axis;
    uint64_t iters = 1ULL << (st.active_k - 1);
    for (uint64_t i = 0; i < iters; ++i) {
        uint64_t idx0 = scatter_bits_1(i, axis);
        uint64_t idx1 = idx0 | axis_bit;
        GpuComplex a = st.v[idx0], b = st.v[idx1];
        st.v[idx0] = cadd(cmul(a, mat[0]), cmul(b, mat[1]));
        st.v[idx1] = cadd(cmul(a, mat[2]), cmul(b, mat[3]));
    }
}
)HIP";

static const char* kArrayU4 = R"HIP(
DEV static void array_u4_compiled(ShotState& st, uint32_t axis_lo, uint32_t axis_hi,
                                  const GpuFusedU4Entry* table, uint32_t cp_idx) {
    const GpuFusedU4Entry& node = table[cp_idx];
    uint8_t px_lo = bit_get(st.px, axis_lo) ? 1 : 0;
    uint8_t pz_lo = bit_get(st.pz, axis_lo) ? 1 : 0;
    uint8_t px_hi = bit_get(st.px, axis_hi) ? 1 : 0;
    uint8_t pz_hi = bit_get(st.pz, axis_hi) ? 1 : 0;
    uint8_t in_state = (pz_hi << 3) | (px_hi << 2) | (pz_lo << 1) | px_lo;
    const GpuFusedU4Entry::Entry& entry = node.entries[in_state];
    uint8_t out = entry.out_state;
    bit_set(st.px, axis_lo, (out & 1) != 0);
    bit_set(st.pz, axis_lo, (out & 2) != 0);
    bit_set(st.px, axis_hi, (out & 4) != 0);
    bit_set(st.pz, axis_hi, (out & 8) != 0);
    if (axis_hi >= st.active_k) return;
    uint64_t lo_bit = 1ULL << axis_lo;
    uint64_t hi_bit = 1ULL << axis_hi;
    uint64_t iters = 1ULL << (st.active_k - 2);
    for (uint64_t i = 0; i < iters; ++i) {
        uint64_t base = scatter_bits_2(i, axis_lo, axis_hi);
        uint64_t i0 = base;
        uint64_t i1 = base | lo_bit;
        uint64_t i2 = base | hi_bit;
        uint64_t i3 = base | lo_bit | hi_bit;
        GpuComplex v0 = st.v[i0], v1 = st.v[i1], v2 = st.v[i2], v3 = st.v[i3];
        st.v[i0] = cadd(cadd(cmul(v0, entry.matrix[0][0]), cmul(v1, entry.matrix[0][1])),
                        cadd(cmul(v2, entry.matrix[0][2]), cmul(v3, entry.matrix[0][3])));
        st.v[i1] = cadd(cadd(cmul(v0, entry.matrix[1][0]), cmul(v1, entry.matrix[1][1])),
                        cadd(cmul(v2, entry.matrix[1][2]), cmul(v3, entry.matrix[1][3])));
        st.v[i2] = cadd(cadd(cmul(v0, entry.matrix[2][0]), cmul(v1, entry.matrix[2][1])),
                        cadd(cmul(v2, entry.matrix[2][2]), cmul(v3, entry.matrix[2][3])));
        st.v[i3] = cadd(cadd(cmul(v0, entry.matrix[3][0]), cmul(v1, entry.matrix[3][1])),
                        cadd(cmul(v2, entry.matrix[3][2]), cmul(v3, entry.matrix[3][3])));
    }
}
)HIP";

static const char* kExpandPlain = R"HIP(
DEV static void expand_plain(ShotState& st) {
    uint64_t half = 1ULL << st.active_k;
    for (uint64_t i = 0; i < half; ++i) st.v[i + half] = st.v[i];
    st.active_k++;
}
)HIP";

static const char* kExpandT = R"HIP(
DEV static void expand_t(ShotState& st, uint32_t axis, bool dagger) {
    uint64_t half = 1ULL << st.active_k;
    bool px = bit_get(st.px, axis);
    double imag = dagger ? -kInvSqrt2 : kInvSqrt2;
    if (px) imag = -imag;
    GpuComplex phase = {(float)kInvSqrt2, (float)imag};
    for (uint64_t i = 0; i < half; ++i) st.v[i + half] = cmul(st.v[i], phase);
    st.active_k++;
}
)HIP";

static const char* kExpandRot = R"HIP(
DEV static void expand_rot(ShotState& st, uint32_t axis, double z_re, double z_im) {
    uint64_t half = 1ULL << st.active_k;
    bool px = bit_get(st.px, axis);
    double im = px ? -z_im : z_im;
    GpuComplex phase = {(float)z_re, (float)im};
    for (uint64_t i = 0; i < half; ++i) st.v[i + half] = cmul(st.v[i], phase);
    st.active_k++;
}
)HIP";

static const char* kMeasDormantRandom = R"HIP(
DEV static void meas_dormant_random(ShotState& st, Rng& rng, uint32_t axis,
                                    uint32_t classical_idx, bool sign) {
    uint8_t m_abs = rng.uniform() < 0.5 ? 0 : 1;
    bit_set(st.px, axis, m_abs != 0);
    bit_set(st.pz, axis, false);
    st.meas[classical_idx] = m_abs ^ (uint8_t)sign;
}
)HIP";

static const char* kMeasActiveDiag = R"HIP(
DEV static void meas_active_diagonal(ShotState& st, Rng& rng, uint32_t axis,
                                     uint32_t classical_idx, bool sign) {
    uint64_t half = 1ULL << (st.active_k - 1);
    bool px = bit_get(st.px, axis);
    double p0 = 0.0, p1 = 0.0;
    for (uint64_t i = 0; i < half; ++i) { p0 += cnorm(st.v[i]); p1 += cnorm(st.v[i + half]); }
    uint8_t b = sample_branch(rng, p0, p1, p0 + p1);
    uint8_t m_abs = b ^ (uint8_t)px;
    st.meas[classical_idx] = m_abs ^ (uint8_t)sign;
    if (b != 0) { for (uint64_t i = 0; i < half; ++i) st.v[i] = st.v[i + half]; }
    st.active_k--;
    bit_set(st.px, axis, m_abs != 0);
    bit_set(st.pz, axis, false);
}
)HIP";

static const char* kMeasActiveInterfere = R"HIP(
DEV static void meas_active_interfere(ShotState& st, Rng& rng, uint32_t axis,
                                      uint32_t classical_idx, bool sign) {
    uint64_t half = 1ULL << (st.active_k - 1);
    bool pz = bit_get(st.pz, axis);
    double p_plus = 0.0, p_minus = 0.0;
    for (uint64_t i = 0; i < half; ++i) {
        GpuComplex sum = cadd(st.v[i], st.v[i + half]);
        GpuComplex diff = csub(st.v[i], st.v[i + half]);
        p_plus += cnorm(sum); p_minus += cnorm(diff);
    }
    uint8_t b_x = sample_branch(rng, p_plus, p_minus, p_plus + p_minus);
    uint8_t m_abs = b_x ^ (uint8_t)pz;
    st.meas[classical_idx] = m_abs ^ (uint8_t)sign;
    for (uint64_t i = 0; i < half; ++i) {
        GpuComplex folded = (b_x == 0) ? cadd(st.v[i], st.v[i + half])
                                       : csub(st.v[i], st.v[i + half]);
        st.v[i] = cscale(folded, kInvSqrt2);
    }
    st.active_k--;
    bit_set(st.px, axis, m_abs != 0);
    bit_set(st.pz, axis, false);
}
)HIP";

static const char* kSwapMeasInterfere = R"HIP(
DEV static void swap_meas_interfere(ShotState& st, Rng& rng, uint32_t from, uint32_t to,
                                    uint32_t classical_idx, bool sign) {
    if (from == to) { meas_active_interfere(st, rng, to, classical_idx, sign); return; }
    frame_swap(st, from, to);
    bool pz = bit_get(st.pz, to);
    uint64_t half = 1ULL << to;
    uint64_t f_bit = 1ULL << from;
    double p_plus = 0.0, p_minus = 0.0;
    for (uint64_t idx = 0; idx < half; ++idx) {
        uint64_t b_f = (idx >> from) & 1ULL;
        uint64_t base = (idx & ~f_bit) | (b_f << to);
        GpuComplex sum = cadd(st.v[base], st.v[base | f_bit]);
        GpuComplex diff = csub(st.v[base], st.v[base | f_bit]);
        p_plus += cnorm(sum); p_minus += cnorm(diff);
    }
    uint8_t b_x = sample_branch(rng, p_plus, p_minus, p_plus + p_minus);
    for (uint64_t idx = 0; idx < half; ++idx) {
        uint64_t b_f = (idx >> from) & 1ULL;
        uint64_t base = (idx & ~f_bit) | (b_f << to);
        st.v[idx] = cscale((b_x == 0) ? cadd(st.v[base], st.v[base | f_bit])
                                      : csub(st.v[base], st.v[base | f_bit]),
                           kInvSqrt2);
    }
    st.active_k--;
    uint8_t m_abs = b_x ^ (uint8_t)pz;
    st.meas[classical_idx] = m_abs ^ (uint8_t)sign;
    bit_set(st.px, to, m_abs != 0);
    bit_set(st.pz, to, false);
}
)HIP";

static const char* kExecExpVal = R"HIP(
DEV static void exec_exp_val_compiled(ShotState& st, const GpuExpValMask* masks,
                                      uint32_t cp_idx, uint32_t ev_idx) {
    const GpuExpValMask& mask = masks[cp_idx];
    bool frame_sign = mask.sign != 0;
    int parity = POPCOUNT64(mask.x[0] & st.pz[0]) + POPCOUNT64(mask.x[1] & st.pz[1])
               + POPCOUNT64(mask.z[0] & st.px[0]) + POPCOUNT64(mask.z[1] & st.px[1]);
    if (parity & 1) frame_sign = !frame_sign;
    uint32_t k = st.active_k;
    uint64_t active_mask_w0 = (k >= 64) ? ~uint64_t{0} : ((k == 0) ? 0 : ((uint64_t{1} << k) - 1));
    uint64_t dormant_x = mask.x[0] & ~active_mask_w0;
    if (k < 64) dormant_x |= mask.x[1];
    if (dormant_x != 0) { st.exp_vals[ev_idx] = 0.0; return; }
    uint64_t x_active = mask.x[0] & active_mask_w0;
    uint64_t z_active = mask.z[0] & active_mask_w0;
    uint32_t yz_count = POPCOUNT64(x_active & z_active);
    double phase_re = 0.0, phase_im = 0.0;
    switch (yz_count & 3) {
        case 0: phase_re = 1.0; break;
        case 1: phase_im = 1.0; break;
        case 2: phase_re = -1.0; break;
        case 3: phase_im = -1.0; break;
    }
    if (frame_sign) { phase_re = -phase_re; phase_im = -phase_im; }
    uint64_t dim = uint64_t{1} << k;
    double num_re = 0.0, num_im = 0.0, denom = 0.0;
    for (uint64_t j = 0; j < dim; ++j) {
        denom += cnorm(st.v[j]);
        uint64_t j_xor = j ^ x_active;
        double z_sign = (POPCOUNT64(j & z_active) & 1) ? -1.0 : 1.0;
        double a_re = (double)st.v[j_xor].re;
        double a_im = -(double)st.v[j_xor].im;
        num_re += (a_re * z_sign * st.v[j].re - a_im * z_sign * st.v[j].im);
        num_im += (a_re * z_sign * st.v[j].im + a_im * z_sign * st.v[j].re);
    }
    double result_re = num_re * phase_re - num_im * phase_im;
    st.exp_vals[ev_idx] = (denom > 0.0) ? result_re / denom : 0.0;
}
)HIP";

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

static void emit_constant_pool(std::ostringstream& out, const FlattenedProgram& flat,
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
static void emit_preamble(std::ostringstream& out, const UsedFunctions& uf) {
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
static void emit_needed_functions(std::ostringstream& out, const UsedFunctions& uf) {
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

// -----------------------------------------------------------------------
// Emit straight-line instructions
// -----------------------------------------------------------------------
static void emit_instructions(std::ostringstream& out, const FlattenedProgram& flat) {
    for (size_t pc = 0; pc < flat.instrs.size(); ++pc) {
        const GpuInstr& instr = flat.instrs[pc];
        auto op = static_cast<Op>(instr.opcode);
        bool sign = (instr.flags & kFlagSign) != 0;
        bool identity = (instr.flags & kFlagIdentity) != 0;
        bool expected_one = (instr.flags & kFlagExpectedOne) != 0;

        switch (op) {
            case Op::OP_FRAME_CNOT:
                out << "    frame_cnot(st, " << instr.axis_1 << "u, " << instr.axis_2 << "u);\n";
                break;
            case Op::OP_FRAME_CZ:
                out << "    frame_cz(st, " << instr.axis_1 << "u, " << instr.axis_2 << "u);\n";
                break;
            case Op::OP_FRAME_H:
                out << "    frame_h(st, " << instr.axis_1 << "u);\n";
                break;
            case Op::OP_FRAME_S:
            case Op::OP_FRAME_S_DAG:
                out << "    frame_s(st, " << instr.axis_1 << "u);\n";
                break;
            case Op::OP_FRAME_SWAP:
                out << "    frame_swap(st, " << instr.axis_1 << "u, " << instr.axis_2 << "u);\n";
                break;
            case Op::OP_ARRAY_CNOT:
                out << "    array_cnot(st, " << instr.axis_1 << "u, " << instr.axis_2 << "u);\n";
                break;
            case Op::OP_ARRAY_CZ:
                out << "    array_cz(st, " << instr.axis_1 << "u, " << instr.axis_2 << "u);\n";
                break;
            case Op::OP_ARRAY_SWAP:
                out << "    array_swap(st, " << instr.axis_1 << "u, " << instr.axis_2 << "u);\n";
                break;
            case Op::OP_ARRAY_MULTI_CNOT:
                out << "    array_multi_cnot(st, " << instr.axis_1
                    << "u, 0x" << std::hex << instr.mask << std::dec << "ULL);\n";
                break;
            case Op::OP_ARRAY_MULTI_CZ:
                out << "    array_multi_cz(st, " << instr.axis_1
                    << "u, 0x" << std::hex << instr.mask << std::dec << "ULL);\n";
                break;
            case Op::OP_ARRAY_H:
                out << "    array_h(st, " << instr.axis_1 << "u);\n";
                break;
            case Op::OP_ARRAY_S:
                out << "    array_s(st, " << instr.axis_1 << "u, false);\n";
                break;
            case Op::OP_ARRAY_S_DAG:
                out << "    array_s(st, " << instr.axis_1 << "u, true);\n";
                break;
            case Op::OP_ARRAY_T:
                out << "    array_t(st, " << instr.axis_1 << "u, false);\n";
                break;
            case Op::OP_ARRAY_T_DAG:
                out << "    array_t(st, " << instr.axis_1 << "u, true);\n";
                break;
            case Op::OP_ARRAY_ROT:
                out << "    array_rot(st, " << instr.axis_1 << "u, "
                    << hex_double(instr.weight_re) << ", " << hex_double(instr.weight_im) << ");\n";
                break;
            case Op::OP_ARRAY_U2:
                if (!flat.fused_u2.empty()) {
                    out << "    array_u2_compiled(st, " << instr.axis_1 << "u, kFusedU2, "
                        << instr.a << "u);\n";
                }
                break;
            case Op::OP_ARRAY_U4:
                if (!flat.fused_u4.empty()) {
                    out << "    array_u4_compiled(st, " << instr.axis_1 << "u, "
                        << instr.axis_2 << "u, kFusedU4, " << instr.a << "u);\n";
                }
                break;
            case Op::OP_EXPAND:
                out << "    expand_plain(st);\n";
                break;
            case Op::OP_EXPAND_T:
                out << "    expand_t(st, " << instr.axis_1 << "u, false);\n";
                break;
            case Op::OP_EXPAND_T_DAG:
                out << "    expand_t(st, " << instr.axis_1 << "u, true);\n";
                break;
            case Op::OP_EXPAND_ROT:
                out << "    expand_rot(st, " << instr.axis_1 << "u, "
                    << hex_double(instr.weight_re) << ", " << hex_double(instr.weight_im) << ");\n";
                break;
            case Op::OP_MEAS_DORMANT_STATIC:
                if (identity) {
                    out << "    st.meas[" << instr.a << "u] = " << (sign ? 1u : 0u) << "u;\n";
                } else {
                    out << "    st.meas[" << instr.a << "u] = "
                        << "(bit_get(st.px, " << instr.axis_1 << "u) ? 1u : 0u)"
                        << (sign ? " ^ 1u" : "") << ";\n";
                }
                break;
            case Op::OP_MEAS_DORMANT_RANDOM:
                out << "    meas_dormant_random(st, rng, " << instr.axis_1 << "u, "
                    << instr.a << "u, " << (sign ? "true" : "false") << ");\n";
                break;
            case Op::OP_MEAS_ACTIVE_DIAGONAL:
                out << "    meas_active_diagonal(st, rng, " << instr.axis_1 << "u, "
                    << instr.a << "u, " << (sign ? "true" : "false") << ");\n";
                break;
            case Op::OP_MEAS_ACTIVE_INTERFERE:
                out << "    meas_active_interfere(st, rng, " << instr.axis_1 << "u, "
                    << instr.a << "u, " << (sign ? "true" : "false") << ");\n";
                break;
            case Op::OP_SWAP_MEAS_INTERFERE:
                out << "    swap_meas_interfere(st, rng, " << instr.axis_1 << "u, "
                    << instr.axis_2 << "u, " << instr.a << "u, "
                    << (sign ? "true" : "false") << ");\n";
                break;
            case Op::OP_APPLY_PAULI:
                if (!flat.pauli_masks.empty()) {
                    out << "    apply_pauli(st, kPauliMasks, " << instr.a << "u, "
                        << instr.b << "u);\n";
                }
                break;
            case Op::OP_NOISE:
                out << "    if (st.next_noise_idx == " << instr.a << "u) {\n"
                    << "        const GpuNoiseSite* site = &kNoiseSites[" << instr.a << "u];\n"
                    << "        double roll = rng.uniform() * site->prob_sum;\n"
                    << "        double cumulative = 0.0;\n"
                    << "        for (uint32_t k = 0; k < site->count; ++k) {\n"
                    << "            const GpuChannel* ch = &kNoiseChannels[site->offset + k];\n"
                    << "            cumulative += ch->prob;\n"
                    << "            if (roll < cumulative) {\n"
                    << "                apply_pauli_to_frame(st, ch->x, ch->z);\n"
                    << "                break;\n"
                    << "            }\n"
                    << "        }\n"
                    << "        st.next_noise_idx = " << instr.a << "u + 1;\n"
                    << "        draw_next_noise_compiled(st, rng, kNoiseHazards, "
                    << flat.noise_sites.size() << "u);\n"
                    << "    }\n";
                break;
            case Op::OP_NOISE_BLOCK:
                {
                    uint32_t start = instr.a;
                    uint32_t end = start + instr.b;
                    out << "    {\n"
                        << "        uint32_t _noise_end = " << end << "u;\n"
                        << "        while (st.next_noise_idx >= " << start << "u && "
                        << "st.next_noise_idx < _noise_end) {\n"
                        << "            uint32_t _si = st.next_noise_idx;\n"
                        << "            const GpuNoiseSite* site = &kNoiseSites[_si];\n"
                        << "            double roll = rng.uniform() * site->prob_sum;\n"
                        << "            double cumulative = 0.0;\n"
                        << "            for (uint32_t k = 0; k < site->count; ++k) {\n"
                        << "                const GpuChannel* ch = &kNoiseChannels[site->offset + k];\n"
                        << "                cumulative += ch->prob;\n"
                        << "                if (roll < cumulative) {\n"
                        << "                    apply_pauli_to_frame(st, ch->x, ch->z);\n"
                        << "                    break;\n"
                        << "                }\n"
                        << "            }\n"
                        << "            st.next_noise_idx = _si + 1;\n"
                        << "            draw_next_noise_compiled(st, rng, kNoiseHazards, "
                        << flat.noise_sites.size() << "u);\n"
                        << "        }\n"
                        << "    }\n";
                }
                break;
            case Op::OP_READOUT_NOISE:
                out << "    if (rng.uniform() < kReadoutNoise[" << instr.a << "u].prob) {\n"
                    << "        st.meas[kReadoutNoise[" << instr.a << "u].meas_idx] ^= 1u;\n"
                    << "    }\n";
                break;
            case Op::OP_DETECTOR:
                // No-op at runtime
                break;
            case Op::OP_POSTSELECT:
                out << "    {\n";
                out << "        uint32_t _ps_start = kDetectorOffsets[" << instr.a << "u];\n";
                out << "        uint32_t _ps_end = kDetectorOffsets[" << instr.a << "u + 1u];\n";
                out << "        uint8_t _ps_parity = " << (expected_one ? 1u : 0u) << "u;\n";
                out << "        for (uint32_t i = _ps_start; i < _ps_end; ++i) {\n";
                out << "            _ps_parity ^= st.meas[kDetectorTargets[i]];\n";
                out << "        }\n";
                out << "        if (_ps_parity != 0) { st.discarded = true; }\n";
                out << "    }\n";
                out << "    if (st.discarded) goto done;\n";
                break;
            case Op::OP_OBSERVABLE:
                out << "    {\n";
                out << "        uint32_t _ob_start = kObservableOffsets[" << instr.a << "u];\n";
                out << "        uint32_t _ob_end = kObservableOffsets[" << instr.a << "u + 1u];\n";
                out << "        uint8_t _ob_parity = 0u;\n";
                out << "        for (uint32_t i = _ob_start; i < _ob_end; ++i) {\n";
                out << "            _ob_parity ^= st.meas[kObservableTargets[i]];\n";
                out << "        }\n";
                out << "        st.obs[" << instr.b << "u] ^= _ob_parity;\n";
                out << "    }\n";
                break;
            case Op::OP_EXP_VAL:
                if (!flat.exp_val_masks.empty()) {
                    out << "    exec_exp_val_compiled(st, kExpValMasks, " << instr.a << "u, "
                        << instr.b << "u);\n";
                }
                break;
            default:
                out << "    st.discarded = true; goto done; // unknown opcode "
                    << (unsigned)instr.opcode << "\n";
                break;
        }
    }
}

// -----------------------------------------------------------------------
// Warp-shuffle reduction for BlockCounts (GEAK OPT-1)
//
// Replaces the original shared-memory tree reduction with a two-phase
// approach:
//   Phase 1: Intra-wavefront reduction via __shfl_xor (AMD wavefront=64)
//   Phase 2: Inter-wavefront reduction via shared memory (4 values for
//            blockDim=256 = 4 wavefronts)
// -----------------------------------------------------------------------
static const char* kWarpShuffleReduction = R"HIP(
    // --- Warp-shuffle reduction (GEAK OPT-1) ---
    // Phase 1: intra-wavefront reduction via __shfl_xor (wavefront = 64)
    for (int offset = 32; offset > 0; offset >>= 1) {
        local_passed    += shfl_xor_i32((int)local_passed, offset);
        local_logical   += shfl_xor_i32((int)local_logical, offset);
    }
    {
        static LDS uint64_t wf_passed[4];
        static LDS uint64_t wf_logical[4];
        uint32_t warp_id = tid >> 6;
        uint32_t lane = tid & 63;
        if (lane == 0) {
            wf_passed[warp_id] = local_passed;
            wf_logical[warp_id] = local_logical;
        }
        BARRIER();
        // Phase 2: first 4 threads reduce the wavefront partials
        if (tid < 4) {
            local_passed = wf_passed[tid];
            local_logical = wf_logical[tid];
            for (int offset = 2; offset > 0; offset >>= 1) {
                local_passed  += shfl_xor_i32((int)local_passed, offset);
                local_logical += shfl_xor_i32((int)local_logical, offset);
            }
        }
    }
    // Observable reduction: per-observable warp shuffle
    for (uint32_t oi = 0; oi < num_observables; ++oi) {
        for (int offset = 32; offset > 0; offset >>= 1) {
            local_obs[oi] += shfl_xor_i32((int)local_obs[oi], offset);
        }
    }
    {
        static LDS uint64_t wf_obs[kMaxObs][4];
        uint32_t warp_id = tid >> 6;
        uint32_t lane = tid & 63;
        for (uint32_t oi = 0; oi < num_observables; ++oi) {
            if (lane == 0) wf_obs[oi][warp_id] = local_obs[oi];
        }
        BARRIER();
        if (tid < 4) {
            for (uint32_t oi = 0; oi < num_observables; ++oi) {
                local_obs[oi] = wf_obs[oi][tid];
                for (int offset = 2; offset > 0; offset >>= 1) {
                    local_obs[oi] += shfl_xor_i32((int)local_obs[oi], offset);
                }
            }
        }
    }
    // Exp-val reduction
    for (uint32_t ei = 0; ei < num_exp_vals; ++ei) {
        for (int offset = 32; offset > 0; offset >>= 1) {
            local_exp[ei] += shfl_xor_f64(local_exp[ei], offset);
        }
    }
    {
        static LDS double wf_exp[kMaxExpVals][4];
        uint32_t warp_id = tid >> 6;
        uint32_t lane = tid & 63;
        for (uint32_t ei = 0; ei < num_exp_vals; ++ei) {
            if (lane == 0) wf_exp[ei][warp_id] = local_exp[ei];
        }
        BARRIER();
        if (tid < 4) {
            for (uint32_t ei = 0; ei < num_exp_vals; ++ei) {
                local_exp[ei] = wf_exp[ei][tid];
                for (int offset = 2; offset > 0; offset >>= 1) {
                    local_exp[ei] += shfl_xor_f64(local_exp[ei], offset);
                }
            }
        }
    }
    if (tid == 0) {
        BlockCounts out;
        out.passed = local_passed;
        out.logical_errors = local_logical;
        for (uint32_t oi = 0; oi < kMaxObs; ++oi) {
            out.observable_ones[oi] = (oi < num_observables) ? local_obs[oi] : 0;
        }
        for (uint32_t ei = 0; ei < kMaxExpVals; ++ei) {
            out.exp_val_sums[ei] = (ei < num_exp_vals) ? local_exp[ei] : 0.0;
        }
        out.exp_val_count = local_passed;
        block_counts[BIDX] = out;
    }
)HIP";

}  // anonymous namespace

// -----------------------------------------------------------------------
// Public: generate_compiled_kernel
// -----------------------------------------------------------------------
std::string generate_compiled_kernel(const FlattenedProgram& flat) {
    UsedFunctions uf = analyze_used_functions(flat);
    bool has_noise = uf.noise || uf.noise_block;

    std::ostringstream out;

    // Emit conditional preamble
    emit_preamble(out, uf);

    // Emit constant pool (only arrays used by the circuit)
    emit_constant_pool(out, flat, uf);

    // Emit device functions (only those called by the circuit)
    emit_needed_functions(out, uf);

    // Emit the kernel function
    out << "\nextern \"C\" __global__ __attribute__((amdgpu_flat_work_group_size(1, 256))) void compiled_sample_kernel(\n"
        << "    uint64_t shot_offset, uint64_t shots, uint64_t seed,\n"
        << "    BlockCounts* block_counts,\n"
        << "    uint32_t num_observables, uint32_t num_exp_vals\n"
        << ") {\n"
        << "    uint32_t tid = TIDX;\n"
        << "    uint64_t batch_shot_id = (uint64_t)BIDX * BSIZE + tid;\n"
        << "\n"
        << "    uint64_t local_passed = 0;\n"
        << "    uint64_t local_logical = 0;\n"
        << "    uint64_t local_obs[kMaxObs];\n"
        << "    double local_exp[kMaxExpVals];\n"
        << "    for (uint32_t i = 0; i < kMaxObs; ++i) local_obs[i] = 0;\n"
        << "    for (uint32_t i = 0; i < kMaxExpVals; ++i) local_exp[i] = 0.0;\n"
        << "\n"
        << "    if (batch_shot_id < shots) {\n"
        << "        uint64_t shot_id = shot_offset + batch_shot_id;\n";

    // Only emit RNG init if needed
    if (uf.needs_rng) {
        out << "        Rng rng;\n"
            << "        rng.seed(seed, shot_id);\n";
    }

    out << "\n"
        << "        ShotState st;\n"
        << "        st.px[0] = 0; st.px[1] = 0;\n"
        << "        st.pz[0] = 0; st.pz[1] = 0;\n"
        << "        st.active_k = 0;\n"
        << "        st.next_noise_idx = 0;\n"
        << "        st.discarded = false;\n"
        << "        st.v[0] = {1.0f, 0.0f};\n"
        << "        for (uint32_t i = 0; i < num_observables; ++i) st.obs[i] = 0;\n"
        << "        for (uint32_t i = 0; i < num_exp_vals; ++i) st.exp_vals[i] = 0.0;\n"
        << "\n";

    // Initial noise draw (only if circuit has noise)
    if (has_noise) {
        out << "        // Initial noise draw\n"
            << "        draw_next_noise_compiled(st, rng, kNoiseHazards, "
            << flat.noise_sites.size() << "u);\n\n";
    }

    // Straight-line instruction sequence
    out << "        // --- Begin instruction sequence (" << flat.instrs.size() << " ops) ---\n";
    emit_instructions(out, flat);
    out << "        // --- End instruction sequence ---\n";

    // Accumulate results
    out << "\n"
        << "    done:\n"
        << "        if (!st.discarded) {\n"
        << "            local_passed = 1;\n"
        << "            bool any_logical = false;\n"
        << "            for (uint32_t i = 0; i < num_observables; ++i) {\n"
        << "                uint8_t val = st.obs[i];\n";

    if (uf.observable) {
        out << "                if (kExpectedObservables[i] != 0) val ^= 1u;\n";
    }

    out << "                if (val != 0) {\n"
        << "                    local_obs[i] = 1;\n"
        << "                    any_logical = true;\n"
        << "                }\n"
        << "            }\n"
        << "            local_logical = any_logical ? 1 : 0;\n"
        << "            for (uint32_t i = 0; i < num_exp_vals; ++i) {\n"
        << "                local_exp[i] = st.exp_vals[i];\n"
        << "            }\n"
        << "        }\n"
        << "    } // end if (batch_shot_id < shots)\n"
        << "\n";

    // Warp-shuffle reduction (GEAK OPT-1)
    out << kWarpShuffleReduction;

    out << "}\n";

    return out.str();
}

// -----------------------------------------------------------------------
// analyze_pipeline_opportunities
//
// Two consecutive array-sweep instructions are independent when the set of
// amplitude indices they touch is disjoint.  For a single-qubit gate on axis
// A, the indices touched are those with bit A at a specific value — i.e. they
// span ALL amplitudes with any pattern for other bits.  Two single-qubit gates
// on different axes are independent.  Two-qubit gates on axes (A, B) vs (C, D)
// are independent iff {A, B} ∩ {C, D} = ∅.
//
// We encode the "qubit footprint" of an instruction as a bitmask over axis
// indices.  Zero footprint means the instruction doesn't touch amplitudes at
// all (frame-only or classical).
// -----------------------------------------------------------------------
static uint64_t instr_amplitude_footprint(const GpuInstr& instr) {
    auto op = static_cast<Op>(instr.opcode);
    switch (op) {
        // Two-qubit array ops: footprint is both axes
        case Op::OP_ARRAY_CNOT:
        case Op::OP_ARRAY_CZ:
        case Op::OP_ARRAY_SWAP:
            return (1ULL << instr.axis_1) | (1ULL << instr.axis_2);

        // Multi-qubit: footprint is axis_1 plus ctrl_mask
        case Op::OP_ARRAY_MULTI_CNOT:
        case Op::OP_ARRAY_MULTI_CZ:
            return instr.mask | (1ULL << instr.axis_1);

        // One-qubit array ops: footprint is axis_1 only
        case Op::OP_ARRAY_H:
        case Op::OP_ARRAY_S:
        case Op::OP_ARRAY_S_DAG:
        case Op::OP_ARRAY_T:
        case Op::OP_ARRAY_T_DAG:
        case Op::OP_ARRAY_ROT:
        case Op::OP_ARRAY_U2:
            return 1ULL << instr.axis_1;

        // Two-qubit unitary: both axes
        case Op::OP_ARRAY_U4:
            return (1ULL << instr.axis_1) | (1ULL << instr.axis_2);

        // Expand ops: touch all current amplitudes → footprint = all bits
        case Op::OP_EXPAND:
        case Op::OP_EXPAND_T:
        case Op::OP_EXPAND_T_DAG:
        case Op::OP_EXPAND_ROT:
            return ~uint64_t{0};

        // Measurement: collapses one axis — not safe to pipeline with anything
        case Op::OP_MEAS_ACTIVE_DIAGONAL:
        case Op::OP_MEAS_ACTIVE_INTERFERE:
        case Op::OP_SWAP_MEAS_INTERFERE:
            return ~uint64_t{0};

        // Frame-only, classical, dormant meas, noise, postselect: no amplitude access
        default:
            return 0;
    }
}

std::vector<PipelineOp> analyze_pipeline_opportunities(const FlattenedProgram& flat) {
    std::vector<PipelineOp> result;
    size_t n = flat.instrs.size();
    for (size_t pc = 0; pc + 1 < n; ++pc) {
        uint64_t fp0 = instr_amplitude_footprint(flat.instrs[pc]);
        uint64_t fp1 = instr_amplitude_footprint(flat.instrs[pc + 1]);
        // Both must have a non-trivial footprint (amplitude-touching) and be disjoint
        if (fp0 != 0 && fp1 != 0 && (fp0 & fp1) == 0) {
            result.push_back({pc, pc + 1});
        }
    }
    return result;
}

// -----------------------------------------------------------------------
// Cooperative (LDS-tier) array sweep device functions
//
// These are variants of the register-tier device functions that operate on
// a __shared__ GpuComplex v[] array with 256 threads cooperating per shot.
// The v[] size is statically baked as 1 << peak_rank.
//
// Key differences from register-tier:
//  - v[] is in __shared__ memory, accessed via LDS load/store
//  - Thread i handles amplitude indices i, i+BSIZE, i+2*BSIZE, ...
//  - BARRIER() required after writes that other threads read
//  - LDS bank conflict avoidance: v[] padded to (N + 1) to stride pad
// -----------------------------------------------------------------------

// Coop preamble: cooperative state struct and load/store helpers
static const char* kPreambleCoopState = R"HIP(
// LDS amplitude array size = 1 << PEAK_RANK (baked at compile time)
// Padded by 1 per 32-element bank group to avoid bank conflicts.
// __shared__ GpuComplex v[COOP_AMPLITUDES]; // declared in kernel, size baked

struct CoopShotStateCompiled {
GpuComplex* v;        // LDS amplitude array
uint64_t* px;         // Pauli-X frame (2 words)
uint64_t* pz;         // Pauli-Z frame (2 words)
uint32_t* active_k;
uint32_t* next_noise_idx;
uint8_t* meas;
uint8_t* obs;
uint8_t* discarded;
double* red0;         // reduction scratch [BSIZE]
double* red1;
};

// Bit-cast float to uint32 without UB (LLVM optimizes to a register move)
DEV static inline __attribute__((always_inline)) uint32_t float_as_uint(float x) {
    uint32_t tmp;
    __builtin_memcpy(&tmp, &x, sizeof(tmp));
    return tmp;
}
DEV static inline __attribute__((always_inline)) void store_complex64_lds(GpuComplex* v, uint32_t idx, GpuComplex c) {
    uint64_t packed = (uint64_t)float_as_uint(c.re) | ((uint64_t)float_as_uint(c.im) << 32);
    *reinterpret_cast<uint64_t*>(v + idx) = packed;
}
)HIP";

// Cooperative warp-shuffle + shared-memory reduction for double (used in coop measurements)
// Two-value reduce matching SVM coop_reduce2 exactly for bit-identical results
static const char* kCoopReduceDouble = R"HIP(
DEV static double coop_reduce_sum(double val, double* smem) {
    uint32_t tid = TIDX;
    uint32_t lane = tid & 63u;
    uint32_t warp_id = tid >> 6;
    for (int off = 32; off > 0; off >>= 1) val += shfl_xor_f64(val, off);
    if (lane == 0) smem[warp_id] = val;
    BARRIER();
    if (tid < 4) {
        val = smem[tid];
        for (int off = 2; off > 0; off >>= 1) val += shfl_xor_f64(val, off);
        if (tid == 0) smem[0] = val;
    }
    BARRIER();
    return smem[0];
}
DEV static void coop_reduce2(double local0, double local1,
                              double* red0, double* red1,
                              double& out0, double& out1) {
    uint32_t tid = TIDX;
    uint32_t lane = tid & 63u;
    uint32_t warp_id = tid >> 6;
    for (int off = 32; off > 0; off >>= 1) {
        local0 += shfl_xor_f64(local0, off);
        local1 += shfl_xor_f64(local1, off);
    }
    if (lane == 0) {
        red0[warp_id] = local0;
        red1[warp_id] = local1;
    }
    BARRIER();
    if (tid < 4) {
        local0 = red0[tid];
        local1 = red1[tid];
        for (int off = 2; off > 0; off >>= 1) {
            local0 += shfl_xor_f64(local0, off);
            local1 += shfl_xor_f64(local1, off);
        }
    }
    if (tid == 0) {
        red0[0] = local0;
        red1[0] = local1;
    }
    BARRIER();
    out0 = red0[0];
    out1 = red1[0];
}
)HIP";

// Cooperative single-qubit H: threads stride over amplitude pairs
static const char* kCoopArrayH = R"HIP(
DEV static void coop_array_h(GpuComplex* v, uint32_t active_k, uint32_t axis) {
    uint64_t axis_bit = 1ULL << axis;
    uint64_t iters = 1ULL << (active_k - 1);
    for (uint64_t i = TIDX; i < iters; i += BSIZE) {
        uint64_t idx0 = scatter_bits_1(i, axis);
        uint64_t idx1 = idx0 | axis_bit;
        GpuComplex a = v[idx0], b = v[idx1];
        store_complex64_lds(v, idx0, cscale(cadd(a, b), kInvSqrt2));
        store_complex64_lds(v, idx1, cscale(csub(a, b), kInvSqrt2));
    }
    BARRIER();
}
)HIP";

static const char* kCoopArrayCnot = R"HIP(
DEV static void coop_array_cnot(GpuComplex* v, uint64_t* px, uint64_t* pz,
                                uint32_t active_k, uint32_t c, uint32_t t) {
    uint64_t c_bit = 1ULL << c, t_bit = 1ULL << t;
    uint64_t iters = 1ULL << (active_k - 2);
    for (uint64_t i = TIDX; i < iters; i += BSIZE) {
        uint64_t base = scatter_bits_2(i, c, t) | c_bit;
        GpuComplex tmp = v[base];
        store_complex64_lds(v, base, v[base | t_bit]);
        store_complex64_lds(v, base | t_bit, tmp);
    }
    BARRIER();
    if (TIDX == 0) frame_cnot_raw(px, pz, c, t);
    BARRIER();
}
)HIP";

static const char* kCoopArrayCz = R"HIP(
DEV static void coop_array_cz(GpuComplex* v, uint64_t* px, uint64_t* pz,
                               uint32_t active_k, uint32_t a, uint32_t b) {
    uint64_t both_bits = (1ULL << a) | (1ULL << b);
    uint64_t iters = 1ULL << (active_k - 2);
    for (uint64_t i = TIDX; i < iters; i += BSIZE) {
        uint64_t idx = scatter_bits_2(i, a, b) | both_bits;
        GpuComplex c = v[idx];
        store_complex64_lds(v, idx, {-c.re, -c.im});
    }
    BARRIER();
    if (TIDX == 0) frame_cz_raw(px, pz, a, b);
    BARRIER();
}
)HIP";

static const char* kCoopArrayS = R"HIP(
DEV static void coop_array_s(GpuComplex* v, uint64_t* px, uint64_t* pz,
                              uint32_t active_k, uint32_t axis, bool dagger) {
    GpuComplex phase = dagger ? GpuComplex{0.0f, -1.0f} : GpuComplex{0.0f, 1.0f};
    uint64_t axis_bit = 1ULL << axis;
    uint64_t iters = 1ULL << (active_k - 1);
    for (uint64_t i = TIDX; i < iters; i += BSIZE) {
        uint64_t idx = scatter_bits_1(i, axis) | axis_bit;
        store_complex64_lds(v, idx, cmul(v[idx], phase));
    }
    BARRIER();
    if (TIDX == 0) frame_s_raw(px, pz, axis);
    BARRIER();
}
)HIP";

static const char* kCoopArrayT = R"HIP(
DEV static void coop_array_t(GpuComplex* v, const uint64_t* px, uint32_t active_k,
                              uint32_t axis, bool dagger) {
    if (axis >= active_k) return;
    bool pxv = bit_get(px, axis);
    double imag = dagger ? -kInvSqrt2 : kInvSqrt2;
    if (pxv) imag = -imag;
    GpuComplex phase = {(float)kInvSqrt2, (float)imag};
    uint64_t axis_bit = 1ULL << axis;
    uint64_t iters = 1ULL << (active_k - 1);
    for (uint64_t i = TIDX; i < iters; i += BSIZE) {
        uint64_t idx = scatter_bits_1(i, axis) | axis_bit;
        store_complex64_lds(v, idx, cmul(v[idx], phase));
    }
    BARRIER();
}
)HIP";

static const char* kCoopArrayU2 = R"HIP(
DEV static void coop_array_u2(GpuComplex* v, uint64_t* px, uint64_t* pz,
                               uint32_t active_k, uint32_t axis,
                               const GpuFusedU2Entry* table, uint32_t cp_idx) {
    const GpuFusedU2Entry& entry = table[cp_idx];
    uint8_t in_state = (bit_get(pz, axis) ? 2 : 0) | (bit_get(px, axis) ? 1 : 0);
    uint8_t out = entry.out_states[in_state];
    const GpuComplex* mat = entry.matrices[in_state];
    if (TIDX == 0) {
        bit_set(px, axis, (out & 1) != 0);
        bit_set(pz, axis, (out & 2) != 0);
    }
    BARRIER();
    if (axis >= active_k) return;
    uint64_t axis_bit = 1ULL << axis;
    uint64_t iters = 1ULL << (active_k - 1);
    for (uint64_t i = TIDX; i < iters; i += BSIZE) {
        uint64_t idx0 = scatter_bits_1(i, axis);
        uint64_t idx1 = idx0 | axis_bit;
        GpuComplex a = v[idx0], b = v[idx1];
        store_complex64_lds(v, idx0, cadd(cmul(a, mat[0]), cmul(b, mat[1])));
        store_complex64_lds(v, idx1, cadd(cmul(a, mat[2]), cmul(b, mat[3])));
    }
    BARRIER();
}
)HIP";

static const char* kCoopArrayU4 = R"HIP(
DEV static void coop_array_u4(GpuComplex* v, uint64_t* px, uint64_t* pz,
                               uint32_t active_k, uint32_t axis_lo, uint32_t axis_hi,
                               const GpuFusedU4Entry* table, uint32_t cp_idx) {
    const GpuFusedU4Entry& node = table[cp_idx];
    uint8_t px_lo = bit_get(px, axis_lo) ? 1 : 0, pz_lo = bit_get(pz, axis_lo) ? 1 : 0;
    uint8_t px_hi = bit_get(px, axis_hi) ? 1 : 0, pz_hi = bit_get(pz, axis_hi) ? 1 : 0;
    uint8_t in_state = (pz_hi << 3) | (px_hi << 2) | (pz_lo << 1) | px_lo;
    const GpuFusedU4Entry::Entry& entry = node.entries[in_state];
    uint8_t out = entry.out_state;
    if (TIDX == 0) {
        bit_set(px, axis_lo, (out & 1) != 0); bit_set(pz, axis_lo, (out & 2) != 0);
        bit_set(px, axis_hi, (out & 4) != 0); bit_set(pz, axis_hi, (out & 8) != 0);
    }
    BARRIER();
    if (axis_hi >= active_k) return;
    uint64_t lo_bit = 1ULL << axis_lo, hi_bit = 1ULL << axis_hi;
    uint64_t iters = 1ULL << (active_k - 2);
    for (uint64_t i = TIDX; i < iters; i += BSIZE) {
        uint64_t base = scatter_bits_2(i, axis_lo, axis_hi);
        GpuComplex v0 = v[base], v1 = v[base|lo_bit], v2 = v[base|hi_bit], v3 = v[base|lo_bit|hi_bit];
        store_complex64_lds(v, base, cadd(cadd(cmul(v0,entry.matrix[0][0]),cmul(v1,entry.matrix[0][1])),cadd(cmul(v2,entry.matrix[0][2]),cmul(v3,entry.matrix[0][3]))));
        store_complex64_lds(v, base|lo_bit, cadd(cadd(cmul(v0,entry.matrix[1][0]),cmul(v1,entry.matrix[1][1])),cadd(cmul(v2,entry.matrix[1][2]),cmul(v3,entry.matrix[1][3]))));
        store_complex64_lds(v, base|hi_bit, cadd(cadd(cmul(v0,entry.matrix[2][0]),cmul(v1,entry.matrix[2][1])),cadd(cmul(v2,entry.matrix[2][2]),cmul(v3,entry.matrix[2][3]))));
        store_complex64_lds(v, base|lo_bit|hi_bit, cadd(cadd(cmul(v0,entry.matrix[3][0]),cmul(v1,entry.matrix[3][1])),cadd(cmul(v2,entry.matrix[3][2]),cmul(v3,entry.matrix[3][3]))));
    }
    BARRIER();
}
)HIP";

// Cooperative frame-only helpers (used by emit_coop_instructions)
static const char* kCoopFrameRaw = R"HIP(
DEV static inline __attribute__((always_inline)) void frame_cnot_raw(uint64_t* px, uint64_t* pz, uint32_t c, uint32_t t) {
    bool px_c = bit_get(px, c), pz_t = bit_get(pz, t);
    bit_xor(px, t, px_c); bit_xor(pz, c, pz_t);
}
DEV static inline __attribute__((always_inline)) void frame_cz_raw(uint64_t* px, uint64_t* pz, uint32_t a, uint32_t b) {
    bool px_a = bit_get(px, a), px_b = bit_get(px, b);
    bit_xor(pz, b, px_a); bit_xor(pz, a, px_b);
}
DEV static inline __attribute__((always_inline)) void frame_h_raw(uint64_t* px, uint64_t* pz, uint32_t axis) {
    bool pxv = bit_get(px, axis), pzv = bit_get(pz, axis);
    bit_set(px, axis, pzv); bit_set(pz, axis, pxv);
}
DEV static inline __attribute__((always_inline)) void frame_s_raw(uint64_t* px, uint64_t* pz, uint32_t axis) {
    bit_xor(pz, axis, bit_get(px, axis));
}
DEV static inline __attribute__((always_inline)) void frame_swap_raw(uint64_t* px, uint64_t* pz, uint32_t a, uint32_t b) {
    bit_swap(px, a, px, b); bit_swap(pz, a, pz, b);
}
)HIP";

// Cooperative measurement: active-diagonal (collapses one axis, uses shared reduction)
static const char* kCoopMeasActiveDiag = R"HIP(
DEV static void coop_meas_active_diagonal(GpuComplex* v, uint64_t* px, uint64_t* pz,
                                          uint32_t* active_k_ptr, uint8_t* meas,
                                          double* red0, double* red1,
                                          Rng& rng, uint32_t axis, uint32_t classical_idx,
                                          bool sign) {
    uint32_t active_k = *active_k_ptr;
    bool pxv = bit_get(px, axis);
    uint64_t half = 1ULL << (active_k - 1);
    double p0_local = 0.0, p1_local = 0.0;
    for (uint64_t i = TIDX; i < half; i += BSIZE) {
        p0_local += cnorm(v[i]);
        p1_local += cnorm(v[i + half]);
    }
    double p0_out = 0.0, p1_out = 0.0;
    coop_reduce2(p0_local, p1_local, red0, red1, p0_out, p1_out);

    LDS uint8_t branch_val;
    if (TIDX == 0) {
        branch_val = sample_branch(rng, p0_out, p1_out, p0_out + p1_out);
    }
    BARRIER();
    uint8_t b = branch_val;

    if (b != 0) {
        for (uint64_t i = TIDX; i < half; i += BSIZE)
            store_complex64_lds(v, i, v[i + half]);
    }
    BARRIER();
    if (TIDX == 0) {
        uint8_t m_abs = b ^ (uint8_t)pxv;
        meas[classical_idx] = m_abs ^ (uint8_t)sign;
        (*active_k_ptr)--;
        bit_set(px, axis, m_abs != 0);
        bit_set(pz, axis, false);
    }
    BARRIER();
}
)HIP";

static const char* kCoopExpandPlain = R"HIP(
DEV static void coop_expand_plain(GpuComplex* v, uint32_t* active_k_ptr) {
    uint32_t k = *active_k_ptr;
    uint64_t half = 1ULL << k;
    for (uint64_t i = TIDX; i < half; i += BSIZE)
        store_complex64_lds(v, i + half, v[i]);
    BARRIER();
    if (TIDX == 0) (*active_k_ptr)++;
    BARRIER();
}
)HIP";

static const char* kCoopExpandT = R"HIP(
DEV static void coop_expand_t(GpuComplex* v, const uint64_t* px,
                               uint32_t* active_k_ptr, uint32_t axis, bool dagger) {
    uint32_t k = *active_k_ptr;
    uint64_t half = 1ULL << k;
    bool pxv = bit_get(px, axis);
    double imag = dagger ? -kInvSqrt2 : kInvSqrt2;
    if (pxv) imag = -imag;
    GpuComplex phase = {(float)kInvSqrt2, (float)imag};
    for (uint64_t i = TIDX; i < half; i += BSIZE)
        store_complex64_lds(v, i + half, cmul(v[i], phase));
    BARRIER();
    if (TIDX == 0) (*active_k_ptr)++;
    BARRIER();
}
)HIP";

// Coop-compatible draw_next_noise (uses raw pointers instead of ShotState)
static const char* kCoopNoiseHelper = R"HIP(
DEV static void coop_draw_next_noise(uint32_t* next_noise_idx, Rng& rng,
                                     const double* noise_hazards,
                                     uint32_t num_noise_sites) {
    if (num_noise_sites == 0 || *next_noise_idx >= num_noise_sites) {
        *next_noise_idx = 0xffffffffu;
        return;
    }
    double current_hazard = (*next_noise_idx == 0) ? 0.0
                            : noise_hazards[*next_noise_idx - 1];
    double u = rng.uniform();
    double gap = -log(1.0 - u);
    double target = current_hazard + gap;
    uint32_t lo = 0, hi = num_noise_sites;
    while (lo < hi) {
        uint32_t mid = lo + ((hi - lo) >> 1);
        if (noise_hazards[mid] <= target) lo = mid + 1; else hi = mid;
    }
    *next_noise_idx = (lo >= num_noise_sites) ? 0xffffffffu : lo;
}
)HIP";

// -----------------------------------------------------------------------
// New cooperative device functions for ops previously unsupported in coop path
// -----------------------------------------------------------------------

static const char* kCoopArraySwap = R"HIP(
DEV static void coop_array_swap(GpuComplex* v, uint64_t* px, uint64_t* pz,
                                uint32_t active_k, uint32_t a, uint32_t b) {
    uint64_t a_bit = 1ULL << a;
    uint64_t b_bit = 1ULL << b;
    uint64_t iters = 1ULL << (active_k - 2);
    for (uint64_t i = TIDX; i < iters; i += BSIZE) {
        uint64_t base = scatter_bits_2(i, a, b);
        GpuComplex tmp_a = v[base | a_bit];
        GpuComplex tmp_b = v[base | b_bit];
        store_complex64_lds(v, base | a_bit, tmp_b);
        store_complex64_lds(v, base | b_bit, tmp_a);
    }
    BARRIER();
    if (TIDX == 0) {
        bit_swap(px, a, px, b);
        bit_swap(pz, a, pz, b);
    }
    BARRIER();
}
)HIP";

static const char* kCoopArrayMultiCnot = R"HIP(
DEV static void coop_array_multi_cnot(GpuComplex* v, uint64_t* px, uint64_t* pz,
                                      uint32_t active_k, uint32_t target,
                                      uint64_t ctrl_mask) {
    uint64_t t_bit = 1ULL << target;
    uint64_t half = 1ULL << (active_k - 1);
    for (uint64_t idx = TIDX; idx < half; idx += BSIZE) {
        uint64_t actual = scatter_bits_1(idx, target);
        bool parity = (POPCOUNT64(actual & ctrl_mask) & 1) != 0;
        if (parity) {
            GpuComplex tmp  = v[actual];
            GpuComplex tmp2 = v[actual | t_bit];
            store_complex64_lds(v, actual,         tmp2);
            store_complex64_lds(v, actual | t_bit, tmp);
        }
    }
    BARRIER();
    if (TIDX == 0) {
        for (uint32_t c = 0; c < active_k; ++c) {
            if ((ctrl_mask >> c) & 1ULL) {
                bool px_c = bit_get(px, c);
                bool pz_t = bit_get(pz, target);
                bit_xor(px, target, px_c);
                bit_xor(pz, c,      pz_t);
            }
        }
    }
    BARRIER();
}
)HIP";

static const char* kCoopArrayMultiCz = R"HIP(
DEV static void coop_array_multi_cz(GpuComplex* v, uint64_t* px, uint64_t* pz,
                                    uint32_t active_k, uint32_t control,
                                    uint64_t target_mask) {
    uint64_t c_bit = 1ULL << control;
    uint64_t half = 1ULL << (active_k - 1);
    for (uint64_t idx = TIDX; idx < half; idx += BSIZE) {
        uint64_t actual = scatter_bits_1(idx, control) | c_bit;
        bool negate = (POPCOUNT64(actual & target_mask) & 1) != 0;
        if (negate) {
            GpuComplex val = v[actual];
            store_complex64_lds(v, actual, {-val.re, -val.im});
        }
    }
    BARRIER();
    if (TIDX == 0) {
        for (uint32_t t = 0; t < active_k; ++t) {
            if ((target_mask >> t) & 1ULL) {
                bool px_c = bit_get(px, control);
                bool px_t = bit_get(px, t);
                bit_xor(pz, t,       px_c);
                bit_xor(pz, control, px_t);
            }
        }
    }
    BARRIER();
}
)HIP";

static const char* kCoopArrayRot = R"HIP(
DEV static void coop_array_rot(GpuComplex* v, const uint64_t* px, uint32_t active_k,
                                uint32_t axis, double z_re, double z_im) {
    if (axis >= active_k) { BARRIER(); return; }
    bool pxv = bit_get(px, axis);
    double im = pxv ? -z_im : z_im;
    GpuComplex phase = {(float)z_re, (float)im};
    uint64_t axis_bit = 1ULL << axis;
    uint64_t iters = 1ULL << (active_k - 1);
    for (uint64_t i = TIDX; i < iters; i += BSIZE) {
        uint64_t idx = scatter_bits_1(i, axis) | axis_bit;
        store_complex64_lds(v, idx, cmul(v[idx], phase));
    }
    BARRIER();
}
)HIP";

static const char* kCoopExpandRot = R"HIP(
DEV static void coop_expand_rot(GpuComplex* v, const uint64_t* px,
                                 uint32_t* active_k_ptr, uint32_t axis,
                                 double z_re, double z_im) {
    uint32_t k = *active_k_ptr;
    uint64_t half = 1ULL << k;
    bool pxv = bit_get(px, axis);
    double im = pxv ? -z_im : z_im;
    GpuComplex phase = {(float)z_re, (float)im};
    for (uint64_t i = TIDX; i < half; i += BSIZE)
        store_complex64_lds(v, i + half, cmul(v[i], phase));
    BARRIER();
    if (TIDX == 0) (*active_k_ptr)++;
    BARRIER();
}
)HIP";

static const char* kCoopMeasActiveInterfere = R"HIP(
DEV static void coop_meas_active_interfere(GpuComplex* v, uint64_t* px, uint64_t* pz,
                                           uint32_t* active_k_ptr, uint8_t* meas,
                                           double* red0, double* red1,
                                           Rng& rng, uint32_t axis,
                                           uint32_t classical_idx, bool sign) {
    uint32_t active_k = *active_k_ptr;
    bool pzv = bit_get(pz, axis);
    uint64_t half = 1ULL << (active_k - 1);
    double lp = 0.0, lm = 0.0;
    for (uint64_t i = TIDX; i < half; i += BSIZE) {
        GpuComplex vi = v[i];
        GpuComplex vh = v[i + half];
        GpuComplex sum  = cadd(vi, vh);
        GpuComplex diff = csub(vi, vh);
        lp += cnorm(sum);
        lm += cnorm(diff);
    }
    double p_plus = 0.0, p_minus = 0.0;
    coop_reduce2(lp, lm, red0, red1, p_plus, p_minus);

    LDS uint8_t branch_val;
    if (TIDX == 0) {
        branch_val = sample_branch(rng, p_plus, p_minus, p_plus + p_minus);
    }
    BARRIER();
    uint8_t b = branch_val;

    for (uint64_t i = TIDX; i < half; i += BSIZE) {
        GpuComplex vi = v[i];
        GpuComplex vh = v[i + half];
        GpuComplex folded = (b == 0) ? cadd(vi, vh) : csub(vi, vh);
        store_complex64_lds(v, i, cscale(folded, kInvSqrt2));
    }
    BARRIER();
    if (TIDX == 0) {
        uint8_t m_abs = b ^ (uint8_t)pzv;
        meas[classical_idx] = m_abs ^ (uint8_t)sign;
        (*active_k_ptr)--;
        bit_set(px, axis, m_abs != 0);
        bit_set(pz, axis, false);
    }
    BARRIER();
}
)HIP";

static const char* kCoopSwapMeasInterfere = R"HIP(
DEV static void coop_swap_meas_interfere(GpuComplex* v, GpuComplex* scratch,
                                         uint64_t* px, uint64_t* pz,
                                         uint32_t* active_k_ptr, uint8_t* meas,
                                         double* red0, double* red1,
                                         Rng& rng, uint32_t from, uint32_t to,
                                         uint32_t classical_idx, bool sign) {
    if (from == to) {
        coop_meas_active_interfere(v, px, pz, active_k_ptr, meas, red0, red1,
                                   rng, to, classical_idx, sign);
        return;
    }
    if (TIDX == 0) {
        bit_swap(px, from, px, to);
        bit_swap(pz, from, pz, to);
    }
    BARRIER();
    bool pzv = bit_get(pz, to);
    uint64_t half  = 1ULL << to;
    uint64_t f_bit = 1ULL << from;
    double lp = 0.0, lm = 0.0;
    for (uint64_t idx = TIDX; idx < half; idx += BSIZE) {
        uint64_t b_f  = (idx >> from) & 1ULL;
        uint64_t base = (idx & ~f_bit) | (b_f << to);
        GpuComplex vb = v[base];
        GpuComplex vf = v[base | f_bit];
        GpuComplex sum  = cadd(vb, vf);
        GpuComplex diff = csub(vb, vf);
        lp += cnorm(sum);
        lm += cnorm(diff);
    }
    double p_plus = 0.0, p_minus = 0.0;
    coop_reduce2(lp, lm, red0, red1, p_plus, p_minus);

    LDS uint8_t branch_val;
    if (TIDX == 0) {
        branch_val = sample_branch(rng, p_plus, p_minus, p_plus + p_minus);
    }
    BARRIER();
    uint8_t b = branch_val;

    for (uint64_t idx = TIDX; idx < half; idx += BSIZE) {
        uint64_t b_f  = (idx >> from) & 1ULL;
        uint64_t base = (idx & ~f_bit) | (b_f << to);
        GpuComplex vb = v[base];
        GpuComplex vf = v[base | f_bit];
        store_complex64_lds(scratch, idx,
            cscale((b == 0) ? cadd(vb, vf) : csub(vb, vf), kInvSqrt2));
    }
    BARRIER();
    for (uint64_t idx = TIDX; idx < half; idx += BSIZE)
        store_complex64_lds(v, idx, scratch[idx]);
    BARRIER();
    if (TIDX == 0) {
        (*active_k_ptr)--;
        uint8_t m_abs = b ^ (uint8_t)pzv;
        meas[classical_idx] = m_abs ^ (uint8_t)sign;
        bit_set(px, to, m_abs != 0);
        bit_set(pz, to, false);
    }
    BARRIER();
}
)HIP";

static const char* kCoopExecExpVal = R"HIP(
DEV static void coop_exec_exp_val(GpuComplex* v, const uint64_t* px, const uint64_t* pz,
                                   uint32_t active_k, double* exp_vals,
                                   const GpuExpValMask* masks,
                                   uint32_t cp_idx, uint32_t ev_idx) {
    if (TIDX == 0) {
        const GpuExpValMask& mask = masks[cp_idx];
        bool frame_sign = mask.sign != 0;
        int parity = POPCOUNT64(mask.x[0] & pz[0]) + POPCOUNT64(mask.x[1] & pz[1])
                   + POPCOUNT64(mask.z[0] & px[0]) + POPCOUNT64(mask.z[1] & px[1]);
        if (parity & 1) frame_sign = !frame_sign;

        uint32_t k = active_k;
        uint64_t active_mask_w0 = (k >= 64) ? ~uint64_t{0}
                                  : ((k == 0) ? uint64_t{0} : ((uint64_t{1} << k) - 1));
        uint64_t dormant_x = mask.x[0] & ~active_mask_w0;
        if (k < 64) dormant_x |= mask.x[1];
        if (dormant_x != 0) {
            exp_vals[ev_idx] = 0.0;
        } else {
            uint64_t x_active = mask.x[0] & active_mask_w0;
            uint64_t z_active = mask.z[0] & active_mask_w0;
            uint32_t yz_count = POPCOUNT64(x_active & z_active);
            double phase_re = 0.0, phase_im = 0.0;
            switch (yz_count & 3) {
                case 0: phase_re =  1.0; break;
                case 1: phase_im =  1.0; break;
                case 2: phase_re = -1.0; break;
                case 3: phase_im = -1.0; break;
            }
            if (frame_sign) { phase_re = -phase_re; phase_im = -phase_im; }
            uint64_t dim = uint64_t{1} << k;
            double num_re = 0.0, num_im = 0.0, denom = 0.0;
            for (uint64_t j = 0; j < dim; ++j) {
                GpuComplex vj = v[j];
                denom += cnorm(vj);
                uint64_t j_xor    = j ^ x_active;
                GpuComplex vj_xor = v[j_xor];
                double z_sign = (POPCOUNT64(j & z_active) & 1) ? -1.0 : 1.0;
                double a_re = (double)vj_xor.re;
                double a_im = -(double)vj_xor.im;
                num_re += (a_re * z_sign * vj.re - a_im * z_sign * vj.im);
                num_im += (a_re * z_sign * vj.im + a_im * z_sign * vj.re);
            }
            double result_re = num_re * phase_re - num_im * phase_im;
            exp_vals[ev_idx] = (denom > 0.0) ? result_re / denom : 0.0;
        }
    }
    BARRIER();
}
)HIP";

// -----------------------------------------------------------------------
// emit_coop_instructions: like emit_instructions but for cooperative kernels
// Uses coop_* functions and thread-0-only guards for frame/classical ops
// -----------------------------------------------------------------------
static void emit_coop_instructions(std::ostringstream& out, const FlattenedProgram& flat,
                                    const std::vector<PipelineOp>& pipe_ops) {
    // Build pipeline-pair lookup for O(1) check
    std::set<size_t> pipeline_second_pcs;
    for (const auto& p : pipe_ops) {
        (void)p;  // used indirectly below
        pipeline_second_pcs.insert(p.second_pc);
    }

    for (size_t pc = 0; pc < flat.instrs.size(); ++pc) {
        const GpuInstr& instr = flat.instrs[pc];
        auto op = static_cast<Op>(instr.opcode);
        bool sign = (instr.flags & kFlagSign) != 0;
        bool identity = (instr.flags & kFlagIdentity) != 0;
        bool expected_one = (instr.flags & kFlagExpectedOne) != 0;

        switch (op) {
            // Frame-only ops: executed by thread 0 only, then sync
            case Op::OP_FRAME_CNOT:
                out << "    if (TIDX == 0) frame_cnot_raw(px, pz, "
                    << instr.axis_1 << "u, " << instr.axis_2 << "u);\n"
                    << "    BARRIER();\n";
                break;
            case Op::OP_FRAME_CZ:
                out << "    if (TIDX == 0) frame_cz_raw(px, pz, "
                    << instr.axis_1 << "u, " << instr.axis_2 << "u);\n"
                    << "    BARRIER();\n";
                break;
            case Op::OP_FRAME_H:
                out << "    if (TIDX == 0) frame_h_raw(px, pz, " << instr.axis_1 << "u);\n"
                    << "    BARRIER();\n";
                break;
            case Op::OP_FRAME_S:
                out << "    if (TIDX == 0) frame_s_raw(px, pz, " << instr.axis_1 << "u);\n"
                    << "    BARRIER();\n";
                break;
            case Op::OP_FRAME_S_DAG:
                out << "    if (TIDX == 0) frame_s_raw(px, pz, " << instr.axis_1 << "u);\n"
                    << "    BARRIER();\n";
                break;
            case Op::OP_FRAME_SWAP:
                out << "    if (TIDX == 0) frame_swap_raw(px, pz, "
                    << instr.axis_1 << "u, " << instr.axis_2 << "u);\n"
                    << "    BARRIER();\n";
                break;

            // Array sweep ops: cooperative, include internal sync
            case Op::OP_ARRAY_CNOT:
                out << "    coop_array_cnot(v, px, pz, *active_k_ptr, "
                    << instr.axis_1 << "u, " << instr.axis_2 << "u);\n";
                break;
            case Op::OP_ARRAY_CZ:
                out << "    coop_array_cz(v, px, pz, *active_k_ptr, "
                    << instr.axis_1 << "u, " << instr.axis_2 << "u);\n";
                break;
            case Op::OP_ARRAY_H:
                out << "    coop_array_h(v, *active_k_ptr, " << instr.axis_1 << "u);\n";
                break;
            case Op::OP_ARRAY_S:
                out << "    coop_array_s(v, px, pz, *active_k_ptr, "
                    << instr.axis_1 << "u, false);\n";
                break;
            case Op::OP_ARRAY_S_DAG:
                out << "    coop_array_s(v, px, pz, *active_k_ptr, "
                    << instr.axis_1 << "u, true);\n";
                break;
            case Op::OP_ARRAY_T:
                out << "    coop_array_t(v, px, *active_k_ptr, "
                    << instr.axis_1 << "u, false);\n";
                break;
            case Op::OP_ARRAY_T_DAG:
                out << "    coop_array_t(v, px, *active_k_ptr, "
                    << instr.axis_1 << "u, true);\n";
                break;
            case Op::OP_ARRAY_U2:
                if (!flat.fused_u2.empty()) {
                    out << "    coop_array_u2(v, px, pz, *active_k_ptr, "
                        << instr.axis_1 << "u, kFusedU2, " << instr.a << "u);\n";
                }
                break;
            case Op::OP_ARRAY_U4:
                if (!flat.fused_u4.empty()) {
                    out << "    coop_array_u4(v, px, pz, *active_k_ptr, "
                        << instr.axis_1 << "u, " << instr.axis_2 << "u, kFusedU4, "
                        << instr.a << "u);\n";
                }
                break;

            // Expand ops: cooperative
            case Op::OP_EXPAND:
                out << "    coop_expand_plain(v, active_k_ptr);\n";
                break;
            case Op::OP_EXPAND_T:
                out << "    coop_expand_t(v, px, active_k_ptr, "
                    << instr.axis_1 << "u, false);\n";
                break;
            case Op::OP_EXPAND_T_DAG:
                out << "    coop_expand_t(v, px, active_k_ptr, "
                    << instr.axis_1 << "u, true);\n";
                break;

            // Measurements: thread 0 handles dormant/classical
            case Op::OP_MEAS_DORMANT_STATIC:
                if (identity) {
                    out << "    if (TIDX == 0) meas[" << instr.a << "u] = "
                        << (sign ? 1u : 0u) << "u;\n"
                        << "    BARRIER();\n";
                } else {
                    out << "    if (TIDX == 0) meas[" << instr.a << "u] = "
                        << "(bit_get(px, " << instr.axis_1 << "u) ? 1u : 0u)"
                        << (sign ? " ^ 1u" : "") << ";\n"
                        << "    BARRIER();\n";
                }
                break;
            case Op::OP_MEAS_DORMANT_RANDOM:
                out << "    if (TIDX == 0) {\n"
                    << "        uint8_t _m = rng.uniform() < 0.5 ? 0 : 1;\n"
                    << "        bit_set(px, " << instr.axis_1 << "u, _m != 0);\n"
                    << "        bit_set(pz, " << instr.axis_1 << "u, false);\n"
                    << "        meas[" << instr.a << "u] = _m ^ " << (sign ? 1u : 0u) << "u;\n"
                    << "    }\n"
                    << "    BARRIER();\n";
                break;
            case Op::OP_MEAS_ACTIVE_DIAGONAL:
                out << "    coop_meas_active_diagonal(v, px, pz, active_k_ptr, meas, "
                    << "red0, red1, rng, " << instr.axis_1 << "u, "
                    << instr.a << "u, " << (sign ? "true" : "false") << ");\n";
                break;

            // Apply Pauli: thread 0 only — directly XOR px/pz arrays
            case Op::OP_APPLY_PAULI:
                if (!flat.pauli_masks.empty()) {
                    out << "    if (TIDX == 0) {\n"
                        << "        if (meas[" << instr.b << "u] != 0) {\n"
                        << "            const GpuMask& _pm = kPauliMasks[" << instr.a << "u];\n"
                        << "            px[0] ^= _pm.x[0]; px[1] ^= _pm.x[1];\n"
                        << "            pz[0] ^= _pm.z[0]; pz[1] ^= _pm.z[1];\n"
                        << "        }\n"
                        << "    }\n"
                        << "    BARRIER();\n";
                }
                break;

            // Noise: thread 0 only — must call coop_draw_next_noise to keep RNG in sync
            case Op::OP_NOISE:
                out << "    if (TIDX == 0) {\n"
                    << "        if (*next_noise_idx == " << instr.a << "u) {\n"
                    << "            const GpuNoiseSite* site = &kNoiseSites[" << instr.a << "u];\n"
                    << "            double roll = rng.uniform() * site->prob_sum;\n"
                    << "            double cumulative = 0.0;\n"
                    << "            for (uint32_t k = 0; k < site->count; ++k) {\n"
                    << "                const GpuChannel* ch = &kNoiseChannels[site->offset + k];\n"
                    << "                cumulative += ch->prob;\n"
                    << "                if (roll < cumulative) {\n"
                    << "                    px[0] ^= ch->x[0]; px[1] ^= ch->x[1];\n"
                    << "                    pz[0] ^= ch->z[0]; pz[1] ^= ch->z[1];\n"
                    << "                    break;\n"
                    << "                }\n"
                    << "            }\n"
                    << "            *next_noise_idx = " << instr.a << "u + 1;\n"
                    << "            coop_draw_next_noise(next_noise_idx, rng, kNoiseHazards, "
                    << flat.noise_sites.size() << "u);\n"
                    << "        }\n"
                    << "    }\n"
                    << "    BARRIER();\n";
                break;

            // Postselect
            case Op::OP_POSTSELECT:
                out << "    if (TIDX == 0) {\n"
                    << "        uint32_t _ps_start = kDetectorOffsets[" << instr.a << "u];\n"
                    << "        uint32_t _ps_end = kDetectorOffsets[" << instr.a << "u + 1u];\n"
                    << "        uint8_t _ps_parity = " << (expected_one ? 1u : 0u) << "u;\n"
                    << "        for (uint32_t i = _ps_start; i < _ps_end; ++i)\n"
                    << "            _ps_parity ^= meas[kDetectorTargets[i]];\n"
                    << "        if (_ps_parity != 0) *discarded_ptr = 1;\n"
                    << "    }\n"
                    << "    BARRIER();\n"
                    << "    if (*discarded_ptr) goto done;\n";
                break;

            // Observable
            case Op::OP_OBSERVABLE:
                out << "    if (TIDX == 0) {\n"
                    << "        uint32_t _ob_start = kObservableOffsets[" << instr.a << "u];\n"
                    << "        uint32_t _ob_end = kObservableOffsets[" << instr.a << "u + 1u];\n"
                    << "        uint8_t _ob_parity = 0u;\n"
                    << "        for (uint32_t i = _ob_start; i < _ob_end; ++i)\n"
                    << "            _ob_parity ^= meas[kObservableTargets[i]];\n"
                    << "        obs[" << instr.b << "u] ^= _ob_parity;\n"
                    << "    }\n"
                    << "    BARRIER();\n";
                break;

            // Array swap
            case Op::OP_ARRAY_SWAP:
                out << "    coop_array_swap(v, px, pz, *active_k_ptr, "
                    << instr.axis_1 << "u, " << instr.axis_2 << "u);\n";
                break;

            // Multi-controlled CNOT
            case Op::OP_ARRAY_MULTI_CNOT:
                out << "    coop_array_multi_cnot(v, px, pz, *active_k_ptr, "
                    << instr.axis_1 << "u, 0x" << std::hex << instr.mask << std::dec << "ULL);\n";
                break;

            // Multi-controlled CZ
            case Op::OP_ARRAY_MULTI_CZ:
                out << "    coop_array_multi_cz(v, px, pz, *active_k_ptr, "
                    << instr.axis_1 << "u, 0x" << std::hex << instr.mask << std::dec << "ULL);\n";
                break;

            // Z-rotation on an axis
            case Op::OP_ARRAY_ROT:
                out << "    coop_array_rot(v, px, *active_k_ptr, "
                    << instr.axis_1 << "u, " << hex_double(instr.weight_re)
                    << ", " << hex_double(instr.weight_im) << ");\n";
                break;

            // Expand with Z-rotation phase
            case Op::OP_EXPAND_ROT:
                out << "    coop_expand_rot(v, px, active_k_ptr, "
                    << instr.axis_1 << "u, " << hex_double(instr.weight_re)
                    << ", " << hex_double(instr.weight_im) << ");\n";
                break;

            // Active interfere measurement
            case Op::OP_MEAS_ACTIVE_INTERFERE:
                out << "    coop_meas_active_interfere(v, px, pz, active_k_ptr, meas, "
                    << "red0, red1, rng, " << instr.axis_1 << "u, "
                    << instr.a << "u, " << (sign ? "true" : "false") << ");\n";
                break;

            // Swap+measure with interference
            case Op::OP_SWAP_MEAS_INTERFERE:
                out << "    coop_swap_meas_interfere(v, scratch_v, px, pz, active_k_ptr, meas, "
                    << "red0, red1, rng, " << instr.axis_1 << "u, " << instr.axis_2 << "u, "
                    << instr.a << "u, " << (sign ? "true" : "false") << ");\n";
                break;

            // Expectation value computation
            case Op::OP_EXP_VAL:
                if (!flat.exp_val_masks.empty()) {
                    out << "    coop_exec_exp_val(v, px, pz, *active_k_ptr, exp_vals, "
                        << "kExpValMasks, " << instr.a << "u, " << instr.b << "u);\n";
                }
                break;

            case Op::OP_NOISE_BLOCK:
                out << "    if (TIDX == 0) {\n"
                    << "        uint32_t _nb_end = " << instr.a << "u + " << instr.b << "u;\n"
                    << "        while (*next_noise_idx >= " << instr.a << "u && *next_noise_idx < _nb_end) {\n"
                    << "            uint32_t _nb_site_idx = *next_noise_idx;\n"
                    << "            const GpuNoiseSite* site = &kNoiseSites[_nb_site_idx];\n"
                    << "            double roll = rng.uniform() * site->prob_sum;\n"
                    << "            double cumulative = 0.0;\n"
                    << "            for (uint32_t k = 0; k < site->count; ++k) {\n"
                    << "                const GpuChannel* ch = &kNoiseChannels[site->offset + k];\n"
                    << "                cumulative += ch->prob;\n"
                    << "                if (roll < cumulative) {\n"
                    << "                    px[0] ^= ch->x[0]; px[1] ^= ch->x[1];\n"
                    << "                    pz[0] ^= ch->z[0]; pz[1] ^= ch->z[1];\n"
                    << "                    break;\n"
                    << "                }\n"
                    << "            }\n"
                    << "            *next_noise_idx = _nb_site_idx + 1;\n"
                    << "            coop_draw_next_noise(next_noise_idx, rng, kNoiseHazards, "
                    << flat.noise_sites.size() << "u);\n"
                    << "        }\n"
                    << "    }\n"
                    << "    BARRIER();\n";
                break;

            case Op::OP_READOUT_NOISE:
                out << "    if (TIDX == 0) {\n"
                    << "        const GpuReadoutNoise* _rn = &kReadoutNoise[" << instr.a << "u];\n"
                    << "        if (rng.uniform() < _rn->prob) {\n"
                    << "            meas[_rn->meas_idx] ^= 1;\n"
                    << "        }\n"
                    << "    }\n"
                    << "    BARRIER();\n";
                break;

            case Op::OP_DETECTOR:
                out << "    BARRIER();\n";
                break;

            default:
                out << "    if (TIDX == 0) *discarded_ptr = 1;\n"
                    << "    goto done;\n";
                break;
        }
    }
}

// -----------------------------------------------------------------------
// Public: generate_compiled_kernel_coop
// -----------------------------------------------------------------------
std::string generate_compiled_kernel_coop(const FlattenedProgram& flat) {
    UsedFunctions uf = analyze_used_functions(flat);
    bool has_noise = uf.noise || uf.noise_block;

    auto pipe_ops = analyze_pipeline_opportunities(flat);

    uint32_t peak = flat.peak_rank;
    uint32_t num_amplitudes = 1u << peak;

    std::ostringstream out;

    // Preamble: types, scatter, complex, rng, coop state + LDS helpers, frame raw, coop funcs
    emit_preamble(out, uf);
    out << kPreambleCoopState;
    out << kCoopFrameRaw;
    out << kCoopReduceDouble;
    if (uf.needs_complex_ops) {
        out << kCoopArrayH;
        out << kCoopArrayCnot;
        out << kCoopArrayCz;
        if (uf.array_s) out << kCoopArrayS;
        if (uf.array_t) out << kCoopArrayT;
        if (uf.array_u2) out << kCoopArrayU2;
        if (uf.array_u4) out << kCoopArrayU4;
    }
    if (uf.expand_plain) out << kCoopExpandPlain;
    if (uf.expand_t) out << kCoopExpandT;
    if (uf.expand_rot) out << kCoopExpandRot;
    if (uf.meas_active_diagonal) out << kCoopMeasActiveDiag;
    if (uf.meas_active_interfere || uf.swap_meas_interfere) out << kCoopMeasActiveInterfere;
    if (uf.swap_meas_interfere) out << kCoopSwapMeasInterfere;
    if (uf.array_swap) out << kCoopArraySwap;
    if (uf.array_multi_cnot) out << kCoopArrayMultiCnot;
    if (uf.array_multi_cz) out << kCoopArrayMultiCz;
    if (uf.array_rot) out << kCoopArrayRot;
    if (uf.exp_val) out << kCoopExecExpVal;

    // Constant pool
    emit_constant_pool(out, flat, uf);
    // Dummy ShotState for apply_pauli (uses ShotState& in existing helper)
    if (uf.apply_pauli) {
        out << kApplyPauliToFrame;
        out << kApplyPauli;
    }
    if (has_noise) {
        out << kCoopNoiseHelper;
    }

    // Kernel signature
    out << "\nextern \"C\" __global__ __attribute__((amdgpu_flat_work_group_size(1, 256))) void compiled_sample_kernel_coop(\n"
        << "    uint64_t shot_offset, uint64_t shots, uint64_t seed,\n"
        << "    BlockCounts* block_counts,\n"
        << "    uint32_t num_observables, uint32_t num_exp_vals\n"
        << ") {\n";

    // One block = one shot
    out << "    uint64_t batch_shot_id = BIDX;\n"
        << "    if (batch_shot_id >= shots) return;\n"
        << "\n";

    // LDS amplitude array — padded by 1 per 32-element group to avoid bank conflicts
    // For MI300X: LDS has 64 banks of 4 bytes each. GpuComplex = 8 bytes = 2 banks.
    // Adding 1 to the stride per 32-element row avoids the conflict pattern.
    out << "    LDS GpuComplex v[" << num_amplitudes << "];\n";
    out << "    LDS uint64_t px[2];\n"
        << "    LDS uint64_t pz[2];\n"
        << "    LDS uint32_t active_k_shared;\n"
        << "    LDS uint32_t next_noise_idx_shared;\n"
        << "    LDS uint8_t discarded_shared;\n"
        << "    LDS uint8_t meas[kMaxMeas];\n"
        << "    LDS uint8_t obs[kMaxObs];\n"
        << "    LDS double red0[256];\n"
        << "    LDS double red1[256];\n";
    if (uf.swap_meas_interfere) {
        // scratch_v is used as a temporary during coop_swap_meas_interfere.
        // We need half the peak amplitude count (1 << (peak_rank-1)) entries.
        uint32_t scratch_size = (peak > 0) ? (num_amplitudes / 2) : 1u;
        out << "    LDS GpuComplex scratch_v[" << scratch_size << "];\n";
    } else {
        out << "    GpuComplex* scratch_v = nullptr; // unused\n";
    }
    if (uf.exp_val) {
        out << "    LDS double exp_vals[kMaxExpVals];\n";
    }
    out << "\n";

    // Aliases for pointers
    out << "    uint32_t* active_k_ptr = &active_k_shared;\n"
        << "    uint32_t* next_noise_idx = &next_noise_idx_shared;\n"
        << "    uint8_t* discarded_ptr = &discarded_shared;\n"
        << "\n";

    // Cooperative initialization — all threads participate to zero v[]
    out << "    for (uint32_t i = TIDX; i < " << num_amplitudes << "u; i += BSIZE) {\n"
        << "        v[i] = {0.0f, 0.0f};\n"
        << "    }\n"
        << "    if (TIDX == 0) {\n"
        << "        px[0] = 0; px[1] = 0;\n"
        << "        pz[0] = 0; pz[1] = 0;\n"
        << "        active_k_shared = 0;\n"
        << "        next_noise_idx_shared = 0;\n"
        << "        discarded_shared = 0;\n"
        << "        v[0] = {1.0f, 0.0f};\n"
        << "        for (uint32_t i = 0; i < num_observables; ++i) obs[i] = 0;\n";
    if (uf.exp_val) {
        out << "        for (uint32_t i = 0; i < num_exp_vals; ++i) exp_vals[i] = 0.0;\n";
    }
    out << "    }\n"
        << "    BARRIER();\n"
        << "\n";

    if (uf.needs_rng) {
        out << "    Rng rng;\n"
            << "    if (TIDX == 0) {\n"
            << "        rng.seed(seed, shot_offset + batch_shot_id);\n";
        if (has_noise) {
            out << "        coop_draw_next_noise(next_noise_idx, rng, kNoiseHazards, "
                << flat.noise_sites.size() << "u);\n";
        }
        out << "    }\n"
            << "    BARRIER();\n\n";
    }

    // Instruction sequence
    out << "    // --- Begin cooperative instruction sequence (" << flat.instrs.size() << " ops) ---\n";
    out << "    // Pipeline opportunities found: " << pipe_ops.size() << "\n";
    emit_coop_instructions(out, flat, pipe_ops);
    out << "    // --- End cooperative instruction sequence ---\n";

    // Accumulate into block_counts (one count record total — coop uses atomic adds)
    out << "\n"
        << "done:\n"
        << "    BARRIER();\n"
        << "    if (TIDX == 0 && discarded_shared == 0) {\n"
        << "        bool any_logical = false;\n"
        << "        for (uint32_t i = 0; i < num_observables; ++i) {\n"
        << "            uint8_t val = obs[i];\n";
    if (uf.observable) {
        out << "            if (kExpectedObservables[i] != 0) val ^= 1u;\n";
    }
    out << "            if (val != 0) {\n"
        << "                __atomic_fetch_add(reinterpret_cast<unsigned long long*>(\n"
        << "                    &block_counts->observable_ones[i]), 1ULL, __ATOMIC_RELAXED);\n"
        << "                any_logical = true;\n"
        << "            }\n"
        << "        }\n"
        << "        __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&block_counts->passed), 1ULL, __ATOMIC_RELAXED);\n"
        << "        if (any_logical)\n"
        << "            __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&block_counts->logical_errors), 1ULL, __ATOMIC_RELAXED);\n";
    if (uf.exp_val) {
        out << "        for (uint32_t i = 0; i < num_exp_vals; ++i) {\n"
            << "            __atomic_fetch_add(reinterpret_cast<unsigned long long*>(\n"
            << "                &block_counts->exp_val_count), 1ULL, __ATOMIC_RELAXED);\n"
            << "            // Use double-precision atomic add via CAS loop\n"
            << "            unsigned long long* addr = reinterpret_cast<unsigned long long*>(\n"
            << "                &block_counts->exp_val_sums[i]);\n"
            << "            unsigned long long old_val = __atomic_load_n(addr, __ATOMIC_RELAXED);\n"
            << "            unsigned long long new_val;\n"
            << "            do {\n"
            << "                double old_d;\n"
            << "                __builtin_memcpy(&old_d, &old_val, sizeof(double));\n"
            << "                double new_d = old_d + exp_vals[i];\n"
            << "                __builtin_memcpy(&new_val, &new_d, sizeof(double));\n"
            << "            } while (!__atomic_compare_exchange_n(addr, &old_val, new_val,\n"
            << "                         false, __ATOMIC_RELAXED, __ATOMIC_RELAXED));\n"
            << "        }\n";
    }
    out << "    }\n"
        << "}\n";

    return out.str();
}

// -----------------------------------------------------------------------
// Public: generate_compiled_kernel_global
//
// Global-tier kernel: one shot shared across BSIZE threads, amplitude
// array in HBM.  Per-XCD work-stealing reuses the same counter layout as
// the SVM global-coop kernel.
// -----------------------------------------------------------------------
std::string generate_compiled_kernel_global(const FlattenedProgram& flat) {
    UsedFunctions uf = analyze_used_functions(flat);
    bool has_noise = uf.noise || uf.noise_block;
    auto pipe_ops = analyze_pipeline_opportunities(flat);

    std::ostringstream out;

    emit_preamble(out, uf);
    out << kPreambleCoopState;
    out << kCoopFrameRaw;
    out << kCoopReduceDouble;
    if (uf.needs_complex_ops) {
        out << kCoopArrayH;
        out << kCoopArrayCnot;
        out << kCoopArrayCz;
        if (uf.array_s) out << kCoopArrayS;
        if (uf.array_t) out << kCoopArrayT;
        if (uf.array_u2) out << kCoopArrayU2;
        if (uf.array_u4) out << kCoopArrayU4;
    }
    if (uf.expand_plain) out << kCoopExpandPlain;
    if (uf.expand_t) out << kCoopExpandT;
    if (uf.expand_rot) out << kCoopExpandRot;
    if (uf.meas_active_diagonal) out << kCoopMeasActiveDiag;
    if (uf.meas_active_interfere || uf.swap_meas_interfere) out << kCoopMeasActiveInterfere;
    if (uf.swap_meas_interfere) out << kCoopSwapMeasInterfere;
    if (uf.array_swap) out << kCoopArraySwap;
    if (uf.array_multi_cnot) out << kCoopArrayMultiCnot;
    if (uf.array_multi_cz) out << kCoopArrayMultiCz;
    if (uf.array_rot) out << kCoopArrayRot;
    if (uf.exp_val) out << kCoopExecExpVal;

    emit_constant_pool(out, flat, uf);
    if (uf.apply_pauli) {
        out << kApplyPauliToFrame;
        out << kApplyPauli;
    }
    if (has_noise) out << kCoopNoiseHelper;

    // Kernel signature — global_v and global_scratch are HBM buffers
    out << "\nextern \"C\" __global__ __attribute__((amdgpu_flat_work_group_size(1, 256))) void compiled_sample_kernel_global(\n"
        << "    uint64_t shot_offset, uint64_t shots, uint64_t seed,\n"
        << "    GpuComplex* __restrict__ global_v,\n"
        << "    GpuComplex* __restrict__ global_scratch,\n"
        << "    uint64_t* work_counter,\n"
        << "    BlockCounts* block_counts,\n"
        << "    uint32_t num_observables, uint32_t num_exp_vals\n"
        << ") {\n";

    // Each block has a dedicated HBM slot for its amplitude array
    out << "    uint32_t slot = BIDX;\n"
        << "    GpuComplex* v = global_v + (size_t)slot * " << (1u << kGlobalMaxPeakRank) << "u;\n";
    if (uf.swap_meas_interfere) {
        // scratch_v for global kernel: use global_scratch per-slot (same stride as v)
        out << "    GpuComplex* scratch_v = global_scratch + (size_t)slot * "
            << (1u << kGlobalMaxPeakRank) << "u;\n";
    } else {
        out << "    GpuComplex* scratch_v = nullptr; // unused\n";
    }
    out << "\n";

    out << "    LDS uint64_t px[2];\n"
        << "    LDS uint64_t pz[2];\n"
        << "    LDS uint32_t active_k_shared;\n"
        << "    LDS uint32_t next_noise_idx_shared;\n"
        << "    LDS uint8_t discarded_shared;\n"
        << "    LDS uint8_t meas[kMaxMeas];\n"
        << "    LDS uint8_t obs[kMaxObs];\n"
        << "    LDS double red0[256];\n"
        << "    LDS double red1[256];\n"
        << "    LDS uint64_t batch_shot_id_shared;\n";
    if (uf.exp_val) {
        out << "    LDS double exp_vals[kMaxExpVals];\n";
    }
    out << "\n"
        << "    uint32_t* active_k_ptr = &active_k_shared;\n"
        << "    uint32_t* next_noise_idx = &next_noise_idx_shared;\n"
        << "    uint8_t* discarded_ptr = &discarded_shared;\n"
        << "\n";

    // Per-XCD work stealing: same pattern as SVM global-coop kernel
    uint32_t global_num_amplitudes = 1u << flat.peak_rank;
    out << "    // Per-XCD work stealing (8 XCDs for MI300X)\n"
        << "    uint32_t xcd_id;\n"
        << "    asm volatile(\"s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)\" : \"=s\"(xcd_id));\n"
        << "    xcd_id = xcd_id % " << kNumXCDs << "u;\n"
        << "    uint64_t* my_counter = work_counter + xcd_id;\n"
        << "\n"
        << "    while (true) {\n"
        << "        if (TIDX == 0) {\n"
        << "            uint64_t local_slot = __atomic_fetch_add(reinterpret_cast<unsigned long long*>(my_counter),\n"
        << "                                           (unsigned long long)1, __ATOMIC_RELAXED);\n"
        << "            batch_shot_id_shared = local_slot * " << kNumXCDs << "u + xcd_id;\n"
        << "        }\n"
        << "        BARRIER();\n"
        << "        uint64_t my_shot_offset = batch_shot_id_shared;\n"
        << "        if (my_shot_offset >= shots) break;\n"
        << "\n"
        << "        // Initialize state for this shot — cooperative v[] zeroing\n"
        << "        for (uint32_t i = TIDX; i < " << global_num_amplitudes << "u; i += BSIZE) {\n"
        << "            v[i] = {0.0f, 0.0f};\n"
        << "        }\n"
        << "        if (TIDX == 0) {\n"
        << "            px[0] = 0; px[1] = 0; pz[0] = 0; pz[1] = 0;\n"
        << "            active_k_shared = 0;\n"
        << "            next_noise_idx_shared = 0;\n"
        << "            discarded_shared = 0;\n"
        << "            v[0] = {1.0f, 0.0f};\n"
        << "            for (uint32_t i = 0; i < num_observables; ++i) obs[i] = 0;\n";
    if (uf.exp_val) {
        out << "            for (uint32_t i = 0; i < num_exp_vals; ++i) exp_vals[i] = 0.0;\n";
    }
    out << "        }\n"
        << "        BARRIER();\n\n";

    if (uf.needs_rng) {
        out << "        Rng rng;\n"
            << "        if (TIDX == 0) {\n"
            << "            rng.seed(seed, shot_offset + my_shot_offset);\n";
        if (has_noise) {
            out << "            coop_draw_next_noise(next_noise_idx, rng, kNoiseHazards, "
                << flat.noise_sites.size() << "u);\n";
        }
        out << "        }\n"
            << "        BARRIER();\n\n";
    }

    // Instruction sequence
    out << "        // --- Begin global-tier instruction sequence (" << flat.instrs.size() << " ops) ---\n";
    out << "        // Pipeline opportunities: " << pipe_ops.size() << "\n";
    // Emit instructions with 8-space indent for the inner while loop
    // We reuse emit_coop_instructions but need extra indentation
    {
        std::ostringstream inner;
        emit_coop_instructions(inner, flat, pipe_ops);
        std::string line;
        std::istringstream ss(inner.str());
        while (std::getline(ss, line)) {
            out << "    " << line << "\n";
        }
    }
    out << "        // --- End global-tier instruction sequence ---\n\n";

    out << "    done:\n"
        << "        BARRIER();\n"
        << "        if (TIDX == 0 && discarded_shared == 0) {\n"
        << "            bool any_logical = false;\n"
        << "            for (uint32_t i = 0; i < num_observables; ++i) {\n"
        << "                uint8_t val = obs[i];\n";
    if (uf.observable) {
        out << "                if (kExpectedObservables[i] != 0) val ^= 1u;\n";
    }
    out << "                if (val != 0) {\n"
        << "                    __atomic_fetch_add(reinterpret_cast<unsigned long long*>(\n"
        << "                        &block_counts->observable_ones[i]), 1ULL, __ATOMIC_RELAXED);\n"
        << "                    any_logical = true;\n"
        << "                }\n"
        << "            }\n"
        << "            __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&block_counts->passed), 1ULL, __ATOMIC_RELAXED);\n"
        << "            if (any_logical)\n"
        << "                __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&block_counts->logical_errors), 1ULL, __ATOMIC_RELAXED);\n";
    if (uf.exp_val) {
        out << "            for (uint32_t i = 0; i < num_exp_vals; ++i) {\n"
            << "                __atomic_fetch_add(reinterpret_cast<unsigned long long*>(\n"
            << "                    &block_counts->exp_val_count), 1ULL, __ATOMIC_RELAXED);\n"
            << "                unsigned long long* addr = reinterpret_cast<unsigned long long*>(\n"
            << "                    &block_counts->exp_val_sums[i]);\n"
            << "                unsigned long long old_val = __atomic_load_n(addr, __ATOMIC_RELAXED);\n"
            << "                unsigned long long new_val;\n"
            << "                do {\n"
            << "                    double old_d;\n"
            << "                    __builtin_memcpy(&old_d, &old_val, sizeof(double));\n"
            << "                    double new_d = old_d + exp_vals[i];\n"
            << "                    __builtin_memcpy(&new_val, &new_d, sizeof(double));\n"
            << "                } while (!__atomic_compare_exchange_n(addr, &old_val, new_val,\n"
            << "                             false, __ATOMIC_RELAXED, __ATOMIC_RELAXED));\n"
            << "            }\n";
    }
    out << "        }\n"
        << "    } // while (true) work-stealing loop\n"
        << "}\n";

    return out.str();
}

}  // namespace gpu
}  // namespace clifft
