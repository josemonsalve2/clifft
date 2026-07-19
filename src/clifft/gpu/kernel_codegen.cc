#include "clifft/gpu/kernel_codegen.h"
#include "clifft/gpu/gpu_types.h"

#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <set>
#include <sstream>
#include <string>
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
static const char* kPreambleTypes = R"HIP(
#include <stdint.h>
#include <math.h>

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

__device__ inline bool bit_get(const uint64_t* words, uint32_t idx) {
    return ((words[idx >> 6] >> (idx & 63u)) & 1ULL) != 0;
}
__device__ inline void bit_set(uint64_t* words, uint32_t idx, bool value) {
    uint64_t mask = 1ULL << (idx & 63u);
    uint32_t word = idx >> 6;
    if (value) words[word] |= mask; else words[word] &= ~mask;
}
__device__ inline void bit_xor(uint64_t* words, uint32_t idx, bool value) {
    if (value) words[idx >> 6] ^= 1ULL << (idx & 63u);
}
__device__ inline void bit_swap(uint64_t* a, uint32_t ia, uint64_t* b, uint32_t ib) {
    bool va = bit_get(a, ia), vb = bit_get(b, ib);
    if (va != vb) {
        a[ia >> 6] ^= 1ULL << (ia & 63u);
        b[ib >> 6] ^= 1ULL << (ib & 63u);
    }
}
)HIP";

// RNG: only needed if measurements or noise use randomness
static const char* kPreambleRng = R"HIP(
__device__ inline uint64_t rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}
__device__ inline uint64_t splitmix64_next(uint64_t& state) {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
struct Rng {
    uint64_t s[4];
    __device__ void seed(uint64_t seed_value, uint64_t shot_id) {
        uint64_t z = seed_value ^ (0x9e3779b97f4a7c15ULL * (shot_id + 1));
        s[0] = splitmix64_next(z);
        s[1] = splitmix64_next(z);
        s[2] = splitmix64_next(z);
        s[3] = splitmix64_next(z);
    }
    __device__ uint64_t next() {
        const uint64_t result = rotl64(s[0] + s[3], 23) + s[0];
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t; s[3] = rotl64(s[3], 45);
        return result;
    }
    __device__ double uniform() {
        return (double)(next() >> 11) * 0x1.0p-53;
    }
};
)HIP";

// Complex arithmetic: only needed when array sweeps transform amplitudes
static const char* kPreambleComplex = R"HIP(
__device__ inline GpuComplex cadd(GpuComplex a, GpuComplex b) { return {a.re+b.re, a.im+b.im}; }
__device__ inline GpuComplex csub(GpuComplex a, GpuComplex b) { return {a.re-b.re, a.im-b.im}; }
__device__ inline GpuComplex cscale(GpuComplex a, double s) {
    return {(float)((double)a.re*s), (float)((double)a.im*s)};
}
__device__ inline GpuComplex cmul(GpuComplex a, GpuComplex b) {
    return {(float)(a.re*b.re - a.im*b.im), (float)(a.re*b.im + a.im*b.re)};
}
__device__ inline double cnorm(GpuComplex a) {
    return (double)a.re*(double)a.re + (double)a.im*(double)a.im;
}
)HIP";

// Scatter/insert-zero-bit: needed for array sweep index calculations
static const char* kPreambleArraySweep = R"HIP(
__device__ inline uint64_t insert_zero_bit(uint64_t val, uint32_t pos) {
    uint64_t mask = (1ULL << pos) - 1ULL;
    return (val & mask) | ((val & ~mask) << 1);
}
__device__ inline uint64_t scatter_bits_1(uint64_t val, uint32_t bit_pos) {
    return insert_zero_bit(val, bit_pos);
}
__device__ inline uint64_t scatter_bits_2(uint64_t val, uint32_t bit1, uint32_t bit2) {
    uint32_t lo = bit1 < bit2 ? bit1 : bit2;
    uint32_t hi = bit1 < bit2 ? bit2 : bit1;
    val = insert_zero_bit(val, lo);
    return insert_zero_bit(val, hi);
}
)HIP";

// sample_branch: needed for active measurements
static const char* kPreambleSampleBranch = R"HIP(
__device__ inline uint8_t sample_branch(Rng& rng, double prob0, double prob1, double total) {
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
__device__ inline void frame_cnot(ShotState& st, uint32_t c, uint32_t t) {
    bool px_c = bit_get(st.px, c), pz_t = bit_get(st.pz, t);
    bit_xor(st.px, t, px_c); bit_xor(st.pz, c, pz_t);
}
__device__ inline void frame_cz(ShotState& st, uint32_t c, uint32_t t) {
    bool px_c = bit_get(st.px, c), px_t = bit_get(st.px, t);
    bit_xor(st.pz, t, px_c); bit_xor(st.pz, c, px_t);
}
__device__ inline void frame_h(ShotState& st, uint32_t axis) {
    bool px = bit_get(st.px, axis), pz = bit_get(st.pz, axis);
    bit_set(st.px, axis, pz); bit_set(st.pz, axis, px);
}
__device__ inline void frame_s(ShotState& st, uint32_t axis) {
    bit_xor(st.pz, axis, bit_get(st.px, axis));
}
__device__ inline void frame_swap(ShotState& st, uint32_t a, uint32_t b) {
    bit_swap(st.px, a, st.px, b); bit_swap(st.pz, a, st.pz, b);
}
)HIP";

static const char* kApplyPauliToFrame = R"HIP(
__device__ void apply_pauli_to_frame(ShotState& st, const uint64_t* x, const uint64_t* z) {
    st.px[0] ^= x[0]; st.px[1] ^= x[1];
    st.pz[0] ^= z[0]; st.pz[1] ^= z[1];
}
)HIP";

static const char* kApplyPauli = R"HIP(
__device__ void apply_pauli(ShotState& st, const GpuMask* masks, uint32_t mask_idx,
                            uint32_t condition_idx) {
    if (st.meas[condition_idx] == 0) return;
    const GpuMask& mask = masks[mask_idx];
    apply_pauli_to_frame(st, mask.x, mask.z);
}
)HIP";

static const char* kNoiseHelper = R"HIP(
__device__ void draw_next_noise_compiled(ShotState& st, Rng& rng,
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
__device__ void apply_phase(ShotState& st, uint32_t axis, GpuComplex phase) {
    uint64_t axis_bit = 1ULL << axis;
    uint64_t iters = 1ULL << (st.active_k - 1);
    for (uint64_t i = 0; i < iters; ++i) {
        uint64_t idx = scatter_bits_1(i, axis) | axis_bit;
        st.v[idx] = cmul(st.v[idx], phase);
    }
}
)HIP";

static const char* kArrayCnot = R"HIP(
__device__ void array_cnot(ShotState& st, uint32_t c, uint32_t t) {
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
__device__ void array_cz(ShotState& st, uint32_t a, uint32_t b) {
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
__device__ void array_swap(ShotState& st, uint32_t a, uint32_t b) {
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
__device__ void array_multi_cnot(ShotState& st, uint32_t target, uint64_t ctrl_mask) {
    uint64_t t_bit = 1ULL << target;
    uint64_t half = 1ULL << (st.active_k - 1);
    for (uint64_t idx = 0; idx < half; ++idx) {
        uint64_t actual = scatter_bits_1(idx, target);
        if ((__popcll(actual & ctrl_mask) & 1) != 0) {
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
__device__ void array_multi_cz(ShotState& st, uint32_t control, uint64_t target_mask) {
    uint64_t c_bit = 1ULL << control;
    uint64_t half = 1ULL << (st.active_k - 1);
    for (uint64_t idx = 0; idx < half; ++idx) {
        uint64_t actual = scatter_bits_1(idx, control) | c_bit;
        if ((__popcll(actual & target_mask) & 1) != 0) {
            st.v[actual].re = -st.v[actual].re; st.v[actual].im = -st.v[actual].im;
        }
    }
    for (uint32_t t = 0; t < st.active_k; ++t) {
        if ((target_mask >> t) & 1ULL) frame_cz(st, control, t);
    }
}
)HIP";

static const char* kArrayH = R"HIP(
__device__ void array_h(ShotState& st, uint32_t axis) {
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
__device__ void array_s(ShotState& st, uint32_t axis, bool dagger) {
    apply_phase(st, axis, dagger ? (GpuComplex){0.0f, -1.0f} : (GpuComplex){0.0f, 1.0f});
    frame_s(st, axis);
}
)HIP";

static const char* kArrayT = R"HIP(
__device__ void array_t(ShotState& st, uint32_t axis, bool dagger) {
    bool px = bit_get(st.px, axis);
    if (axis >= st.active_k) return;
    double imag = dagger ? -kInvSqrt2 : kInvSqrt2;
    if (px) imag = -imag;
    apply_phase(st, axis, (GpuComplex){(float)kInvSqrt2, (float)imag});
}
)HIP";

static const char* kArrayRot = R"HIP(
__device__ void array_rot(ShotState& st, uint32_t axis, double z_re, double z_im) {
    if (axis < st.active_k) {
        bool px = bit_get(st.px, axis);
        double im = px ? -z_im : z_im;
        apply_phase(st, axis, (GpuComplex){(float)z_re, (float)im});
    }
}
)HIP";

static const char* kArrayU2 = R"HIP(
__device__ void array_u2_compiled(ShotState& st, uint32_t axis,
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
__device__ void array_u4_compiled(ShotState& st, uint32_t axis_lo, uint32_t axis_hi,
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
__device__ void expand_plain(ShotState& st) {
    uint64_t half = 1ULL << st.active_k;
    for (uint64_t i = 0; i < half; ++i) st.v[i + half] = st.v[i];
    st.active_k++;
}
)HIP";

static const char* kExpandT = R"HIP(
__device__ void expand_t(ShotState& st, uint32_t axis, bool dagger) {
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
__device__ void expand_rot(ShotState& st, uint32_t axis, double z_re, double z_im) {
    uint64_t half = 1ULL << st.active_k;
    bool px = bit_get(st.px, axis);
    double im = px ? -z_im : z_im;
    GpuComplex phase = {(float)z_re, (float)im};
    for (uint64_t i = 0; i < half; ++i) st.v[i + half] = cmul(st.v[i], phase);
    st.active_k++;
}
)HIP";

static const char* kMeasDormantRandom = R"HIP(
__device__ void meas_dormant_random(ShotState& st, Rng& rng, uint32_t axis,
                                    uint32_t classical_idx, bool sign) {
    uint8_t m_abs = rng.uniform() < 0.5 ? 0 : 1;
    bit_set(st.px, axis, m_abs != 0);
    bit_set(st.pz, axis, false);
    st.meas[classical_idx] = m_abs ^ (uint8_t)sign;
}
)HIP";

static const char* kMeasActiveDiag = R"HIP(
__device__ void meas_active_diagonal(ShotState& st, Rng& rng, uint32_t axis,
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
__device__ void meas_active_interfere(ShotState& st, Rng& rng, uint32_t axis,
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
__device__ void swap_meas_interfere(ShotState& st, Rng& rng, uint32_t from, uint32_t to,
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
__device__ void exec_exp_val_compiled(ShotState& st, const GpuExpValMask* masks,
                                      uint32_t cp_idx, uint32_t ev_idx) {
    const GpuExpValMask& mask = masks[cp_idx];
    bool frame_sign = mask.sign != 0;
    int parity = __popcll(mask.x[0] & st.pz[0]) + __popcll(mask.x[1] & st.pz[1])
               + __popcll(mask.z[0] & st.px[0]) + __popcll(mask.z[1] & st.px[1]);
    if (parity & 1) frame_sign = !frame_sign;
    uint32_t k = st.active_k;
    uint64_t active_mask_w0 = (k >= 64) ? ~uint64_t{0} : ((k == 0) ? 0 : ((uint64_t{1} << k) - 1));
    uint64_t dormant_x = mask.x[0] & ~active_mask_w0;
    if (k < 64) dormant_x |= mask.x[1];
    if (dormant_x != 0) { st.exp_vals[ev_idx] = 0.0; return; }
    uint64_t x_active = mask.x[0] & active_mask_w0;
    uint64_t z_active = mask.z[0] & active_mask_w0;
    uint32_t yz_count = __popcll(x_active & z_active);
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
        double z_sign = (__popcll(j & z_active) & 1) ? -1.0 : 1.0;
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
        out << "static __device__ const GpuMask kPauliMasks["
            << flat.pauli_masks.size() << "] = {\n";
        for (size_t i = 0; i < flat.pauli_masks.size(); ++i) {
            out << "    " << emit_gpu_mask(flat.pauli_masks[i]);
            if (i + 1 < flat.pauli_masks.size()) out << ",";
            out << "\n";
        }
        out << "};\n";
    }

    if ((uf.noise || uf.noise_block) && !flat.noise_sites.empty()) {
        out << "static __device__ const GpuNoiseSite kNoiseSites["
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

        out << "static __device__ const GpuChannel kNoiseChannels["
            << flat.noise_channels.size() << "] = {\n";
        for (size_t i = 0; i < flat.noise_channels.size(); ++i) {
            out << "    " << emit_gpu_channel(flat.noise_channels[i]);
            if (i + 1 < flat.noise_channels.size()) out << ",";
            out << "\n";
        }
        out << "};\n";

        out << "static __device__ const double kNoiseHazards["
            << flat.noise_hazards.size() << "] = {\n";
        for (size_t i = 0; i < flat.noise_hazards.size(); ++i) {
            out << "    " << hex_double(flat.noise_hazards[i]);
            if (i + 1 < flat.noise_hazards.size()) out << ",";
            out << "\n";
        }
        out << "};\n";
    }

    if (uf.readout_noise && !flat.readout_noise.empty()) {
        out << "static __device__ const GpuReadoutNoise kReadoutNoise["
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
        out << "static __device__ const uint32_t kDetectorOffsets["
            << flat.detector_offsets.size() << "] = {";
        for (size_t i = 0; i < flat.detector_offsets.size(); ++i) {
            if (i) out << ", ";
            out << flat.detector_offsets[i] << "u";
        }
        out << "};\n";
        out << "static __device__ const uint32_t kDetectorTargets["
            << flat.detector_targets.size() << "] = {";
        for (size_t i = 0; i < flat.detector_targets.size(); ++i) {
            if (i) out << ", ";
            out << flat.detector_targets[i] << "u";
        }
        out << "};\n";
    }

    if (uf.observable && !flat.observable_offsets.empty()) {
        out << "static __device__ const uint32_t kObservableOffsets["
            << flat.observable_offsets.size() << "] = {";
        for (size_t i = 0; i < flat.observable_offsets.size(); ++i) {
            if (i) out << ", ";
            out << flat.observable_offsets[i] << "u";
        }
        out << "};\n";
        out << "static __device__ const uint32_t kObservableTargets["
            << flat.observable_targets.size() << "] = {";
        for (size_t i = 0; i < flat.observable_targets.size(); ++i) {
            if (i) out << ", ";
            out << flat.observable_targets[i] << "u";
        }
        out << "};\n";
        out << "static __device__ const uint8_t kExpectedObservables["
            << flat.expected_observables.size() << "] = {";
        for (size_t i = 0; i < flat.expected_observables.size(); ++i) {
            if (i) out << ", ";
            out << (unsigned)flat.expected_observables[i] << "u";
        }
        out << "};\n";
    }

    if (uf.array_u2 && !flat.fused_u2.empty()) {
        out << "static __device__ const GpuFusedU2Entry kFusedU2["
            << flat.fused_u2.size() << "] = {\n";
        for (size_t i = 0; i < flat.fused_u2.size(); ++i) {
            out << "          " << emit_fused_u2(flat.fused_u2[i]);
            if (i + 1 < flat.fused_u2.size()) out << ",";
            out << "\n";
        }
        out << "};\n";
    }

    if (uf.array_u4 && !flat.fused_u4.empty()) {
        out << "static __device__ const GpuFusedU4Entry kFusedU4["
            << flat.fused_u4.size() << "] = {\n";
        for (size_t i = 0; i < flat.fused_u4.size(); ++i) {
            out << "          " << emit_fused_u4(flat.fused_u4[i]);
            if (i + 1 < flat.fused_u4.size()) out << ",";
            out << "\n";
        }
        out << "};\n";
    }

    if (uf.exp_val && !flat.exp_val_masks.empty()) {
        out << "static __device__ const GpuExpValMask kExpValMasks["
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
        local_passed    += __shfl_xor((int)local_passed, offset);
        local_logical   += __shfl_xor((int)local_logical, offset);
    }
    {
        __shared__ uint64_t wf_passed[4];
        __shared__ uint64_t wf_logical[4];
        uint32_t warp_id = tid >> 6;
        uint32_t lane = tid & 63;
        if (lane == 0) {
            wf_passed[warp_id] = local_passed;
            wf_logical[warp_id] = local_logical;
        }
        __syncthreads();
        // Phase 2: first 4 threads reduce the wavefront partials
        if (tid < 4) {
            local_passed = wf_passed[tid];
            local_logical = wf_logical[tid];
            for (int offset = 2; offset > 0; offset >>= 1) {
                local_passed  += __shfl_xor((int)local_passed, offset);
                local_logical += __shfl_xor((int)local_logical, offset);
            }
        }
    }
    // Observable reduction: per-observable warp shuffle
    for (uint32_t oi = 0; oi < num_observables; ++oi) {
        for (int offset = 32; offset > 0; offset >>= 1) {
            local_obs[oi] += __shfl_xor((int)local_obs[oi], offset);
        }
    }
    {
        __shared__ uint64_t wf_obs[kMaxObs][4];
        uint32_t warp_id = tid >> 6;
        uint32_t lane = tid & 63;
        for (uint32_t oi = 0; oi < num_observables; ++oi) {
            if (lane == 0) wf_obs[oi][warp_id] = local_obs[oi];
        }
        __syncthreads();
        if (tid < 4) {
            for (uint32_t oi = 0; oi < num_observables; ++oi) {
                local_obs[oi] = wf_obs[oi][tid];
                for (int offset = 2; offset > 0; offset >>= 1) {
                    local_obs[oi] += __shfl_xor((int)local_obs[oi], offset);
                }
            }
        }
    }
    // Exp-val reduction
    for (uint32_t ei = 0; ei < num_exp_vals; ++ei) {
        for (int offset = 32; offset > 0; offset >>= 1) {
            local_exp[ei] += __shfl_xor(local_exp[ei], offset);
        }
    }
    {
        __shared__ double wf_exp[kMaxExpVals][4];
        uint32_t warp_id = tid >> 6;
        uint32_t lane = tid & 63;
        for (uint32_t ei = 0; ei < num_exp_vals; ++ei) {
            if (lane == 0) wf_exp[ei][warp_id] = local_exp[ei];
        }
        __syncthreads();
        if (tid < 4) {
            for (uint32_t ei = 0; ei < num_exp_vals; ++ei) {
                local_exp[ei] = wf_exp[ei][tid];
                for (int offset = 2; offset > 0; offset >>= 1) {
                    local_exp[ei] += __shfl_xor(local_exp[ei], offset);
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
        block_counts[blockIdx.x] = out;
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
    out << "\n__global__ void compiled_sample_kernel(\n"
        << "    uint64_t shot_offset, uint64_t shots, uint64_t seed,\n"
        << "    BlockCounts* block_counts,\n"
        << "    uint32_t num_observables, uint32_t num_exp_vals\n"
        << ") {\n"
        << "    uint32_t tid = threadIdx.x;\n"
        << "    uint64_t batch_shot_id = (uint64_t)blockIdx.x * blockDim.x + tid;\n"
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

}  // namespace gpu
}  // namespace clifft
