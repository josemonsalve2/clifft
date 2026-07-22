#include "clifft/gpu/gpu_sampler.h"
#include "clifft/gpu/device_program.h"
#include "clifft/gpu/codegen/kernel_cache.h"
#include "clifft/gpu/codegen/kernel_codegen.h"
#include "clifft/gpu/runtime/hsa_kernel_dispatch.h"
#include "clifft/gpu/runtime/hsa_persistent_dispatch.h"
#ifdef CLIFFT_ENABLE_MLIR
#include "clifft/gpu/mlir/mlir_kernel_cache.h"
#endif

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace clifft {
namespace gpu {
namespace {

using clifft::gpu::GpuComplex;
using clifft::gpu::GpuInstr;
using clifft::gpu::GpuMask;
using clifft::gpu::GpuChannel;
using clifft::gpu::GpuNoiseSite;
using clifft::gpu::GpuReadoutNoise;
using clifft::gpu::GpuFusedU2Entry;
using clifft::gpu::GpuFusedU4Entry;
using clifft::gpu::GpuExpValMask;
using clifft::gpu::GpuProgram;
using clifft::gpu::BlockCounts;
using clifft::gpu::DeviceBuffers;

inline void check_hip(hipError_t err, const char* what) {
    if (err != hipSuccess) {
        std::ostringstream ss;
        ss << what << ": " << hipGetErrorString(err);
        throw std::runtime_error(ss.str());
    }
}

template <typename T>
void copy_to_device(T*& dst, const std::vector<T>& src, const char* name) {
    if (src.empty()) {
        dst = nullptr;
        return;
    }
    check_hip(hipMalloc(reinterpret_cast<void**>(&dst), src.size() * sizeof(T)), name);
    check_hip(hipMemcpy(dst, src.data(), src.size() * sizeof(T), hipMemcpyHostToDevice), name);
}

void free_buffers(DeviceBuffers& b) {
    hipFree(b.instrs);
    hipFree(b.pauli_masks);
    hipFree(b.noise_sites);
    hipFree(b.noise_channels);
    hipFree(b.noise_hazards);
    hipFree(b.readout_noise);
    hipFree(b.detector_offsets);
    hipFree(b.detector_targets);
    hipFree(b.observable_offsets);
    hipFree(b.observable_targets);
    hipFree(b.expected_observables);
    hipFree(b.block_counts);
    hipFree(b.global_v);
    hipFree(b.global_scratch);
    hipFree(b.work_counter);
    hipFree(b.fused_u2);
    hipFree(b.fused_u4);
    hipFree(b.exp_val_masks);
}

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

        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];

        s[2] ^= t;
        s[3] = rotl64(s[3], 45);
        return result;
    }

    __device__ double uniform() {
        return static_cast<double>(next() >> 11) * 0x1.0p-53;
    }
};

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

__device__ inline bool bit_get(const uint64_t* words, uint32_t idx) {
    return ((words[idx >> 6] >> (idx & 63u)) & 1ULL) != 0;
}

__device__ inline void bit_set(uint64_t* words, uint32_t idx, bool value) {
    uint64_t mask = 1ULL << (idx & 63u);
    uint32_t word = idx >> 6;
    if (value) {
        words[word] |= mask;
    } else {
        words[word] &= ~mask;
    }
}

__device__ inline void bit_xor(uint64_t* words, uint32_t idx, bool value) {
    if (value) {
        words[idx >> 6] ^= 1ULL << (idx & 63u);
    }
}

__device__ inline void bit_swap(uint64_t* a, uint32_t ia, uint64_t* b, uint32_t ib) {
    bool va = bit_get(a, ia);
    bool vb = bit_get(b, ib);
    if (va != vb) {
        a[ia >> 6] ^= 1ULL << (ia & 63u);
        b[ib >> 6] ^= 1ULL << (ib & 63u);
    }
}

__device__ inline GpuComplex cadd(GpuComplex a, GpuComplex b) {
    return {a.re + b.re, a.im + b.im};
}

__device__ inline GpuComplex csub(GpuComplex a, GpuComplex b) {
    return {a.re - b.re, a.im - b.im};
}

__device__ inline GpuComplex cscale(GpuComplex a, double s) {
    return {static_cast<float>(static_cast<double>(a.re) * s),
            static_cast<float>(static_cast<double>(a.im) * s)};
}

__device__ inline GpuComplex cmul(GpuComplex a, GpuComplex b) {
    return {static_cast<float>(a.re * b.re - a.im * b.im),
            static_cast<float>(a.re * b.im + a.im * b.re)};
}

__device__ inline double cnorm(GpuComplex a) {
    return static_cast<double>(a.re) * static_cast<double>(a.re) +
           static_cast<double>(a.im) * static_cast<double>(a.im);
}

// 64-bit LDS load/store helpers: load/store both re and im in a single
// ds_read_b64/ds_write_b64 instead of two ds_read_b32 instructions.
// GpuComplex must be aligned(8) for this to work correctly.
__device__ __forceinline__ GpuComplex load_complex64(const GpuComplex* ptr) {
    uint64_t raw = *reinterpret_cast<const uint64_t*>(ptr);
    GpuComplex c;
    c.re = __uint_as_float(static_cast<uint32_t>(raw));
    c.im = __uint_as_float(static_cast<uint32_t>(raw >> 32));
    return c;
}

__device__ __forceinline__ void store_complex64(GpuComplex* ptr, GpuComplex c) {
    uint64_t raw = static_cast<uint64_t>(__float_as_uint(c.re)) |
                   (static_cast<uint64_t>(__float_as_uint(c.im)) << 32);
    *reinterpret_cast<uint64_t*>(ptr) = raw;
}

__device__ inline uint64_t insert_zero_bit(uint64_t val, uint32_t pos) {
    uint64_t mask = (1ULL << pos) - 1ULL;
    return (val & mask) | ((val & ~mask) << 1);
}

__device__ inline uint64_t scatter_bits_1(uint64_t val, uint32_t bit_pos) {
    return insert_zero_bit(val, bit_pos);
}

__device__ inline uint64_t scatter_bits_2(uint64_t val, uint32_t bit1, uint32_t bit2) {
    uint32_t lo = min(bit1, bit2);
    uint32_t hi = max(bit1, bit2);
    val = insert_zero_bit(val, lo);
    return insert_zero_bit(val, hi);
}

__device__ inline uint8_t sample_branch(Rng& rng, double prob0, double prob1, double total) {
    double eps = kDustEpsilon * total;
    if (prob1 <= eps) {
        return 0;
    }
    if (prob0 <= eps) {
        return 1;
    }
    return (rng.uniform() * total < prob0) ? 0 : 1;
}

__device__ void draw_next_noise(ShotState& st, const GpuProgram& program, Rng& rng) {
    if (program.num_noise_sites == 0 || st.next_noise_idx >= program.num_noise_sites) {
        st.next_noise_idx = 0xffffffffu;
        return;
    }
    double current_hazard =
        (st.next_noise_idx == 0) ? 0.0 : program.noise_hazards[st.next_noise_idx - 1];
    double u = rng.uniform();
    double gap = -log(1.0 - u);
    double target = current_hazard + gap;

    uint32_t lo = 0;
    uint32_t hi = program.num_noise_sites;
    while (lo < hi) {
        uint32_t mid = lo + ((hi - lo) >> 1);
        if (program.noise_hazards[mid] <= target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    st.next_noise_idx = (lo >= program.num_noise_sites) ? 0xffffffffu : lo;
}

__device__ inline void frame_cnot(ShotState& st, uint32_t c, uint32_t t) {
    bool px_c = bit_get(st.px, c);
    bool pz_t = bit_get(st.pz, t);
    bit_xor(st.px, t, px_c);
    bit_xor(st.pz, c, pz_t);
}

__device__ inline void frame_cz(ShotState& st, uint32_t c, uint32_t t) {
    bool px_c = bit_get(st.px, c);
    bool px_t = bit_get(st.px, t);
    bit_xor(st.pz, t, px_c);
    bit_xor(st.pz, c, px_t);
}

__device__ inline void frame_h(ShotState& st, uint32_t axis) {
    bool px = bit_get(st.px, axis);
    bool pz = bit_get(st.pz, axis);
    bit_set(st.px, axis, pz);
    bit_set(st.pz, axis, px);
}

__device__ inline void frame_s(ShotState& st, uint32_t axis) {
    bool px = bit_get(st.px, axis);
    bit_xor(st.pz, axis, px);
}

__device__ inline void frame_swap(ShotState& st, uint32_t a, uint32_t b) {
    bit_swap(st.px, a, st.px, b);
    bit_swap(st.pz, a, st.pz, b);
}

__device__ void array_cnot(ShotState& st, uint32_t c, uint32_t t) {
    uint64_t c_bit = 1ULL << c;
    uint64_t t_bit = 1ULL << t;
    uint64_t iters = 1ULL << (st.active_k - 2);
    for (uint64_t i = 0; i < iters; ++i) {
        uint64_t base = scatter_bits_2(i, c, t) | c_bit;
        GpuComplex tmp = st.v[base];
        st.v[base] = st.v[base | t_bit];
        st.v[base | t_bit] = tmp;
    }
    frame_cnot(st, c, t);
}

__device__ void array_cz(ShotState& st, uint32_t a, uint32_t b) {
    uint64_t both_bits = (1ULL << a) | (1ULL << b);
    uint64_t iters = 1ULL << (st.active_k - 2);
    for (uint64_t i = 0; i < iters; ++i) {
        uint64_t idx = scatter_bits_2(i, a, b) | both_bits;
        st.v[idx].re = -st.v[idx].re;
        st.v[idx].im = -st.v[idx].im;
    }
    frame_cz(st, a, b);
}

__device__ void array_swap(ShotState& st, uint32_t a, uint32_t b) {
    uint64_t a_bit = 1ULL << a;
    uint64_t b_bit = 1ULL << b;
    uint64_t iters = 1ULL << (st.active_k - 2);
    for (uint64_t i = 0; i < iters; ++i) {
        uint64_t base = scatter_bits_2(i, a, b);
        GpuComplex tmp = st.v[base | a_bit];
        st.v[base | a_bit] = st.v[base | b_bit];
        st.v[base | b_bit] = tmp;
    }
    frame_swap(st, a, b);
}

__device__ void array_multi_cnot(ShotState& st, uint32_t target, uint64_t ctrl_mask) {
    uint64_t t_bit = 1ULL << target;
    uint64_t half = 1ULL << (st.active_k - 1);
    for (uint64_t idx = 0; idx < half; ++idx) {
        uint64_t actual = scatter_bits_1(idx, target);
        bool parity = (__popcll(actual & ctrl_mask) & 1) != 0;
        if (parity) {
            GpuComplex tmp = st.v[actual];
            st.v[actual] = st.v[actual | t_bit];
            st.v[actual | t_bit] = tmp;
        }
    }
    for (uint32_t c = 0; c < st.active_k; ++c) {
        if ((ctrl_mask >> c) & 1ULL) {
            frame_cnot(st, c, target);
        }
    }
}

__device__ void array_multi_cz(ShotState& st, uint32_t control, uint64_t target_mask) {
    uint64_t c_bit = 1ULL << control;
    uint64_t half = 1ULL << (st.active_k - 1);
    for (uint64_t idx = 0; idx < half; ++idx) {
        uint64_t actual = scatter_bits_1(idx, control) | c_bit;
        bool negate = (__popcll(actual & target_mask) & 1) != 0;
        if (negate) {
            st.v[actual].re = -st.v[actual].re;
            st.v[actual].im = -st.v[actual].im;
        }
    }
    for (uint32_t t = 0; t < st.active_k; ++t) {
        if ((target_mask >> t) & 1ULL) {
            frame_cz(st, control, t);
        }
    }
}

__device__ void array_h(ShotState& st, uint32_t axis) {
    uint64_t axis_bit = 1ULL << axis;
    uint64_t iters = 1ULL << (st.active_k - 1);
    for (uint64_t i = 0; i < iters; ++i) {
        uint64_t idx0 = scatter_bits_1(i, axis);
        uint64_t idx1 = idx0 | axis_bit;
        GpuComplex a = st.v[idx0];
        GpuComplex b = st.v[idx1];
        st.v[idx0] = cscale(cadd(a, b), kInvSqrt2);
        st.v[idx1] = cscale(csub(a, b), kInvSqrt2);
    }
    frame_h(st, axis);
}

__device__ void apply_phase(ShotState& st, uint32_t axis, GpuComplex phase) {
    uint64_t axis_bit = 1ULL << axis;
    uint64_t iters = 1ULL << (st.active_k - 1);
    for (uint64_t i = 0; i < iters; ++i) {
        uint64_t idx = scatter_bits_1(i, axis) | axis_bit;
        st.v[idx] = cmul(st.v[idx], phase);
    }
}

__device__ void array_s(ShotState& st, uint32_t axis, bool dagger) {
    apply_phase(st, axis, dagger ? GpuComplex{0.0f, -1.0f} : GpuComplex{0.0f, 1.0f});
    frame_s(st, axis);
}

__device__ void array_t(ShotState& st, uint32_t axis, bool dagger) {
    bool px = bit_get(st.px, axis);
    if (axis >= st.active_k) {
        return;
    }
    double imag = dagger ? -kInvSqrt2 : kInvSqrt2;
    if (px) {
        imag = -imag;
    }
    apply_phase(st, axis, {static_cast<float>(kInvSqrt2), static_cast<float>(imag)});
}

__device__ void expand_plain(ShotState& st) {
    uint64_t half = 1ULL << st.active_k;
    for (uint64_t i = 0; i < half; ++i) {
        st.v[i + half] = st.v[i];
    }
    st.active_k++;
}

__device__ void expand_t(ShotState& st, uint32_t axis, bool dagger) {
    uint64_t half = 1ULL << st.active_k;
    bool px = bit_get(st.px, axis);
    double imag = dagger ? -kInvSqrt2 : kInvSqrt2;
    if (px) {
        imag = -imag;
    }
    GpuComplex phase{static_cast<float>(kInvSqrt2), static_cast<float>(imag)};
    for (uint64_t i = 0; i < half; ++i) {
        st.v[i + half] = cmul(st.v[i], phase);
    }
    st.active_k++;
}

__device__ void meas_dormant_random(ShotState& st, Rng& rng, uint32_t axis,
                                    uint32_t classical_idx, bool sign) {
    uint8_t m_abs = rng.uniform() < 0.5 ? 0 : 1;
    bit_set(st.px, axis, m_abs != 0);
    bit_set(st.pz, axis, false);
    st.meas[classical_idx] = m_abs ^ static_cast<uint8_t>(sign);
}

__device__ void meas_active_diagonal(ShotState& st, Rng& rng, uint32_t axis,
                                     uint32_t classical_idx, bool sign) {
    uint64_t half = 1ULL << (st.active_k - 1);
    bool px = bit_get(st.px, axis);
    double p0 = 0.0;
    double p1 = 0.0;
    for (uint64_t i = 0; i < half; ++i) {
        p0 += cnorm(st.v[i]);
        p1 += cnorm(st.v[i + half]);
    }
    double total = p0 + p1;
    uint8_t b = sample_branch(rng, p0, p1, total);
    uint8_t m_abs = b ^ static_cast<uint8_t>(px);
    st.meas[classical_idx] = m_abs ^ static_cast<uint8_t>(sign);
    if (b != 0) {
        for (uint64_t i = 0; i < half; ++i) {
            st.v[i] = st.v[i + half];
        }
    }
    st.active_k--;
    bit_set(st.px, axis, m_abs != 0);
    bit_set(st.pz, axis, false);
}

__device__ void meas_active_interfere(ShotState& st, Rng& rng, uint32_t axis,
                                      uint32_t classical_idx, bool sign) {
    uint64_t half = 1ULL << (st.active_k - 1);
    bool pz = bit_get(st.pz, axis);
    double p_plus = 0.0;
    double p_minus = 0.0;
    for (uint64_t i = 0; i < half; ++i) {
        GpuComplex sum = cadd(st.v[i], st.v[i + half]);
        GpuComplex diff = csub(st.v[i], st.v[i + half]);
        p_plus += cnorm(sum);
        p_minus += cnorm(diff);
    }
    double total = p_plus + p_minus;
    uint8_t b_x = sample_branch(rng, p_plus, p_minus, total);
    uint8_t m_abs = b_x ^ static_cast<uint8_t>(pz);
    st.meas[classical_idx] = m_abs ^ static_cast<uint8_t>(sign);

    for (uint64_t i = 0; i < half; ++i) {
        GpuComplex folded = (b_x == 0) ? cadd(st.v[i], st.v[i + half])
                                       : csub(st.v[i], st.v[i + half]);
        st.v[i] = cscale(folded, kInvSqrt2);
    }
    st.active_k--;
    bit_set(st.px, axis, m_abs != 0);
    bit_set(st.pz, axis, false);
}

__device__ void swap_meas_interfere(ShotState& st, Rng& rng, uint32_t from, uint32_t to,
                                    uint32_t classical_idx, bool sign) {
    if (from == to) {
        meas_active_interfere(st, rng, to, classical_idx, sign);
        return;
    }

    frame_swap(st, from, to);
    bool pz = bit_get(st.pz, to);
    uint64_t half = 1ULL << to;
    uint64_t f_bit = 1ULL << from;

    double p_plus = 0.0;
    double p_minus = 0.0;
    for (uint64_t idx = 0; idx < half; ++idx) {
        uint64_t b_f = (idx >> from) & 1ULL;
        uint64_t base = (idx & ~f_bit) | (b_f << to);
        GpuComplex sum = cadd(st.v[base], st.v[base | f_bit]);
        GpuComplex diff = csub(st.v[base], st.v[base | f_bit]);
        p_plus += cnorm(sum);
        p_minus += cnorm(diff);
    }
    double total = p_plus + p_minus;
    uint8_t b_x = sample_branch(rng, p_plus, p_minus, total);

    for (uint64_t idx = 0; idx < half; ++idx) {
        uint64_t b_f = (idx >> from) & 1ULL;
        uint64_t base = (idx & ~f_bit) | (b_f << to);
        st.v[idx] = cscale((b_x == 0) ? cadd(st.v[base], st.v[base | f_bit])
                                      : csub(st.v[base], st.v[base | f_bit]),
                           kInvSqrt2);
    }

    st.active_k--;
    uint8_t m_abs = b_x ^ static_cast<uint8_t>(pz);
    st.meas[classical_idx] = m_abs ^ static_cast<uint8_t>(sign);
    bit_set(st.px, to, m_abs != 0);
    bit_set(st.pz, to, false);
}

__device__ void apply_pauli_to_frame(ShotState& st, const uint64_t* x, const uint64_t* z) {
    st.px[0] ^= x[0];
    st.px[1] ^= x[1];
    st.pz[0] ^= z[0];
    st.pz[1] ^= z[1];
}

__device__ void apply_pauli(ShotState& st, const GpuProgram& program, uint32_t mask_idx,
                            uint32_t condition_idx) {
    if (st.meas[condition_idx] == 0) {
        return;
    }
    const GpuMask& mask = program.pauli_masks[mask_idx];
    apply_pauli_to_frame(st, mask.x, mask.z);
}

__device__ void exec_noise(ShotState& st, const GpuProgram& program, Rng& rng, uint32_t site_idx) {
    if (site_idx != st.next_noise_idx) {
        return;
    }
    const GpuNoiseSite& site = program.noise_sites[site_idx];
    double roll = rng.uniform() * site.prob_sum;
    double cumulative = 0.0;
    for (uint32_t k = 0; k < site.count; ++k) {
        const GpuChannel& ch = program.noise_channels[site.offset + k];
        cumulative += ch.prob;
        if (roll < cumulative) {
            apply_pauli_to_frame(st, ch.x, ch.z);
            break;
        }
    }
    st.next_noise_idx = site_idx + 1;
    draw_next_noise(st, program, rng);
}

__device__ void exec_noise_block(ShotState& st, const GpuProgram& program, Rng& rng,
                                 uint32_t start, uint32_t count) {
    uint32_t end = start + count;
    while (st.next_noise_idx >= start && st.next_noise_idx < end) {
        exec_noise(st, program, rng, st.next_noise_idx);
    }
}

__device__ void exec_postselect(ShotState& st, const GpuProgram& program, const GpuInstr& instr) {
    uint32_t start = program.detector_offsets[instr.a];
    uint32_t end = program.detector_offsets[instr.a + 1];
    uint8_t parity = (instr.flags & kFlagExpectedOne) ? 1 : 0;
    for (uint32_t i = start; i < end; ++i) {
        parity ^= st.meas[program.detector_targets[i]];
    }
    if (parity != 0) {
        st.discarded = true;
    }
}

__device__ void exec_observable(ShotState& st, const GpuProgram& program, const GpuInstr& instr) {
    uint32_t start = program.observable_offsets[instr.a];
    uint32_t end = program.observable_offsets[instr.a + 1];
    uint8_t parity = 0;
    for (uint32_t i = start; i < end; ++i) {
        parity ^= st.meas[program.observable_targets[i]];
    }
    st.obs[instr.b] ^= parity;
}

__device__ __noinline__ void array_rot(ShotState& st, uint32_t axis, double z_re, double z_im) {
    bool px = bit_get(st.px, axis);
    if (axis >= st.active_k) return;
    double im = px ? -z_im : z_im;
    apply_phase(st, axis, {static_cast<float>(z_re), static_cast<float>(im)});
}

__device__ __noinline__ void expand_rot(ShotState& st, uint32_t axis, double z_re, double z_im) {
    uint64_t half = 1ULL << st.active_k;
    bool px = bit_get(st.px, axis);
    double im = px ? -z_im : z_im;
    GpuComplex phase{static_cast<float>(z_re), static_cast<float>(im)};
    for (uint64_t i = 0; i < half; ++i) {
        st.v[i + half] = cmul(st.v[i], phase);
    }
    st.active_k++;
}

__device__ __noinline__ void array_u2(ShotState& st, const GpuProgram& program, uint32_t axis, uint32_t cp_idx) {
    const auto& entry = program.fused_u2[cp_idx];
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

__device__ __noinline__ void array_u4(ShotState& st, const GpuProgram& program,
                         uint32_t axis_lo, uint32_t axis_hi, uint32_t cp_idx) {
    const auto& node = program.fused_u4[cp_idx];
    uint8_t px_lo = bit_get(st.px, axis_lo) ? 1 : 0;
    uint8_t pz_lo = bit_get(st.pz, axis_lo) ? 1 : 0;
    uint8_t px_hi = bit_get(st.px, axis_hi) ? 1 : 0;
    uint8_t pz_hi = bit_get(st.pz, axis_hi) ? 1 : 0;
    uint8_t in_state = (pz_hi << 3) | (px_hi << 2) | (pz_lo << 1) | px_lo;
    const auto& entry = node.entries[in_state];
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
        const auto& mat = entry.matrix;
        st.v[i0] = cadd(cadd(cmul(v0, mat[0][0]), cmul(v1, mat[0][1])),
                        cadd(cmul(v2, mat[0][2]), cmul(v3, mat[0][3])));
        st.v[i1] = cadd(cadd(cmul(v0, mat[1][0]), cmul(v1, mat[1][1])),
                        cadd(cmul(v2, mat[1][2]), cmul(v3, mat[1][3])));
        st.v[i2] = cadd(cadd(cmul(v0, mat[2][0]), cmul(v1, mat[2][1])),
                        cadd(cmul(v2, mat[2][2]), cmul(v3, mat[2][3])));
        st.v[i3] = cadd(cadd(cmul(v0, mat[3][0]), cmul(v1, mat[3][1])),
                        cadd(cmul(v2, mat[3][2]), cmul(v3, mat[3][3])));
    }
}

__device__ __noinline__ void exec_exp_val(ShotState& st, const GpuProgram& program,
                             uint32_t cp_idx, uint32_t ev_idx) {
    const GpuExpValMask& mask = program.exp_val_masks[cp_idx];
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
        double norm_j = cnorm(st.v[j]);
        denom += norm_j;
        uint64_t j_xor = j ^ x_active;
        double z_sign = (__popcll(j & z_active) & 1) ? -1.0 : 1.0;
        double a_re = static_cast<double>(st.v[j_xor].re);
        double a_im = static_cast<double>(-st.v[j_xor].im);
        double b_re = z_sign * static_cast<double>(st.v[j].re);
        double b_im = z_sign * static_cast<double>(st.v[j].im);
        num_re += a_re * b_re - a_im * b_im;
        num_im += a_re * b_im + a_im * b_re;
    }
    double result_re = num_re * phase_re - num_im * phase_im;
    st.exp_vals[ev_idx] = (denom > 0.0) ? result_re / denom : 0.0;
}

__device__ void execute_shot(const GpuProgram& program, Rng& rng, ShotState& st) {
    for (uint32_t pc = 0; pc < program.num_instrs; ++pc) {
        const GpuInstr& instr = program.instrs[pc];
        switch (instr.opcode) {
            case static_cast<uint8_t>(clifft::Opcode::OP_FRAME_CNOT):
                frame_cnot(st, instr.axis_1, instr.axis_2);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_FRAME_CZ):
                frame_cz(st, instr.axis_1, instr.axis_2);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_FRAME_H):
                frame_h(st, instr.axis_1);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_FRAME_S):
            case static_cast<uint8_t>(clifft::Opcode::OP_FRAME_S_DAG):
                frame_s(st, instr.axis_1);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_FRAME_SWAP):
                frame_swap(st, instr.axis_1, instr.axis_2);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_ARRAY_CNOT):
                array_cnot(st, instr.axis_1, instr.axis_2);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_ARRAY_CZ):
                array_cz(st, instr.axis_1, instr.axis_2);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_ARRAY_SWAP):
                array_swap(st, instr.axis_1, instr.axis_2);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_ARRAY_MULTI_CNOT):
                array_multi_cnot(st, instr.axis_1, instr.mask);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_ARRAY_MULTI_CZ):
                array_multi_cz(st, instr.axis_1, instr.mask);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_ARRAY_H):
                array_h(st, instr.axis_1);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_ARRAY_S):
                array_s(st, instr.axis_1, false);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_ARRAY_S_DAG):
                array_s(st, instr.axis_1, true);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_ARRAY_T):
                array_t(st, instr.axis_1, false);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_ARRAY_T_DAG):
                array_t(st, instr.axis_1, true);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_EXPAND):
                expand_plain(st);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_EXPAND_T):
                expand_t(st, instr.axis_1, false);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_EXPAND_T_DAG):
                expand_t(st, instr.axis_1, true);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_MEAS_DORMANT_STATIC):
                if (instr.flags & kFlagIdentity) {
                    st.meas[instr.a] = (instr.flags & kFlagSign) ? 1 : 0;
                } else {
                    uint8_t outcome = bit_get(st.px, instr.axis_1) ? 1 : 0;
                    st.meas[instr.a] = outcome ^ static_cast<uint8_t>((instr.flags & kFlagSign) != 0);
                }
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_MEAS_DORMANT_RANDOM):
                meas_dormant_random(st, rng, instr.axis_1, instr.a,
                                    (instr.flags & kFlagSign) != 0);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_MEAS_ACTIVE_DIAGONAL):
                meas_active_diagonal(st, rng, instr.axis_1, instr.a,
                                     (instr.flags & kFlagSign) != 0);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_MEAS_ACTIVE_INTERFERE):
                meas_active_interfere(st, rng, instr.axis_1, instr.a,
                                      (instr.flags & kFlagSign) != 0);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_SWAP_MEAS_INTERFERE):
                swap_meas_interfere(st, rng, instr.axis_1, instr.axis_2, instr.a,
                                    (instr.flags & kFlagSign) != 0);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_APPLY_PAULI):
                apply_pauli(st, program, instr.a, instr.b);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_NOISE):
                exec_noise(st, program, rng, instr.a);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_NOISE_BLOCK):
                exec_noise_block(st, program, rng, instr.a, instr.b);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_READOUT_NOISE): {
                const GpuReadoutNoise& r = program.readout_noise[instr.a];
                if (rng.uniform() < r.prob) {
                    st.meas[r.meas_idx] ^= 1;
                }
                break;
            }
            case static_cast<uint8_t>(clifft::Opcode::OP_DETECTOR):
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_POSTSELECT):
                exec_postselect(st, program, instr);
                if (st.discarded) {
                    return;
                }
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_OBSERVABLE):
                exec_observable(st, program, instr);
                break;
            default:
                switch (instr.opcode) {
                    case static_cast<uint8_t>(clifft::Opcode::OP_EXPAND_ROT):
                        expand_rot(st, instr.axis_1, instr.weight_re, instr.weight_im);
                        break;
                    case static_cast<uint8_t>(clifft::Opcode::OP_ARRAY_ROT):
                        array_rot(st, instr.axis_1, instr.weight_re, instr.weight_im);
                        break;
                    case static_cast<uint8_t>(clifft::Opcode::OP_ARRAY_U2):
                        array_u2(st, program, instr.axis_1, instr.a);
                        break;
                    case static_cast<uint8_t>(clifft::Opcode::OP_ARRAY_U4):
                        array_u4(st, program, instr.axis_1, instr.axis_2, instr.a);
                        break;
                    case static_cast<uint8_t>(clifft::Opcode::OP_EXP_VAL):
                        exec_exp_val(st, program, instr.a, instr.b);
                        break;
                    default:
                        st.discarded = true;
                        return;
                }
                break;
        }
    }
}

struct CoopShotState {
    GpuComplex* v;
    GpuComplex* scratch;
    uint8_t* meas;
    uint8_t* obs;
    uint64_t* px;
    uint64_t* pz;
    uint32_t* active_k;
    uint32_t* next_noise_idx;
    uint8_t* discarded;
    uint8_t* branch;
    double* red0;
    double* red1;
    double* exp_vals;
    uint32_t* scatter_lut;       // precomputed index LUT in LDS (nullptr for global-coop)
    uint32_t* cached_axis1;      // last axis used for scatter LUT
    uint32_t* cached_axis2;      // last axis2 used for 2-axis LUT
    uint32_t* cached_lut_k;      // last active_k used for LUT
    uint32_t* cached_lut_mode;   // 0=none, 1=scatter_1, 2=scatter_2
};

// Precompute scatter_bits_1 LUT for all i in [0, 2^(active_k-1)).
// After this call, lut[i] == insert_zero_bit(i, axis).
__device__ void precompute_scatter_1(uint32_t* lut, uint32_t active_k, uint32_t axis) {
    uint32_t n = 1u << (active_k - 1);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        lut[i] = static_cast<uint32_t>(insert_zero_bit(i, axis));
    }
    __syncthreads();
}

// Precompute scatter_bits_2 LUT for all i in [0, 2^(active_k-2)).
// After this call, lut[i] == scatter_bits_2(i, axis1, axis2).
__device__ void precompute_scatter_2(uint32_t* lut, uint32_t active_k,
                                     uint32_t axis1, uint32_t axis2) {
    uint32_t n = 1u << (active_k - 2);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        lut[i] = static_cast<uint32_t>(scatter_bits_2(i, axis1, axis2));
    }
    __syncthreads();
}

// Ensure scatter_1 LUT is populated for the given (active_k, axis).
// Uses cached state to skip recomputation when possible.
__device__ void ensure_scatter_1(CoopShotState& st, uint32_t axis) {
    if (st.scatter_lut == nullptr) return;
    uint32_t k = *st.active_k;
    if (*st.cached_lut_mode == 1 && *st.cached_axis1 == axis && *st.cached_lut_k == k) {
        return;  // already populated
    }
    precompute_scatter_1(st.scatter_lut, k, axis);
    if (threadIdx.x == 0) {
        *st.cached_axis1 = axis;
        *st.cached_lut_k = k;
        *st.cached_lut_mode = 1;
    }
    // No extra syncthreads needed: precompute_scatter_1 already did one
}

// Ensure scatter_2 LUT is populated for the given (active_k, axis1, axis2).
__device__ void ensure_scatter_2(CoopShotState& st, uint32_t axis1, uint32_t axis2) {
    if (st.scatter_lut == nullptr) return;
    uint32_t k = *st.active_k;
    if (*st.cached_lut_mode == 2 && *st.cached_axis1 == axis1 &&
        *st.cached_axis2 == axis2 && *st.cached_lut_k == k) {
        return;  // already populated
    }
    precompute_scatter_2(st.scatter_lut, k, axis1, axis2);
    if (threadIdx.x == 0) {
        *st.cached_axis1 = axis1;
        *st.cached_axis2 = axis2;
        *st.cached_lut_k = k;
        *st.cached_lut_mode = 2;
    }
}

__device__ void coop_reduce2(CoopShotState& st, double local0, double local1, double& out0,
                             double& out1) {
    uint32_t tid = threadIdx.x;
    // Phase 1: intra-wavefront reduction via warp shuffle (AMD wavefront = 64 threads)
    for (int offset = 32; offset > 0; offset >>= 1) {
        local0 += __shfl_xor(local0, offset);
        local1 += __shfl_xor(local1, offset);
    }
    // Phase 2: inter-wavefront reduction via shared memory (4 wavefronts for blockDim=256)
    uint32_t warp_id = tid >> 6;   // tid / 64
    uint32_t lane = tid & 63;      // tid % 64
    if (lane == 0) {
        st.red0[warp_id] = local0;
        st.red1[warp_id] = local1;
    }
    __syncthreads();
    if (tid < 4) {
        local0 = st.red0[tid];
        local1 = st.red1[tid];
        // reduce 4 values with shuffle
        for (int offset = 2; offset > 0; offset >>= 1) {
            local0 += __shfl_xor(local0, offset);
            local1 += __shfl_xor(local1, offset);
        }
    }
    if (tid == 0) {
        st.red0[0] = local0;
        st.red1[0] = local1;
    }
    __syncthreads();
    out0 = st.red0[0];
    out1 = st.red1[0];
}

__device__ void coop_draw_next_noise(CoopShotState& st, const GpuProgram& program, Rng& rng) {
    if (program.num_noise_sites == 0 || *st.next_noise_idx >= program.num_noise_sites) {
        *st.next_noise_idx = 0xffffffffu;
        return;
    }
    double current_hazard =
        (*st.next_noise_idx == 0) ? 0.0 : program.noise_hazards[*st.next_noise_idx - 1];
    double target = current_hazard + (-log(1.0 - rng.uniform()));
    uint32_t lo = 0;
    uint32_t hi = program.num_noise_sites;
    while (lo < hi) {
        uint32_t mid = lo + ((hi - lo) >> 1);
        if (program.noise_hazards[mid] <= target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    *st.next_noise_idx = (lo >= program.num_noise_sites) ? 0xffffffffu : lo;
}

__device__ void coop_array_cnot(CoopShotState& st, uint32_t c, uint32_t t) {
    uint64_t c_bit = 1ULL << c;
    uint64_t t_bit = 1ULL << t;
    uint64_t iters = 1ULL << (*st.active_k - 2);
    ensure_scatter_2(st, c, t);
    for (uint64_t i = threadIdx.x; i < iters; i += blockDim.x) {
        uint64_t base = (st.scatter_lut ? st.scatter_lut[i]
                                        : static_cast<uint32_t>(scatter_bits_2(i, c, t)))
                        | c_bit;
        GpuComplex tmp = load_complex64(&st.v[base]);
        GpuComplex tmp2 = load_complex64(&st.v[base | t_bit]);
        store_complex64(&st.v[base], tmp2);
        store_complex64(&st.v[base | t_bit], tmp);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        bool px_c = bit_get(st.px, c);
        bool pz_t = bit_get(st.pz, t);
        bit_xor(st.px, t, px_c);
        bit_xor(st.pz, c, pz_t);
    }
    __syncthreads();
}

__device__ void coop_array_cz(CoopShotState& st, uint32_t a, uint32_t b) {
    uint64_t both_bits = (1ULL << a) | (1ULL << b);
    uint64_t iters = 1ULL << (*st.active_k - 2);
    ensure_scatter_2(st, a, b);
    for (uint64_t i = threadIdx.x; i < iters; i += blockDim.x) {
        uint64_t idx = (st.scatter_lut ? st.scatter_lut[i]
                                       : static_cast<uint32_t>(scatter_bits_2(i, a, b)))
                       | both_bits;
        GpuComplex c = load_complex64(&st.v[idx]);
        c.re = -c.re;
        c.im = -c.im;
        store_complex64(&st.v[idx], c);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        bool px_a = bit_get(st.px, a);
        bool px_b = bit_get(st.px, b);
        bit_xor(st.pz, b, px_a);
        bit_xor(st.pz, a, px_b);
    }
    __syncthreads();
}

__device__ void coop_array_swap(CoopShotState& st, uint32_t a, uint32_t b) {
    uint64_t a_bit = 1ULL << a;
    uint64_t b_bit = 1ULL << b;
    uint64_t iters = 1ULL << (*st.active_k - 2);
    ensure_scatter_2(st, a, b);
    for (uint64_t i = threadIdx.x; i < iters; i += blockDim.x) {
        uint64_t base = st.scatter_lut ? st.scatter_lut[i]
                                       : static_cast<uint32_t>(scatter_bits_2(i, a, b));
        GpuComplex tmp_a = load_complex64(&st.v[base | a_bit]);
        GpuComplex tmp_b = load_complex64(&st.v[base | b_bit]);
        store_complex64(&st.v[base | a_bit], tmp_b);
        store_complex64(&st.v[base | b_bit], tmp_a);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        bit_swap(st.px, a, st.px, b);
        bit_swap(st.pz, a, st.pz, b);
    }
    __syncthreads();
}

__device__ void coop_array_multi_cnot(CoopShotState& st, uint32_t target, uint64_t ctrl_mask) {
    uint64_t t_bit = 1ULL << target;
    uint64_t half = 1ULL << (*st.active_k - 1);
    ensure_scatter_1(st, target);
    for (uint64_t idx = threadIdx.x; idx < half; idx += blockDim.x) {
        uint64_t actual = st.scatter_lut ? st.scatter_lut[idx]
                                         : static_cast<uint32_t>(scatter_bits_1(idx, target));
        bool parity = (__popcll(actual & ctrl_mask) & 1) != 0;
        if (parity) {
            GpuComplex tmp = load_complex64(&st.v[actual]);
            GpuComplex tmp2 = load_complex64(&st.v[actual | t_bit]);
            store_complex64(&st.v[actual], tmp2);
            store_complex64(&st.v[actual | t_bit], tmp);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (uint32_t c = 0; c < *st.active_k; ++c) {
            if ((ctrl_mask >> c) & 1ULL) {
                bool px_c = bit_get(st.px, c);
                bool pz_t = bit_get(st.pz, target);
                bit_xor(st.px, target, px_c);
                bit_xor(st.pz, c, pz_t);
            }
        }
    }
    __syncthreads();
}

__device__ void coop_array_multi_cz(CoopShotState& st, uint32_t control, uint64_t target_mask) {
    uint64_t c_bit = 1ULL << control;
    uint64_t half = 1ULL << (*st.active_k - 1);
    ensure_scatter_1(st, control);
    for (uint64_t idx = threadIdx.x; idx < half; idx += blockDim.x) {
        uint64_t actual = (st.scatter_lut ? st.scatter_lut[idx]
                                          : static_cast<uint32_t>(scatter_bits_1(idx, control)))
                          | c_bit;
        bool negate = (__popcll(actual & target_mask) & 1) != 0;
        if (negate) {
            GpuComplex val = load_complex64(&st.v[actual]);
            val.re = -val.re;
            val.im = -val.im;
            store_complex64(&st.v[actual], val);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (uint32_t t = 0; t < *st.active_k; ++t) {
            if ((target_mask >> t) & 1ULL) {
                bool px_c = bit_get(st.px, control);
                bool px_t = bit_get(st.px, t);
                bit_xor(st.pz, t, px_c);
                bit_xor(st.pz, control, px_t);
            }
        }
    }
    __syncthreads();
}

__device__ void coop_array_h(CoopShotState& st, uint32_t axis) {
    uint64_t axis_bit = 1ULL << axis;
    uint64_t iters = 1ULL << (*st.active_k - 1);
    ensure_scatter_1(st, axis);
    for (uint64_t i = threadIdx.x; i < iters; i += blockDim.x) {
        uint64_t idx0 = st.scatter_lut ? st.scatter_lut[i]
                                       : static_cast<uint32_t>(scatter_bits_1(i, axis));
        uint64_t idx1 = idx0 | axis_bit;
        GpuComplex a = load_complex64(&st.v[idx0]);
        GpuComplex b = load_complex64(&st.v[idx1]);
        store_complex64(&st.v[idx0], cscale(cadd(a, b), kInvSqrt2));
        store_complex64(&st.v[idx1], cscale(csub(a, b), kInvSqrt2));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        bool px = bit_get(st.px, axis);
        bool pz = bit_get(st.pz, axis);
        bit_set(st.px, axis, pz);
        bit_set(st.pz, axis, px);
    }
    __syncthreads();
}

__device__ void coop_apply_phase(CoopShotState& st, uint32_t axis, GpuComplex phase) {
    uint64_t axis_bit = 1ULL << axis;
    uint64_t iters = 1ULL << (*st.active_k - 1);
    ensure_scatter_1(st, axis);
    for (uint64_t i = threadIdx.x; i < iters; i += blockDim.x) {
        uint64_t idx = (st.scatter_lut ? st.scatter_lut[i]
                                       : static_cast<uint32_t>(scatter_bits_1(i, axis)))
                       | axis_bit;
        GpuComplex val = load_complex64(&st.v[idx]);
        store_complex64(&st.v[idx], cmul(val, phase));
    }
    __syncthreads();
}

__device__ void coop_array_s(CoopShotState& st, uint32_t axis, bool dagger) {
    coop_apply_phase(st, axis, dagger ? GpuComplex{0.0f, -1.0f} : GpuComplex{0.0f, 1.0f});
    if (threadIdx.x == 0) {
        bool px = bit_get(st.px, axis);
        bit_xor(st.pz, axis, px);
    }
    __syncthreads();
}

__device__ void coop_array_t(CoopShotState& st, uint32_t axis, bool dagger) {
    bool px = bit_get(st.px, axis);
    if (axis >= *st.active_k) {
        __syncthreads();
        return;
    }
    double imag = dagger ? -kInvSqrt2 : kInvSqrt2;
    if (px) {
        imag = -imag;
    }
    coop_apply_phase(st, axis, {static_cast<float>(kInvSqrt2), static_cast<float>(imag)});
}

__device__ void coop_expand_plain(CoopShotState& st) {
    uint64_t half = 1ULL << *st.active_k;
    for (uint64_t i = threadIdx.x; i < half; i += blockDim.x) {
        GpuComplex val = load_complex64(&st.v[i]);
        store_complex64(&st.v[i + half], val);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        ++*st.active_k;
    }
    __syncthreads();
}

__device__ void coop_expand_t(CoopShotState& st, uint32_t axis, bool dagger) {
    uint64_t half = 1ULL << *st.active_k;
    bool px = bit_get(st.px, axis);
    double imag = dagger ? -kInvSqrt2 : kInvSqrt2;
    if (px) {
        imag = -imag;
    }
    GpuComplex phase{static_cast<float>(kInvSqrt2), static_cast<float>(imag)};
    for (uint64_t i = threadIdx.x; i < half; i += blockDim.x) {
        GpuComplex val = load_complex64(&st.v[i]);
        store_complex64(&st.v[i + half], cmul(val, phase));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        ++*st.active_k;
    }
    __syncthreads();
}

__device__ void coop_meas_active_diagonal(CoopShotState& st, Rng& rng, uint32_t axis,
                                          uint32_t classical_idx, bool sign) {
    uint64_t half = 1ULL << (*st.active_k - 1);
    bool px = bit_get(st.px, axis);
    double l0 = 0.0;
    double l1 = 0.0;
    for (uint64_t i = threadIdx.x; i < half; i += blockDim.x) {
        l0 += cnorm(load_complex64(&st.v[i]));
        l1 += cnorm(load_complex64(&st.v[i + half]));
    }
    double p0 = 0.0;
    double p1 = 0.0;
    coop_reduce2(st, l0, l1, p0, p1);
    if (threadIdx.x == 0) {
        uint8_t b = sample_branch(rng, p0, p1, p0 + p1);
        *st.branch = b;
        uint8_t m_abs = b ^ static_cast<uint8_t>(px);
        st.meas[classical_idx] = m_abs ^ static_cast<uint8_t>(sign);
    }
    __syncthreads();
    if (*st.branch != 0) {
        for (uint64_t i = threadIdx.x; i < half; i += blockDim.x) {
            store_complex64(&st.v[i], load_complex64(&st.v[i + half]));
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        --*st.active_k;
        uint8_t m_abs = st.meas[classical_idx] ^ static_cast<uint8_t>(sign);
        bit_set(st.px, axis, m_abs != 0);
        bit_set(st.pz, axis, false);
    }
    __syncthreads();
}

__device__ void coop_meas_active_interfere(CoopShotState& st, Rng& rng, uint32_t axis,
                                           uint32_t classical_idx, bool sign) {
    uint64_t half = 1ULL << (*st.active_k - 1);
    bool pz = bit_get(st.pz, axis);
    double lp = 0.0;
    double lm = 0.0;
    for (uint64_t i = threadIdx.x; i < half; i += blockDim.x) {
        GpuComplex vi = load_complex64(&st.v[i]);
        GpuComplex vh = load_complex64(&st.v[i + half]);
        GpuComplex sum = cadd(vi, vh);
        GpuComplex diff = csub(vi, vh);
        lp += cnorm(sum);
        lm += cnorm(diff);
    }
    double p_plus = 0.0;
    double p_minus = 0.0;
    coop_reduce2(st, lp, lm, p_plus, p_minus);
    if (threadIdx.x == 0) {
        uint8_t b = sample_branch(rng, p_plus, p_minus, p_plus + p_minus);
        *st.branch = b;
        uint8_t m_abs = b ^ static_cast<uint8_t>(pz);
        st.meas[classical_idx] = m_abs ^ static_cast<uint8_t>(sign);
    }
    __syncthreads();
    for (uint64_t i = threadIdx.x; i < half; i += blockDim.x) {
        GpuComplex vi = load_complex64(&st.v[i]);
        GpuComplex vh = load_complex64(&st.v[i + half]);
        GpuComplex folded =
            (*st.branch == 0) ? cadd(vi, vh) : csub(vi, vh);
        store_complex64(&st.v[i], cscale(folded, kInvSqrt2));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        --*st.active_k;
        uint8_t m_abs = st.meas[classical_idx] ^ static_cast<uint8_t>(sign);
        bit_set(st.px, axis, m_abs != 0);
        bit_set(st.pz, axis, false);
    }
    __syncthreads();
}

__device__ void coop_swap_meas_interfere(CoopShotState& st, Rng& rng, uint32_t from, uint32_t to,
                                         uint32_t classical_idx, bool sign) {
    if (from == to) {
        coop_meas_active_interfere(st, rng, to, classical_idx, sign);
        return;
    }
    if (threadIdx.x == 0) {
        bit_swap(st.px, from, st.px, to);
        bit_swap(st.pz, from, st.pz, to);
    }
    __syncthreads();
    bool pz = bit_get(st.pz, to);
    uint64_t half = 1ULL << to;
    uint64_t f_bit = 1ULL << from;
    double lp = 0.0;
    double lm = 0.0;
    for (uint64_t idx = threadIdx.x; idx < half; idx += blockDim.x) {
        uint64_t b_f = (idx >> from) & 1ULL;
        uint64_t base = (idx & ~f_bit) | (b_f << to);
        GpuComplex vb = load_complex64(&st.v[base]);
        GpuComplex vf = load_complex64(&st.v[base | f_bit]);
        GpuComplex sum = cadd(vb, vf);
        GpuComplex diff = csub(vb, vf);
        lp += cnorm(sum);
        lm += cnorm(diff);
    }
    double p_plus = 0.0;
    double p_minus = 0.0;
    coop_reduce2(st, lp, lm, p_plus, p_minus);
    if (threadIdx.x == 0) {
        uint8_t b = sample_branch(rng, p_plus, p_minus, p_plus + p_minus);
        *st.branch = b;
        uint8_t m_abs = b ^ static_cast<uint8_t>(pz);
        st.meas[classical_idx] = m_abs ^ static_cast<uint8_t>(sign);
    }
    __syncthreads();
    for (uint64_t idx = threadIdx.x; idx < half; idx += blockDim.x) {
        uint64_t b_f = (idx >> from) & 1ULL;
        uint64_t base = (idx & ~f_bit) | (b_f << to);
        GpuComplex vb = load_complex64(&st.v[base]);
        GpuComplex vf = load_complex64(&st.v[base | f_bit]);
        store_complex64(&st.scratch[idx],
            cscale((*st.branch == 0) ? cadd(vb, vf)
                                    : csub(vb, vf),
                   kInvSqrt2));
    }
    __syncthreads();
    for (uint64_t idx = threadIdx.x; idx < half; idx += blockDim.x) {
        store_complex64(&st.v[idx], load_complex64(&st.scratch[idx]));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        --*st.active_k;
        uint8_t m_abs = st.meas[classical_idx] ^ static_cast<uint8_t>(sign);
        bit_set(st.px, to, m_abs != 0);
        bit_set(st.pz, to, false);
    }
    __syncthreads();
}

__device__ void coop_array_rot(CoopShotState& st, uint32_t axis, double z_re, double z_im) {
    if (axis < *st.active_k) {
        bool px = bit_get(st.px, axis);
        double im = px ? -z_im : z_im;
        uint64_t axis_bit = 1ULL << axis;
        uint64_t iters = 1ULL << (*st.active_k - 1);
        GpuComplex phase{static_cast<float>(z_re), static_cast<float>(im)};
        ensure_scatter_1(st, axis);
        for (uint64_t i = threadIdx.x; i < iters; i += blockDim.x) {
            uint64_t idx = (st.scatter_lut ? st.scatter_lut[i]
                                           : static_cast<uint32_t>(scatter_bits_1(i, axis)))
                           | axis_bit;
            GpuComplex val = load_complex64(&st.v[idx]);
            store_complex64(&st.v[idx], cmul(val, phase));
        }
    }
    __syncthreads();
}

__device__ void coop_expand_rot(CoopShotState& st, uint32_t axis, double z_re, double z_im) {
    uint64_t half = 1ULL << *st.active_k;
    bool px = bit_get(st.px, axis);
    double im = px ? -z_im : z_im;
    GpuComplex phase{static_cast<float>(z_re), static_cast<float>(im)};
    for (uint64_t i = threadIdx.x; i < half; i += blockDim.x) {
        GpuComplex val = load_complex64(&st.v[i]);
        store_complex64(&st.v[i + half], cmul(val, phase));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        (*st.active_k)++;
    }
    __syncthreads();
}

__device__ __noinline__ void coop_u2_sweep(GpuComplex* v, uint32_t active_k,
                                           uint32_t axis, const GpuComplex* mat,
                                           const uint32_t* lut) {
    uint64_t axis_bit = 1ULL << axis;
    uint64_t iters = 1ULL << (active_k - 1);
    for (uint64_t i = threadIdx.x; i < iters; i += blockDim.x) {
        uint64_t idx0 = lut ? lut[i]
                            : static_cast<uint32_t>(scatter_bits_1(i, axis));
        uint64_t idx1 = idx0 | axis_bit;
        GpuComplex a = load_complex64(&v[idx0]), b = load_complex64(&v[idx1]);
        store_complex64(&v[idx0], cadd(cmul(a, mat[0]), cmul(b, mat[1])));
        store_complex64(&v[idx1], cadd(cmul(a, mat[2]), cmul(b, mat[3])));
    }
}

__device__ void coop_array_u2(CoopShotState& st, const GpuProgram& program,
                              uint32_t axis, uint32_t cp_idx) {
    const auto& entry = program.fused_u2[cp_idx];
    uint8_t in_state = (bit_get(st.pz, axis) ? 2 : 0) | (bit_get(st.px, axis) ? 1 : 0);
    if (axis < *st.active_k) {
        ensure_scatter_1(st, axis);
        coop_u2_sweep(st.v, *st.active_k, axis, entry.matrices[in_state], st.scatter_lut);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        uint8_t out = entry.out_states[in_state];
        bit_set(st.px, axis, (out & 1) != 0);
        bit_set(st.pz, axis, (out & 2) != 0);
    }
    __syncthreads();
}

__device__ __noinline__ void coop_u4_sweep(GpuComplex* v, uint32_t active_k,
                                           uint32_t axis_lo, uint32_t axis_hi,
                                           const GpuComplex mat[4][4],
                                           const uint32_t* lut) {
    uint64_t lo_bit = 1ULL << axis_lo;
    uint64_t hi_bit = 1ULL << axis_hi;
    uint64_t iters = 1ULL << (active_k - 2);
    for (uint64_t i = threadIdx.x; i < iters; i += blockDim.x) {
        uint64_t base = lut ? lut[i]
                            : static_cast<uint32_t>(scatter_bits_2(i, axis_lo, axis_hi));
        uint64_t i0 = base;
        uint64_t i1 = base | lo_bit;
        uint64_t i2 = base | hi_bit;
        uint64_t i3 = base | lo_bit | hi_bit;
        GpuComplex v0 = load_complex64(&v[i0]), v1 = load_complex64(&v[i1]);
        GpuComplex v2 = load_complex64(&v[i2]), v3 = load_complex64(&v[i3]);
        store_complex64(&v[i0], cadd(cadd(cmul(v0, mat[0][0]), cmul(v1, mat[0][1])),
                     cadd(cmul(v2, mat[0][2]), cmul(v3, mat[0][3]))));
        store_complex64(&v[i1], cadd(cadd(cmul(v0, mat[1][0]), cmul(v1, mat[1][1])),
                     cadd(cmul(v2, mat[1][2]), cmul(v3, mat[1][3]))));
        store_complex64(&v[i2], cadd(cadd(cmul(v0, mat[2][0]), cmul(v1, mat[2][1])),
                     cadd(cmul(v2, mat[2][2]), cmul(v3, mat[2][3]))));
        store_complex64(&v[i3], cadd(cadd(cmul(v0, mat[3][0]), cmul(v1, mat[3][1])),
                     cadd(cmul(v2, mat[3][2]), cmul(v3, mat[3][3]))));
    }
}

__device__ void coop_array_u4(CoopShotState& st, const GpuProgram& program,
                              uint32_t axis_lo, uint32_t axis_hi, uint32_t cp_idx) {
    const auto& node = program.fused_u4[cp_idx];
    uint8_t px_lo = bit_get(st.px, axis_lo) ? 1 : 0;
    uint8_t pz_lo = bit_get(st.pz, axis_lo) ? 1 : 0;
    uint8_t px_hi = bit_get(st.px, axis_hi) ? 1 : 0;
    uint8_t pz_hi = bit_get(st.pz, axis_hi) ? 1 : 0;
    uint8_t in_state = (pz_hi << 3) | (px_hi << 2) | (pz_lo << 1) | px_lo;
    const auto& entry = node.entries[in_state];
    if (axis_hi < *st.active_k) {
        ensure_scatter_2(st, axis_lo, axis_hi);
        coop_u4_sweep(st.v, *st.active_k, axis_lo, axis_hi, entry.matrix, st.scatter_lut);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        uint8_t out = entry.out_state;
        bit_set(st.px, axis_lo, (out & 1) != 0);
        bit_set(st.pz, axis_lo, (out & 2) != 0);
        bit_set(st.px, axis_hi, (out & 4) != 0);
        bit_set(st.pz, axis_hi, (out & 8) != 0);
    }
    __syncthreads();
}

__device__ void coop_exec_exp_val(CoopShotState& st, const GpuProgram& program,
                                  uint32_t cp_idx, uint32_t ev_idx) {
    if (threadIdx.x == 0) {
        const GpuExpValMask& mask = program.exp_val_masks[cp_idx];
        bool frame_sign = mask.sign != 0;
        int parity = __popcll(mask.x[0] & st.pz[0]) + __popcll(mask.x[1] & st.pz[1])
                   + __popcll(mask.z[0] & st.px[0]) + __popcll(mask.z[1] & st.px[1]);
        if (parity & 1) frame_sign = !frame_sign;

        uint32_t k = *st.active_k;
        uint64_t active_mask_w0 = (k >= 64) ? ~uint64_t{0} : ((k == 0) ? 0 : ((uint64_t{1} << k) - 1));
        uint64_t dormant_x = mask.x[0] & ~active_mask_w0;
        if (k < 64) dormant_x |= mask.x[1];
        if (dormant_x != 0) { st.exp_vals[ev_idx] = 0.0; }
        else {
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
                GpuComplex vj = load_complex64(&st.v[j]);
                denom += cnorm(vj);
                uint64_t j_xor = j ^ x_active;
                GpuComplex vj_xor = load_complex64(&st.v[j_xor]);
                double z_sign = (__popcll(j & z_active) & 1) ? -1.0 : 1.0;
                double a_re = static_cast<double>(vj_xor.re);
                double a_im = -static_cast<double>(vj_xor.im);
                num_re += (a_re * z_sign * vj.re - a_im * z_sign * vj.im);
                num_im += (a_re * z_sign * vj.im + a_im * z_sign * vj.re);
            }
            double result_re = num_re * phase_re - num_im * phase_im;
            st.exp_vals[ev_idx] = (denom > 0.0) ? result_re / denom : 0.0;
        }
    }
    __syncthreads();
}

__device__ bool is_frame_only_op(uint8_t opcode) {
    switch (opcode) {
        case static_cast<uint8_t>(clifft::Opcode::OP_FRAME_CNOT):
        case static_cast<uint8_t>(clifft::Opcode::OP_FRAME_CZ):
        case static_cast<uint8_t>(clifft::Opcode::OP_FRAME_H):
        case static_cast<uint8_t>(clifft::Opcode::OP_FRAME_S):
        case static_cast<uint8_t>(clifft::Opcode::OP_FRAME_S_DAG):
        case static_cast<uint8_t>(clifft::Opcode::OP_FRAME_SWAP):
            return true;
        default:
            return false;
    }
}

__device__ void execute_shot_coop(const GpuProgram& program, Rng& rng, CoopShotState& st) {
    for (uint32_t pc = 0; pc < program.num_instrs; ++pc) {
        const GpuInstr& instr = program.instrs[pc];
        switch (instr.opcode) {
            case static_cast<uint8_t>(clifft::Opcode::OP_FRAME_CNOT):
                if (threadIdx.x == 0) {
                    bool px_c = bit_get(st.px, instr.axis_1);
                    bool pz_t = bit_get(st.pz, instr.axis_2);
                    bit_xor(st.px, instr.axis_2, px_c);
                    bit_xor(st.pz, instr.axis_1, pz_t);
                }
                if (pc + 1 >= program.num_instrs || !is_frame_only_op(program.instrs[pc + 1].opcode)) {
                    __syncthreads();
                }
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_FRAME_CZ):
                if (threadIdx.x == 0) {
                    bool px_a = bit_get(st.px, instr.axis_1);
                    bool px_b = bit_get(st.px, instr.axis_2);
                    bit_xor(st.pz, instr.axis_2, px_a);
                    bit_xor(st.pz, instr.axis_1, px_b);
                }
                if (pc + 1 >= program.num_instrs || !is_frame_only_op(program.instrs[pc + 1].opcode)) {
                    __syncthreads();
                }
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_FRAME_H):
                if (threadIdx.x == 0) {
                    bool px = bit_get(st.px, instr.axis_1);
                    bool pz = bit_get(st.pz, instr.axis_1);
                    bit_set(st.px, instr.axis_1, pz);
                    bit_set(st.pz, instr.axis_1, px);
                }
                if (pc + 1 >= program.num_instrs || !is_frame_only_op(program.instrs[pc + 1].opcode)) {
                    __syncthreads();
                }
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_FRAME_S):
            case static_cast<uint8_t>(clifft::Opcode::OP_FRAME_S_DAG):
                if (threadIdx.x == 0) {
                    bool px = bit_get(st.px, instr.axis_1);
                    bit_xor(st.pz, instr.axis_1, px);
                }
                if (pc + 1 >= program.num_instrs || !is_frame_only_op(program.instrs[pc + 1].opcode)) {
                    __syncthreads();
                }
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_FRAME_SWAP):
                if (threadIdx.x == 0) {
                    bit_swap(st.px, instr.axis_1, st.px, instr.axis_2);
                    bit_swap(st.pz, instr.axis_1, st.pz, instr.axis_2);
                }
                if (pc + 1 >= program.num_instrs || !is_frame_only_op(program.instrs[pc + 1].opcode)) {
                    __syncthreads();
                }
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_ARRAY_CNOT):
                coop_array_cnot(st, instr.axis_1, instr.axis_2);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_ARRAY_CZ):
                coop_array_cz(st, instr.axis_1, instr.axis_2);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_ARRAY_SWAP):
                coop_array_swap(st, instr.axis_1, instr.axis_2);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_ARRAY_MULTI_CNOT):
                coop_array_multi_cnot(st, instr.axis_1, instr.mask);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_ARRAY_MULTI_CZ):
                coop_array_multi_cz(st, instr.axis_1, instr.mask);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_ARRAY_H):
                coop_array_h(st, instr.axis_1);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_ARRAY_S):
                coop_array_s(st, instr.axis_1, false);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_ARRAY_S_DAG):
                coop_array_s(st, instr.axis_1, true);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_ARRAY_T):
                coop_array_t(st, instr.axis_1, false);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_ARRAY_T_DAG):
                coop_array_t(st, instr.axis_1, true);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_EXPAND):
                coop_expand_plain(st);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_EXPAND_T):
                coop_expand_t(st, instr.axis_1, false);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_EXPAND_T_DAG):
                coop_expand_t(st, instr.axis_1, true);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_EXPAND_ROT):
                coop_expand_rot(st, instr.axis_1, instr.weight_re, instr.weight_im);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_ARRAY_ROT):
                coop_array_rot(st, instr.axis_1, instr.weight_re, instr.weight_im);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_ARRAY_U2):
                coop_array_u2(st, program, instr.axis_1, instr.a);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_ARRAY_U4):
                coop_array_u4(st, program, instr.axis_1, instr.axis_2, instr.a);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_MEAS_DORMANT_STATIC):
                if (threadIdx.x == 0) {
                    if (instr.flags & kFlagIdentity) {
                        st.meas[instr.a] = (instr.flags & kFlagSign) ? 1 : 0;
                    } else {
                        uint8_t outcome = bit_get(st.px, instr.axis_1) ? 1 : 0;
                        st.meas[instr.a] =
                            outcome ^ static_cast<uint8_t>((instr.flags & kFlagSign) != 0);
                    }
                }
                __syncthreads();
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_MEAS_DORMANT_RANDOM):
                if (threadIdx.x == 0) {
                    uint8_t m_abs = rng.uniform() < 0.5 ? 0 : 1;
                    bit_set(st.px, instr.axis_1, m_abs != 0);
                    bit_set(st.pz, instr.axis_1, false);
                    st.meas[instr.a] = m_abs ^ static_cast<uint8_t>((instr.flags & kFlagSign) != 0);
                }
                __syncthreads();
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_MEAS_ACTIVE_DIAGONAL):
                coop_meas_active_diagonal(st, rng, instr.axis_1, instr.a,
                                          (instr.flags & kFlagSign) != 0);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_MEAS_ACTIVE_INTERFERE):
                coop_meas_active_interfere(st, rng, instr.axis_1, instr.a,
                                           (instr.flags & kFlagSign) != 0);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_SWAP_MEAS_INTERFERE):
                coop_swap_meas_interfere(st, rng, instr.axis_1, instr.axis_2, instr.a,
                                         (instr.flags & kFlagSign) != 0);
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_APPLY_PAULI):
                if (threadIdx.x == 0 && st.meas[instr.b] != 0) {
                    const GpuMask& mask = program.pauli_masks[instr.a];
                    st.px[0] ^= mask.x[0];
                    st.px[1] ^= mask.x[1];
                    st.pz[0] ^= mask.z[0];
                    st.pz[1] ^= mask.z[1];
                }
                __syncthreads();
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_NOISE):
                if (threadIdx.x == 0 && instr.a == *st.next_noise_idx) {
                    const GpuNoiseSite& site = program.noise_sites[instr.a];
                    double roll = rng.uniform() * site.prob_sum;
                    double cumulative = 0.0;
                    for (uint32_t k = 0; k < site.count; ++k) {
                        const GpuChannel& ch = program.noise_channels[site.offset + k];
                        cumulative += ch.prob;
                        if (roll < cumulative) {
                            st.px[0] ^= ch.x[0];
                            st.px[1] ^= ch.x[1];
                            st.pz[0] ^= ch.z[0];
                            st.pz[1] ^= ch.z[1];
                            break;
                        }
                    }
                    *st.next_noise_idx = instr.a + 1;
                    coop_draw_next_noise(st, program, rng);
                }
                __syncthreads();
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_NOISE_BLOCK):
                if (threadIdx.x == 0) {
                    uint32_t end = instr.a + instr.b;
                    while (*st.next_noise_idx >= instr.a && *st.next_noise_idx < end) {
                        uint32_t site_idx = *st.next_noise_idx;
                        const GpuNoiseSite& site = program.noise_sites[site_idx];
                        double roll = rng.uniform() * site.prob_sum;
                        double cumulative = 0.0;
                        for (uint32_t k = 0; k < site.count; ++k) {
                            const GpuChannel& ch = program.noise_channels[site.offset + k];
                            cumulative += ch.prob;
                            if (roll < cumulative) {
                                st.px[0] ^= ch.x[0];
                                st.px[1] ^= ch.x[1];
                                st.pz[0] ^= ch.z[0];
                                st.pz[1] ^= ch.z[1];
                                break;
                            }
                        }
                        *st.next_noise_idx = site_idx + 1;
                        coop_draw_next_noise(st, program, rng);
                    }
                }
                __syncthreads();
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_READOUT_NOISE):
                if (threadIdx.x == 0) {
                    const GpuReadoutNoise& r = program.readout_noise[instr.a];
                    if (rng.uniform() < r.prob) {
                        st.meas[r.meas_idx] ^= 1;
                    }
                }
                __syncthreads();
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_DETECTOR):
                __syncthreads();
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_POSTSELECT):
                if (threadIdx.x == 0) {
                    uint32_t start = program.detector_offsets[instr.a];
                    uint32_t end = program.detector_offsets[instr.a + 1];
                    uint8_t parity = (instr.flags & kFlagExpectedOne) ? 1 : 0;
                    for (uint32_t i = start; i < end; ++i) {
                        parity ^= st.meas[program.detector_targets[i]];
                    }
                    if (parity != 0) {
                        *st.discarded = 1;
                    }
                }
                __syncthreads();
                if (*st.discarded) {
                    return;
                }
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_OBSERVABLE):
                if (threadIdx.x == 0) {
                    uint32_t start = program.observable_offsets[instr.a];
                    uint32_t end = program.observable_offsets[instr.a + 1];
                    uint8_t parity = 0;
                    for (uint32_t i = start; i < end; ++i) {
                        parity ^= st.meas[program.observable_targets[i]];
                    }
                    st.obs[instr.b] ^= parity;
                }
                __syncthreads();
                break;
            case static_cast<uint8_t>(clifft::Opcode::OP_EXP_VAL):
                coop_exec_exp_val(st, program, instr.a, instr.b);
                break;
            default:
                if (threadIdx.x == 0) {
                    *st.discarded = 1;
                }
                __syncthreads();
                return;
        }
    }
}

__global__ void sample_kernel(GpuProgram program, uint64_t shot_offset, uint64_t shots,
                              uint64_t seed, BlockCounts* block_counts) {
    extern __shared__ uint64_t shared_counts[];
    uint32_t tid = threadIdx.x;
    uint64_t* s_passed = shared_counts;
    uint64_t* s_logical = s_passed + blockDim.x;
    uint64_t* s_obs = s_logical + blockDim.x;

    uint64_t local_passed = 0;
    uint64_t local_logical = 0;
    uint64_t local_obs[kMaxObs];
    for (uint32_t i = 0; i < kMaxObs; ++i) {
        local_obs[i] = 0;
    }

    uint64_t batch_shot_id = static_cast<uint64_t>(blockIdx.x) * blockDim.x + tid;
    if (batch_shot_id < shots) {
        uint64_t shot_id = shot_offset + batch_shot_id;
        Rng rng;
        rng.seed(seed, shot_id);

        ShotState st;
        st.px[0] = 0;
        st.px[1] = 0;
        st.pz[0] = 0;
        st.pz[1] = 0;
        st.active_k = 0;
        st.next_noise_idx = 0;
        st.discarded = false;
        st.v[0] = {1.0f, 0.0f};
        for (uint32_t i = 0; i < program.num_observables; ++i) {
            st.obs[i] = 0;
        }
        for (uint32_t i = 0; i < kMaxExpVals; ++i) {
            st.exp_vals[i] = 0.0;
        }
        draw_next_noise(st, program, rng);
        execute_shot(program, rng, st);

        if (!st.discarded) {
            local_passed = 1;
            bool any_logical = false;
            for (uint32_t i = 0; i < program.num_observables; ++i) {
                uint8_t val = st.obs[i];
                if (program.expected_observables[i] != 0) {
                    val ^= 1;
                }
                if (val != 0) {
                    local_obs[i] = 1;
                    any_logical = true;
                }
            }
            local_logical = any_logical ? 1 : 0;
        }
    }

    s_passed[tid] = local_passed;
    s_logical[tid] = local_logical;
    for (uint32_t i = 0; i < kMaxObs; ++i) {
        s_obs[i * blockDim.x + tid] = local_obs[i];
    }
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_passed[tid] += s_passed[tid + stride];
            s_logical[tid] += s_logical[tid + stride];
            for (uint32_t i = 0; i < kMaxObs; ++i) {
                s_obs[i * blockDim.x + tid] += s_obs[i * blockDim.x + tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        BlockCounts out{};
        out.passed = s_passed[0];
        out.logical_errors = s_logical[0];
        for (uint32_t i = 0; i < kMaxObs; ++i) {
            out.observable_ones[i] = s_obs[i * blockDim.x];
        }
        out.exp_val_count = out.passed;
        block_counts[blockIdx.x] = out;
    }
}

__global__ void sample_kernel_coop(GpuProgram program, uint64_t shot_offset, uint64_t shots,
                                   uint64_t seed, BlockCounts* counts) {
    uint64_t batch_shot_id = blockIdx.x;
    if (batch_shot_id >= shots) {
        return;
    }

    __shared__ GpuComplex v[kSharedMaxAmplitudes];
    __shared__ GpuComplex scratch[kSharedMaxAmplitudes / 2];
    __shared__ uint8_t meas[kMaxMeas];
    __shared__ uint8_t obs[kMaxObs];
    __shared__ uint64_t px[2];
    __shared__ uint64_t pz[2];
    __shared__ uint32_t active_k;
    __shared__ uint32_t next_noise_idx;
    __shared__ uint8_t discarded;
    __shared__ uint8_t branch;
    __shared__ double red0[256];
    __shared__ double red1[256];
    // Precomputed scatter-bits LUT: max 2^(k-1) = 512 entries for rank 10
    __shared__ uint32_t scatter_lut[kSharedMaxAmplitudes / 2];
    __shared__ uint32_t cached_axis1;
    __shared__ uint32_t cached_axis2;
    __shared__ uint32_t cached_lut_k;
    __shared__ uint32_t cached_lut_mode;

    if (threadIdx.x == 0) {
        px[0] = 0;
        px[1] = 0;
        pz[0] = 0;
        pz[1] = 0;
        active_k = 0;
        next_noise_idx = 0;
        discarded = 0;
        branch = 0;
        v[0] = {1.0f, 0.0f};
        cached_lut_mode = 0;  // invalidate LUT cache
        for (uint32_t i = 0; i < program.num_observables; ++i) {
            obs[i] = 0;
        }
    }
    __syncthreads();

    Rng rng;
    if (threadIdx.x == 0) {
        rng.seed(seed, shot_offset + batch_shot_id);
        CoopShotState st{v, scratch, meas, obs, px, pz, &active_k, &next_noise_idx,
                         &discarded, &branch, red0, red1, nullptr,
                         scatter_lut, &cached_axis1, &cached_axis2,
                         &cached_lut_k, &cached_lut_mode};
        coop_draw_next_noise(st, program, rng);
    }
    __syncthreads();

    CoopShotState st{v, scratch, meas, obs, px, pz, &active_k, &next_noise_idx,
                     &discarded, &branch, red0, red1, nullptr,
                     scatter_lut, &cached_axis1, &cached_axis2,
                     &cached_lut_k, &cached_lut_mode};
    execute_shot_coop(program, rng, st);

    if (threadIdx.x == 0 && discarded == 0) {
        bool any_logical = false;
        for (uint32_t i = 0; i < program.num_observables; ++i) {
            uint8_t val = obs[i];
            if (program.expected_observables[i] != 0) {
                val ^= 1;
            }
            if (val != 0) {
                atomicAdd(reinterpret_cast<unsigned long long*>(&counts->observable_ones[i]), 1ULL);
                any_logical = true;
            }
        }
        atomicAdd(reinterpret_cast<unsigned long long*>(&counts->passed), 1ULL);
        if (any_logical) {
            atomicAdd(reinterpret_cast<unsigned long long*>(&counts->logical_errors), 1ULL);
        }
    }
}

__global__ void sample_kernel_global_coop(GpuProgram program, uint64_t shot_offset, uint64_t shots,
                                          uint64_t seed, GpuComplex* global_v,
                                          GpuComplex* global_scratch, uint64_t* work_counter,
                                          BlockCounts* counts) {
    uint32_t slot = blockIdx.x;
    GpuComplex* v = global_v + static_cast<size_t>(slot) * kGlobalMaxAmplitudes;
    GpuComplex* scratch =
        global_scratch + static_cast<size_t>(slot) * (kGlobalMaxAmplitudes / 2);

    __shared__ uint8_t meas[kMaxMeas];
    __shared__ uint8_t obs[kMaxObs];
    __shared__ uint64_t px[2];
    __shared__ uint64_t pz[2];
    __shared__ uint32_t active_k;
    __shared__ uint32_t next_noise_idx;
    __shared__ uint8_t discarded;
    __shared__ uint8_t branch;
    __shared__ uint64_t batch_shot_id;
    __shared__ double red0[256];
    __shared__ double red1[256];

    // Read hardware XCD ID once per block (uniform across all threads in a block).
    // work_counter is an array of kNumXCDs counters; each XCD atomics on its own
    // counter to avoid cross-XCD atomic latency (202 ns vs 116 ns same-XCD).
    __shared__ uint32_t xcd_id;
    if (threadIdx.x == 0) {
        asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s"(xcd_id));
        xcd_id = xcd_id % kNumXCDs;  // safety clamp
    }
    __syncthreads();

    while (true) {
        if (threadIdx.x == 0) {
            // Per-XCD work counter: each XCD increments its own counter
            uint64_t local_shot = atomicAdd(
                reinterpret_cast<unsigned long long*>(&work_counter[xcd_id]), 1ULL);
            // Interleaved mapping: shot = local_shot * kNumXCDs + xcd_id
            // This distributes shots round-robin across XCDs, ensuring even load
            batch_shot_id = local_shot * kNumXCDs + xcd_id;
        }
        __syncthreads();
        if (batch_shot_id >= shots) {
            return;
        }

        if (threadIdx.x == 0) {
            px[0] = 0;
            px[1] = 0;
            pz[0] = 0;
            pz[1] = 0;
            active_k = 0;
            next_noise_idx = 0;
            discarded = 0;
            branch = 0;
            v[0] = {1.0f, 0.0f};
            for (uint32_t i = 0; i < program.num_observables; ++i) {
                obs[i] = 0;
            }
        }
        __syncthreads();

        Rng rng;
        if (threadIdx.x == 0) {
            rng.seed(seed, shot_offset + batch_shot_id);
            CoopShotState st{v, scratch, meas, obs, px, pz, &active_k, &next_noise_idx,
                             &discarded, &branch, red0, red1, nullptr,
                             nullptr, nullptr, nullptr, nullptr, nullptr};
            coop_draw_next_noise(st, program, rng);
        }
        __syncthreads();

        CoopShotState st{v, scratch, meas, obs, px, pz, &active_k, &next_noise_idx,
                         &discarded, &branch, red0, red1, nullptr,
                         nullptr, nullptr, nullptr, nullptr, nullptr};
        execute_shot_coop(program, rng, st);

        if (threadIdx.x == 0 && discarded == 0) {
            bool any_logical = false;
            for (uint32_t i = 0; i < program.num_observables; ++i) {
                uint8_t val = obs[i];
                if (program.expected_observables[i] != 0) {
                    val ^= 1;
                }
                if (val != 0) {
                    atomicAdd(reinterpret_cast<unsigned long long*>(&counts->observable_ones[i]),
                              1ULL);
                    any_logical = true;
                }
            }
            atomicAdd(reinterpret_cast<unsigned long long*>(&counts->passed), 1ULL);
            if (any_logical) {
                atomicAdd(reinterpret_cast<unsigned long long*>(&counts->logical_errors), 1ULL);
            }
        }
        __syncthreads();
    }
}

}  // namespace

SurvivorResult gpu_sample_survivors(const CompiledModule& program, uint64_t shots,
                                    const GpuSamplerOptions& options) {
    auto flat = flatten_program(program);

    uint32_t block_size = options.block_size == 0 ? 256 : options.block_size;
    if ((block_size & (block_size - 1)) != 0 || block_size > 1024) {
        throw std::runtime_error("GPU block size must be a power of two <= 1024");
    }
    if (options.keep_records) {
        throw std::runtime_error("GPU aggregate sampler does not keep per-shot records yet");
    }

    bool use_thread_kernel = flat.peak_rank <= kThreadMaxPeakRank;
    bool use_shared_coop_kernel =
        flat.peak_rank > kThreadMaxPeakRank && flat.peak_rank <= kSharedMaxPeakRank;
    bool use_global_coop_kernel = flat.peak_rank > kSharedMaxPeakRank;
    uint64_t max_batch_shots = std::min<uint64_t>(shots, kMaxBatchShots);
    uint32_t max_blocks = use_thread_kernel
                              ? static_cast<uint32_t>((max_batch_shots + block_size - 1) /
                                                      block_size)
                              : 1u;
    uint32_t global_worker_blocks = 0;
    if (use_global_coop_kernel) {
        int device = 0;
        hipDeviceProp_t prop{};
        check_hip(hipGetDevice(&device), "get HIP device");
        check_hip(hipGetDeviceProperties(&prop, device), "get HIP device properties");
        global_worker_blocks = std::max(1, prop.multiProcessorCount * 2);
        if (max_batch_shots < global_worker_blocks) {
            global_worker_blocks = static_cast<uint32_t>(max_batch_shots);
        }
    }

    DeviceBuffers buffers;
    try {
        copy_to_device(buffers.instrs, flat.instrs, "copy instrs");
        copy_to_device(buffers.pauli_masks, flat.pauli_masks, "copy pauli masks");
        copy_to_device(buffers.noise_sites, flat.noise_sites, "copy noise sites");
        copy_to_device(buffers.noise_channels, flat.noise_channels, "copy noise channels");
        copy_to_device(buffers.noise_hazards, flat.noise_hazards, "copy noise hazards");
        copy_to_device(buffers.readout_noise, flat.readout_noise, "copy readout noise");
        copy_to_device(buffers.detector_offsets, flat.detector_offsets, "copy detector offsets");
        copy_to_device(buffers.detector_targets, flat.detector_targets, "copy detector targets");
        copy_to_device(buffers.observable_offsets, flat.observable_offsets, "copy observable offsets");
        copy_to_device(buffers.observable_targets, flat.observable_targets, "copy observable targets");
        copy_to_device(buffers.expected_observables, flat.expected_observables, "copy expected observables");
        copy_to_device(buffers.fused_u2, flat.fused_u2, "copy fused u2");
        copy_to_device(buffers.fused_u4, flat.fused_u4, "copy fused u4");
        copy_to_device(buffers.exp_val_masks, flat.exp_val_masks, "copy exp val masks");
        check_hip(hipMalloc(reinterpret_cast<void**>(&buffers.block_counts),
                              static_cast<size_t>(max_blocks) * sizeof(BlockCounts)),
                   "allocate block counts");
        bool need_global_buffers =
            use_global_coop_kernel ||
            (options.hybrid && flat.peak_rank > kSharedMaxPeakRank &&
             flat.peak_rank <= kGlobalMaxPeakRank);
        if (need_global_buffers) {
            size_t v_count = static_cast<size_t>(global_worker_blocks) * kGlobalMaxAmplitudes;
            size_t scratch_count =
                static_cast<size_t>(global_worker_blocks) * (kGlobalMaxAmplitudes / 2);
            check_hip(hipMalloc(reinterpret_cast<void**>(&buffers.global_v),
                                  v_count * sizeof(GpuComplex)),
                       "allocate global cooperative state");
            check_hip(hipMalloc(reinterpret_cast<void**>(&buffers.global_scratch),
                                  scratch_count * sizeof(GpuComplex)),
                       "allocate global cooperative scratch");
            check_hip(hipMalloc(reinterpret_cast<void**>(&buffers.work_counter),
                                  kNumXCDs * sizeof(uint64_t)),
                       "allocate per-XCD work counters");
        }

        GpuProgram gpu_program{};
        gpu_program.instrs = buffers.instrs;
        gpu_program.num_instrs = static_cast<uint32_t>(flat.instrs.size());
        gpu_program.peak_rank = flat.peak_rank;
        gpu_program.total_meas_slots = flat.total_meas_slots;
        gpu_program.num_observables = flat.num_observables;
        gpu_program.num_exp_vals = flat.num_exp_vals;
        gpu_program.pauli_masks = buffers.pauli_masks;
        gpu_program.noise_sites = buffers.noise_sites;
        gpu_program.noise_channels = buffers.noise_channels;
        gpu_program.num_noise_sites = static_cast<uint32_t>(flat.noise_sites.size());
        gpu_program.noise_hazards = buffers.noise_hazards;
        gpu_program.readout_noise = buffers.readout_noise;
        gpu_program.detector_offsets = buffers.detector_offsets;
        gpu_program.detector_targets = buffers.detector_targets;
        gpu_program.observable_offsets = buffers.observable_offsets;
        gpu_program.observable_targets = buffers.observable_targets;
        gpu_program.expected_observables = buffers.expected_observables;
        gpu_program.fused_u2 = buffers.fused_u2;
        gpu_program.fused_u4 = buffers.fused_u4;
        gpu_program.exp_val_masks = buffers.exp_val_masks;
        gpu_program.has_extended_opcodes = flat.has_extended_opcodes;

        hipEvent_t start{};
        hipEvent_t stop{};
        check_hip(hipEventCreate(&start), "create start event");
        check_hip(hipEventCreate(&stop), "create stop event");

        size_t shared_bytes = static_cast<size_t>(block_size) * (2 + kMaxObs) * sizeof(uint64_t);
        std::vector<BlockCounts> host_counts(max_blocks);
        uint64_t passed = 0;
        uint64_t logical = 0;
        std::vector<uint64_t> observable_ones(flat.num_observables, 0);
        std::vector<double> exp_val_sums(flat.num_exp_vals, 0.0);
        uint64_t exp_val_count = 0;
        double kernel_seconds = 0.0;

        for (uint64_t shot_offset = 0; shot_offset < shots; shot_offset += kMaxBatchShots) {
            uint64_t batch_shots = std::min<uint64_t>(kMaxBatchShots, shots - shot_offset);
            uint32_t batch_blocks = 0;
            if (use_thread_kernel) {
                batch_blocks =
                    static_cast<uint32_t>((batch_shots + block_size - 1) / block_size);
            } else if (use_shared_coop_kernel) {
                batch_blocks = static_cast<uint32_t>(batch_shots);
            } else {
                batch_blocks = global_worker_blocks;
            }

            // Compiled megakernel paths (--hybrid) — checked FIRST so they
            // take priority over the SVM interpreter paths below.
            // For coop/global tiers, fall back to SVM if the circuit uses
            // operations not yet implemented in the compiled codegen path —
            // this is detected by compile_or_load returning !valid when the
            // generated kernel would discard every shot.
            if (options.hybrid && flat.peak_rank <= kThreadMaxPeakRank) {
                // Compiled megakernel via persistent HSA dispatch (register tier, rank 0-4)
                static HsaLoadedKernel hsa_ck{};
                static PersistentDispatcher pd_ck;
                if (!hsa_ck.valid) {
                    hsa_ck = compile_or_load_kernel(flat);
                }
                if (!hsa_ck.valid) {
                    std::cerr << "[clifft-gpu] FATAL: compiled kernel (register tier) failed to load for rank "
                              << flat.peak_rank << " — aborting (--hybrid requires compiled kernel)\n";
                    std::abort();
                }
                if (!pd_ck.is_bound()) pd_ck.bind(hsa_ck, 0);
                struct { uint64_t so; uint64_t bs; uint64_t sd;
                         BlockCounts* bc; uint32_t nobs; uint32_t nexp; } kargs;
                kargs.so = shot_offset; kargs.bs = batch_shots;
                kargs.sd = options.seed; kargs.bc = buffers.block_counts;
                kargs.nobs = flat.num_observables; kargs.nexp = flat.num_exp_vals;
                kernel_seconds += pd_ck.dispatch(
                    batch_blocks * block_size, block_size,
                    &kargs, sizeof(kargs));
            } else if (options.hybrid && flat.peak_rank <= kSharedMaxPeakRank) {
                // Compiled megakernel via persistent HSA dispatch (LDS/coop tier, rank 5-10)
                static HsaLoadedKernel hsa_ck_coop{};
                static PersistentDispatcher pd_coop;
                if (!hsa_ck_coop.valid) {
                    hsa_ck_coop = compile_or_load_kernel_coop(flat);
                }
                if (!hsa_ck_coop.valid) {
                    std::cerr << "[clifft-gpu] FATAL: compiled kernel (coop tier) failed to load for rank "
                              << flat.peak_rank << " — aborting (--hybrid requires compiled kernel)\n";
                    std::abort();
                }
                if (!pd_coop.is_bound()) pd_coop.bind(hsa_ck_coop, 0);
                check_hip(hipMemset(buffers.block_counts, 0, sizeof(BlockCounts)),
                           "clear coop compiled counts");
                struct { uint64_t so; uint64_t bs; uint64_t sd;
                         BlockCounts* bc; uint32_t nobs; uint32_t nexp; } kargs;
                kargs.so = shot_offset; kargs.bs = batch_shots;
                kargs.sd = options.seed; kargs.bc = buffers.block_counts;
                kargs.nobs = flat.num_observables; kargs.nexp = flat.num_exp_vals;
                uint32_t coop_blocks = static_cast<uint32_t>(batch_shots);
                kernel_seconds += pd_coop.dispatch(
                    coop_blocks * block_size, block_size,
                    &kargs, sizeof(kargs));
            } else if (options.hybrid && flat.peak_rank <= kGlobalMaxPeakRank) {
                // Compiled megakernel via persistent HSA dispatch (global-coop tier, rank 11-19)
                static HsaLoadedKernel hsa_ck_global{};
                static PersistentDispatcher pd_global;
                if (!hsa_ck_global.valid) {
                    hsa_ck_global = compile_or_load_kernel_global(flat);
                }
                if (!hsa_ck_global.valid) {
                    std::cerr << "[clifft-gpu] FATAL: compiled kernel (global tier) failed to load for rank "
                              << flat.peak_rank << " — aborting (--hybrid requires compiled kernel)\n";
                    std::abort();
                }
                if (!pd_global.is_bound()) pd_global.bind(hsa_ck_global, 0);
                check_hip(hipMemset(buffers.block_counts, 0, sizeof(BlockCounts)),
                           "clear global compiled counts");
                check_hip(hipMemset(buffers.work_counter, 0, kNumXCDs * sizeof(uint64_t)),
                           "clear per-XCD work counters compiled");
                struct { uint64_t so; uint64_t bs; uint64_t sd;
                         GpuComplex* gv; GpuComplex* gs; uint64_t* wc;
                         BlockCounts* bc; uint32_t nobs; uint32_t nexp; } kargs;
                kargs.so = shot_offset; kargs.bs = batch_shots;
                kargs.sd = options.seed;
                kargs.gv = buffers.global_v; kargs.gs = buffers.global_scratch;
                kargs.wc = buffers.work_counter;
                kargs.bc = buffers.block_counts;
                kargs.nobs = flat.num_observables; kargs.nexp = flat.num_exp_vals;
                kernel_seconds += pd_global.dispatch(
                    batch_blocks * block_size, block_size,
                    &kargs, sizeof(kargs));
            } else if (options.mlir && flat.peak_rank <= kThreadMaxPeakRank) {
#ifdef CLIFFT_ENABLE_MLIR
                // MLIR→LLVM-IR megakernel via HSA dispatch (register tier)
                static HsaLoadedKernel hsa_ck_mlir{};
                if (!hsa_ck_mlir.valid) {
                    hsa_ck_mlir = compile_or_load_mlir_kernel(flat);
                }
                if (hsa_ck_mlir.valid) {
                    struct { uint64_t so; uint64_t bs; uint64_t sd;
                             BlockCounts* bc; uint32_t nobs; uint32_t nexp; } kargs;
                    kargs.so = shot_offset; kargs.bs = batch_shots;
                    kargs.sd = options.seed; kargs.bc = buffers.block_counts;
                    kargs.nobs = flat.num_observables; kargs.nexp = flat.num_exp_vals;
                    kernel_seconds += hsa_dispatch_and_wait(hsa_ck_mlir, 0,
                        batch_blocks * block_size, block_size,
                        &kargs, sizeof(kargs));
                } else {
                    std::cerr << "[clifft-gpu] FATAL: MLIR kernel failed to load — aborting\n";
                    std::abort();
                }
#else
                std::cerr << "[clifft-gpu] FATAL: MLIR NOT COMPILED IN (CLIFFT_ENABLE_MLIR=OFF)\n";
                std::abort();
#endif
            // SVM interpreter paths (no --hybrid / --mlir)
            } else if (use_shared_coop_kernel) {
                check_hip(hipEventRecord(start), "record start event");
                check_hip(hipMemset(buffers.block_counts, 0, sizeof(BlockCounts)),
                           "clear coop counts");
                sample_kernel_coop<<<batch_blocks, block_size>>>(
                    gpu_program, shot_offset, batch_shots, options.seed, buffers.block_counts);
            } else if (use_global_coop_kernel) {
                check_hip(hipEventRecord(start), "record start event");
                check_hip(hipMemset(buffers.block_counts, 0, sizeof(BlockCounts)),
                           "clear global coop counts");
                check_hip(hipMemset(buffers.work_counter, 0, kNumXCDs * sizeof(uint64_t)),
                           "clear per-XCD work counters");
                sample_kernel_global_coop<<<batch_blocks, block_size>>>(
                    gpu_program, shot_offset, batch_shots, options.seed, buffers.global_v,
                    buffers.global_scratch, buffers.work_counter, buffers.block_counts);
            } else {
                check_hip(hipEventRecord(start), "record start event");
                sample_kernel<<<batch_blocks, block_size, shared_bytes>>>(
                    gpu_program, shot_offset, batch_shots, options.seed, buffers.block_counts);
            }

            // Timing: HSA paths already accumulated kernel_seconds via hsa_dispatch_and_wait.
            // SVM/fallback <<<>>> paths need HIP event timing.
            bool used_hsa_dispatch = (options.hybrid || options.mlir) &&
                flat.peak_rank <= kGlobalMaxPeakRank;
            if (!used_hsa_dispatch) {
                check_hip(hipGetLastError(), "launch sample kernel");
                check_hip(hipEventRecord(stop), "record stop event");
                check_hip(hipEventSynchronize(stop), "synchronize sample kernel");
                float ms = 0.0f;
                check_hip(hipEventElapsedTime(&ms, start, stop), "measure kernel time");
                kernel_seconds += static_cast<double>(ms) / 1000.0;
            }

            uint32_t copy_blocks = use_thread_kernel ? batch_blocks : 1u;
            check_hip(hipMemcpy(host_counts.data(), buffers.block_counts,
                                  static_cast<size_t>(copy_blocks) * sizeof(BlockCounts),
                                  hipMemcpyDeviceToHost),
                       "copy block counts");

            for (uint32_t block = 0; block < copy_blocks; ++block) {
                const auto& bc = host_counts[block];
                passed += bc.passed;
                logical += bc.logical_errors;
                for (uint32_t i = 0; i < flat.num_observables; ++i) {
                    observable_ones[i] += bc.observable_ones[i];
                }
                for (uint32_t i = 0; i < flat.num_exp_vals; ++i) {
                    exp_val_sums[i] += bc.exp_val_sums[i];
                }
                exp_val_count += bc.exp_val_count;
            }
            if (shots >= kMaxBatchShots) {
                uint64_t done = shot_offset + batch_shots;
                std::cerr << "[clifft-gpu] sampled " << done << "/" << shots
                          << " shots, passed=" << passed << ", logical=" << logical
                          << ", kernel_seconds=" << kernel_seconds << "\n";
            }
        }
        hipEventDestroy(start);
        hipEventDestroy(stop);

        SurvivorResult result;
        result.total_shots = static_cast<uint32_t>(std::min<uint64_t>(shots, UINT32_MAX));
        result.passed_shots = static_cast<uint32_t>(std::min<uint64_t>(passed, UINT32_MAX));
        result.logical_errors = static_cast<uint32_t>(std::min<uint64_t>(logical, UINT32_MAX));
        result.observable_ones.resize(flat.num_observables);
        for (uint32_t i = 0; i < flat.num_observables; ++i) {
            result.observable_ones[i] = observable_ones[i];
        }
        if (flat.num_exp_vals > 0 && exp_val_count > 0) {
            result.exp_vals.resize(flat.num_exp_vals);
            for (uint32_t i = 0; i < flat.num_exp_vals; ++i) {
                result.exp_vals[i] = exp_val_sums[i] / static_cast<double>(exp_val_count);
            }
        }

        free_buffers(buffers);
        return result;
    } catch (...) {
        free_buffers(buffers);
        throw;
    }
}

std::string gpu_backend_info() {
    std::ostringstream ss;
    int count = 0;
    hipError_t err = hipGetDeviceCount(&count);
    if (err != hipSuccess) {
        ss << "hip runtime unavailable: " << hipGetErrorString(err);
        return ss.str();
    }
    ss << "hip devices: " << count;
    for (int i = 0; i < count; ++i) {
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, i) == hipSuccess) {
            ss << "\n[" << i << "] " << prop.name << " " << prop.gcnArchName
               << " global_mem=" << static_cast<unsigned long long>(prop.totalGlobalMem);
        }
    }
    return ss.str();
}

}  // namespace gpu
}  // namespace clifft
