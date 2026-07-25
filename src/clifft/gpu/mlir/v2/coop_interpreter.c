// coop_interpreter.c — MLIR-V2 coop-tier bytecode interpreter. Plain C, NO HIP.
// A RUNTIME loop over the circuit's operand bytecode (never unrolled), mirroring
// the proven SVM coop interpreter (hip_sampler.hip execute_shot_coop). One
// workgroup (256 threads) cooperates on ONE shot; the statevector lives in LDS.
//
// This is the correctness-first first slice (CODEX_REVIEW.md §7). Opcodes are
// added incrementally; any opcode not yet handled marks the shot discarded so
// mismatches are loud, never silent.
//
// Compiled to amdgcn via cmake/ClifftAmdgcn.cmake (clang/llc/ld.lld, no -x hip).

#include "clifft/gpu/mlir/v2/device_abi.h"

typedef unsigned int   u32;
typedef unsigned long  u64;
typedef unsigned char  u8;

// ----- opcodes (MUST match src/clifft/backend/backend.h enum order EXACTLY;
// verified against the authoritative enum — see device_abi_checks). --------
enum {
    OP_FRAME_CNOT = 0, OP_FRAME_CZ, OP_FRAME_H, OP_FRAME_S, OP_FRAME_S_DAG,
    OP_FRAME_SWAP,                                   // 5
    OP_ARRAY_CNOT, OP_ARRAY_CZ, OP_ARRAY_SWAP, OP_ARRAY_MULTI_CNOT,
    OP_ARRAY_MULTI_CZ, OP_ARRAY_H, OP_ARRAY_S, OP_ARRAY_S_DAG, OP_ARRAY_T,
    OP_ARRAY_T_DAG, OP_ARRAY_ROT, OP_ARRAY_U2, OP_ARRAY_U4,   // 6..18
    OP_EXPAND, OP_EXPAND_T, OP_EXPAND_T_DAG, OP_EXPAND_ROT,   // 19..22
    OP_MEAS_DORMANT_STATIC, OP_MEAS_DORMANT_RANDOM,           // 23,24
    OP_MEAS_ACTIVE_DIAGONAL, OP_MEAS_ACTIVE_INTERFERE, OP_SWAP_MEAS_INTERFERE, // 25,26,27
    OP_MEAS_DORMANT_STATIC_FORCED, OP_MEAS_DORMANT_RANDOM_FORCED,      // 28,29
    OP_MEAS_ACTIVE_DIAGONAL_FORCED, OP_MEAS_ACTIVE_INTERFERE_FORCED,   // 30,31
    OP_SWAP_MEAS_INTERFERE_FORCED,                            // 32
    OP_APPLY_PAULI, OP_NOISE, OP_NOISE_BLOCK, OP_READOUT_NOISE,  // 33..36
    OP_DETECTOR, OP_POSTSELECT, OP_OBSERVABLE, OP_EXP_VAL,    // 37..40
};

// flags
enum { FLAG_SIGN = 1u << 0, FLAG_IDENTITY = 1u << 2, FLAG_EXPECTED_ONE = 1u << 3 };

// ----- LDS shared state (per workgroup / per shot) ---------------------------
// Sized for coop tier: rank <= 10 -> 1024 amplitudes. extern = uninitialized
// (llc rejects an initializer on addrspace(3)).
//
// P1 LDS reclamation (2026-07-25): the coop kernel was LDS-occupancy-bound at
// 25 KB -> 2 wg/CU. Shrinking the reduction buffers is free + byte-exact:
//   - lds_red0/1: coop_reduce2 only writes warps 0..3 and reads t<4 -> [4] is
//     the true size (was [256] = 4 KB wasted). Sized to 8 (>=4, aligned).
//   - lds_red_scratch: only OP_SWAP_MEAS_INTERFERE uses it, and at coop rank<=10
//     the fold half = 1<<to <= 1<<9 = 512. [512] is exact (was [1024] = 4 KB).
// This is the SAME per-circuit sizing the MLIR specializer will bake in at
// compile time; here it is a conservative static bound for the runtime tier.
#define V2_MAX_AMP     1024
#define V2_MAX_MEAS    4096
#define V2_MEAS_WORDS  (V2_MAX_MEAS / 64)  // meas records bit-packed: 4096b=64 u64
#define V2_SCRATCH_AMP 512    // coop SWAP_MEAS fold half <= 2^(10-1)
#define V2_RED_WARPS   8      // coop_reduce2 touches only warps 0..3
extern __attribute__((address_space(3))) CV2Complex lds_v[V2_MAX_AMP];
extern __attribute__((address_space(3))) CV2Complex lds_red_scratch[V2_SCRATCH_AMP];
// Measurement records are boolean (0/1) and every access is tid0-only, so they
// bit-pack safely: 4096 bytes -> 512 bytes (~3.5 KB reclaimed). Accessors below.
extern __attribute__((address_space(3))) u64  lds_meas[V2_MEAS_WORDS];
extern __attribute__((address_space(3))) u8   lds_obs[CLIFFT_V2_MAX_OBS];
extern __attribute__((address_space(3))) u64  lds_px[CLIFFT_V2_PAULI_WORDS];
extern __attribute__((address_space(3))) u64  lds_pz[CLIFFT_V2_PAULI_WORDS];
extern __attribute__((address_space(3))) u32  lds_active_k;
extern __attribute__((address_space(3))) u8   lds_discarded;
extern __attribute__((address_space(3))) u64  lds_rng[4];   // tid0-owned RNG state
extern __attribute__((address_space(3))) u8   lds_branch;   // sampled branch broadcast
extern __attribute__((address_space(3))) u32  lds_next_noise;// next scheduled noise site
extern __attribute__((address_space(3))) double lds_red0[V2_RED_WARPS];
extern __attribute__((address_space(3))) double lds_red1[V2_RED_WARPS];
extern __attribute__((address_space(3))) u32 lds_xcd;        // global tier: XCD id
extern __attribute__((address_space(3))) u64 lds_shot;       // global tier: claimed shot

// ----- intrinsics ------------------------------------------------------------
static inline u32 tid(void)  { return __builtin_amdgcn_workitem_id_x(); }
static inline u32 bid(void)  { return __builtin_amdgcn_workgroup_id_x(); }
static inline void barrier(void) { __builtin_amdgcn_s_barrier(); }

// ----- RNG (xoshiro256++ seeded by splitmix64), byte-exact with SVM ----------
static inline u64 rotl64(u64 x, int k) { return (x << k) | (x >> (64 - k)); }
static inline u64 splitmix64(u64* state) {
    u64 z = (*state += 0x9e3779b97f4a7c15UL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9UL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebUL;
    return z ^ (z >> 31);
}
// rng state lives in LDS (tid0 only draws); s points at lds_rng.
static inline void rng_seed(__attribute__((address_space(3))) u64* s, u64 seed, u64 shot_id) {
    u64 z = seed ^ (0x9e3779b97f4a7c15UL * (shot_id + 1));
    s[0] = splitmix64(&z); s[1] = splitmix64(&z);
    s[2] = splitmix64(&z); s[3] = splitmix64(&z);
}
static inline u64 rng_next(__attribute__((address_space(3))) u64* s) {
    u64 result = rotl64(s[0] + s[3], 23) + s[0];
    u64 t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t;    s[3] = rotl64(s[3], 45);
    return result;
}
static inline double rng_uniform(__attribute__((address_space(3))) u64* s) {
    return (double)(rng_next(s) >> 11) * 0x1.0p-53;
}

// ----- Pauli-frame bit helpers (on LDS words) --------------------------------
static inline int fget(__attribute__((address_space(3))) u64* w, u32 i) {
    return (int)((w[i >> 6] >> (i & 63u)) & 1UL);
}
static inline void fset(__attribute__((address_space(3))) u64* w, u32 i, int v) {
    u64 m = 1UL << (i & 63u);
    if (v) w[i >> 6] |= m; else w[i >> 6] &= ~m;
}
static inline void fxor(__attribute__((address_space(3))) u64* w, u32 i, int v) {
    if (v) w[i >> 6] ^= 1UL << (i & 63u);
}
static inline void fswap(__attribute__((address_space(3))) u64* w, u32 a, u32 b) {
    int va = fget(w, a), vb = fget(w, b);
    fset(w, a, vb); fset(w, b, va);
}
// ----- measurement-record bit helpers (bit-packed lds_meas, tid0-only) -------
static inline u8 mget(u32 i) {
    return (u8)((lds_meas[i >> 6] >> (i & 63u)) & 1UL);
}
static inline void mset(u32 i, u8 v) {
    u64 m = 1UL << (i & 63u);
    if (v & 1u) lds_meas[i >> 6] |= m; else lds_meas[i >> 6] &= ~m;
}
static inline void mxor1(u32 i) { lds_meas[i >> 6] ^= 1UL << (i & 63u); }

// ----- complex + amplitude helpers -------------------------------------------
#define V2_INV_SQRT2 0.70710678118654752440
#define V2_DUST_EPS  1e-18

static inline CV2Complex cmul(CV2Complex a, CV2Complex b) {
    CV2Complex r;
    r.re = a.re * b.re - a.im * b.im;
    r.im = a.re * b.im + a.im * b.re;
    return r;
}
static inline CV2Complex cadd(CV2Complex a, CV2Complex b) {
    CV2Complex r; r.re = a.re + b.re; r.im = a.im + b.im; return r;
}
static inline CV2Complex csub(CV2Complex a, CV2Complex b) {
    CV2Complex r; r.re = a.re - b.re; r.im = a.im - b.im; return r;
}
// NOTE: scalar multiply happens in f64 then narrows to f32 — byte-exact with
// SVM cscale (static_cast<float>(static_cast<double>(a.re) * s)). Do NOT
// simplify to an f32 multiply; that desyncs measurement branches vs the gold.
static inline CV2Complex cscale(CV2Complex a, double s) {
    CV2Complex r;
    r.re = (float)((double)a.re * s);
    r.im = (float)((double)a.im * s);
    return r;
}
// |v|^2 accumulated in f64 (match gold cnorm: extend each component first).
static inline double cnorm(CV2Complex v) {
    double re = (double)v.re, im = (double)v.im;
    return re * re + im * im;
}

// Insert a 0 bit at position `pos`: bits below stay, bits >=pos shift up by one.
static inline u64 insert_zero_bit(u64 val, u32 pos) {
    u64 lo = val & ((1ull << pos) - 1ull);
    u64 hi = (val & ~((1ull << pos) - 1ull)) << 1;
    return lo | hi;
}

// Scatter i over one zeroed bit slot at `pos` — mirrors SVM scatter_bits_1.
static inline u64 scatter_bits_1(u64 val, u32 pos) { return insert_zero_bit(val, pos); }

// Scatter i over two zeroed bit slots (lo first, then hi) — mirrors SVM.
static inline u64 scatter_bits_2(u64 val, u32 b1, u32 b2) {
    u32 lo = b1 < b2 ? b1 : b2;
    u32 hi = b1 < b2 ? b2 : b1;
    val = insert_zero_bit(val, lo);
    return insert_zero_bit(val, hi);
}

// Cooperative diagonal phase on an active axis: v[idx | axis_bit] *= phase,
// strided over the 2^(active_k-1) lower half. Matches SVM coop_apply_phase.
static inline void coop_apply_phase(u32 t, CV2Complex* v, u32 axis, CV2Complex phase) {
    u64 axis_bit = 1ull << axis;
    u64 iters = 1ull << (lds_active_k - 1u);
    for (u64 i = t; i < iters; i += 256u) {
        u64 idx = insert_zero_bit(i, axis) | axis_bit;
        v[idx] = cmul(v[idx], phase);
    }
    barrier();
}

// Wavefront butterfly shuffle of an f64 (lane l receives lane (l^offset)'s
// value) via ds_bpermute — the amdgcn primitive HIP's __shfl_xor lowers to.
static inline double shfl_xor_f64(double v, u32 lane, int offset) {
    u64 bits;
    __builtin_memcpy(&bits, &v, 8);
    u32 lo = (u32)bits, hi = (u32)(bits >> 32);
    u32 addr = ((lane ^ (u32)offset) & 63u) << 2;  // ds_bpermute addr = lane*4
    u32 rlo = __builtin_amdgcn_ds_bpermute((int)addr, (int)lo);
    u32 rhi = __builtin_amdgcn_ds_bpermute((int)addr, (int)hi);
    u64 rbits = ((u64)rhi << 32) | (u64)rlo;
    double r;
    __builtin_memcpy(&r, &rbits, 8);
    return r;
}

// Cooperative reduction of two per-thread f64 partials across 256 threads.
// MUST reproduce SVM coop_reduce2's exact summation order (intra-wave butterfly
// 32..1, then a 4-warp butterfly 2..1) or f64 rounding diverges at measurement
// branch points on reduction-heavy circuits (rank-10 QEC). All 256 threads call
// it; result broadcast via lds_red[0].
static inline void coop_reduce2(u32 t, double l0, double l1, double* out0, double* out1) {
    u32 lane = t & 63u;
    u32 warp = t >> 6;
    for (int off = 32; off > 0; off >>= 1) {
        l0 += shfl_xor_f64(l0, lane, off);
        l1 += shfl_xor_f64(l1, lane, off);
    }
    if (lane == 0u) { lds_red0[warp] = l0; lds_red1[warp] = l1; }
    barrier();
    if (t < 4u) {
        l0 = lds_red0[t]; l1 = lds_red1[t];
        for (int off = 2; off > 0; off >>= 1) {
            l0 += shfl_xor_f64(l0, lane, off);
            l1 += shfl_xor_f64(l1, lane, off);
        }
    }
    if (t == 0u) { lds_red0[0] = l0; lds_red1[0] = l1; }
    barrier();
    *out0 = lds_red0[0]; *out1 = lds_red1[0];
    barrier();
}

// sample_branch — byte-exact with SVM (dust clamp + rng draw only when needed).
static inline u8 sample_branch(double p0, double p1, double total) {
    double eps = V2_DUST_EPS * total;
    if (p1 <= eps) return 0;
    if (p0 <= eps) return 1;
    return (rng_uniform(lds_rng) * total < p0) ? 0u : 1u;
}

// ROCm device-library transcendental (linked from ocml.bc). SVM's log() lowers
// to this same symbol, so calling it directly keeps V2 byte-exact.
extern double __ocml_log_f64(double);
static inline double ocml_log_f64(double x) { return __ocml_log_f64(x); }

// tid0-only: advance lds_next_noise via exponential-hazard sampling (byte-exact
// with SVM coop_draw_next_noise). Binary-searches the cumulative-hazard table.
static inline void draw_next_noise(const double* hazards, u32 num_sites) {
    if (num_sites == 0u || lds_next_noise >= num_sites) { lds_next_noise = 0xffffffffu; return; }
    double current_hazard = (lds_next_noise == 0u) ? 0.0 : hazards[lds_next_noise - 1u];
    double target = current_hazard + (-ocml_log_f64(1.0 - rng_uniform(lds_rng)));
    u32 lo = 0u, hi = num_sites;
    while (lo < hi) {
        u32 mid = lo + ((hi - lo) >> 1);
        if (hazards[mid] <= target) lo = mid + 1u; else hi = mid;
    }
    lds_next_noise = (lo >= num_sites) ? 0xffffffffu : lo;
}

// tid0-only: apply the Pauli channel drawn at noise site `site_idx` into the
// frame (matches SVM coop OP_NOISE body).
static inline void apply_noise_site(const CV2NoiseSite* sites, const CV2Channel* channels,
                                    u32 site_idx) {
    CV2NoiseSite site = sites[site_idx];
    double roll = rng_uniform(lds_rng) * site.prob_sum;
    double cumulative = 0.0;
    for (u32 k = 0; k < site.count; ++k) {
        const CV2Channel* ch = &channels[site.offset + k];
        cumulative += ch->prob;
        if (roll < cumulative) {
            for (u32 w = 0; w < CLIFFT_V2_PAULI_WORDS; ++w) {
                lds_px[w] ^= ch->x[w];
                lds_pz[w] ^= ch->z[w];
            }
            break;
        }
    }
}

// =============================================================================
// execute_shot — the SHARED interpreter body. One workgroup = one shot; 256
// threads cooperate on the amplitude buffer `v` (flat pointer: LDS for the coop
// tier, HBM for the global tier). `scratch` is a half-size fold buffer in the
// same space. `amp_capacity` is the number of amplitudes to zero at init
// (1<<peak_rank). Classical state (frame, meas, obs, rng) always lives in LDS.
// Both kernels below call this; the opcode logic is written ONCE.
// =============================================================================
static void execute_shot(CV2Complex* v, CV2Complex* scratch, u32 amp_capacity,
                         u64 shot_id,
                         const CV2Instr* instrs, u32 num_instrs,
                         u32 total_meas_slots, u32 num_observables,
                         u64 seed,
                         CV2BlockCounts* block_counts,
                         const CV2FusedU2Entry* fused_u2,
                         const CV2FusedU4Entry* fused_u4,
                         const u32* observable_offsets,
                         const u32* observable_targets,
                         const CV2NoiseSite* noise_sites,
                         const CV2Channel* noise_channels,
                         const double* noise_hazards,
                         u32 num_noise_sites,
                         const CV2Mask* pauli_masks,
                         const CV2ReadoutNoise* readout_noise,
                         const u32* detector_offsets,
                         const u32* detector_targets) {
    u32 t = tid();

    // --- cooperative init ---
    for (u32 i = t; i < amp_capacity; i += 256u) { v[i].re = 0.0f; v[i].im = 0.0f; }
    barrier();
    if (t == 0) {
        for (u32 w = 0; w < CLIFFT_V2_PAULI_WORDS; ++w) { lds_px[w] = 0; lds_pz[w] = 0; }
        lds_active_k = 0;
        lds_discarded = 0;
        v[0].re = 1.0f; v[0].im = 0.0f;
        for (u32 i = 0; i < num_observables; ++i) lds_obs[i] = 0;
        for (u32 w = 0; w < V2_MEAS_WORDS; ++w) lds_meas[w] = 0;
        lds_next_noise = 0;
        rng_seed(lds_rng, seed, shot_id);
        draw_next_noise(noise_hazards, num_noise_sites);
    }
    barrier();

    // --- interpreter loop (runtime, never unrolled) ---
    for (u32 pc = 0; pc < num_instrs; ++pc) {
        CV2Instr ins = instrs[pc];
        switch (ins.opcode) {
        case OP_FRAME_CNOT:
            if (t == 0) {
                int px_c = fget(lds_px, ins.axis_1);
                int pz_tt = fget(lds_pz, ins.axis_2);
                fxor(lds_px, ins.axis_2, px_c);
                fxor(lds_pz, ins.axis_1, pz_tt);
            }
            barrier();
            break;
        case OP_FRAME_CZ:
            if (t == 0) {
                int px_a = fget(lds_px, ins.axis_1);
                int px_b = fget(lds_px, ins.axis_2);
                fxor(lds_pz, ins.axis_2, px_a);
                fxor(lds_pz, ins.axis_1, px_b);
            }
            barrier();
            break;
        case OP_FRAME_H:
            if (t == 0) {
                int px = fget(lds_px, ins.axis_1);
                int pz = fget(lds_pz, ins.axis_1);
                fset(lds_px, ins.axis_1, pz);
                fset(lds_pz, ins.axis_1, px);
            }
            barrier();
            break;
        case OP_FRAME_S:
        case OP_FRAME_S_DAG:
            if (t == 0) { int px = fget(lds_px, ins.axis_1); fxor(lds_pz, ins.axis_1, px); }
            barrier();
            break;
        case OP_FRAME_SWAP:
            if (t == 0) {
                int px_a = fget(lds_px, ins.axis_1), px_b = fget(lds_px, ins.axis_2);
                fset(lds_px, ins.axis_1, px_b); fset(lds_px, ins.axis_2, px_a);
                int pz_a = fget(lds_pz, ins.axis_1), pz_b = fget(lds_pz, ins.axis_2);
                fset(lds_pz, ins.axis_1, pz_b); fset(lds_pz, ins.axis_2, pz_a);
            }
            barrier();
            break;
        case OP_MEAS_DORMANT_STATIC:
            // outcome from px[axis] (deterministic); tid0-only. flags: identity/sign.
            if (t == 0) {
                u8 mval;
                if (ins.flags & FLAG_IDENTITY) {
                    mval = (ins.flags & FLAG_SIGN) ? 1u : 0u;
                } else {
                    u8 outcome = (u8)fget(lds_px, ins.axis_1);
                    mval = outcome ^ (u8)((ins.flags & FLAG_SIGN) != 0);
                }
                if (ins.a < V2_MAX_MEAS) mset(ins.a, mval);
            }
            barrier();
            break;
        case OP_MEAS_DORMANT_RANDOM:
            // Random outcome (qubit in superposition, dormant). tid0 draws RNG.
            // Mirrors SVM meas_dormant_random.
            if (t == 0) {
                u8 m_abs = (rng_uniform(lds_rng) < 0.5) ? 0u : 1u;
                fset(lds_px, ins.axis_1, m_abs != 0);
                fset(lds_pz, ins.axis_1, 0);
                if (ins.a < V2_MAX_MEAS)
                    mset(ins.a, m_abs ^ (u8)((ins.flags & FLAG_SIGN) != 0));
            }
            barrier();
            break;
        case OP_EXPAND: {
            // Virtual H on a dormant axis: duplicate v[0,half) -> v[half,2half).
            u32 half = 1u << lds_active_k;
            for (u32 i = t; i < half; i += 256u) v[i + half] = v[i];
            barrier();
            if (t == 0) lds_active_k += 1;
            barrier();
            break;
        }
        case OP_EXPAND_T:
        case OP_EXPAND_T_DAG: {
            // Fused EXPAND + T phase. v[i+half] = v[i] * (1/sqrt2, +-1/sqrt2).
            u32 half = 1u << lds_active_k;
            int px = fget(lds_px, ins.axis_1);
            double imag = (ins.opcode == OP_EXPAND_T_DAG) ? -V2_INV_SQRT2 : V2_INV_SQRT2;
            if (px) imag = -imag;
            CV2Complex phase; phase.re = (float)V2_INV_SQRT2; phase.im = (float)imag;
            for (u32 i = t; i < half; i += 256u) v[i + half] = cmul(v[i], phase);
            barrier();
            if (t == 0) lds_active_k += 1;
            barrier();
            break;
        }
        case OP_MEAS_ACTIVE_DIAGONAL: {
            // Z-basis measurement of an active axis (top axis). Cooperative norm
            // reduction -> tid0 sample -> conditional compaction -> active_k--.
            u32 half = 1u << (lds_active_k - 1u);
            int px = fget(lds_px, ins.axis_1);
            double l0 = 0.0, l1 = 0.0;
            for (u32 i = t; i < half; i += 256u) {
                l0 += cnorm(v[i]);
                l1 += cnorm(v[i + half]);
            }
            double p0, p1;
            coop_reduce2(t, l0, l1, &p0, &p1);
            if (t == 0) {
                u8 b = sample_branch(p0, p1, p0 + p1);
                lds_branch = b;
                u8 m_abs = b ^ (u8)px;
                if (ins.a < V2_MAX_MEAS)
                    mset(ins.a, m_abs ^ (u8)((ins.flags & FLAG_SIGN) != 0));
            }
            barrier();
            if (lds_branch != 0) {
                for (u32 i = t; i < half; i += 256u) v[i] = v[i + half];
            }
            barrier();
            if (t == 0) {
                lds_active_k -= 1;
                u8 m_abs = mget(ins.a) ^ (u8)((ins.flags & FLAG_SIGN) != 0);
                fset(lds_px, ins.axis_1, m_abs != 0);
                fset(lds_pz, ins.axis_1, 0);
            }
            barrier();
            break;
        }
        case OP_MEAS_ACTIVE_INTERFERE: {
            // X-basis fold of an active axis: (v[i]+-v[i+half])/sqrt2, k -> k-1.
            u32 half = 1u << (lds_active_k - 1u);
            int pz = fget(lds_pz, ins.axis_1);
            double lp = 0.0, lm = 0.0;
            for (u32 i = t; i < half; i += 256u) {
                CV2Complex vi = v[i], vh = v[i + half];
                lp += cnorm(cadd(vi, vh));
                lm += cnorm(csub(vi, vh));
            }
            double p_plus, p_minus;
            coop_reduce2(t, lp, lm, &p_plus, &p_minus);
            if (t == 0) {
                u8 b = sample_branch(p_plus, p_minus, p_plus + p_minus);
                lds_branch = b;
                u8 m_abs = b ^ (u8)pz;
                if (ins.a < V2_MAX_MEAS)
                    mset(ins.a, m_abs ^ (u8)((ins.flags & FLAG_SIGN) != 0));
            }
            barrier();
            for (u32 i = t; i < half; i += 256u) {
                CV2Complex vi = v[i], vh = v[i + half];
                CV2Complex folded = (lds_branch == 0) ? cadd(vi, vh) : csub(vi, vh);
                v[i] = cscale(folded, V2_INV_SQRT2);
            }
            barrier();
            if (t == 0) {
                lds_active_k -= 1;
                u8 m_abs = mget(ins.a) ^ (u8)((ins.flags & FLAG_SIGN) != 0);
                fset(lds_px, ins.axis_1, m_abs != 0);
                fset(lds_pz, ins.axis_1, 0);
            }
            barrier();
            break;
        }
        case OP_ARRAY_CNOT: {
            u32 c = ins.axis_1, tg = ins.axis_2;
            u64 c_bit = 1ull << c, t_bit = 1ull << tg;
            u64 iters = 1ull << (lds_active_k - 2u);
            for (u64 i = t; i < iters; i += 256u) {
                u64 base = scatter_bits_2(i, c, tg) | c_bit;
                CV2Complex a = v[base], b = v[base | t_bit];
                v[base] = b; v[base | t_bit] = a;
            }
            barrier();
            if (t == 0) {
                int px_c = fget(lds_px, c), pz_t = fget(lds_pz, tg);
                fxor(lds_px, tg, px_c); fxor(lds_pz, c, pz_t);
            }
            barrier();
            break;
        }
        case OP_ARRAY_CZ: {
            u32 a = ins.axis_1, b = ins.axis_2;
            u64 both = (1ull << a) | (1ull << b);
            u64 iters = 1ull << (lds_active_k - 2u);
            for (u64 i = t; i < iters; i += 256u) {
                u64 idx = scatter_bits_2(i, a, b) | both;
                CV2Complex c = v[idx]; c.re = -c.re; c.im = -c.im; v[idx] = c;
            }
            barrier();
            if (t == 0) {
                int px_a = fget(lds_px, a), px_b = fget(lds_px, b);
                fxor(lds_pz, b, px_a); fxor(lds_pz, a, px_b);
            }
            barrier();
            break;
        }
        case OP_ARRAY_SWAP: {
            u32 a = ins.axis_1, b = ins.axis_2;
            u64 a_bit = 1ull << a, b_bit = 1ull << b;
            u64 iters = 1ull << (lds_active_k - 2u);
            for (u64 i = t; i < iters; i += 256u) {
                u64 base = scatter_bits_2(i, a, b);
                CV2Complex ta = v[base | a_bit], tb = v[base | b_bit];
                v[base | a_bit] = tb; v[base | b_bit] = ta;
            }
            barrier();
            if (t == 0) { fswap(lds_px, a, b); fswap(lds_pz, a, b); }
            barrier();
            break;
        }
        case OP_ARRAY_MULTI_CNOT: {
            u32 tg = ins.axis_1; u64 ctrl_mask = ins.mask;
            u64 t_bit = 1ull << tg;
            u64 half = 1ull << (lds_active_k - 1u);
            for (u64 idx = t; idx < half; idx += 256u) {
                u64 actual = scatter_bits_1(idx, tg);
                if (__builtin_popcountll(actual & ctrl_mask) & 1) {
                    CV2Complex a = v[actual], b = v[actual | t_bit];
                    v[actual] = b; v[actual | t_bit] = a;
                }
            }
            barrier();
            if (t == 0) {
                for (u32 c = 0; c < lds_active_k; ++c) if ((ctrl_mask >> c) & 1ull) {
                    int px_c = fget(lds_px, c), pz_t = fget(lds_pz, tg);
                    fxor(lds_px, tg, px_c); fxor(lds_pz, c, pz_t);
                }
            }
            barrier();
            break;
        }
        case OP_ARRAY_MULTI_CZ: {
            u32 ctrl = ins.axis_1; u64 target_mask = ins.mask;
            u64 c_bit = 1ull << ctrl;
            u64 half = 1ull << (lds_active_k - 1u);
            for (u64 idx = t; idx < half; idx += 256u) {
                u64 actual = scatter_bits_1(idx, ctrl) | c_bit;
                if (__builtin_popcountll(actual & target_mask) & 1) {
                    CV2Complex vv = v[actual]; vv.re = -vv.re; vv.im = -vv.im; v[actual] = vv;
                }
            }
            barrier();
            if (t == 0) {
                for (u32 tg = 0; tg < lds_active_k; ++tg) if ((target_mask >> tg) & 1ull) {
                    int px_c = fget(lds_px, ctrl), px_t = fget(lds_px, tg);
                    fxor(lds_pz, tg, px_c); fxor(lds_pz, ctrl, px_t);
                }
            }
            barrier();
            break;
        }
        case OP_ARRAY_H: {
            u32 axis = ins.axis_1; u64 axis_bit = 1ull << axis;
            u64 iters = 1ull << (lds_active_k - 1u);
            for (u64 i = t; i < iters; i += 256u) {
                u64 i0 = scatter_bits_1(i, axis), i1 = i0 | axis_bit;
                CV2Complex a = v[i0], b = v[i1];
                v[i0] = cscale(cadd(a, b), V2_INV_SQRT2);
                v[i1] = cscale(csub(a, b), V2_INV_SQRT2);
            }
            barrier();
            if (t == 0) {
                int px = fget(lds_px, axis), pz = fget(lds_pz, axis);
                fset(lds_px, axis, pz); fset(lds_pz, axis, px);
            }
            barrier();
            break;
        }
        case OP_ARRAY_S:
        case OP_ARRAY_S_DAG: {
            u32 axis = ins.axis_1;
            CV2Complex ph; ph.re = 0.0f;
            ph.im = (ins.opcode == OP_ARRAY_S_DAG) ? -1.0f : 1.0f;
            coop_apply_phase(t, v, axis, ph);
            if (t == 0) { int px = fget(lds_px, axis); fxor(lds_pz, axis, px); }
            barrier();
            break;
        }
        case OP_ARRAY_ROT: {
            u32 axis = ins.axis_1;
            if (axis < lds_active_k) {
                int px = fget(lds_px, axis);
                double im = px ? -ins.weight_im : ins.weight_im;
                CV2Complex phase; phase.re = (float)ins.weight_re; phase.im = (float)im;
                coop_apply_phase(t, v, axis, phase);
            } else barrier();
            break;
        }
        case OP_EXPAND_ROT: {
            u32 half = 1u << lds_active_k;
            int px = fget(lds_px, ins.axis_1);
            double im = px ? -ins.weight_im : ins.weight_im;
            CV2Complex phase; phase.re = (float)ins.weight_re; phase.im = (float)im;
            for (u32 i = t; i < half; i += 256u) v[i + half] = cmul(v[i], phase);
            barrier();
            if (t == 0) lds_active_k += 1;
            barrier();
            break;
        }
        case OP_ARRAY_U2: {
            u32 axis = ins.axis_1;
            int in_state = (fget(lds_pz, axis) ? 2 : 0) | (fget(lds_px, axis) ? 1 : 0);
            const CV2Complex* mat = fused_u2[ins.a].matrices[in_state];
            if (axis < lds_active_k) {
                u64 axis_bit = 1ull << axis;
                u64 iters = 1ull << (lds_active_k - 1u);
                for (u64 i = t; i < iters; i += 256u) {
                    u64 i0 = scatter_bits_1(i, axis), i1 = i0 | axis_bit;
                    CV2Complex a = v[i0], b = v[i1];
                    v[i0] = cadd(cmul(a, mat[0]), cmul(b, mat[1]));
                    v[i1] = cadd(cmul(a, mat[2]), cmul(b, mat[3]));
                }
            }
            barrier();
            if (t == 0) {
                u8 out = fused_u2[ins.a].out_states[in_state];
                fset(lds_px, axis, (out & 1) != 0);
                fset(lds_pz, axis, (out & 2) != 0);
            }
            barrier();
            break;
        }
        case OP_ARRAY_U4: {
            u32 lo = ins.axis_1, hi = ins.axis_2;
            int in_state = (fget(lds_pz, hi) << 3) | (fget(lds_px, hi) << 2)
                         | (fget(lds_pz, lo) << 1) | fget(lds_px, lo);
            const CV2Complex (*mat)[4] = fused_u4[ins.a].entries[in_state].matrix;
            if (hi < lds_active_k) {
                u64 lo_bit = 1ull << lo, hi_bit = 1ull << hi;
                u64 iters = 1ull << (lds_active_k - 2u);
                for (u64 i = t; i < iters; i += 256u) {
                    u64 base = scatter_bits_2(i, lo, hi);
                    CV2Complex v0 = v[base], v1 = v[base | lo_bit];
                    CV2Complex v2 = v[base | hi_bit], v3 = v[base | lo_bit | hi_bit];
                    v[base] = cadd(cadd(cmul(v0, mat[0][0]), cmul(v1, mat[0][1])),
                                       cadd(cmul(v2, mat[0][2]), cmul(v3, mat[0][3])));
                    v[base | lo_bit] = cadd(cadd(cmul(v0, mat[1][0]), cmul(v1, mat[1][1])),
                                       cadd(cmul(v2, mat[1][2]), cmul(v3, mat[1][3])));
                    v[base | hi_bit] = cadd(cadd(cmul(v0, mat[2][0]), cmul(v1, mat[2][1])),
                                       cadd(cmul(v2, mat[2][2]), cmul(v3, mat[2][3])));
                    v[base | lo_bit | hi_bit] = cadd(cadd(cmul(v0, mat[3][0]), cmul(v1, mat[3][1])),
                                       cadd(cmul(v2, mat[3][2]), cmul(v3, mat[3][3])));
                }
            }
            barrier();
            if (t == 0) {
                u8 out = fused_u4[ins.a].entries[in_state].out_state;
                fset(lds_px, lo, (out & 1) != 0); fset(lds_pz, lo, (out & 2) != 0);
                fset(lds_px, hi, (out & 4) != 0); fset(lds_pz, hi, (out & 8) != 0);
            }
            barrier();
            break;
        }
        case OP_SWAP_MEAS_INTERFERE: {
            u32 from = ins.axis_1, to = ins.axis_2;
            if (from == to) {
                // degenerate: identical to MEAS_ACTIVE_INTERFERE on `to`
                u32 half = 1u << (lds_active_k - 1u);
                int pz = fget(lds_pz, to);
                double lp = 0.0, lm = 0.0;
                for (u32 i = t; i < half; i += 256u) {
                    CV2Complex vi = v[i], vh = v[i + half];
                    lp += cnorm(cadd(vi, vh)); lm += cnorm(csub(vi, vh));
                }
                double pp, pm; coop_reduce2(t, lp, lm, &pp, &pm);
                if (t == 0) {
                    u8 bb = sample_branch(pp, pm, pp + pm); lds_branch = bb;
                    u8 m_abs = bb ^ (u8)pz;
                    if (ins.a < V2_MAX_MEAS) mset(ins.a, m_abs ^ (u8)((ins.flags & FLAG_SIGN) != 0));
                }
                barrier();
                for (u32 i = t; i < half; i += 256u) {
                    CV2Complex vi = v[i], vh = v[i + half];
                    v[i] = cscale((lds_branch == 0) ? cadd(vi, vh) : csub(vi, vh), V2_INV_SQRT2);
                }
                barrier();
                if (t == 0) {
                    lds_active_k -= 1;
                    u8 m_abs = mget(ins.a) ^ (u8)((ins.flags & FLAG_SIGN) != 0);
                    fset(lds_px, to, m_abs != 0); fset(lds_pz, to, 0);
                }
                barrier();
                break;
            }
            if (t == 0) { fswap(lds_px, from, to); fswap(lds_pz, from, to); }
            barrier();
            int pz = fget(lds_pz, to);
            u64 half = 1ull << to, f_bit = 1ull << from;
            double lp = 0.0, lm = 0.0;
            for (u64 idx = t; idx < half; idx += 256u) {
                u64 b_f = (idx >> from) & 1ull;
                u64 base = (idx & ~f_bit) | (b_f << to);
                CV2Complex vb = v[base], vf = v[base | f_bit];
                lp += cnorm(cadd(vb, vf)); lm += cnorm(csub(vb, vf));
            }
            double pp, pm; coop_reduce2(t, lp, lm, &pp, &pm);
            if (t == 0) {
                u8 bb = sample_branch(pp, pm, pp + pm); lds_branch = bb;
                u8 m_abs = bb ^ (u8)pz;
                if (ins.a < V2_MAX_MEAS) mset(ins.a, m_abs ^ (u8)((ins.flags & FLAG_SIGN) != 0));
            }
            barrier();
            // fold into contiguous lower half (write to [idx], read strided)
            for (u64 idx = t; idx < half; idx += 256u) {
                u64 b_f = (idx >> from) & 1ull;
                u64 base = (idx & ~f_bit) | (b_f << to);
                CV2Complex vb = v[base], vf = v[base | f_bit];
                scratch[idx] = cscale((lds_branch == 0) ? cadd(vb, vf) : csub(vb, vf), V2_INV_SQRT2);
            }
            barrier();
            for (u64 idx = t; idx < half; idx += 256u) v[idx] = scratch[idx];
            barrier();
            if (t == 0) {
                lds_active_k -= 1;
                u8 m_abs = mget(ins.a) ^ (u8)((ins.flags & FLAG_SIGN) != 0);
                fset(lds_px, to, m_abs != 0); fset(lds_pz, to, 0);
            }
            barrier();
            break;
        }
        case OP_APPLY_PAULI:
            if (t == 0 && mget(ins.b) != 0) {
                const CV2Mask* m = &pauli_masks[ins.a];
                for (u32 w = 0; w < CLIFFT_V2_PAULI_WORDS; ++w) {
                    lds_px[w] ^= m->x[w]; lds_pz[w] ^= m->z[w];
                }
            }
            barrier();
            break;
        case OP_NOISE:
            if (t == 0 && ins.a == lds_next_noise) {
                apply_noise_site(noise_sites, noise_channels, ins.a);
                lds_next_noise = ins.a + 1u;
                draw_next_noise(noise_hazards, num_noise_sites);
            }
            barrier();
            break;
        case OP_NOISE_BLOCK:
            if (t == 0) {
                u32 end = ins.a + ins.b;
                while (lds_next_noise >= ins.a && lds_next_noise < end) {
                    u32 site_idx = lds_next_noise;
                    apply_noise_site(noise_sites, noise_channels, site_idx);
                    lds_next_noise = site_idx + 1u;
                    draw_next_noise(noise_hazards, num_noise_sites);
                }
            }
            barrier();
            break;
        case OP_ARRAY_T:
        case OP_ARRAY_T_DAG: {
            // Diagonal T phase on an active axis. No-op if axis is dormant.
            if (ins.axis_1 >= lds_active_k) { barrier(); break; }
            int px = fget(lds_px, ins.axis_1);
            double imag = (ins.opcode == OP_ARRAY_T_DAG) ? -V2_INV_SQRT2 : V2_INV_SQRT2;
            if (px) imag = -imag;
            CV2Complex phase; phase.re = (float)V2_INV_SQRT2; phase.im = (float)imag;
            coop_apply_phase(t, v, ins.axis_1, phase);
            break;
        }
        case OP_OBSERVABLE:
            if (t == 0) {
                u32 s0 = observable_offsets[ins.a];
                u32 e0 = observable_offsets[ins.a + 1];
                u8 parity = 0;
                for (u32 k = s0; k < e0; ++k) parity ^= mget(observable_targets[k]);
                if (ins.b < CLIFFT_V2_MAX_OBS) lds_obs[ins.b] ^= parity;
            }
            barrier();
            break;
        case OP_READOUT_NOISE:
            if (t == 0) {
                CV2ReadoutNoise r = readout_noise[ins.a];
                if (rng_uniform(lds_rng) < r.prob) mxor1(r.meas_idx);
            }
            barrier();
            break;
        case OP_POSTSELECT:
            if (t == 0) {
                u32 s0 = detector_offsets[ins.a];
                u32 e0 = detector_offsets[ins.a + 1];
                u8 parity = (ins.flags & FLAG_EXPECTED_ONE) ? 1u : 0u;
                for (u32 k = s0; k < e0; ++k) parity ^= mget(detector_targets[k]);
                if (parity != 0) lds_discarded = 1;
            }
            barrier();
            if (lds_discarded) return;
            break;
        case OP_DETECTOR:
            barrier();
            break;
        default:
            // Not yet implemented in V2 — mark discarded (loud, never silent).
            if (t == 0) lds_discarded = 1;
            barrier();
            break;
        }
    }

    // --- result aggregation (tid0 -> atomic add to block_counts) ---
    barrier();
    if (t == 0) {
        if (!lds_discarded) {
            __atomic_fetch_add(&block_counts->passed, 1UL, __ATOMIC_RELAXED);
            for (u32 i = 0; i < num_observables && i < CLIFFT_V2_MAX_OBS; ++i) {
                if (lds_obs[i] != 0) {
                    __atomic_fetch_add(&block_counts->observable_ones[i], 1UL, __ATOMIC_RELAXED);
                }
            }
        }
    }
}

// =============================================================================
// COOP tier kernel (rank <= 10). One workgroup per shot; amplitudes in LDS.
// =============================================================================
__attribute__((amdgpu_kernel, visibility("default")))
void clifft_v2_coop(const CV2Instr* instrs, u32 num_instrs, u32 peak_rank,
                    u32 total_meas_slots, u32 num_observables,
                    u64 seed, u64 shot_offset, u64 shots,
                    CV2BlockCounts* block_counts,
                    const CV2FusedU2Entry* fused_u2,
                    const CV2FusedU4Entry* fused_u4,
                    const u32* observable_offsets,
                    const u32* observable_targets,
                    const CV2NoiseSite* noise_sites,
                    const CV2Channel* noise_channels,
                    const double* noise_hazards,
                    u32 num_noise_sites,
                    const CV2Mask* pauli_masks,
                    const CV2ReadoutNoise* readout_noise,
                    const u32* detector_offsets,
                    const u32* detector_targets) {
    (void)peak_rank;
    u64 shot_id = shot_offset + (u64)bid();
    if (shot_id >= shots) return;
    // LDS globals decay to flat pointers; execute_shot addresses them generically.
    execute_shot((CV2Complex*)lds_v, (CV2Complex*)lds_red_scratch, V2_MAX_AMP,
                 shot_id, instrs, num_instrs, total_meas_slots, num_observables,
                 seed, block_counts, fused_u2, fused_u4,
                 observable_offsets, observable_targets,
                 noise_sites, noise_channels, noise_hazards, num_noise_sites,
                 pauli_masks, readout_noise, detector_offsets, detector_targets);
}

// =============================================================================
// GLOBAL tier kernel (rank 11-19). Amplitudes in HBM (one slice per workgroup);
// XCD-aware work-stealing so a fixed grid drains all shots. Interpreter body is
// identical (execute_shot). global_v/global_scratch are sized peak-rank amps.
// =============================================================================
#define V2_NUM_XCDS 8u
__attribute__((amdgpu_kernel, visibility("default")))
void clifft_v2_global(const CV2Instr* instrs, u32 num_instrs, u32 peak_rank,
                      u32 total_meas_slots, u32 num_observables,
                      u64 seed, u64 shot_offset, u64 shots,
                      CV2BlockCounts* block_counts,
                      const CV2FusedU2Entry* fused_u2,
                      const CV2FusedU4Entry* fused_u4,
                      const u32* observable_offsets,
                      const u32* observable_targets,
                      const CV2NoiseSite* noise_sites,
                      const CV2Channel* noise_channels,
                      const double* noise_hazards,
                      u32 num_noise_sites,
                      const CV2Mask* pauli_masks,
                      const CV2ReadoutNoise* readout_noise,
                      const u32* detector_offsets,
                      const u32* detector_targets,
                      CV2Complex* global_v,
                      CV2Complex* global_scratch,
                      u64* work_counter) {
    u32 t = tid();
    u32 slot = bid();
    u64 amp_capacity = 1ull << peak_rank;
    CV2Complex* v = global_v + (u64)slot * amp_capacity;
    CV2Complex* scratch = global_scratch + (u64)slot * (amp_capacity >> 1);

    // Correctness-first work queue: a SINGLE global atomic counter hands out
    // shot ids. Every shot is processed exactly once regardless of XCD topology.
    // (The per-XCD interleaved variant is a P2 perf optimization; it requires
    // all XCDs be populated to cover the shot space and is topology-fragile.)
    (void)V2_NUM_XCDS; (void)lds_xcd;
    for (;;) {
        if (t == 0) {
            lds_shot = __atomic_fetch_add(&work_counter[0], 1UL, __ATOMIC_RELAXED);
        }
        barrier();
        u64 batch_shot = lds_shot;
        if (batch_shot >= shots) return;
        execute_shot(v, scratch, (u32)amp_capacity,
                     shot_offset + batch_shot, instrs, num_instrs,
                     total_meas_slots, num_observables, seed, block_counts,
                     fused_u2, fused_u4, observable_offsets, observable_targets,
                     noise_sites, noise_channels, noise_hazards, num_noise_sites,
                     pauli_masks, readout_noise, detector_offsets, detector_targets);
        barrier();
    }
}
