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
#define V2_MAX_AMP  1024
#define V2_MAX_MEAS 4096
extern __attribute__((address_space(3))) CV2Complex lds_v[V2_MAX_AMP];
extern __attribute__((address_space(3))) u8   lds_meas[V2_MAX_MEAS];
extern __attribute__((address_space(3))) u8   lds_obs[CLIFFT_V2_MAX_OBS];
extern __attribute__((address_space(3))) u64  lds_px[CLIFFT_V2_PAULI_WORDS];
extern __attribute__((address_space(3))) u64  lds_pz[CLIFFT_V2_PAULI_WORDS];
extern __attribute__((address_space(3))) u32  lds_active_k;
extern __attribute__((address_space(3))) u8   lds_discarded;
extern __attribute__((address_space(3))) u64  lds_rng[4];   // tid0-owned RNG state
extern __attribute__((address_space(3))) u8   lds_branch;   // sampled branch broadcast
extern __attribute__((address_space(3))) u32  lds_next_noise;// next scheduled noise site
extern __attribute__((address_space(3))) double lds_red0[256];
extern __attribute__((address_space(3))) double lds_red1[256];

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
static inline CV2Complex cscale(CV2Complex a, float s) {
    CV2Complex r; r.re = a.re * s; r.im = a.im * s; return r;
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

// Cooperative diagonal phase on an active axis: v[idx | axis_bit] *= phase,
// strided over the 2^(active_k-1) lower half. Matches SVM coop_apply_phase.
static inline void coop_apply_phase(u32 t, u32 axis, CV2Complex phase) {
    u64 axis_bit = 1ull << axis;
    u64 iters = 1ull << (lds_active_k - 1u);
    for (u64 i = t; i < iters; i += 256u) {
        u64 idx = insert_zero_bit(i, axis) | axis_bit;
        lds_v[idx] = cmul(lds_v[idx], phase);
    }
    barrier();
}

// Cooperative reduction of two per-thread f64 partials across 256 threads.
// LDS tree reduce (correctness-first; ds_bpermute optimization later). All 256
// threads call it; result broadcast to all via lds_red[0].
static inline void coop_reduce2(u32 t, double l0, double l1, double* out0, double* out1) {
    lds_red0[t] = l0; lds_red1[t] = l1;
    barrier();
    for (u32 stride = 128u; stride > 0u; stride >>= 1) {
        if (t < stride) {
            lds_red0[t] += lds_red0[t + stride];
            lds_red1[t] += lds_red1[t + stride];
        }
        barrier();
    }
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
// The coop interpreter kernel. One workgroup = one shot.
//   args: packed CV2KernArgs (see device_abi.h).
// For P1 milestone 1 this handles: init, RNG seed, FRAME ops, OBSERVABLE, and
// result aggregation. Amplitude/measurement ops are added incrementally; any
// unhandled opcode sets discarded=1 (loud failure, never silent-wrong).
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
                    u32 num_noise_sites) {
    (void)peak_rank; (void)fused_u2; (void)fused_u4;
    u64 shot_id = shot_offset + (u64)bid();
    if (shot_id >= shots) return;

    u32 t = tid();

    // --- cooperative init ---
    for (u32 i = t; i < V2_MAX_AMP; i += 256u) { lds_v[i].re = 0.0f; lds_v[i].im = 0.0f; }
    barrier();
    if (t == 0) {
        for (u32 w = 0; w < CLIFFT_V2_PAULI_WORDS; ++w) { lds_px[w] = 0; lds_pz[w] = 0; }
        lds_active_k = 0;
        lds_discarded = 0;
        lds_v[0].re = 1.0f; lds_v[0].im = 0.0f;
        for (u32 i = 0; i < num_observables; ++i) lds_obs[i] = 0;
        for (u32 i = 0; i < total_meas_slots && i < V2_MAX_MEAS; ++i) lds_meas[i] = 0;
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
                if (ins.a < V2_MAX_MEAS) lds_meas[ins.a] = mval;
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
                    lds_meas[ins.a] = m_abs ^ (u8)((ins.flags & FLAG_SIGN) != 0);
            }
            barrier();
            break;
        case OP_EXPAND: {
            // Virtual H on a dormant axis: duplicate v[0,half) -> v[half,2half).
            u32 half = 1u << lds_active_k;
            for (u32 i = t; i < half; i += 256u) lds_v[i + half] = lds_v[i];
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
            for (u32 i = t; i < half; i += 256u) lds_v[i + half] = cmul(lds_v[i], phase);
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
                l0 += cnorm(lds_v[i]);
                l1 += cnorm(lds_v[i + half]);
            }
            double p0, p1;
            coop_reduce2(t, l0, l1, &p0, &p1);
            if (t == 0) {
                u8 b = sample_branch(p0, p1, p0 + p1);
                lds_branch = b;
                u8 m_abs = b ^ (u8)px;
                if (ins.a < V2_MAX_MEAS)
                    lds_meas[ins.a] = m_abs ^ (u8)((ins.flags & FLAG_SIGN) != 0);
            }
            barrier();
            if (lds_branch != 0) {
                for (u32 i = t; i < half; i += 256u) lds_v[i] = lds_v[i + half];
            }
            barrier();
            if (t == 0) {
                lds_active_k -= 1;
                u8 m_abs = lds_meas[ins.a] ^ (u8)((ins.flags & FLAG_SIGN) != 0);
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
                CV2Complex vi = lds_v[i], vh = lds_v[i + half];
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
                    lds_meas[ins.a] = m_abs ^ (u8)((ins.flags & FLAG_SIGN) != 0);
            }
            barrier();
            for (u32 i = t; i < half; i += 256u) {
                CV2Complex vi = lds_v[i], vh = lds_v[i + half];
                CV2Complex folded = (lds_branch == 0) ? cadd(vi, vh) : csub(vi, vh);
                lds_v[i] = cscale(folded, (float)V2_INV_SQRT2);
            }
            barrier();
            if (t == 0) {
                lds_active_k -= 1;
                u8 m_abs = lds_meas[ins.a] ^ (u8)((ins.flags & FLAG_SIGN) != 0);
                fset(lds_px, ins.axis_1, m_abs != 0);
                fset(lds_pz, ins.axis_1, 0);
            }
            barrier();
            break;
        }
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
            coop_apply_phase(t, ins.axis_1, phase);
            break;
        }
        case OP_OBSERVABLE:
            if (t == 0) {
                u32 s0 = observable_offsets[ins.a];
                u32 e0 = observable_offsets[ins.a + 1];
                u8 parity = 0;
                for (u32 k = s0; k < e0; ++k) parity ^= lds_meas[observable_targets[k]];
                if (ins.b < CLIFFT_V2_MAX_OBS) lds_obs[ins.b] ^= parity;
            }
            barrier();
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
