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
                    const u32* observable_targets) {
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
        rng_seed(lds_rng, seed, shot_id);
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
