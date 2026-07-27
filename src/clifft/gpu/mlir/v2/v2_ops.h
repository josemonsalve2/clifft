// v2_ops.h — MLIR-V2 shared operand library (plain C -> amdgcn, NO HIP).
//
// THE single source of truth for V2 opcode semantics. Both consumers include it:
//   1. coop_interpreter.c — the RUNTIME interpreter: a for(pc)switch that calls
//      v2_op_*() with runtime operands (register/coop/global tiers).
//   2. generated specialized_<hash>.c — the per-circuit SPECIALIZER output: a
//      straight-line sequence of v2_op_*() calls with COMPILE-TIME-CONSTANT
//      operands and active_k. -O2 constant-propagation then folds loop bounds
//      (1<<(active_k-2) -> a literal), matrix indices, and axis math per call.
//      Because both call the identical inline bodies, the specialized kernel is
//      byte-exact with the interpreter (and thus with GPU-SVM) BY CONSTRUCTION.
//
// R5: we specialize the operand BODY (constants, rank) and straight-line the
// SEQUENCE; we do NOT unroll the 2^k amplitude sweeps (that was V1's disease).
//
// Every op takes `active_k` (the value BEFORE the op) by value so the loop
// bounds fold when it is a constant. Ops that grow/shrink k also write st->
// active_k so the interpreter's next iteration sees the update.
#ifndef CLIFFT_GPU_MLIR_V2_OPS_H
#define CLIFFT_GPU_MLIR_V2_OPS_H

#include "clifft/gpu/mlir/v2/device_abi.h"

typedef unsigned int   u32;
typedef unsigned long  u64;
typedef unsigned char  u8;
typedef unsigned short u16;

// ----- opcodes (MUST match src/clifft/backend/backend.h enum order EXACTLY) ---
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

// ----- sizing (shared by interpreter + specialized kernels) ------------------
#define V2_MAX_AMP     1024
#define V2_MAX_MEAS    4096
#define V2_MEAS_WORDS  (V2_MAX_MEAS / 64)
#define V2_SCRATCH_AMP 512
#define V2_RED_WARPS   8
#define V2_INV_SQRT2   0.70710678118654752440

// Relative epsilon for detecting floating-point dust in measurement
// probabilities. This is NOT the SVM's kDustEpsilon and must not be synced to
// it: the two backends store amplitudes at different precisions, and the
// threshold has to sit above the dust floor of whichever one is in use.
//
//   SVM: std::complex<double> -> analytically-zero interference lands at
//        1e-30..1e-24, and kDustEpsilon = 1e-18 clears it by six decades.
//   V2:  CV2Complex = {float re; float im;} -> the same interference bottoms
//        out on fp32_eps^2 = 1.4e-14 (see sizing below). 1e-18 is four decades
//        BELOW that, so it can never fire.
//
// Why a threshold that never fires is a correctness bug, not just dead code:
// sample_branch() returns WITHOUT drawing when a branch is dust. If the SVM
// clamps where V2 does not, V2 consumes a PRNG draw the SVM never did. Both
// still pick the same outcome (p0/total is 1-1e-15, so any uniform draw lands
// the same way), but every subsequent draw in that shot is shifted by one and
// the two streams never resynchronize. In a surface-code circuit nearly every
// stabilizer measurement is deterministic, so this fires constantly -- it is
// what made the d5 fixtures disagree with the CPU reference.
//
// Sizing. A branch probability is a sum of `half = 1 << (active_k - 1)` squared
// fp32 magnitudes. The rounding error is RELATIVE to each amplitude, so the
// dust floor does not grow with the term count -- summing more terms averages
// the residuals rather than accumulating them. Measured by
// V2_performance/tools/dust_floor.py (p1/total over 2000 trials per rank):
//
//     rank  1 -> 9.7e-15 median, 1.2e-13 max      rank 12 -> 1.42e-14, 1.6e-14
//     rank  4 -> 1.3e-14 median, 5.8e-14 max      rank 26 -> 1.42e-14, 1.5e-14
//
// It concentrates on fp32_eps^2 = 1.42e-14 and the spread TIGHTENS with rank;
// the low-rank tail is the widest at ~1.2e-13. So one constant covers the whole
// 1..26 range.
//
// Those rows come from the tool's default --model residual, a STATISTICAL model
// that assumes every term rounds at full eps and that the errors never cancel:
// resid_i = a_i * eps * z_i, so p1/total = eps^2 * a weighted mean of Exp(1).
// It is an upper envelope. Directly simulating the butterfly the kernel
// actually performs (--model butterfly) puts the true floor ~36x lower, near
// 3e-16 with a ~5e-15 low-rank tail. Both agree on everything this constant
// rests on -- rank-independence, the tail being widest at LOW rank, and
// insensitivity to circuit depth (<5% across 256 gates) -- and differ only in
// absolute location, so 1e-11 is sized against the conservative one and clears
// the simulated one by ~4 decades.
//
// So 1e-11 sits ~2 decades above that tail and ~5 decades below the smallest
// probability fp32 can carry meaningfully, leaving a wide margin on both sides.
// Genuine small probabilities (the SVM comment cites R_ZZ angles producing
// ~1e-16) are not representable in fp32 storage in the first place, so nothing
// real is lost to the clamp.
#define V2_DUST_EPS    1e-11

// ----- per-shot classical state ----------------------------------------------
typedef struct {
    u64 px[CLIFFT_V2_PAULI_WORDS];
    u64 pz[CLIFFT_V2_PAULI_WORDS];
    u64 meas[V2_MEAS_WORDS];
    u64 rng[4];
    u8  obs[CLIFFT_V2_MAX_OBS];
    u32 active_k;
    u32 next_noise;
    u8  discarded;
    u8  branch;
} V2State;

// ----- cooperation primitives (tier-parameterized) ---------------------------
// coop reduction scratch lives in LDS (coop/global only).
#ifndef V2_REGISTER
extern __attribute__((address_space(3))) double lds_red0[V2_RED_WARPS];
extern __attribute__((address_space(3))) double lds_red1[V2_RED_WARPS];
#endif

#ifdef V2_REGISTER
#  define V2_STRIDE 1u
static inline u32 v2_tid(void)  { return 0u; }
static inline void v2_barrier(void) {}
#  define V2_REDUCE2(T, L0, L1, O0, O1) do { *(O0) = (L0); *(O1) = (L1); } while (0)
#  define IS_OWNER 1
#else
#  define V2_STRIDE 256u
static inline u32 v2_tid(void)  { return __builtin_amdgcn_workitem_id_x(); }
// s_barrier alone is an EXECUTION barrier only: LLVM models the intrinsic as
// IntrNoMem, so it neither emits `s_waitcnt lgkmcnt(0)` nor stops the scheduler
// from sinking/hoisting LDS accesses across it. A wave can therefore retire the
// barrier with a ds_write still in flight while a peer wave reads the stale
// word. The release/acquire fence pair is what makes it a MEMORY barrier: the
// release forces the waitcnt before s_barrier, the acquire keeps later loads
// from being hoisted above it. HIP's __syncthreads() expands to exactly this,
// which is why the SVM backend never saw the race.
static inline void v2_barrier(void) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}
#  define V2_REDUCE2(T, L0, L1, O0, O1) coop_reduce2((T), (L0), (L1), (O0), (O1))
#  define IS_OWNER (t == 0)
#endif
static inline u32 v2_bid(void)  { return __builtin_amdgcn_workgroup_id_x(); }

// ----- RNG (xoshiro256++ seeded by splitmix64), byte-exact with SVM ----------
static inline u64 v2_rotl64(u64 x, int k) { return (x << k) | (x >> (64 - k)); }
static inline u64 v2_splitmix64(u64* state) {
    u64 z = (*state += 0x9e3779b97f4a7c15UL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9UL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebUL;
    return z ^ (z >> 31);
}
static inline void rng_seed(u64* s, u64 seed, u64 shot_id) {
    u64 z = seed ^ (0x9e3779b97f4a7c15UL * (shot_id + 1));
    s[0] = v2_splitmix64(&z); s[1] = v2_splitmix64(&z);
    s[2] = v2_splitmix64(&z); s[3] = v2_splitmix64(&z);
}
static inline u64 rng_next(u64* s) {
    u64 result = v2_rotl64(s[0] + s[3], 23) + s[0];
    u64 t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t;    s[3] = v2_rotl64(s[3], 45);
    return result;
}
static inline double rng_uniform(u64* s) {
    return (double)(rng_next(s) >> 11) * 0x1.0p-53;
}

// ----- Pauli-frame + meas bit helpers (generic pointer) ----------------------
static inline int fget(u64* w, u32 i) { return (int)((w[i >> 6] >> (i & 63u)) & 1UL); }
static inline void fset(u64* w, u32 i, int v) {
    u64 m = 1UL << (i & 63u);
    if (v) w[i >> 6] |= m; else w[i >> 6] &= ~m;
}
static inline void fxor(u64* w, u32 i, int v) { if (v) w[i >> 6] ^= 1UL << (i & 63u); }
static inline void fswap(u64* w, u32 a, u32 b) {
    int va = fget(w, a), vb = fget(w, b); fset(w, a, vb); fset(w, b, va);
}
static inline u8 mget(u64* meas, u32 i) { return (u8)((meas[i >> 6] >> (i & 63u)) & 1UL); }
static inline void mset(u64* meas, u32 i, u8 v) {
    u64 m = 1UL << (i & 63u);
    if (v & 1u) meas[i >> 6] |= m; else meas[i >> 6] &= ~m;
}
static inline void mxor1(u64* meas, u32 i) { meas[i >> 6] ^= 1UL << (i & 63u); }

// ----- complex + index helpers -----------------------------------------------
static inline CV2Complex cmul(CV2Complex a, CV2Complex b) {
    CV2Complex r; r.re = a.re * b.re - a.im * b.im; r.im = a.re * b.im + a.im * b.re; return r;
}
static inline CV2Complex cadd(CV2Complex a, CV2Complex b) {
    CV2Complex r; r.re = a.re + b.re; r.im = a.im + b.im; return r;
}
static inline CV2Complex csub(CV2Complex a, CV2Complex b) {
    CV2Complex r; r.re = a.re - b.re; r.im = a.im - b.im; return r;
}
// f64 scalar multiply then narrow — byte-exact with SVM cscale. Do NOT relax.
static inline CV2Complex cscale(CV2Complex a, double s) {
    CV2Complex r; r.re = (float)((double)a.re * s); r.im = (float)((double)a.im * s); return r;
}
static inline double cnorm(CV2Complex v) {
    double re = (double)v.re, im = (double)v.im; return re * re + im * im;
}
static inline u64 insert_zero_bit(u64 val, u32 pos) {
    u64 lo = val & ((1ull << pos) - 1ull);
    u64 hi = (val & ~((1ull << pos) - 1ull)) << 1;
    return lo | hi;
}
static inline u64 scatter_bits_1(u64 val, u32 pos) { return insert_zero_bit(val, pos); }
static inline u64 scatter_bits_2(u64 val, u32 b1, u32 b2) {
    u32 lo = b1 < b2 ? b1 : b2, hi = b1 < b2 ? b2 : b1;
    val = insert_zero_bit(val, lo); return insert_zero_bit(val, hi);
}

// ----- cooperative reduction (coop/global only; register uses identity) ------
#ifndef V2_REGISTER
static inline double shfl_xor_f64(double v, u32 lane, int offset) {
    u64 bits; __builtin_memcpy(&bits, &v, 8);
    u32 lo = (u32)bits, hi = (u32)(bits >> 32);
    u32 addr = ((lane ^ (u32)offset) & 63u) << 2;
    u32 rlo = __builtin_amdgcn_ds_bpermute((int)addr, (int)lo);
    u32 rhi = __builtin_amdgcn_ds_bpermute((int)addr, (int)hi);
    u64 rbits = ((u64)rhi << 32) | (u64)rlo;
    double r; __builtin_memcpy(&r, &rbits, 8); return r;
}
// MUST reproduce SVM coop_reduce2's exact summation order or f64 rounding
// diverges at measurement branch points.
static inline void coop_reduce2(u32 t, double l0, double l1, double* out0, double* out1) {
    u32 lane = t & 63u, warp = t >> 6;
    for (int off = 32; off > 0; off >>= 1) {
        l0 += shfl_xor_f64(l0, lane, off);
        l1 += shfl_xor_f64(l1, lane, off);
    }
    // v2_barrier(), not a bare s_barrier: every one of these three guards a
    // read-after-write on lds_red0/lds_red1, so the release fence (== the
    // s_waitcnt lgkmcnt(0) before s_barrier) is load-bearing. Without it a wave
    // retires the barrier with its partial-sum ds_write still in flight and a
    // peer reads the previous op's value -- which perturbs the reduction total
    // and flips sample_branch.
    if (lane == 0u) { lds_red0[warp] = l0; lds_red1[warp] = l1; }
    v2_barrier();
    if (t < 4u) {
        l0 = lds_red0[t]; l1 = lds_red1[t];
        for (int off = 2; off > 0; off >>= 1) {
            l0 += shfl_xor_f64(l0, lane, off);
            l1 += shfl_xor_f64(l1, lane, off);
        }
    }
    if (t == 0u) { lds_red0[0] = l0; lds_red1[0] = l1; }
    v2_barrier();
    *out0 = lds_red0[0]; *out1 = lds_red1[0];
    v2_barrier();
}
#endif

// The clamp decision must match the SVM's on every call: a branch clamped on
// one side and rolled on the other consumes a PRNG draw the other never did,
// and the two streams never resynchronize. See V2_DUST_EPS.
static inline u8 sample_branch(u64* rng, double p0, double p1, double total) {
    double eps = V2_DUST_EPS * total;
    if (p1 <= eps) return 0;
    if (p0 <= eps) return 1;
    return (rng_uniform(rng) * total < p0) ? 0u : 1u;
}

extern double __ocml_log_f64(double);
static inline double ocml_log_f64(double x) { return __ocml_log_f64(x); }

// V2_NOISE_ATTR: always_inline for the interpreter/register build; the
// specializer defines it to noinline (before including this header) so each
// emitted noise OP is a fenced call the -O2 scheduler cannot reassociate FP
// across — reproducing the interpreter's loop-body-per-noise-op semantics. The
// FP-carrying helpers below stay plain static-inline (fencing at the op level is
// what matters; fencing the helpers too was empirically worse).
#ifndef V2_NOISE_ATTR
#define V2_NOISE_ATTR __attribute__((always_inline))
#endif
static inline void draw_next_noise(V2State* st, const double* hazards, u32 num_sites) {
    if (num_sites == 0u || st->next_noise >= num_sites) { st->next_noise = 0xffffffffu; return; }
    double current_hazard = (st->next_noise == 0u) ? 0.0 : hazards[st->next_noise - 1u];
    double target = current_hazard + (-ocml_log_f64(1.0 - rng_uniform(st->rng)));
    u32 lo = 0u, hi = num_sites;
    while (lo < hi) { u32 mid = lo + ((hi - lo) >> 1); if (hazards[mid] <= target) lo = mid + 1u; else hi = mid; }
    st->next_noise = (lo >= num_sites) ? 0xffffffffu : lo;
}
static inline void apply_noise_site(V2State* st, const CV2NoiseSite* sites,
                                    const CV2Channel* channels, u32 site_idx) {
    CV2NoiseSite site = sites[site_idx];
    double roll = rng_uniform(st->rng) * site.prob_sum;
    double cumulative = 0.0;
    for (u32 k = 0; k < site.count; ++k) {
        const CV2Channel* ch = &channels[site.offset + k];
        cumulative += ch->prob;
        if (roll < cumulative) {
            for (u32 w = 0; w < CLIFFT_V2_PAULI_WORDS; ++w) { st->px[w] ^= ch->x[w]; st->pz[w] ^= ch->z[w]; }
            break;
        }
    }
}
// Cooperative diagonal phase on active axis; `active_k` is the pre-op rank.
static inline void coop_apply_phase(u32 t, CV2Complex* v, u32 active_k, u32 axis, CV2Complex phase) {
    u64 axis_bit = 1ull << axis;
    u64 iters = 1ull << (active_k - 1u);
    for (u64 i = t; i < iters; i += V2_STRIDE) {
        u64 idx = insert_zero_bit(i, axis) | axis_bit;
        v[idx] = cmul(v[idx], phase);
    }
    v2_barrier();
}

#include "clifft/gpu/mlir/v2/v2_ops_body.inc"

#endif  // CLIFFT_GPU_MLIR_V2_OPS_H
