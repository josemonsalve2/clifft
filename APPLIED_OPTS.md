# Applied Optimizations — Clifft GPU Backend

Annotated list of performance-significant optimizations applied during the
GPU backend development, ordered by impact. Each entry includes the measured
delta, the commit tag, and a brief explanation of why it works.

---

## OPT-1: Warp-Shuffle Coop Reduction (+40% coop tier)

**Tag:** `perf-geak-warp-shuffle-40pct`
**Commit:** `2e92b47` (gpu-compiled-kernel)
**Circuit:** cultivation_d5 (rank=10, 1720 instructions)
**Measured:** 4.0M → 5.6M shots/s on MI300X

The shared-cooperative kernel uses `coop_reduce2()` to compute measurement
probabilities (p0, p1) by summing `|v[i]|^2` across the amplitude array.
The original implementation used a shared-memory tree reduction with 8
`__syncthreads()` barriers per call (log2(256) iterations for blockDim=256).

**Replacement:** Two-phase warp-shuffle:
1. **Intra-wavefront** (no sync): `__shfl_xor` across 64 threads (AMD wavefront)
2. **Inter-wavefront** (1 barrier): 4 wavefront leaders write to 4 LDS slots,
   then thread 0 reduces those 4 values via another `__shfl_xor`

Net: 8 barriers → 1 barrier per reduce. Measurement ops dominate d5 execution
(~40% of instructions), so reducing their sync overhead yields +40%.

**Also applied:** Frame barrier batching (skip `__syncthreads` between consecutive
frame ops in the coop interpreter) and LDS right-sizing (`red0[1024]` → `red0[256]`).

**Found by:** GEAK kernel_workflow (14 agents, budget=3, 1M tokens)

**Before:**
```cpp
__device__ void coop_reduce2(CoopShotState& st, double& out0, double& out1) {
    st.red0[threadIdx.x] = out0;
    st.red1[threadIdx.x] = out1;
    __syncthreads();
    for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            st.red0[threadIdx.x] += st.red0[threadIdx.x + s];
            st.red1[threadIdx.x] += st.red1[threadIdx.x + s];
        }
        __syncthreads();  // 8 barriers for blockDim=256
    }
    out0 = st.red0[0]; out1 = st.red1[0];
}
```

**After:**
```cpp
__device__ void coop_reduce2(CoopShotState& st, double& out0, double& out1) {
    // Phase 1: intra-wavefront (no barriers, no shared memory)
    for (int offset = 32; offset > 0; offset >>= 1) {
        out0 += __shfl_xor(out0, offset);
        out1 += __shfl_xor(out1, offset);
    }
    // Phase 2: inter-wavefront (1 barrier, 4 LDS slots)
    uint32_t warp_id = threadIdx.x / 64;
    uint32_t lane_id = threadIdx.x % 64;
    if (lane_id == 0) { st.red0[warp_id] = out0; st.red1[warp_id] = out1; }
    __syncthreads();  // 1 barrier total
    if (threadIdx.x < 4) {
        out0 = st.red0[threadIdx.x]; out1 = st.red1[threadIdx.x];
        for (int offset = 2; offset > 0; offset >>= 1) {
            out0 += __shfl_xor(out0, offset);
            out1 += __shfl_xor(out1, offset);
        }
    }
}
```

---

## OPT-2: `__noinline__` on Cooperative Sweep Functions (+1.4% all tiers)

**Tag:** `perf-noinline-coop-sweeps`
**Commit:** `e2e20a4`
**Measured:** VGPRs 128 → 108, SGPRs 112 → 96

The `coop_array_u2` and `coop_array_u4` functions perform 4x4 complex matrix
sweeps. When inlined into `execute_shot_coop`, the compiler kept all matrix
temporaries live across the switch dispatch, inflating VGPRs from 72 (baseline
clifft-amd) to 128.

**Fix:** Extract the computation into `__noinline__` helper functions
(`coop_u2_sweep`, `coop_u4_sweep`) that take raw `GpuComplex*` pointers.
The `__syncthreads()` barriers remain in the wrapper functions (HIP rejects
`__noinline__` on functions containing `__syncthreads`).

**Why raw pointers:** Passing `CoopShotState&` to a `__noinline__` function
triggers "unsupported expression in static initializer: addrspacecast" because
HIP can't pass shared-memory pointers through struct initialization in
`__noinline__` function prologues.

**Before:**
```cpp
__device__ void coop_array_u2(CoopShotState& st, const GpuProgram& program,
                               uint32_t axis, uint32_t cp_idx) {
    // ... frame update ...
    if (axis >= *st.active_k) return;
    const GpuComplex* mat = program.fused_u2[cp_idx].matrices[in_state];
    uint32_t axis_bit = 1u << axis;
    uint32_t iters = 1u << (*st.active_k - 1);
    for (uint32_t i = threadIdx.x; i < iters; i += blockDim.x) {
        uint32_t idx0 = scatter_bits_1(i, axis);
        uint32_t idx1 = idx0 | axis_bit;
        GpuComplex a = st.v[idx0], b = st.v[idx1];
        // 4x4 complex matrix-vector multiply inlined here
        st.v[idx0] = cadd(cmul(a, mat[0]), cmul(b, mat[1]));
        st.v[idx1] = cadd(cmul(a, mat[2]), cmul(b, mat[3]));
    }
    __syncthreads();
}
```

**After:**
```cpp
// Computation extracted into __noinline__ helper with raw pointers
__device__ __noinline__ void coop_u2_sweep(
    GpuComplex* v, uint32_t active_k, uint32_t axis, const GpuComplex* mat) {
    uint32_t axis_bit = 1u << axis;
    uint32_t iters = 1u << (active_k - 1);
    for (uint32_t i = threadIdx.x; i < iters; i += blockDim.x) {
        uint32_t idx0 = scatter_bits_1(i, axis);
        uint32_t idx1 = idx0 | axis_bit;
        GpuComplex a = v[idx0], b = v[idx1];
        v[idx0] = cadd(cmul(a, mat[0]), cmul(b, mat[1]));
        v[idx1] = cadd(cmul(a, mat[2]), cmul(b, mat[3]));
    }
    // No __syncthreads here — caller handles it
}

__device__ void coop_array_u2(CoopShotState& st, ...) {
    // ... frame update ...
    coop_u2_sweep(st.v, *st.active_k, axis, mat);
    __syncthreads();  // Barrier stays in wrapper
}
```

---

## OPT-3: Precomputed Scatter-Bits LUT — REVERTED

**Tag:** `svm-opt-4-scatter-lut`
**Commit:** `d2d22a4` (reverted)

**Status:** Reverted due to VGPR regression. The LUT's 512-entry `__shared__`
array plus 4 cache-state variables increased VGPR pressure from 108 to 128,
causing a -3.8% throughput regression on the coop tier. The extra LDS footprint
reduced occupancy enough to outweigh the SALU savings.

**Original idea:** Precompute `scatter_bits_1`/`scatter_bits_2` into a shared LUT
once per axis change, replacing ~15 SALU instructions per loop iteration with
1 LDS load. PMC counters showed SALU-heavy execution (13.6B vs 7.5B VALU), but
the register spill cost dominated.

**Lesson:** LDS-cached index tables must be evaluated against VGPR budget.
On MI300X, crossing the 108-VGPR boundary drops occupancy from 8 to 7 waves/CU.

---

## OPT-4: Compiled Megakernel (+5-18% per-thread tier)

**Tag:** (on gpu-compiled-kernel branch)
**Commit:** `756ec32`

For the per-thread tier (peak_rank ≤ 4), walk the bytecode at host time and emit
a straight-line HIP kernel with all opcode handlers inlined and constants baked in.
Compile via clang++ subprocess (NOT HIPRTC — HIPRTC produces 14% worse code).

**Why it helps:** Eliminates switch dispatch overhead, allows compiler to optimize
the full instruction sequence (LICM, CSE, dead code elimination across op boundaries).
Most impactful on deeper circuits (depth ≥ 3) with T-gates.

**Template specialization:** Only emit the device functions the circuit actually uses.
A pure Clifford circuit generates a kernel with ~20 VGPRs instead of 84.

---

## OPT-5: `__noinline__` on Extended Opcode Handlers (foundational)

**Tag:** `svm-opt-1-noinline-extended`
**Commit:** `406a992`

New opcodes (OP_ARRAY_ROT, OP_ARRAY_U2, OP_ARRAY_U4, OP_EXP_VAL, OP_EXPAND_ROT)
added to the per-thread kernel were marked `__noinline__` to prevent register
inflation from inlining large functions into the switch dispatch body.

---

## What Didn't Work

### Manual SVM micro-optimizations (0% improvement)
- Hot/cold dispatch split, frame-op batching, `__launch_bounds__`, instruction
  prefetching, meas[] reduction — the HIP compiler at -O3 already does all of these.

### MFMA tensor accelerators (45-70x slower)
- Gate matrices (2x2 / 4x4 complex) are too small for MFMA's tile size
- Gather/scatter overhead for butterfly pattern exceeds compute savings
- Complex decomposition requires 4x real MFMAs per complex operation

### HIPRTC JIT compilation (-14% vs AOT)
- HIPRTC has fewer optimization passes than the AOT clang++ compiler
- Always use clang++ subprocess with disk-cached .hsaco files
