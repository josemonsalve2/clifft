# Lessons Learned — Clifft GPU Backend Optimization

Comprehensive record of what worked, what didn't, and why. Ordered by
discovery date. Each entry includes the measured impact and root cause analysis.

---

## Successes

### 1. GEAK Warp-Shuffle Reduction: +40% coop tier
**Tag:** `perf-geak-warp-shuffle-40pct`
- Replaced 8-barrier shared-memory tree reduction with 2-phase `__shfl_xor`
- Root cause: measurement operations dominate d5 execution (~40% of instructions)
  and each measurement calls `coop_reduce2()` twice (p0, p1 computation)
- Net: 8 `__syncthreads` → 1 per reduce call
- **Lesson:** Synchronization overhead is the #1 bottleneck in the coop tier,
  not compute or memory bandwidth. Always minimize barriers.

### 2. `__noinline__` on Coop Sweeps: VGPRs 128→108
**Tag:** `perf-noinline-coop-sweeps`
- Extracted coop_u2_sweep and coop_u4_sweep into `__noinline__` functions
  taking raw `GpuComplex*` pointers (not `CoopShotState&`)
- Root cause: HIP compiler inlined the 4x4 matrix operations into the main
  switch dispatch, keeping all matrix temporaries live across the switch
- **Lesson:** `__noinline__` is a surgical tool for controlling register pressure.
  Must use raw pointers because HIP can't pass shared-memory references through
  `__noinline__` function prologues (addrspacecast error).

### 3. Compiled Megakernel: +5-18% per-thread tier
- Straight-line HIP source with constants baked in, compiled via clang++ subprocess
- **Lesson:** clang++ subprocess >> HIPRTC JIT (-14% with HIPRTC). Always AOT-compile
  when possible. The switch dispatch overhead IS measurable (+5-18%) but only for
  the per-thread tier — coop tier is dominated by array sweeps and sync.

### 4. SVM+GEAK vs clifft-amd: +39% overall
**Tag:** `perf-svm-vs-clifftamd-39pct`
- Combined: `__noinline__` sweeps + warp-shuffle + frame barrier batching +
  new opcode support (ROT, U2, U4, EXP_VAL, EXPAND_ROT)
- **Lesson:** The original clifft-amd SVM was a clean implementation but left
  significant performance on the table in the reduction path.

---

## Failures (Optimizations That Didn't Work)

### F1. Precomputed Scatter-Bits LUT: -3.8% regression
**Tag:** `svm-opt-4-scatter-lut` (reverted)
- Precomputed `scatter_bits_1(i, axis)` into `__shared__ uint32_t lut[512]`
- **Expected:** Reduce SALU from 13.6B by replacing ~15 SALU instructions per
  loop iteration with 1 LDS load
- **Actual:** VGPRs increased from 108→128 (the LUT pointer, cache variables,
  and address computation added live registers). Throughput dropped 3.8%.
- **Root cause:** On MI300X, crossing the ~108 VGPR boundary doesn't change
  occupancy (both 108 and 128 give 4 waves/SIMD for 512-VGPR budget), but the
  increased register pressure caused more instruction scheduling stalls.
- **Lesson:** LDS-cached index tables must be evaluated against VGPR budget.
  The "SALU is too high" diagnosis was correct but the cure was worse than the
  disease. The compiler's inline scatter_bits is register-efficient.

### F2. Manual SVM Micro-Optimizations: 0% improvement
- Approach F: 7 optimizations (hot/cold split, frame batching, `__launch_bounds__`,
  instruction prefetch, meas[] reduction, compact px/pz)
- **Actual:** Zero measurable improvement on any circuit
- **Root cause:** The HIP compiler at -O3 already applies all these optimizations
  automatically. Frame batching, instruction scheduling, dead code elimination —
  the compiler does it better.
- **Lesson:** Don't compete with the compiler on micro-optimizations. Focus on
  data structure and algorithmic changes that the compiler CAN'T do.

### F3. MFMA Tensor Accelerators: 45-70x SLOWER (not applied)
- Attempted to map 2x2/4x4 gate butterflies to MFMA instructions
- **Expected:** Matrix cores should accelerate matrix-vector products
- **Actual:** Estimated 45-70x slower due to:
  1. Tiny K dimension (K=2 for 1-qubit gates) vs MFMA minimum K=4
  2. Gather/scatter overhead for butterfly stride pattern (~80 cycles)
  3. Complex arithmetic decomposition (4 real MFMAs per complex MMA)
- **Root cause:** MFMA is designed for large GEMM tiles (M≥16, N≥16, K≥4).
  Quantum gate operations are inherently small (2x2 or 4x4 complex).
- **Lesson:** Tensor/matrix cores are not applicable to quantum gate operations.
  The workload is element-wise/butterfly, not GEMM-shaped.

### F4. HIPRTC JIT Compilation: -14% vs AOT
- HIPRTC produces inferior code compared to full clang++ compiler
- **Root cause:** HIPRTC has fewer optimization passes, no whole-program analysis,
  and limited register allocation tuning
- **Lesson:** Always use clang++ subprocess (or disk-cached .hsaco) over HIPRTC
  for performance-critical kernels.

### F5. HipGraph Pipeline: -1% per-thread, +1% coop (noise)
- `hipGraph_t` with kernel nodes and dependency edges
- **Expected:** Reduced launch overhead from batching all ops in one graph
- **Actual:** Near-zero improvement; the graph dispatch overhead exactly cancels
  any launch overhead savings
- **Root cause:** Our bottleneck is execution time (register pressure, array
  sweeps), not launch overhead. HipGraph solves the wrong problem.

### F6. Persistent Kernel Phase-Sorted Dispatch: -3% coop, 0% thread
- Replaced 35-case switch with 6-case phase dispatch
- **Expected:** Fewer branch mispredictions
- **Actual:** The phase list traversal adds overhead that offsets the simpler switch
- **Root cause:** GPU branch predictor handles regular circuits well. QEC circuits
  have repetitive opcode patterns that the predictor learns.

### F7. Hybrid Split/Persistent Kernel: +0.8% on d5 (noise)
- k-aware circuit splitting with persistent work-stealing dispatch
- **Expected:** 3-5x on circuits with sawtooth k-profile (75-85% at k=0)
- **Actual:** +0.8% on cultivation_d5 which sustains k=10 throughout
- **Root cause:** cultivation_d5 does NOT have the sawtooth pattern we expected.
  It's all-coop all the time. The hybrid benefit requires circuits with genuinely
  mixed high/low k segments.
- **Lesson:** The k-profile hypothesis was correct but the test circuit (d5) is
  the wrong test case. Need circuits with actual mixed k-profiles.

---

## Key Architectural Insights

### I1. Peak_rank determines everything
rank=0 runs at 48M shots/s, rank=10 at 5.6M shots/s, rank=19 at 145K shots/s.
This is a **330x range** entirely from array size. No dispatch optimization can
overcome this fundamental scaling.

### I2. The compiler is really good
ROCm 7.2.3 clang++ at -O3 produces near-optimal code for the SVM interpreter.
Manual micro-optimizations that replicate compiler behavior yield 0%. Only
algorithmic/structural changes (warp-shuffle, `__noinline__`) help.

### I3. SALU dominance is structural, not from the interpreter
The 1.81x SALU/VALU ratio comes from address computation (`scatter_bits`,
`insert_zero_bit`, `bit_get/set/xor`), not from the switch dispatch. Eliminating
the switch (compiled kernel) reduces SALU by only ~11% while the structural
address computation remains.

### I4. Synchronization is the #1 coop bottleneck
The warp-shuffle optimization (+40%) was the biggest single win because it
reduced `__syncthreads` barriers from 8 to 1 per measurement. On MI300X with
304 CUs, barrier overhead compounds catastrophically.

### I5. AMD GPU families are compatible for our optimizations
`__shfl_xor` works identically on CDNA3 (gfx942, MI300X/MI325X) and CDNA4
(gfx950, MI350X). Wavefront width is 64 on all AMD CDNA architectures.
RDNA (gfx1100) would need 32-wide handling but is not a target.

### I6. Clifft's compiler minimizes peak_rank aggressively
`StatevectorSqueezePass` reduces synthetic circuits to rank ≤ 1 regardless of
T-gate count. Only real QEC circuits with genuinely entangled T-gate blocks
achieve rank > 4. This means the per-thread tier handles nearly all workloads.

---

## Re-evaluation: Scatter LUT at Rank=19

**Date:** 2026-07-19
**Context:** The scatter-bits LUT (F1, OPT-3) was reverted at rank=10 due to a
VGPR regression (108->128, -3.8%). At rank=19 (d7), VGPRs are already at 128
and occupancy is 4 waves/SIMD regardless, so the VGPR concern no longer applies.
The workload is VALU-dominated (SALU/VALU = 0.82x vs 1.81x at rank=10). Does the
LUT help here?

### Answer: No --- the LUT physically cannot fit in LDS at rank=19.

At rank=19, `scatter_bits_1` iterates over 2^18 = 262,144 pairs. The LUT would
need 262,144 x 4 bytes = **1,024 KB**. MI300X provides **64 KB of LDS per
workgroup**. After existing shared-memory allocations (reduction buffers, Pauli
frames, meas/obs arrays = ~5 KB), only ~59 KB remains. The LUT exceeds the
available LDS by **16x**.

`scatter_bits_2` is worse: 2^17 entries = 512 KB.

The maximum rank where a scatter_bits_1 LUT fits is **rank=14** (2^13 = 8,192
entries = 32 KB). Above rank=14, the LUT is infeasible in LDS.

### Even if it could fit, the benefit is marginal

The user's back-of-envelope calculation is correct in approach but the conclusion
shifts with the 500K-shot benchmark data from RESULTS.md:

- SALU at rank=19: 19.2B instructions (500K shots)
- Hypothetical 30% SALU reduction: 5.8B fewer instructions
- Time saved: 5.8B / (304 CUs x 2.1 GHz) = 9.1 ms
- Total duration (500K shots at 143,916 shots/s): 3,474 ms
- **Saving: 0.26%** (below measurement noise)

At rank=10 (where the LUT fits), it saved ~30% SALU but the VGPR cost outweighed
the benefit. At rank=19 (where VGPRs are neutral), the LUT cannot fit. This is an
inherent mismatch: the ranks where the LUT is physically feasible (rank <= 14) are
the same ranks where the VGPR cost matters most.

### What the SALU/VALU inversion actually means

At rank=10, SALU/VALU = 1.81x because the array is small (1,024 elements) and
each sweep finishes in a few iterations, making address computation dominate. At
rank=19, SALU/VALU = 0.82x because each sweep processes 262K elements, and the
VALU cost of complex multiply-add over 262K elements dwarfs the SALU cost of
`scatter_bits` index computation. The SALU is **not** the bottleneck at rank=19;
HBM bandwidth is (confirmed by all approaches converging to ~143K shots/s).

### Recommendation

**Do not pursue.** The scatter-bits LUT is fundamentally incompatible with
rank >= 15 due to LDS capacity, and at rank <= 14 the VGPR cost makes it a net
negative. The SALU address computation at rank=19 accounts for ~30% of cycles
(per MI300X_NUMA.md section 6.3) but this is better addressed by the compiled
megakernel (which bakes axis constants, enabling the compiler to strength-reduce
`insert_zero_bit` to a constant shift-and-mask) than by an LDS lookup table.

### F8. LDS-Pipelined Tiling for Global-Coop Sweeps: -66% coop, -17% global-coop

**Commit:** b70b90d (reverted in eb77d0f)
**Expected:** Replace random HBM accesses with coalesced tile loads through LDS
**Actual:**
- D7 (rank=19): 143.9K → 120.1K shots/s (**-16.5%**)
- D5 (rank=10): 5.59M → 1.90M shots/s (**-66.1%**)
- QEC (rank=0): 48.3M → 45.1M shots/s (-6.6%)

**Root causes:**
1. **Cooperative load/store overhead**: Each tile requires 3 `__syncthreads` (before load, before compute, before store). For the shared-coop kernel where the amplitude array ALREADY lives in LDS, adding tile boundaries introduces entirely new barriers where none existed before.
2. **Shared-coop double-penalty**: The existing coop kernel (`sample_kernel_coop`) passed `nullptr` for tile pointers and used the direct-LDS path. But the CoopShotState struct extension added overhead even on the null path.
3. **False premise for global-coop**: The butterfly access pattern (pairs at `v[i]` and `v[i | axis_bit]`) is not inherently cache-hostile — the access stride is `axis_bit * 8` bytes. For large axes (axis > tile_bits), the two tiles load exactly the needed data; but the barrier synchronization (3 per sweep vs 0 before) dominates.
4. **Tile boundary irregularity**: For "mixed" cases (one axis qubit inside tile, one outside), fallback to direct HBM removes the tiling benefit while the overhead remains.

**Lesson:** LDS tiling only helps when:
1. The untiled access pattern causes L2/cache thrashing (random access)
2. The tiling barrier cost << memory latency saved
3. The existing code doesn't already use LDS (the coop kernel already does!)

For quantum amplitude butterflies, the access is strided (predictable), not random, so the L2 hardware prefetcher handles it reasonably. Adding explicit tiling barriers destroys the benefit.
