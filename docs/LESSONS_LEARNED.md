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
