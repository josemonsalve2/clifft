# V2 Requirements Addendum (owner directives, 2026-07-24)

These are hard requirements/questions from the project owner that the final
V2.md must resolve. They refine pre_V2.md.

## R0. SVM is already a runtime — a new runtime must justify itself
The GPU SVM path (`src/clifft/gpu/sampler/hip_sampler.hip:729`) is ALREADY a
runtime bytecode interpreter: one kernel, `for(pc) switch(op)`, fast, correct.
Therefore:
- A V2 "runtime interpreter" (pre_V2 Option A) risks simply rebuilding SVM. It
  must differentiate ENOUGH to justify existing — e.g. by being the *single
  source* SVM+Hybrid both draw from, or by measurably beating the current SVM
  interpreter. If it can't, the right answer may be: keep SVM (runtime) +
  Hybrid (compiled), delete MLIR, unify their operand math.
- Frame the decision as: what does each PATH earn? SVM = the runtime/reference;
  Hybrid = compiled-per-circuit-loop (already specialized, not unrolled). The
  open slot MLIR tried to fill (device-agnostic + faster than both) was never
  delivered. V2.md must state plainly whether that slot should exist at all.

## R1. Tier reuse — do we still need 3 tiers, or a unified mode?
The 3 tiers (register/coop/global) differ ONLY in (a) where the statevector
lives (alloca / LDS addrspace3 / HBM) and (b) cooperation (1 thread/shot vs
256 threads/shot). The operand MATH is identical across tiers.
V2.md must analyze:
- Can the operand library be written ONCE, tier-parameterized (`if constexpr`),
  so the 3 tiers share source even if they compile to 2-3 specializations?
- Can we UNIFY to fewer modes? e.g. can register-tier fold into coop (fewer
  threads), or an adaptive kernel branch on rank at runtime? Quantify the
  occupancy/divergence cost of unification vs the maintenance win of fewer
  kernels. Default hypothesis: unify the SOURCE, keep the minimum number of
  compiled specializations, prove empirically whether register can merge into
  coop.

## R2. Correctness FIRST, then performance — O0 -> O3 incremental
- ALL circuits of interest + all test circuits must WORK (match gold) before
  any performance work. No perf optimization on a path that isn't correct.
- Bring up correctness at O0 (fast compile; we confirmed bugs reproduce at O0
  so it is a valid correctness target), then raise the opt level incrementally
  O0 -> O1 -> O2 -> O3, verifying correctness is preserved at each step and
  measuring the perf delta each level buys. CLIFFT_MLIR_OPT already supports
  this for the current MLIR path; V2 must keep an equivalent knob.

## R3. Correctness gold = BOTH CPU and SVM
- Validate against the CPU reference (`clifft::sample_survivors`, f64,
  `--cpu-reference`) AND the GPU SVM path (f32). Both are gold standards.
- Note the two can legitimately differ (CPU f64 vs GPU f32 accumulation). V2.md
  must define which is authoritative where, and the tolerance model (exact
  passed_shots match expected when RNG + f32 model match SVM; CPU f64 may differ
  by borderline-branch shots). Differential tests run against both.

## R4. k / max-rank limits — design them out from the start
Current hard limits:
- `kGlobalMaxPeakRank = 19` (`gpu_types.h:12`) -> statevector cap 2^19 = 524288
  amplitudes = 4 MB/slot. Blocks qv20 (rank 20).
- `kPauliWords` (now 5 = 320 qubits) -> Pauli-frame word count; independent of
  rank. Just widened from 2.
V2.md must:
- Make max rank a TUNABLE bounded only by HBM (rank 20 = 8 MB/slot,
  rank 21 = 16 MB/slot, ... 512 global slots * size must fit ~288 GB HBM), not
  a hardcoded constant. Design the global-tier storage rank-parametric from day
  one so raising the cap is a config change, not a rewrite.
- Make kPauliWords similarly config-driven (already partly done) so qubit count
  isn't a recompile-the-world change.
- Confirm which limits are FUNDAMENTAL (HBM size, LDS 64KB for coop) vs
  ARTIFICIAL (hardcoded constants) and eliminate the artificial ones up front.

## Cross-cutting: what "V2 done" means
All 22 BENCHMARKS.md circuits correct vs CPU+SVM, at O0 first, with a single
operand source of truth, the fewest kernels/tiers that the hardware justifies,
and no artificial rank/qubit caps. Performance parity with Hybrid is the bar
AFTER correctness, pursued O0->O3.

## R5. Progressive lowering + kernel specialization (owner directive)
There IS specialization opportunity and V2 should exploit it. Key reframing:
the design axis is NOT "runtime vs compiled" but SPECIALIZATION GRANULARITY.
- What is specializable at compile time (per circuit): axis indices, U2/U4
  matrix VALUES (ConstantPool), amplitude-loop trip counts (1<<active_k),
  which Pauli-frame words are touched, noise/no-noise branch, and the EXACT
  rank (fixes inner-butterfly trip count -> compiler can unroll+vectorize the
  innermost sweep). SVM specializes none of this; Hybrid specializes it via
  C++ template + clang per circuit; V1-MLIR specialized the operand SEQUENCE
  too (unrolled) which is what killed it.
- The sweet spot V1 missed: SPECIALIZE THE OPERAND BODY, LOOP THE OPERAND
  SEQUENCE (and the amplitude sweep, and shots). Progressive lowering is the
  right tool for this: a high-level op ("apply U4 at axes a,b with matrix M")
  lowers through stages to a specialized-but-LOOPED sweep — burning matrix
  constants + axis math into immediates while keeping loops over
  shots/bytecode/amplitudes. Each TIER is a lowering target, not a hand kernel.
- This is where MLIR-the-FRAMEWORK (dialects + lowering passes), as opposed to
  our hand-emitted MLIR TEXT, could actually earn its keep.
- DEVIL'S ADVOCATE (must resolve by measurement, not a priori): building a
  real lowering pipeline (custom dialects, tile ops, passes) is a large
  investment, and HipKittens reports fancy schedules often UNDERPERFORM simple
  ones on CDNA3/4. Meanwhile letting clang specialize a templated C++ operand
  library per circuit (Hybrid's model) gets ~80% of the specialization for
  near-zero infra. V2.md must weigh: does custom progressive-lowering
  specialization beat "templated C++ + clang per circuit" by enough to justify
  the infra? Prototype/measure the highest-value specialization (inner-sweep
  unroll for fixed rank + constant U2/U4 matrices) both ways before committing.

## Observation from the Hybrid IR oracle (docs/v2/ir_reference/*.hybrid.*.ll)
Even Hybrid's HIP source is LARGE for U2/U4-heavy circuits (coop_qv10 = 33.5k
lines HIP; coop_surface_d7_t10 = 75k lines HIP, 398k O0.ll / 146k O2.ll). So
Hybrid ALSO partially unrolls (its templates emit per-instruction code, just
more compactly than MLIR and via a real compiler). This means "Hybrid is the
compact ideal" is only partly true — a V2 that keeps the operand SEQUENCE as a
true runtime loop (not per-instruction emission) could be MORE compact than
Hybrid, not just match it. Worth confirming against these oracle files.

## R6. AVOID HIP AT ALL COSTS — HSA-only dispatch + runtime (HARD constraint)
The project owner mandates: NO HIP. HSA-based dispatch and runtime management
ONLY. This reshapes the option space:
- INVALIDATES "double down on Hybrid" and "templated __device__ HIP operand
  library compiled by clang++ -x hip" (pre_V2 Options A/B as literally written).
  Hybrid IS a HIP kernel; -x hip IS HIP. Both are disallowed as the V2 device
  path.
- STILL ALLOWED and already in place: the HSA runtime for dispatch/memory/
  signals (src/clifft/gpu/runtime/hsa_runtime.cc, PersistentDispatcher). Keep.
- The device-code SOURCE OF TRUTH therefore cannot be HIP C++. It must be one
  of: (a) LLVM-IR (hand/lib-emitted) -> llc -> lld -> .hsaco (the current MLIR
  pipeline's back half, which is HSA-loaded, no HIP runtime); (b) MLIR dialects
  lowered to LLVM-IR then the same; (c) plain C / OpenCL-C compiled to
  amdgcn-amd-amdhsa via clang WITHOUT the HIP runtime (device-libs only, not
  -x hip). The operand MATH is simple arithmetic+memory — it never needed HIP's
  runtime; HIP was just a convenient device-C++ dialect.
- CONSEQUENCE: the cleanest maintainability idea (one templated HIP library)
  must be re-homed to a non-HIP authoring language. Candidates for the single
  operand source of truth under HSA-only:
    1. An LLVM-IR / MLIR operand-function library (llvm.func per operand),
       called by a runtime interpreter loop — HSA-loaded. Device-agnostic-ish.
    2. Plain-C/OpenCL-C operand library -> amdgcn via clang device-only ->
       one .bc/.ll linked into the interpreter kernel -> HSA. Keeps C-level
       single-source without HIP runtime.
    3. MLIR custom dialect (quantum ops) + progressive lowering passes ->
       LLVM-IR -> llc/lld -> HSA. This is where MLIR-the-FRAMEWORK earns its
       keep (R5) AND satisfies HSA-only.
- This REORDERS the recommendation: it pushes V2 toward an IR/MLIR device path
  (what just failed) but now for a PRINCIPLED reason (HSA-only), with the fix
  for why it failed baked in (loop don't unroll; specialize body via lowering;
  shared operand functions not inline unrolling). The SVM-CPU (f64) and the
  operand MATH from SVM/Hybrid remain the correctness reference, but their HIP
  device form cannot be the shipped V2 device path.
- OPEN QUESTION for V2.md: is option (2) plain-C-to-amdgcn the pragmatic
  single-source win (C-level authoring, no HIP, HSA dispatch, compiler
  specializes per circuit via a runtime loop), vs option (3) full MLIR dialect
  (more infra, more device-agnostic, progressive-lowering specialization)?
  Measure the simplest viable path first.

## R7. Tile / fine-grained-task model — GPUs are not just SPMD (owner directive)
The through-line from HipKittens / ThunderKittens / Composable Kernels / cuTile
is NOT their code (HK/CK are HIP/CUDA C++ — R6 forbids vendoring them) but a
PROGRAMMING MODEL:
- **Tiles are the unit of data**, not scalars. A workgroup/wave owns a tile and
  cooperates on it. GPUs are increasingly task-parallel + tile-structured, not
  flat SPMD lockstep.
- **Fine-grained tasks with an explicit memory<->compute split**: some
  waves/warps move data (producer), others compute (consumer), overlapped via
  async pipelines. Different from V1's bulk-synchronous "all 256 threads do the
  same op, then barrier."

Why this fits quantum simulation directly:
- The AMPLITUDE SWEEP is a tile op: a U4 gate reads 4 amplitude planes, 4x4
  mul, writes 4 planes, over a tile of 2^(k-2) groups = tile-load -> tile-
  compute -> tile-store. V1 does it as a flat strided SPMD loop + hand-rolled
  ds_bpermute. A tile op expresses it natively.
- The MEMORY/COMPUTE SPLIT is real per operand class: MEASUREMENT = bandwidth-
  bound reduction (move amps, sum norms); GATE = compute-bound butterfly;
  NOISE/FRAME = tiny scalar bookkeeping. A producer/consumer tile schedule can
  overlap the measurement reduction of stage N with the gate math of stage N+1
  — exactly what V1's barriers prevent.
- The COOP TIER is a tile-cooperation problem: 256 threads on one LDS
  statevector = "a workgroup owns a tile" (HK shared-tile). Getting this wrong
  is where V1 hit divergent-barrier hangs.

MLIR V2 IMPLICATION: the custom `quantum` dialect should express operands as
TILE OPS (e.g. `quantum.gate_sweep` over a statevector tile) that lower through
stages to tile-load/compute/store, with the memory/compute split and optional
async pipelining as LOWERING choices — not hand-written barriers. This is the
strongest argument for MLIR-the-framework over plain-C: MLIR can carry the tile
abstraction + a scheduling/pipelining pass; plain-C would hand-roll it again.
Study for PATTERNS (not code): HipKittens/ThunderKittens (tile primitives,
producer/consumer, register pinning), Composable Kernels (tile pipeline
abstraction on AMD), cuTile (tile IR). Reconcile with R6: patterns only, since
all are HIP/CUDA C++.

CORRECTION to runtime_patterns_iris_hipkittens.md: it recommended "vendor HK as
a header library." Under R6 that is INVALID (HK is HIP C++). Take the tile
MODEL, reimplement in the MLIR-dialect / non-HIP device path.
