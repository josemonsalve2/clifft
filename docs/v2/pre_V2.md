# pre_V2.md — Initial Design Plan for the clifft GPU Code Generator (V2)

**Status:** starting point for an ultrathink/ultracode effort — NOT the final design.
**Date:** 2026-07-24
**Scope:** the GPU execution path only (SVM interpreter, Hybrid compiled-HIP, MLIR IR emission). Front-end HIR, the back-end lowerer (`backend.cc`), and the optimizer (`StatevectorSqueezePass`) are out of scope except as consumers.

This document is decision support for the choice: **reimplement the GPU code generator from scratch vs. re-adapt V1.** It diagnoses why V1 is slow and unmaintainable, proposes a single-source-of-truth operand model, analyzes four design options, and lays out a verification and migration framework. The evidence points strongly toward a **runtime bytecode interpreter on the GPU** as the primary path, with the compiled-per-circuit path demoted to an opt-in specialization — but the tradeoffs are laid out fairly below.

> **Hard constraint (project owner, 2026-07-24):** V2 must execute the circuit bytecode via **runtime mechanisms** — a persistent GPU kernel looping over operands and dispatching to per-operand device functions — **NOT** by compiling a bespoke fully-unrolled kernel per circuit. The whole reason V1 fails (20 MB IR, register spills, slow compile, 5–60× slower) is the static per-circuit unrolling. This document centers that constraint (§3, §4) and folds in runtime-execution patterns from ROCm IRIS and HipKittens (§3.5), analyzed in the companion brief [`runtime_patterns_iris_hipkittens.md`](./runtime_patterns_iris_hipkittens.md).

> **A note on submissiveness.** The owner explicitly asked that the runtime-interpreter idea be *challenged*, not rubber-stamped. §8 is a dedicated devil's-advocate section that argues, with in-tree measurements, that (a) the interpreter will very likely **not** beat Hybrid, (b) "compiled-per-circuit" is **not** the disease — unrolling-all-ops-inline is — and Hybrid already proves a compiled kernel can be fast, and (c) the sharpest reading of the evidence may be *"delete MLIR, promote the interpreter that already exists (SVM), and stop building a bespoke IR path at all."* Read §8 before treating the Option-A recommendation as settled.

---

## 1. Problem statement & goals

clifft simulates quantum circuits by Pauli-frame tracking. The back-end lowers HIR to a flat **bytecode** of ~40 opcodes (`Opcode` enum, `src/clifft/backend/backend.h:25-86`), each a 32-byte `Instruction` (`backend.h:95-163`) plus a side `ConstantPool`. Three GPU backends must produce bit-identical results:

- **SVM** — a bytecode interpreter. On CPU it uses a computed-goto dispatch loop (`src/clifft/svm/svm_kernels.inl:2196` dispatch table, `:2500` switch fallback); the GPU dispatch/host glue lives in `src/clifft/gpu/sampler/hip_sampler.hip` (note it *already* contains a `for (pc) switch(opcode)` interpreter loop at `hip_sampler.hip:729`). This is the **reference**.
- **Hybrid** — generates a self-contained HIP C++ kernel per circuit from string templates (`src/clifft/gpu/codegen/`), compiles it with `clang++ -x hip` to `.hsaco` (`kernel_cache.cc:75-181`), dispatches via HSA. Correct and fast.
- **MLIR** — emits textual MLIR LLVM-dialect IR per circuit (`src/clifft/gpu/mlir/mlir_emit.cc`, 3,424 lines), runs it through `mlir-opt → mlir-translate → opt → llc → lld → .hsaco`. Slow, fragile, incomplete.

Three tiers keyed on compile-time `peak_rank`: **register** (0–4, amplitudes in stack `alloca`), **coop** (5–10, amplitudes in LDS, 256 threads cooperate on one shot), **global** (11–19, amplitudes in HBM with XCD work-stealing).

### Goals, in priority order

1. **Correctness first.** All GPU paths must match SVM bit-for-bit (same seed → same `passed_shots`, same measurement records) across all three tiers and all ~40 opcodes. Correctness must be *structurally* guaranteed, not re-established per backend.
2. **Performance parity with Hybrid.** Hybrid is the fast reference (49 ms end-to-end, 0.16 ms kernel on a rank-1 register circuit; `PERFORMANCE_OPTIONS.md:30`). V2's steady-state kernel time and compile time must be within a small constant of Hybrid.
3. **Maintainability — single source of truth per operand.** Each opcode's math should be written **once**, not three times.

### Non-goals

- Not re-deriving the frame algebra or the bytecode ISA. The 32-byte `Instruction` and the ConstantPool layout are stable and correct; keep them.
- Not optimizing the front-end/optimizer.
- Not chasing sub-Hybrid kernel time in V1's first cut. Parity is the bar; beating Hybrid is a later specialization.
- Not supporting hand-written MLIR as a permanent artifact. If MLIR survives at all, it is *derived*, never hand-maintained.

---

## 2. Root-cause diagnosis — why V1 is slow and unmaintainable

### 2.1 Triple duplication → divergence bugs (maintainability failure)

Every opcode is implemented three times: once in the SVM (`svm_kernels.inl`, e.g. `exec_array_multi_cnot` at `:464`), once in the Hybrid HIP templates (e.g. `kArrayMultiCnot` / `kCoopArrayMultiCnot` in `codegen/ops/multi_qubit_ops.inc:4,42`), and once in MLIR IR emission (`mlir_emit.cc` plus `mlir/ops/*.inc`). Worse, coop and global tiers each get their *own* variant (`kArrayMultiCnot` vs `kCoopArrayMultiCnot`), so an opcode is effectively hand-written **5–7 times** across backends × tiers.

The predictable consequence is drift. The task list still carries "MULTI_CNOT frame px/pz swapped in MLIR only" as a live class of bug (#136), and the frame-swap logic in `mlir/ops/mlir_frame_ops.inc:39-42` is a hand-transcription of the C++ at `multi_qubit_ops.inc:59-68` — a transcription that has been wrong before. Every opcode addition or fix is an N-way edit with N-way test surface. This is the dominant maintenance tax.

### 2.2 Full unrolling → IR bloat, spills, slow compile (the MLIR performance failure)

MLIR emits **straight-line IR, one inlined block per bytecode instruction**, via three separate unroll loops (`mlir_emit.cc:2377`, `:2867`, `:3297` — one per tier), each a `for (pc) switch(op)` that appends inline IR. There is no runtime loop over the bytecode. Consequences, all confirmed:

- **IR bloat that won't link.** A 16,432-op circuit (`surface_d11_t5`) unrolls to ~20 MB of LLVM-IR. The `ir_reference/` samples already show this scale: `coop_qv10.mlir` is 20 MB for a 140-op circuit once fully expanded, and `coop_qv10.O0.ll` is 13 MB.
- **Register spills.** Straight-line global-tier code spills ~115 KB/thread of private memory — catastrophic for occupancy.
- **Compile-time explosion.** `surface_d5_r5`: MLIR **55 s** vs Hybrid **0.06 s**; `surface_d7_r7`: **193 s** vs **0.06 s**; `cultivation_d7`: **224 s** vs 24 s (`PERFORMANCE_OPTIONS.md`). Hybrid stays fast because HIP templates emit a compact runtime loop that hipcc/LLVM handle in one pass; MLIR hands LLVM a giant already-unrolled function.
- **Runtime slowness.** Measured MLIR is 5–60× slower than SVM/Hybrid on the tiers that matter (qv10 MLIR 3.54 s vs Hybrid 0.058 s; surface_d9_t10 MLIR 5.5 s vs 0.11 s). A bloated, spill-heavy kernel with poor occupancy loses at runtime too, not just at compile time.

The Hybrid path proves this is self-inflicted — but the mechanism is subtler than "MLIR unrolls, Hybrid loops," and the subtlety matters for the recommendation. **Hybrid *also* unrolls.** `--hybrid` calls `compile_or_load_kernel*` → `emit_register_kernel.cc:77` → `emit_instructions(out, flat)` (`codegen/emit_instructions.cc:19`), which loops over the bytecode **at codegen time** and emits one *call line* per operand, e.g. `frame_cnot(st, 3u, 5u);`. So both compiled paths unroll the operand sequence. **Only MLIR explodes.** The difference:

- **Hybrid emits `call`s to shared `__device__` functions, many `__noinline__`** (`array_u2`, `array_u4`, `exec_exp_val` at `hip_sampler.hip:626/645/682`). Those boundaries **cap live-register pressure per operand** — `hipcc` allocates registers within each small callee, not across the whole circuit. `PERF_ANALYSIS.md` shows this was a *deliberate* fix: inlining U2/U4/EXP_VAL pushed coop VGPRs 72→128 (occupancy 7→4 waves); extracting `__noinline__ coop_u2_sweep` dropped them to 108 and restored parity.
- **MLIR emits inline IR text with no call/`__noinline__` structure** — every operand body lands in one monolithic function, so llc sees one giant live range → 115 KB spill.

So the precise disease is **"inline-all-operand-bodies-into-one-un-partitioned-function,"** not "unrolling" per se. This has a sharp consequence explored in §8: the spill can be fixed *either* by a runtime loop (the directive's path) *or* — as Hybrid already demonstrates — by keeping the unroll but emitting `call`s to shared per-operand functions. **The unrolling-into-one-body is the disease; MLIR is where it is most acute; and Hybrid is the proof that "compiled per circuit" is not itself the problem.**

### 2.3 Correctness fragility (cross-cutting)

Divergent barriers hang AMDGPU (coop/global tiers); coop RNG threading (tid0-only vs redundant-full) is subtle; f32 vs f64 accumulation differs across backends; struct byte-layout assumptions (`ShotState`, `GpuComplex{float re; float im;}` at `emit_preamble.cc:90-95`) must match between the interpreter and the compiled kernels; HBM stride bugs recur. Each cost days. These are *shared* hazards, but today each backend re-encounters them independently because none of them share code.

**Synthesis:** V1's three failures have one common cause — **there is no shared, compiled artifact of the operand logic that all paths reuse.** Everything downstream (bloat, spills, slow compile, drift bugs) follows from re-expressing each operand in a different representation per backend and, for MLIR, doing so without a runtime loop.

---

## 3. Design options

### Option A — Runtime bytecode interpreter on the GPU (recommended primary)

One `__global__` kernel per tier (register/coop/global). Each kernel runs `for (pc = 0; pc < num_instrs; ++pc) dispatch(bytecode[pc])`, where `dispatch` is a `switch` calling **one `__device__` function per opcode** drawn from a single shared operand library. The bytecode and ConstantPool are uploaded to the GPU as data; **no per-circuit codegen at all.**

This is not speculative: the host already runs exactly this loop (`hip_sampler.hip:729`), and the SVM CPU interpreter is a computed-goto version of it (`svm_kernels.inl:2196`). V2 moves that loop into device code and points it at device-side operand functions.

- **Single source of truth:** the ~40 `__device__` operand functions are the *only* implementation. SVM (via the same functions compiled for host), Hybrid, and MLIR-equivalents all vanish into one library.
- **No IR bloat / no per-circuit compile:** one kernel binary, compiled once (or shipped precompiled). Compile time drops from minutes to zero-per-circuit. Bloat, spills-from-unrolling, and the 20 MB IR all disappear.
- **Tiers:** the kernel is templated/specialized on tier only (3 binaries, not per-circuit). Register tier keeps `alloca`; coop passes `(v, px, pz, active_k, ...)` and cooperates across 256 threads with barriers factored *inside* the coop operand functions (the coop variants already exist — `kCoopArrayMultiCnot` — they become device functions, not strings); global tier keeps the HBM pointer + work-stealing loop.
- **Cooperation:** the coop/global operand ABI takes the thread's role implicitly (via `TIDX`/`BSIZE`) exactly as the current coop templates do; barriers live inside the operand function so divergence is structurally impossible if the loop never conditionally skips an operand across threads (a key invariant to enforce).
- **Performance:** dispatch overhead is a `switch` per instruction — the same overhead SVM already pays and Hybrid mostly avoids by inlining. For the array/coop/global tiers the per-operand *work* (looping over 2^k amplitudes) dominates dispatch, so parity with Hybrid is realistic. For tiny register-tier circuits, dispatch overhead is relatively larger; this is where Option B specialization can be layered on later.

**Tradeoffs:** (−) loses per-circuit constant-folding and inlining that Hybrid gets for free — arguments like axis indices become runtime loads instead of immediates, and the compiler can't fuse across operands. (−) an indirect/`switch` dispatch per instruction. (−) coop/global operand functions with internal barriers must be written carefully so *all* threads execute the same operand sequence. (+) by far the smallest code surface and the strongest correctness story.

### Option B — Per-circuit compiled kernel from ONE operand library, with a runtime loop (not unrolled)

Keep compiling a kernel per circuit (Hybrid-style), but (i) source every operand from the *same* shared `__device__` library as Option A, and (ii) emit a **runtime loop over the bytecode**, not unrolled straight-line code. This is essentially "Hybrid, de-duplicated and de-unrolled."

- Recovers per-circuit specialization opportunities (the compiler sees the actual `num_instrs`, can constant-fold circuit-wide metadata, can inline the operand library into the loop) while avoiding the unrolling blowup.
- Still pays per-circuit compile time (Hybrid's ~0.06–0.4 s for small circuits, up to 24 s for `cultivation_d7`), and still needs the on-disk `.hsaco` cache.
- **Best used as a *specialization layer on top of Option A***: ship the interpreter as the always-correct baseline; optionally JIT a specialized loop for hot circuits.

**Tradeoffs:** (−) reintroduces a compile step and its caching/latency; (−) more moving parts than A. (+) can close the register-tier dispatch-overhead gap where A is weakest.

### Option C — Derive MLIR/LLVM-IR from the Hybrid kernel's post-hipcc IR

Because Hybrid compiles with `clang++ -x hip` (`kernel_cache.cc:112`), its **post-hipcc LLVM-IR is already exactly the artifact the MLIR path was hand-emitting badly.** So: compile the shared operand library (or a whole Hybrid kernel) with hipcc, capture the LLVM-IR, and use *that* as the source of truth for any IR-level path — instead of hand-writing MLIR in `mlir_emit.cc`. The `ir_reference/` database (`docs/v2/ir_reference/*.O0.ll`, `*.O2.ll`) is the seed of this: per-circuit and eventually per-operand reference IR generated *by the trusted compiler*.

- Kills the 3,424-line hand-emitter entirely. IR is generated by a correct compiler, so the MULTI_CNOT-style transcription bugs cannot occur.
- Useful even if we never ship a bespoke MLIR path: the reference IR is the **oracle** for verifying that A/B produce equivalent code, and a corpus for any future custom lowering.

**Tradeoffs:** (−) doesn't by itself buy anything over just *running* the Hybrid kernel — if you have the IR, you have the kernel. Its value is as a **verification oracle and salvage of the MLIR investment**, not as a third runtime path. (−) hipcc-emitted IR is HIP-flavored (address spaces, intrinsics) and not trivially portable to a device-agnostic MLIR dialect. Recommendation: adopt C as the *verification* strategy (§5), not as a primary runtime.

### Option D — Re-adapt V1 incrementally

Roll the MLIR unrolled emitters into runtime loops op-by-op, and dedupe the three backends into shared functions gradually, keeping V1's structure.

- (+) lowest immediate risk; keeps the working Hybrid/SVM paths untouched; salvages all HSA/persistent-dispatch infra.
- (−) the MLIR emitter's core architecture *is* the unrolling; converting it to a runtime loop is close to rewriting `mlir_emit.cc` anyway. Incrementally deduping three divergent implementations into one is often harder than writing the one and deleting the three.
- **Verdict:** D is attractive only if the shared operand library (needed for A and B) is hard to extract. It is not — the SVM and Hybrid operand bodies are already close to the desired form.

### 3.5 Folding in runtime-execution patterns from IRIS and HipKittens

The companion brief [`runtime_patterns_iris_hipkittens.md`](./runtime_patterns_iris_hipkittens.md) analyzes two Nov-2025 AMD-native projects for patterns that upgrade Option A's runtime kernels *beyond* what SVM does today. These are **additive to Option A** and independent of the interpreter-vs-compiled debate — they apply to any kernel. Summary of what to adopt and where:

**From ROCm IRIS (persistent kernels, XCD scheduling, memory model — brief §1):**

- **Persistent grid-stride loop (brief §1.1).** SVM/Hybrid launch **one block per shot** (`sample_kernel_coop`, `blockIdx.x == shot_id`, `hip_sampler.hip:1863`) and rely on HIP relaunch. IRIS's idiom — launch one workgroup per CU (`get_cu_count()`: 304 on MI300X, 256 on MI355X) and *stride* over the work list on-device — makes the kernel resident and removes relaunch/tail overhead. **This is the "persistent kernel" half of the directive's phrasing** that SVM does *not* yet satisfy. The outer loop strides over shots (register/coop) or amplitude-tiles (global); the inner loop is the operand loop.
- **Agent-scoped atomic work-stealing (brief §1.2).** The global tier already has scaffolding — `buffers.work_counter` with `kNumXCDs` slots, memset before dispatch (`hip_sampler.hip:2333`). IRIS's disciplined version is a single agent-scoped `atomic_add` work-index that every workgroup pulls from, with **release/acquire** flags on any tile-ready dependency (never system scope on a single GPU, never workgroup scope across workgroups).
- **SC-HRF scoped-sync rulebook (brief §1.3) — the fix for V1's divergent-barrier hangs.** Use **workgroup-scope** barriers for coop operand stages, sequenced *at the operand-stage boundary, outside any per-thread branch*, so all threads converge before each barrier. This is exactly the §4.3/§5 "barrier inside the operand, uniform dispatch loop" invariant, now with IRIS's scope discipline as its formal justification. Narrow scopes are also a perf lever (device/system fences are far costlier).
- **XCD pid-remap (brief §1.1/§2.5)** so consecutive shot ranges stay L2-resident on one XCD (8 XCDs × 32 CUs on MI355X).

**From HipKittens (tile abstractions for the coop sweep — brief §2):**

- **Tile-backed statevector (brief §2.1–2.2).** Replace the coop tier's raw `__shared__ GpuComplex v[kSharedMaxAmplitudes]` (`hip_sampler.hip:1868`) with an HK-style **shared (LDS) tile** using bank-conflict-free **swizzling**, stage per-gate operands into **register tiles**, and use **register pinning** to *guarantee* the register-tier statevector never spills (HK's pinning took an MHA kernel 855→1024 TFLOPS by bypassing HIPCC's allocator). This directly attacks V1's 115 KB spill at its root.
- **64-wide cooperative reductions/broadcasts (brief §2.4).** Replace the coop tier's hand-rolled `coop_reduce2` (`hip_sampler.hip:940`) and MLIR's hard-coded shuffle tree `{32,16,8,4,2,1}` (`mlir_emit.cc:572`) with HK's layout-aware `row_reduce`/`packed_shfl` at `WARP_THREADS=64` — the exact primitive the measurement probability-sum + sampled-outcome broadcast need, correct across gfx942/gfx950.

**What does NOT transfer (brief §2.5, §3, honest caveats):**

- **IRIS's actual code** — Triton/Python, multi-GPU symmetric heap. V2 is single-GPU C++/HSA; take the *structure and memory model*, zero lines of code. And IRIS's Three-Taxes headline gains (10–20%) are modest here because **clifft already fuses the whole circuit into one kernel per shot** — the launch/inter-kernel taxes are *already* mostly paid. IRIS's value is the residency loop + sync toolkit, not the "fuse many kernels" thesis.
- **HipKittens' MFMA/matrix-core compute** — a 2×2/4×4 complex-double gate butterfly is a strided gather/scatter AXPY, **not** an MFMA bf16 contraction. Take HK's tile *containers, movement, swizzle, reductions*; not its `mma`. Whether multi-qubit gates can be reformulated as small GEMMs to actually use the matrix cores is an open question (brief §4 Q1).

**Net:** Option A + IRIS persistence/work-stealing (global tier) + HK tiles (coop tier) = the full runtime-mechanism design the directive envisions. But note (foreshadowing §8) that these upgrades apply to a **compiled** kernel just as well as an interpreter — they do not, by themselves, settle whether V2 needs a runtime interpreter or just a better-emitted compiled kernel.

### Recommendation

**Adopt Option A as the primary runtime**, extract the shared operand library from the battle-tested SVM/Hybrid logic, and use **Option C as the verification strategy** to salvage the MLIR/IR investment as an oracle. Keep **Option B in reserve** as a per-circuit specialization for register-tier hot paths only if measurements show A can't reach parity there. Do **not** carry the hand-written MLIR emitter forward. **This recommendation is provisional until the §8 devil's-advocate case is weighed** — in particular, §8 argues the interpreter's win is over *MLIR*, not over *Hybrid*, and that the decisive variable is circuit-variety vs shot-count in the real RL workload.

---

## 4. Single-source-of-truth operand model

The center of V2 is **one `__device__`/`__host__` function per opcode**, written once, compiled for every target.

### 4.1 Operand ABI

Model the ABI on the existing coop template signatures, which already pass state explicitly (`multi_qubit_ops.inc:43`):

```cpp
// Per-tier state view. One struct; tier chooses the storage backing `v`.
struct StateView {
    GpuComplex*  v;          // amplitudes: alloca | LDS ptr | HBM ptr
    uint64_t*    px;         // Pauli-X frame words
    uint64_t*    pz;         // Pauli-Z frame words
    GpuComplex   gamma;      // global phase/weight accumulator
    uint32_t     active_k;   // current active rank
    // + measurement record ptr, RNG state, obs/exp_val accumulators, forced-record ptr
};

// One implementation per opcode. `tier` and cooperation are compile-time.
template <Tier T>
DEV void op_array_multi_cnot(StateView& st, const Instruction& I, RngState& rng);
```

- **Arguments** come straight from the 32-byte `Instruction` union (`backend.h:111-160`) and the ConstantPool by index — no re-encoding. The operand reads `I.axis_1`, `I.multi_gate.mask`, etc., exactly as SVM does today.
- **Tier is a template parameter**, not a copy-paste. `if constexpr (T == Tier::Coop)` selects the LDS/barrier path; register tier compiles the barriers away. This collapses the current `kArrayMultiCnot` / `kCoopArrayMultiCnot` split into one function with a compile-time branch.
- **Cooperation/barriers live inside the operand**, so the dispatch loop is uniform across threads and can never introduce divergent barriers (the AMDGPU hang hazard) — provided the loop itself is not thread-divergent.

### 4.2 How one definition serves all paths

- **Interpreter (Option A):** the dispatch `switch` calls `op_*<Tier>` directly. This library, compiled with the host compiler, *is* the SVM — the CPU and GPU interpreters share source.
- **Compiled/JIT (Option B):** the same functions are `#include`d into a per-circuit loop kernel and inlined by hipcc.
- **IR/verification (Option C):** the same functions, compiled by hipcc with `-emit-llvm`, yield the reference IR corpus (`docs/v2/ir_reference/`), including **per-operand unit IR** (compile each `op_*` in isolation).

One edit to `op_array_multi_cnot` propagates to every path and every tier. The px/pz-swap class of bug becomes impossible because there is exactly one px/pz swap in the tree.

### 4.3 What must be nailed down

- **Exact numeric model.** Pick f32 vs f64 per accumulator and enforce it in the one place (today `GpuComplex` is f32, `emit_preamble.cc:90`, but gamma/probabilities need f64 in places). Divergence here is a top cause of SVM≠GPU mismatches.
- **Struct layout parity.** `Instruction`, `ConstantPool`-derived device tables, and `StateView` must have one canonical device layout shared by interpreter and any compiled path.
- **RNG threading contract.** Coop tier: define once whether the RNG advances on tid0-only or redundantly-full, and bake it into the coop operand functions.

---

## 5. Cross-backend verification strategy

V2 stays correct by construction (one operand library) *and* by continuous differential testing.

1. **SVM as golden oracle.** SVM (the same operand library compiled for host) is the reference. Every GPU tier is diffed against it: same seed → identical `passed_shots`, measurement records, observable parities, exp-vals. This is already the project's rule (CLAUDE.md: "Always verify correctness against CPU/SVM").
2. **Per-operand unit tests.** For each opcode, construct a minimal bytecode exercising it at each tier, run interpreter vs SVM vs (optionally) a compiled loop, and assert bit-identity of the full state vector, not just measurements. Because there is now *one* implementation, a per-operand test failing localizes to one function.
3. **LLVM-IR reference database (Option C).** `docs/v2/ir_reference/` already holds `*.mlir`, `*.O0.ll`, `*.O2.ll` for representative circuits per tier (coop_qv10, coop_circuit_d5, coop_surface_d7_t10, …). Extend this to **per-operand unit IR** generated by hipcc from the shared library. Uses: (a) detect when a refactor changes generated code unexpectedly (IR diff in CI); (b) provide the oracle that a future custom-lowering path must match; (c) document what "correct fast code" looks like for each operand.
4. **Differential fuzzing.** Random valid circuits (respecting rank/tier constraints) run interpreter vs SVM across seeds; any mismatch is a reduced test case. Especially targets the tier boundaries (rank 4→5, 10→11) where storage backing changes.
5. **Barrier/divergence lint.** Static check that the dispatch loop executes an identical operand sequence on all cooperating threads (no data-dependent `continue`/`break` across the barrier), closing the AMDGPU-hang class structurally.

---

## 6. Migration / decision framework

### Reimplement-from-scratch vs re-adapt — criteria

**Reimplement the GPU code generator** (recommended for the MLIR path) when:
- The component's core architecture is the problem. `mlir_emit.cc`'s unrolling *is* its architecture; there is little to salvage from its 3,424 lines beyond the knowledge encoded in `ir_reference/`.
- A shared operand library can be extracted cheaply from existing correct code — which it can, from SVM (`svm_kernels.inl`) and the Hybrid templates (`codegen/ops/*.inc`).

**Re-adapt** when a subsystem is correct and orthogonal to the codegen problem:
- **HSA runtime & PersistentDispatcher** (`gpu/runtime/hsa_*`) — works, saves ~3.3 µs/dispatch (`hsa_persistent_dispatch.h:42`). **Keep as-is.**
- **Bytecode ISA + ConstantPool** (`backend.h`) — stable, correct. **Keep.**
- **Back-end lowerer + StatevectorSqueezePass** — orthogonal to GPU codegen. **Keep.**
- **The `.hsaco` disk cache** (`kernel_cache.cc`) — reusable if Option B survives.

### What is salvaged (highest-value assets)

1. **The operand math itself** — SVM's `exec_*` functions and Hybrid's `k*` device strings are battle-tested and *correct*. V2's shared library is a re-homing of this logic, not a re-derivation. This is the single most valuable salvage.
2. **HSA dispatch / persistent-kernel infra** — untouched.
3. **The `ir_reference/` corpus** — the salvaged value of the MLIR effort, repurposed as a verification oracle.

### Suggested phasing

- **Phase 0:** freeze the ABI (§4.1), numeric model, and struct layouts. Write the differential-test harness (§5) against current SVM/Hybrid so V2 has a regression net *before* any code moves.
- **Phase 1:** extract the shared `op_*<Tier>` library from SVM + Hybrid; make SVM consume it (prove host parity).
- **Phase 2:** stand up the Option A register-tier interpreter kernel; diff vs SVM.
- **Phase 3:** coop then global tiers (barrier/cooperation inside operands).
- **Phase 4:** delete the MLIR emitter; regenerate `ir_reference/` from the shared library as the oracle.
- **Phase 5 (optional):** Option B specialization for register-tier hot paths if measurements demand it.

Each phase is independently shippable and independently verifiable — and each ends at a commit (per project rule "commit at every step").

---

## 7. Risks & open questions for the ultrathink phase

1. **Dispatch overhead on tiny circuits.** For rank-0/1 register circuits, per-instruction `switch` dispatch may cost more than Hybrid's inlined straight-line code (Hybrid kernel 0.16 ms vs SVM 0.67 ms on rank-1 — some of that gap is dispatch). Is Option A parity-acceptable there, or is Option B specialization mandatory for the 331 register-tier circuits? **Measure early.**
2. **Loss of per-circuit constant folding.** Runtime operand args (axis indices, masks) become loads, not immediates. How much does this cost on the array/coop/global tiers where amplitude-loop work dominates? Likely negligible, but unproven.
3. **Coop/global cooperation ABI.** Getting 256-thread cooperation *and* internal barriers correct inside templated operand functions, with a uniform dispatch loop, is the hardest correctness problem. The rank 4→5 and 10→11 tier transitions (storage changes from alloca→LDS→HBM) are the highest-risk seams.
4. **Numeric model unification.** Where exactly does f64 matter (gamma, probability accumulation, renormalization on deep trajectories — see forced-measurement note `backend.h:63-69`) vs f32 amplitudes? A single wrong choice silently breaks SVM=GPU.
5. **Precompiled vs JIT interpreter binary.** Can the 3 tier kernels be precompiled and shipped (zero per-circuit compile), or is per-arch (gfx942/gfx950) JIT still needed? The HSA loader cost (170–206 ms one-time, `PERFORMANCE_OPTIONS.md`) must be amortized.
6. **Forced-measurement / bytecode-rewrite variants** (`OP_*_FORCED`, synthesized at runtime by `record_probabilities()`) must flow through the interpreter unchanged — verify the rewrite path targets the interpreter, not a compiled kernel.
7. **Is Option C worth any runtime role at all,** or purely verification? Current reading: purely verification — if you have the IR you already have the Hybrid kernel, so a bespoke IR runtime adds fragility without a clear win.
8. **U2/U4 fused-unitary tables** (ConstantPool `fused_u2_nodes`/`fused_u4_nodes`) drove code bloat (task #134, "switch to runtime table lookup"). The interpreter model *naturally* fixes this — the tables become runtime data the operand indexes, not unrolled constants. Confirm this holds at all tiers.
9. **(N2 — decisive, from §8.6) Circuit-variety vs shot-count profile of the real RL workload.** This single measurement decides interpreter (Option A) vs keep-Hybrid. Measure amortized per-circuit compile cost across a realistic `clifft_rl` rollout distribution against steady-state shot throughput. **Do this before committing to any rewrite.**
10. **(N1 — from §8.1) Interpreter tax at coop/global tiers under HSA dispatch.** Benchmark SVM-as-HSA-kernel vs Hybrid at coop/global (not just register tier — prior rank-0/1 numbers hide GPU compute, per `PERFORMANCE_OPTIONS.md`). Confirms whether the measured ~1.3×/+35%-SALU register-tier tax holds where amplitude-loop work dominates (it should shrink there).
11. **(N3 — from §8.3) Can Option D′ hit Hybrid parity?** If MLIR (or any per-circuit path) emits `call`s to shared pre-compiled op functions instead of inlining, does it match Hybrid *and* keep constant folding? If yes, it removes the last reason to either keep or replace MLIR, and reframes V2 as "one shared op library, consumed by an interpreter *or* a call-emitting compiled loop" — the choice then purely driven by N2.

---

## 8. Devil's advocate — the case *against* a V2 runtime interpreter

The owner asked for this section explicitly: do not be submissive to the runtime-interpreter idea; argue against it with evidence and be willing to recommend against it. Here is the honest counter-case. Parts of it are, in my judgement, correct, and they should change how confidently §3's recommendation is held.

### 8.1 A GPU `switch(opcode)` interpreter loses ILP and constant specialization — measurably

A runtime interpreter dispatches per operand through a ~40-way `switch` in the hot loop. Against a compiled kernel this costs:

- **Loss of constant specialization.** In Hybrid/MLIR the axis indices, loop trip counts (`iters = 2^k`), and unitary matrix values are **compile-time immediates**, unlocking inner-loop vectorization, strength reduction, and loop-invariant hoisting. `mlir_emit.cc:337` explicitly exploits this — it "receives [active_k] so it can emit loop trip counts as constants." The interpreter reads `instr.axis_1`, the trip count, and the matrix from **memory** every iteration; all of that specialization is gone.
- **Worse ILP / i-cache pressure.** A straight-line compiled body can be software-pipelined across operands; the interpreter re-fetches the dispatch table and can't schedule across the `switch` boundary.
- **Warp divergence: a wash, not a strike.** Worth stating precisely because it's the obvious objection: the top-level opcode `switch` does **not** diverge across lanes. The circuit is shared by all shots in a block, so all lanes are at the same `pc` with the same opcode (register tier packs one shot per lane; coop packs one shot per block). Data-dependent branches *inside* operands (measurement, noise) diverge equally in both the interpreter and the compiled kernel. So divergence does not favor the compiled path — but the ILP/constant-specialization loss does, one-directionally.

**How big?** We have a direct in-tree measurement, not a guess. **SVM is the interpreter, and it is already benchmarked:** on `qv10` SVM = 0.069 s vs Hybrid = 0.055 s (**~1.25×**), and `PERF_ANALYSIS.md` attributes **+35% SALU** in the coop kernel specifically to "switch dispatch." So the interpreter tax is **real and measured at ~25–35%.** Not fatal — but it is *away* from, not toward, the performance target.

### 8.2 The interpreter will almost certainly NOT beat Hybrid — Hybrid is the ceiling

The directive asks directly: will the interpreter beat Hybrid? **No.** Hybrid is a compiled-per-circuit kernel with **full constant specialization** *and* it already avoids the spill (via `__noinline__` op boundaries, §2.2). SVM — literally the interpreter — is measured *slower than Hybrid on every tier*. There is no mechanism by which adding a runtime dispatch loop makes a kernel faster than the same kernel with constants folded and the loop specialized, **when the specialized kernel does not spill.** The interpreter's only structural edge over Hybrid is **zero per-circuit compile** and **bounded IR size** — and those edges are real against *MLIR* (193 s compiles, 20 MB IR), but Hybrid **already has them** (0.06 s AOT compiles, because its `.inc` op bodies are pre-written and only the thin call-list is generated per circuit).

**Reframed honestly: the interpreter's win is over MLIR, not over Hybrid.** Against Hybrid, the interpreter is a small, measured regression, bought in exchange for not compiling per circuit.

### 8.3 The real fix might be "keep per-circuit compile, emit calls" — which Hybrid already does

The owner's own framing is sharp: the problem was **MLIR-specific inline unrolling**, not compilation. §2.2 shows there are *two* ways to kill the spill without a runtime interpreter:

1. **(Directive path) Runtime loop.** Emit one function per operand + a runtime dispatch loop calling them. There is even in-tree precedent that MLIR *can* emit runtime loops: `OP_NOISE_BLOCK` was converted from compile-time site-unroll to a **runtime channel loop** (`mlir/ops/mlir_noise_ops.inc`), shrinking `circuit_d5` IR **32 MB (non-compiling) → 4 MB** and restoring correctness (baseline doc). So the top-level operand loop could be converted the same way.
2. **(Option D′, not in the directive) Keep unrolling, but emit `call`s to pre-compiled per-operand functions** — i.e. do exactly what Hybrid does. This *also* bounds register pressure (each callee allocated independently) **and keeps per-operand constant folding at the call site** (axis indices etc. remain immediates). It plausibly gets the best of both: no spill **and** specialization. It does **not** literally satisfy the "runtime loop" directive — but it may be the technically superior answer if the sole goal is "fix MLIR/keep speed."

This is a genuine tension with the directive: the evidence (Hybrid unrolls and is fast) shows a runtime loop is *sufficient* but **not necessary** to fix the spill.

### 8.4 Does MLIR — or a V2 rewrite — earn its keep at all? (the strongest counter)

Step all the way back. There are **three** paths. **SVM and Hybrid are both correct and fast** across all tiers (0.05–0.08 s). **MLIR is correct on 14/14 compiling circuits but 5–60× slower and cannot compile the largest circuits in tolerable time.** What does MLIR buy over Hybrid?

- The original rationale was pure-HSA dispatch (no HIP) + progressive-lowering headroom. But **Hybrid also dispatches via HSA** — `--hybrid` uses `compile_or_load_kernel*` → `PersistentDispatcher` (`hip_sampler.hip:2276`). HIP is used only at *compile* time to produce the cached `.hsaco`; the CLAUDE.md "no HIP in the dispatch path" rule is *already met* by Hybrid's loaded object.
- MLIR's theoretical optimization headroom **has not materialized** — it is the slowest path by 1–2 orders of magnitude and nowhere near the 2×/3× target in `project_mlir_all_tiers.md`.

So the **sharpest devil's-advocate recommendation is: delete the MLIR backend, keep Hybrid, and — if a runtime loop is wanted for compile-time/IR-size reasons — obtain it by promoting SVM's `execute_shot` into the HSA dispatch path (Option A), which is a few hundred lines of plumbing over code that already exists and already works.** A ground-up "V2 runtime-interpreter compiler" risks re-deriving SVM, plus paying the §8.1 interpreter tax, to replace a path (MLIR) that never earned its keep.

### 8.5 What *would* justify the runtime-interpreter direction anyway

To keep this fair, three conditions genuinely favor the interpreter — and the first is strong:

1. **Circuit variety dominates shot count.** If clifft runs **thousands of distinct circuits** (this is `clifft_rl` — RL rollouts), Hybrid pays compile *per distinct circuit*: 0.06 s small, **24 s** for `cultivation_d7`, and the tier memo counts 331 register + 11 coop + 6 global *distinct* circuit classes with many parameterizations. A **single AOT interpreter** amortizes that to zero, and the §8.1 runtime tax is dwarfed by eliminated compile time. **This is the real case for V2 — and it is a case for Option A (promote SVM), not for a new MLIR compiler.**
2. **Persistence/work-stealing** (IRIS, §3.5) that neither SVM nor Hybrid exploits today — but those layer onto *any* kernel, interpreter or compiled, so they don't discriminate.
3. **Single-source-of-truth** (§4) — an interpreter naturally has one op definition; but Option D′ (shared pre-compiled op functions) also collapses to one copy.

### 8.6 The decision this actually hinges on

The interpreter-vs-Hybrid question reduces to **one measurable variable**: *is the workload circuit-variety-bound or shot-count-bound?*

- **Variety-bound** (many distinct circuits, moderate shots each) → the interpreter/Option A wins big on amortized compile; the ~1.3× steady-state tax is irrelevant. **Recommend Option A, delete MLIR.**
- **Shot-count-bound** (few circuits, huge shot counts) → Hybrid's specialized compiled kernels win at steady state; V2's marginal value is small, and the right move is **keep Hybrid, delete MLIR, and don't build a bespoke interpreter compiler at all** (optionally add IRIS persistence to Hybrid).

The `clifft_rl` framing and the tier memo suggest variety is high, which favors Option A — **but this must be measured (open question N2), not assumed, before committing to a rewrite.** Either way, the one conclusion the evidence supports unconditionally is: **retire the hand-written MLIR unroller.** It is dominated by both alternatives.

---

### One-line recommendation

If the workload is circuit-variety-bound (likely, for RL rollouts): build **one templated `__device__` operand library** (salvaged from SVM/Hybrid), drive it with a **persistent runtime bytecode-interpreter kernel per tier** (Option A + IRIS persistence + HK tiles, §3.5), verify against SVM with the **hipcc-generated IR corpus as oracle** (Option C), keep the HSA/persistent-dispatch infra, and **retire the hand-written MLIR unroller.** But per §8: the interpreter's win is over **MLIR, not Hybrid** — so **first measure circuit-variety-vs-shot-count (N2)**; if shot-count-bound, keep Hybrid and skip the interpreter rewrite. In *all* scenarios, **delete the MLIR unroller** — it is dominated. Hold per-circuit specialization (Option B / Option D′) in reserve.
