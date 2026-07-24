# Runtime GPU Execution Patterns from IRIS and HipKittens — Mapped to the V2 Quantum Simulator

**Audience:** V2 architecture ultrathink. **Target HW:** AMD MI350X / MI355X (gfx950, CDNA4).
**Date:** 2026-07-24.

## 0. The V2 problem, restated for context

V1's "MLIR" backend fully *unrolls* every circuit: one inlined LLVM-IR block per operand. A 16,000-operand
circuit becomes ~20 MB of IR that will not compile; register spills hit 115 KB/thread; runtime is 5–60× slower
than a hand-written interpreter. V2 replaces this with a **persistent interpreter kernel** that reads the
circuit's flat bytecode (~40 operand types) from memory and dispatches at runtime to per-operand device
functions, across three tiers: **register** (few qubits, VGPR-private statevector), **coop** (256 threads
share one shot's statevector in LDS), and **global** (statevector in HBM, work-stealing across workgroups).
The amplitude sweep (touch 2^k complex amplitudes per gate) is the heavy compute; Pauli-frame updates are
light bookkeeping.

Both reference projects are recent (Nov 2025), target the exact silicon we care about (MI350X/gfx950), and
solve problems structurally close to ours. Neither is a quantum simulator; the value is in **transferable
patterns**, and in a couple of cases **directly reusable primitives**.

---

## 1. IRIS (ROCm) — persistence, device-side scheduling, and the memory model

IRIS is a **Triton/Python SHMEM-like RMA library for multi-GPU** communication (paper arXiv:2511.12500;
"Three Taxes" companion arXiv:2511.02168; repo github.com/ROCm/iris). Its *domain* — inter-GPU communication
overlap — is **not our domain** (V2 is single-GPU C++/HSA). So the caveat up front: **IRIS's code is not
reusable for us** (Triton-Python, multi-rank symmetric heap, `iris.put/get/store`, agent/system-scoped
atomics for cross-GPU visibility). What transfers is its **execution-structure vocabulary** and its
**memory-model discipline**, both of which map cleanly onto a single-GPU persistent interpreter.

### 1.1 Persistent kernel design (§4.2.1–4.2.2)

IRIS's fused "mega/uber kernel" (§4.2.1) is exactly the persistent-grid idiom we need. The canonical shape
(Listing 4):

```python
pid = tl.program_id(0)
total_tiles = num_pid_m * num_pid_n
for tile_id in range(pid, total_tiles, NUM_SMS):   # grid-stride persistent loop
    c = gemm_loop(...)                              # do work unit
    # ... signal / consume ...
```

Key structural takeaways for V2:

- **Launch one workgroup per CU** (MI300X = 304 CUs; MI355X = 256 CUs across 8 XCDs of 32 CUs each — HK §2.1)
  and keep it resident. The grid is sized to the machine, *not* to the problem.
- **Work is pulled, not pushed:** each workgroup strides `pid, pid+NUM_SMS, …` over a work list. For V2 the
  work list is **shots** (global/coop tiers) or **statevector amplitude tiles** (global tier heavy sweep).
- **Termination** is implicit: the loop exits when the strided index passes `total_tiles`. No host round-trip.
- IRIS explicitly notes the cost this introduces — **tail-latency / "tail occupancy inefficiency"** (§4.2.1):
  the last wave must finish its whole work item before the kernel ends. For V2 with heterogeneous circuits
  this matters, and it is the argument for **work-stealing** (below) over static grid-stride.

**Mapping to V2:** the interpreter kernel is literally this: a persistent for-loop, but the inner body is a
*second* loop over the circuit's operand array, with a runtime `switch(opcode)` dispatch to a per-operand
device function. Two nested loops: outer over shots/tiles (pulled), inner over operands (sequential, from
bytecode in memory). This is the single most important pattern to adopt.

### 1.2 Device-side work scheduling / work-stealing (§4.2.2, "Workgroup Specialization")

IRIS's *global-tier-relevant* idea is **workgroup specialization**: within one persistent kernel, workgroups
are partitioned by `pid` into roles (e.g. CUs 0–255 compute GEMM tiles, CUs 256–303 wait on a spin-lock and
then scatter). Producers signal with `tl.atomic_cas(..., sem="release")`; consumers spin with `acquire`
semantics.

What IRIS does *better* than V1's "crude work-stealing":

- **Explicit producer/consumer separation with atomic hand-off**, rather than every workgroup racing on the
  same global counter. V1's global tier already does crude stealing; IRIS shows the disciplined version — a
  **released-then-acquired flag per tile**, so consumers never read a half-written tile.
- **Role specialization by workgroup index** is a cheap, deterministic scheduler with no central queue.

**Honest caveat:** IRIS's specialization is producer(compute)→consumer(communicate). V2 has no communication
consumer. The transferable piece is the *mechanism* (atomic-flag hand-off, `pid`-based role assignment, a
released work-counter), not the producer/consumer decomposition itself. For V2's global tier the natural
form is a **single global atomic work-index that every workgroup `atomic_add`s to grab the next shot/tile** —
this is classic persistent work-stealing, and IRIS's atomic-scope discipline (next section) is what makes it
correct.

### 1.3 Fine-grain synchronization / memory model (§2.1.3) — **the most directly useful part**

This is IRIS's most reusable contribution for us, and it addresses V1's **divergent-barrier hangs** head-on.
IRIS adopts AMD's **SC-HRF** (Sequentially Consistent Heterogeneous-Race-Free) model, the C++ model extended
with GPU memory scopes:

- **Orderings:** `relaxed`, `acquire`, `release`, `acq_rel`, `seq_cst` — standard acquire/release semantics.
- **Hierarchical scopes (critical):** **wavefront → workgroup → agent(device) → system.** You pick the
  *narrowest* scope that gives correctness. IRIS uses agent/system scope only for cross-GPU visibility.

**For V2 this is a design rulebook:**

- **Coop tier** (256 threads cooperating in LDS on one statevector): barriers between operand stages need
  only **workgroup scope**. Use `__syncthreads()`/workgroup-scoped fences, never device/system fences.
- The **divergent-barrier hang in V1** is the classic bug of calling a workgroup barrier inside divergent
  control flow. IRIS's discipline — *the barrier is a property of the whole workgroup, sequenced at the
  operand-stage boundary, outside any per-thread branch* — is the fix. Do the runtime `switch(opcode)` so
  that **all threads reach the same barrier** after each operand stage; branch on opcode *inside* the stage
  but converge before the barrier.
- **Global tier work-stealing** needs **agent(device)-scoped** atomics on the work counter and **release/
  acquire** on any tile-ready flags. Not system scope (single GPU), not workgroup scope (cross-workgroup).
- Choosing narrow scopes is also a *performance* lever: device/system fences are far more expensive than
  workgroup fences, and V1 likely over-synchronized.

**Reusable vs reimplement:** the *model* is directly usable (it is just AMD's HIP/`__hip_atomic_*` +
`__builtin_amdgcn_fence` with scope strings `"workgroup"`, `"agent"`, `"system"`). We reimplement in C++/HSA;
IRIS validates the scope choices.

### 1.4 The Three Taxes (arXiv:2511.02168, §2.3) — which does V1 pay?

The taxes are framed for distributed LLMs but map startlingly well onto V1's unrolled-per-circuit approach:

| Tax | IRIS definition (§2.3.1–2.3.3) | Does V1 pay it? |
|---|---|---|
| **Kernel Launch Overhead** | Fixed host-side dispatch/setup cost per kernel; dominates when kernels are short | **Yes, catastrophically** — but V1's version is worse than launch: it pays a **per-circuit compile tax** (emit 20 MB IR + LLVM compile) *before* it can even launch. A persistent interpreter pays this **once**, at build time, for all circuits. |
| **Bulk Synchronous** | GPU idle at global barriers waiting on the slowest participant | **Partially** — any per-gate kernel-boundary sync would pay it; the interpreter avoids it by keeping all operand stages inside one resident kernel with fine-grained (workgroup-scope) sync instead of global barriers. |
| **Inter-Kernel Data Locality** | Producer output evicted to HBM, reloaded by consumer kernel | **Yes, if gates were separate kernels** — the statevector would round-trip to HBM between gates. The persistent interpreter keeps the statevector **resident in VGPR (register tier) / LDS (coop tier)** across all operands, paying this tax zero times. This is precisely the register/coop-tier value proposition. |

The V1 unrolled approach "avoids" the inter-kernel tax by inlining everything into one giant kernel — but
pays for it with the **compile/IR-size explosion and register spills** (which are themselves a locality tax:
spilled registers *are* an HBM round-trip). **The persistent runtime interpreter is the only design that pays
none of the three taxes and also avoids the IR explosion.** This table is the core justification for V2.
Measured payoff in the paper is modest for their workload (10–20% end-to-end on Flash Decode over BSP), but
our starting point (5–60× slower, won't-compile) makes the structural win far larger.

---

## 2. HipKittens (Stanford HazyResearch) — the structurally correct fit

HK is a **C++-embedded, header-only tile primitive library for AMD CDNA3/CDNA4** (paper arXiv:2511.08083;
blog hazyresearch.stanford.edu/blog/2025-11-09-hk; repo github.com/HazyResearch/HipKittens). It is
**single-GPU, C++, gfx950-native** — the same shape as V2. This is the better structural match, and pieces of
it are **directly usable as a header library**.

### 2.1 Tile abstraction on 64-wide wavefronts + LDS (§3.1) — maps to the amplitude sweep

HK's basic datatype is a **tile**, parameterized by `dtype` (FP32/BF16/FP16/FP8/FP6), `rows`, `cols`, and
`layout` (row/col major), living in **register** or **shared (LDS)** memory (§3.1). Bulk PyTorch-like
operators (`mma`, `exp`, `add`, reductions) run over tiles; each wraps raw CDNA assembly/HIP with **no
overhead** (they are thin inline wrappers, §3.1 "Compute"). Terminology note: AMD **waves = 64 threads**
(HK deliberately says "wave" not "warp"); a thread block is several waves co-scheduled on a CU's 4 SIMDs.

**Mapping to V2:** the quantum amplitude sweep — "process 2^k complex amplitudes cooperatively" — **is
structurally a tiled loop.** A gate on qubit `q` pairs amplitudes `|…0…⟩` and `|…1…⟩`; a block of 2^k
amplitudes is a **tile** and the 256-thread workgroup is the cooperating worker set. HK's
`register-tile`/`shared-tile` split is exactly the **register-tier vs coop-tier** split:

- **Register tier** ⇒ HK **register tile** (statevector held in VGPRs, private).
- **Coop tier** ⇒ HK **shared tile** in LDS, loaded/stored with HK's bank-conflict-free swizzled access.
- **Global tier** ⇒ HK's **load/store operators across the hierarchy** (HBM→LDS→register) with async buffer
  loads.

The catch: HK's compute operators are **matrix-core (MFMA) centric** — GEMM and attention. A quantum gate is
a 2×2 (or 4×4) complex butterfly, *not* an MFMA-shaped contraction, so we would **not** use HK's `mma`
operator for the sweep. We use HK's **tile data-movement + layout + vector ops**, and write the butterfly as
a custom bulk operator over the tile. This is honest: **HK's tile *container and memory primitives* transfer;
its *matrix-core compute* mostly does not** (unless we reformulate multi-qubit gate application as small
GEMMs, which is an open question below).

### 2.2 Register/LDS/HBM movement, spill avoidance, occupancy (§2.1, §3.2, §3.3) — fixes V1's 115 KB spill

This section is a direct antidote to V1's global-tier 115 KB/thread spill.

- **Register budget (§2.1):** each SIMD has **512×32-bit VGPRs = 512 KB/CU**; with one wave/SIMD the hardware
  splits into 256 VGPR + 256 AGPR. V1 spilling 115 KB/thread means it blew the per-thread register budget
  many times over — a direct symptom of unrolling. A **runtime interpreter with a bounded working set per
  operand** keeps live registers tiny (only the current gate's operands + loop state), structurally
  preventing the spill.
- **Developer-controlled register pinning (§3.2.1):** HK lets you **pin register tiles to explicit
  registers**, bypassing HIPCC's allocator, and even use **AGPRs as matrix inputs** (which HIPCC refuses,
  forcing redundant `v_accvgpr_read`). Pinning took their MHA-backwards kernel from 855→1024 TFLOPS
  (Table 1), matching AITER assembly. For V2 this is the tool to **guarantee** the register-tier statevector
  stays in VGPRs and never spills.
- **Async buffer loads to LDS (§3.2, Appendix):** HK uses **direct buffer loads** (`buffer_load` →
  LDS) to hide latency and address-gen; it notes buffer loads aren't even Triton's default on AMD, a reason
  HK beats Triton. V2's global tier should stream statevector tiles HBM→LDS with these.
- **Swizzling (§3.2.2, Fig. 4):** bank-conflict-free LDS layouts via a swizzle that swaps column halves. The
  coop-tier LDS statevector must be swizzled the same way to avoid the bank conflicts that would otherwise
  serialize the 256-thread cooperative access.

### 2.3 Programming model — header library, write-once-per-operand (§3.1, blog)

HK **is a C++ header library** in the ThunderKittens lineage (blog: "C++ embedded", attention fwd ≈ 500 LOC,
GEMM loop < 100 LOC; "we might not need raw assembly for peak AMD kernels any more"). Interface is separated
from implementation: the **tile API is uniform**, while the **swizzle/register-scheduling underneath differs
per arch** (blog).

**This is the strongest single idea for V2's maintainability:** write each of the ~40 per-operand device
functions **once**, as a bulk operator over an HK-style tile, and have the **same source compile for all three
tiers** by instantiating over a register-tile vs shared-tile vs HBM-streamed-tile type. That is exactly HK's
"same tile interface, different memory backing" model. It kills the V1 problem of divergent per-tier code and
per-circuit codegen in one move.

### 2.4 Wavefront-cooperative reductions/broadcasts — fixes V1's hand-rolled ds_bpermute bugs

The measurement step needs a **cooperative probability reduction** (sum |amp|² across the workgroup) plus a
**sampled-branch broadcast** (all threads learn the sampled outcome). V1 hand-rolled these with
`ds_bpermute`/`shfl` and hit bugs.

HK provides **tile reductions as bulk operators** (`row_reduce`/`col_reduce`-style, the PyTorch-inspired
reduction set, §3.1) that are **layout-aware**: they know which thread owns which element of a register tile
(§ register-layout discussion, Fig. 20) and emit the correct cross-lane shuffle/`ds_permute` sequence for
that layout. HK's whole thesis is that **getting these cross-lane ops right by hand is exactly where people
introduce bugs**, so they encapsulate it once. HK's online-softmax path (§ attention, line ~760) does
precisely a cooperative max/sum reduction interleaved with compute — the same primitive the measurement
reduction needs.

**Mapping:** back the coop-tier statevector with an HK shared/register tile and use HK's **reduction operator
for the probability sum** and a **tile broadcast for the sampled outcome**, instead of hand-rolled
`ds_bpermute`. **Honest caveat:** HK's reductions are tuned for MFMA register layouts; the quantum
statevector layout may not match a native MFMA layout, so we'd either (a) adopt an MFMA-compatible amplitude
layout, or (b) reimplement the reduction using HK's *layout metadata* but our own indexing. Either way HK
gives a **correct, tested reference** for the cross-lane sequence — reimplement-with-reference, not
drop-in.

### 2.5 Scheduling: 8-wave ping-pong / 4-wave interleave, XCD swizzle (§3.3–3.4)

HK's two scheduling patterns (§3.3.2): **8-wave ping-pong** (two waves/SIMD alternate compute vs memory via a
conditional barrier — balanced workloads, large tiles) and **4-wave interleave** (one wave/SIMD, staggered
compute+memory — imbalanced workloads, small tiles). Plus **chiplet/XCD swizzling** (Algorithm 1, Table 4):
schedule consecutive workgroup IDs onto the same XCD (32 CUs, shared 4 MB L2) to maximize cache reuse.

**Relevance to V2:** the amplitude sweep is **memory-bound** (streaming amplitudes), not MFMA-balanced, so
**4-wave interleave** (memory-heavy staggering) is the closer template for the global tier. The **XCD swizzle
matters for the global-tier work-stealer**: assign contiguous shot/tile ranges to workgroups on the same XCD
so the statevector stays L2-resident. **Honest caveat:** ping-pong assumes a compute (MFMA) vs memory split;
our sweep is mostly memory + light ALU butterflies, so ping-pong's compute wave is under-fed. Treat these as
*inspiration for overlap structure*, not as directly reusable schedules. Note also the blog's finding that
**wave specialization underperforms on CDNA3/CDNA4** — a caution against over-engineering the coop tier with
producer/consumer wave roles.

---

## 3. Synthesis — what V2 should adopt, ranked

**Tier 1 — adopt directly, foundational:**

1. **Persistent interpreter kernel** (IRIS §4.2.1 mega-kernel + grid-stride loop). One resident workgroup
   per CU; outer pulled loop over shots/tiles, inner sequential loop over the circuit's operand bytecode with
   runtime `switch(opcode)` → per-operand device function. *This is the entire V2 thesis and it pays none of
   the Three Taxes.* **Pattern to reimplement (C++/HSA).**
2. **SC-HRF scoped-atomics/fence discipline** (IRIS §2.1.3): workgroup-scope barriers for coop-tier
   operand-stage sync (fixes divergent-barrier hangs by converging all threads before each barrier);
   agent-scope atomics for global-tier work-stealing. **Directly usable model** via `__hip_atomic_*` /
   `__builtin_amdgcn_fence` with `"workgroup"`/`"agent"` scopes.
3. **HK tile abstraction as the per-tier statevector container** (HK §3.1–3.2): register-tile (register tier)
   vs shared-tile+swizzle (coop tier) vs HBM-streamed tile (global tier), with **explicit register pinning**
   to kill the 115 KB spill. **Directly usable header library** for containers + movement + layout;
   **reimplement** the gate butterfly as a custom bulk operator (not MFMA).

**Tier 2 — adopt, high value:**

4. **Write-once per-operand device-function library** in HK style (§2.3): ~40 operand functions written once
   as bulk tile operators, instantiated across all three tiers. Kills per-circuit codegen and per-tier code
   duplication.
5. **HK cooperative reduction/broadcast for measurement** (HK §3.1, §2.4 above): replace hand-rolled
   `ds_bpermute`/`shfl` with HK's layout-aware reduction + broadcast. **Reimplement-with-reference** (HK gives
   the correct cross-lane sequence).
6. **Global-tier work-stealing via a single agent-scoped atomic work-index** (IRIS §4.2.2 mechanism, minus
   the producer/consumer role split) + **XCD-swizzled tile assignment** (HK §3.4) for L2 reuse.

**Tier 3 — inspiration only, verify before use:**

7. **4-wave interleave overlap** (HK §3.3.2) for the memory-bound global-tier sweep — template, not drop-in.
8. **Async buffer loads HBM→LDS** (HK §3.2) for global-tier statevector streaming.

**Explicitly NOT applicable:**

- **All of IRIS's actual code** — Triton/Python, multi-GPU symmetric heap, `iris.put/get/store`, cross-GPU
  agent/system atomics. V2 is single-GPU C++/HSA. We take IRIS's *structure and memory model*, zero lines of
  its code.
- **HK's MFMA/matrix-core compute operators** — a 2×2/4×4 gate butterfly is not an MFMA contraction. Take
  HK's tile *containers/movement/layout/reductions*, not its `mma`.
- **HK's 8-wave ping-pong** — assumes balanced MFMA-vs-memory; our sweep under-feeds the compute wave.
- **Wave specialization** generally — HK reports it *underperforms* on CDNA3/CDNA4.

**Directly-usable-library vs reimplement summary:** *Library:* HK tile containers, memory/swizzle
primitives, register pinning, reductions (single-GPU C++ header, gfx950 — can vendor into V2). *Reimplement:*
the persistent interpreter loop, work-stealer, and scoped-sync — patterns from IRIS, but IRIS's code doesn't
transfer.

---

## 4. Open questions for the V2 ultrathink

1. **Can multi-qubit gate application be reformulated as small GEMMs** to actually use HK's MFMA path (and
   the MI350X matrix cores), or is the butterfly fundamentally the wrong shape? If yes, register/coop tiers
   could ride HK's fastest primitives; if no, we only use HK's memory layer.
2. **Statevector amplitude layout vs MFMA register layout** — do we adopt an MFMA-compatible layout so HK's
   reductions/swizzles apply verbatim, or keep a natural qubit-index layout and reimplement the cross-lane
   ops using HK's layout metadata? (§2.1, §2.4.)
3. **Runtime `switch(opcode)` dispatch cost vs divergence** — 40 operand types in a hot inner loop: does the
   branch table serialize the wave? Does an opcode-sorted bytecode (group like operands) or a computed-goto /
   function-pointer table reduce divergence? IRIS/HK don't answer this (they have no runtime dispatch).
4. **Coop-tier barrier granularity** — one workgroup-scope barrier per operand stage may be too many for a
   16k-operand circuit. Can we batch operands between barriers when they touch disjoint amplitudes? SC-HRF
   scope discipline (§1.3) tells us the barriers are *correct*; it doesn't tell us the minimal count.
5. **Global-tier work-stealing granularity** — steal per shot, per amplitude-tile, or hierarchically (XCD
   owns a shot range, CUs within steal tiles)? IRIS's `pid`-role model is coarse; the right granularity for
   heterogeneous circuit tail-latency (IRIS's own §4.2.1 warning) is unresolved.
6. **Do we vendor HK or reimplement its primitives?** HK is header-only C++ for gfx950 (good), but MFMA-
   centric and research-grade (Nov 2025). Vendoring gets swizzle/pinning/reductions for free; reimplementing
   gets a lean quantum-specific subset. Cost/benefit unclear until Q1–Q2 are answered.
7. **Tail-latency mitigation** (IRIS §4.2.1) — with persistent kernels the last workgroup gates completion.
   For wildly heterogeneous circuit depths, is dynamic work-stealing enough, or do we need work-splitting
   (subdivide the longest circuit's shots across idle workgroups)?

---

## Sources

- IRIS paper: https://arxiv.org/abs/2511.12500 (full PDF read; §2.1.3 memory model, §3.3.3 atomics API,
  §4.1–4.2 fused/unfused taxonomy, Listings 3–5, Figs. 3–5)
- "Three Taxes" paper: https://arxiv.org/abs/2511.02168 (full PDF read; §2.3.1 Kernel Launch Overhead, §2.3.2
  Bulk Synchronous, §2.3.3 Inter-Kernel taxes, Fig. 2)
- IRIS repo/README: https://github.com/ROCm/iris (API, supported GPUs MI300X/MI350X/MI355X, Gluon backend)
- HipKittens paper: https://arxiv.org/abs/2511.08083 (full PDF read; §2.1 HW hierarchy, §3.1 tile interface,
  §3.2 register pinning/swizzle, §3.3 8-wave/4-wave schedules, §3.4 XCD swizzle, Tables 1–4, Fig. 4)
- HipKittens repo/README: https://github.com/HazyResearch/HipKittens (CDNA3/CDNA4, gfx950 docker, primitive
  list, AITER backend landing)
- HipKittens blog: https://hazyresearch.stanford.edu/blog/2025-11-09-hk (tile generalization, C++ model, wave
  specialization underperforms on CDNA3/4, interface-vs-implementation split)
