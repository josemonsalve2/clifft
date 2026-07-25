# V2 GPU Backend — Optimization Plan (Claude)

**Date:** 2026-07-25 · **Node:** mi350x-es (gfx950 / MI355X / CDNA4, 64 KB LDS/CU)
**Kernel:** `src/clifft/gpu/mlir/v2/coop_interpreter.c` (`execute_shot`, `clifft_v2_coop`, `clifft_v2_global`)
**Host driver:** `src/clifft/gpu/mlir/v2/v2_kernel.cc`
**Correctness contract:** byte-exact `passed_shots` + `observable_ones` vs GPU-SVM (`run_gpu --no-postselection`, same seed).

This plan is analysis + planning only. No source was modified.

---

## 0. The one-paragraph problem statement

V2 is a runtime bytecode interpreter where **one 256-thread workgroup cooperates on ONE shot**.
The measured profile is *inverted from intuition* (SYNTHESIS.md:6-20):

| circuit | rank | tier | V2/SVM | why |
|---|---|---|---|---|
| frame_h | 0 | reg→coop | **28.3×** slower | 256 threads spun up so tid0 twiddles Pauli bits; 255 idle |
| circuit_d3 | 4 | reg→coop | **15.2×** slower | 16-amplitude statevector, 256-thread WG, ~barrier-per-op |
| qv10 | 10 | coop | 1.31× slower | LDS=25 KB caps occupancy at 2 wg/CU; 500× more L2 misses |
| surface_d7_t15 | 10 | coop | 1.65× slower | same, measurement/reduce-heavy |
| surface_d9_t10 | 7 | coop | 1.64× slower | same |
| surface_d9_t19 | 13 | global | **0.91× (WINS)** | amps in HBM → LDS only 8.7 KB → high occupancy |
| surface_d11_t15 | 11 | global | **0.94× (WINS)** | same |

Root cause is **not compute** (qv10: V2 executes *fewer* VALU 964M vs 1128M, *fewer* LDS-insts 60.9M vs 89.2M; MFMA=0; VGPR=64) — it is **occupancy + barrier/serial latency**: V2 burns 32% more wave-cycles (11.9B vs 9.0B, qv10.json:29/63) and **555× more L2 misses** (TCC_MISS 17.37M vs 31.3K, qv10.json:58/59). The global tier already proves the fix: move amplitudes out of LDS → occupancy rises → V2 wins. The whole plan follows from three levers: **(A) stop launching 256 threads per low-rank shot, (B) reclaim coop LDS to double occupancy, (C) cut barriers/tid0-serial regions.**

---

## 1. Prioritized optimization ranking

Score = (expected speedup on affected circuits) × (fraction of 22-circuit suite affected) × (1 / risk-effort).
Tier populations are from the memory note "Tier classification" (331 register, 11 coop, 6 global circuits after `StatevectorSqueezePass`) and the 8-circuit profiled sample. **The register tier is by far the largest population and carries the largest per-circuit gap (15-28×) — it dominates the score.**

| # | Optimization | Expected speedup | Suite fraction | Risk/effort | Correctness | Priority |
|---|---|---|---|---|---|---|
| **P0** | **Shot-packed register kernel (rank ≤ 6): 1 shot/thread like SVM** | **5-25×** on rank≤6 | ~85% of circuits (331/~348 register) | MEDIUM-HIGH (new kernel + host route) | **SAFE** (same math, per-thread) | **HIGHEST** |
| **P1** | **Coop LDS reclamation → 2 wg/CU → 4 wg/CU** (shrink `lds_meas`, `lds_red_scratch`; dynamic-size) | 1.5-2× on coop | ~11 coop circuits | LOW-MEDIUM | **SAFE** (layout only) | HIGH |
| **P2** | **Barrier + tid0-serial reduction** (fuse `active_k++`, batch frame ops, trim reduce2 to 2 barriers) | 1.15-1.6× on frame/meas-dense | coop + register + global | MEDIUM | **SAFE** if summation order preserved | HIGH |
| **P3** | **Two-pass measurement fusion** (share reduce/fold streaming pass) | 1.1-1.3× on meas-heavy | coop + QEC | MEDIUM | SAFE (fold is post-branch) | MEDIUM |
| **P4** | **Selective FMA restore** (`-ffp-contract=fast` on U2/U4/H amplitude math only) | 1.2-1.4× on U-gate-heavy (qv10) | coop + some global | MEDIUM | **RISKY** (branch determinism) | MEDIUM |
| **P5** | **f64→f32 in branch-independent paths** (`cscale`, ARRAY_H fold, post-branch INTERFERE fold) | 1.1-1.25× | coop + global | MEDIUM | **RISKY** (branch desync) | MEDIUM-LOW |
| **P6** | **ARRAY_U4 register/AGPR pressure** (split 4×4 into two 2-row passes) | 1.05-1.2× on U4-heavy | few | MEDIUM | SAFE (associativity preserved) | LOW |
| **P7** | **`coop_reduce2` `ds_bpermute` packing / DPP** (16→~8 shuffles) | 1.05-1.15× on reduce-heavy | coop QEC | MEDIUM-HIGH | **RISKY** (exact sum order) | LOW |
| **P8** | **Global-tier XCD-interleaved work-steal** (deferred; v2_kernel.cc:82-105/878-880) | 1.0-1.1× on global | 6 global circuits | MEDIUM | SAFE (each shot once) | LOW (protect the win first) |

**Why P0 outranks everything:** the register tier is ~85% of the suite AND carries a 15-28× gap. Even a conservative 8× recovery on 331 circuits dwarfs a 2× win on 11 coop circuits. P1 is the highest-leverage *coop* move and the cheapest to implement. P4/P5/P7 are the risky precision moves — deliberately ranked below the structural moves because the structural bottleneck is latency, not VALU count (qv10 already does *fewer* VALU instructions than SVM yet loses).

---

## 2. The low-rank catastrophe (P0) — concrete design

### 2.1 Diagnosis, quantified
For frame_h (rank 0), V2 launches Grid_Size_X = 5,120,000 = 20000 shots × 256 threads (frame_h.json:14), i.e. **80,000 waves** (SQ_WAVES 80000) to do work that SVM does with **316 waves** (frame_h.json:57) — one thread per shot, 20224 threads. V2 issues 26.2M VALU + 22.8M SALU instructions and 900K LDS instructions for what is *pure tid0 Pauli-frame bookkeeping* (all FRAME_* ops, no amplitude math): 255/256 lanes sit at `s_barrier` the entire kernel. This is a **255× thread-waste factor**, and it is the mechanism behind the 28× wall-clock gap. circuit_d3 (rank 4, 16 amplitudes) is the same pathology with a tiny amount of real work added.

### 2.2 Recommendation: **(a) a shot-packed register kernel + (c) dispatch routing by rank.**
Reject (b)-as-primary (a separate register kernel that still cooperates) — the problem is *cooperation itself* at low rank, not the kernel identity. The fix is to abandon 256-threads-per-shot and adopt SVM's embarrassingly-parallel model for `peak_rank ≤ kRegPackMax` (propose **kRegPackMax = 6**, statevector ≤ 64 amps = 512 B in registers/private).

**New kernel `clifft_v2_register` — 1 shot per thread:**
- Grid = `ceil(shots/256)*256` threads; **each thread owns a whole shot** (like `sample_kernel` in the SVM reference, frame_h.json:38 — note SVM's register path uses **LDS_Block_Size=0**, all state in registers/scratch).
- Statevector: a per-thread `CV2Complex vloc[1u << kRegPackMax]` in **registers/private** (≤ 64 complex = 128 f32). No LDS `lds_v`. No `barrier()`. No `coop_reduce2` (the per-thread norm is a plain scalar sum over ≤64 amps — no cross-lane shuffle, no 3-barrier reduction).
- The Pauli frame, RNG, meas records become **per-thread locals**, not LDS singletons. `rng_seed(seed, shot_id)` with `shot_id = global_thread_id` reproduces SVM's per-shot stream seeding exactly (rng_seed at coop_interpreter.c:74 already keys on shot_id).
- **`execute_shot` refactor:** hoist the opcode `switch` body into an `execute_shot_scalar()` that takes `CV2Complex* v` (private), local `u64 px[W]/pz[W]`, local rng, and **compiles out every `if(t==0)`/`barrier()`** (they become unconditional scalar code because there is exactly one thread per shot). The two bodies share the *same opcode semantics* — keep one source-of-truth by making `barrier()` and the `t==0` guard no-ops in the scalar build (e.g. `#ifdef V2_SCALAR` → `#define barrier()` empty, `#define IS_OWNER 1`; strided loops `for(i=t;i<N;i+=256)` collapse to `for(i=0;i<N;i++)` when `t≡0, stride≡1`). This preserves the "written once" invariant (coop_interpreter.c:262) and the exact arithmetic, so it is **byte-exact-SAFE by construction**.

**Host dispatch changes (`v2_kernel.cc`):**
- Add a third tier band at line 52-60:
  ```
  constexpr uint32_t kRegPackMax = 6;   // NEW
  enum Tier { REG, COOP, GLOBAL };
  Tier tier = flat.peak_rank <= kRegPackMax ? REG
            : flat.peak_rank <= kCoopMaxRank ? COOP : GLOBAL;
  const char* sym = tier==REG ? "clifft_v2_register"
                  : tier==GLOBAL ? "clifft_v2_global" : "clifft_v2_coop";
  ```
- For REG: `grid = ceil(shots/256)*256`, `block = 256`, **one shot per thread** (contrast the coop path's `grid = shots*256` at v2_kernel.cc:150). Kernarg layout is the coop prefix (v2_kernel.cc:151 already truncates to `offsetof(...,global_v)`), so no global buffers.
- No LDS amplitude buffers to size; `group_segment_size` for this kernel is ~the classical frame words only (a few hundred bytes), so occupancy is register-bound (VGPR 64→~128 for 64 amps still allows ≥4 waves/SIMD).

**Expected:** collapses the 255× thread-waste to 1× → frame_h from 28× to ~1× (parity with SVM's own register kernel), circuit_d3 from 15× toward ~1-2×. This is the single biggest absolute win in the suite.

**Risk note:** register pressure at kRegPackMax=6 (64 complex amps = 128 VGPR of state + interpreter temporaries) may force scratch spills. Mitigation: start kRegPackMax at **4** (16 amps, matches CLAUDE.md register tier boundary and the existing `Stack alloca (≤16)` design), measure VGPR/scratch via `check_registers(binary_path=<.hsaco>)`, then raise to 5-6 only if spill-free. SVM's own register kernel runs this way with VGPR=60, Scratch=4480 (frame_h.json:46-47) — a scratch-backed statevector is acceptable and still 15-28× faster than the coop path.

---

## 3. Coop-tier LDS reclamation (P1) — concrete steps

**Current LDS budget (25088 B = 2 wg/CU), from the static char. §2 / device_abi lines 44-58:**

| global | bytes | line | needed? |
|---|---|---|---|
| `lds_v[1024]` | 8192 | :44 | **yes** (the statevector) |
| `lds_red_scratch[1024]` | 8192 | :45 | only SWAP_MEAS_INTERFERE fold (coop_interpreter.c:702-709) |
| `lds_meas[4096]` | 4096 | :46 | worst-case cap; typical use ≪ 4096 |
| `lds_red0[256]`+`lds_red1[256]` | 4096 | :55-56 | yes (reduce2 partials) |
| others (px/pz/rng/obs/…) | ~142 | :47-58 | yes |

### Step 3.1 — Dynamic-size `lds_meas` via kernarg [cheapest, ~3 KB saved]
`total_meas_slots` is already a kernarg (v2_kernel.cc:123, used at coop_interpreter.c:293 to bound init zeroing). But the LDS *reservation* is static 4096 B because `lds_meas` is an `extern addrspace(3)` array. Two implementation options:
- **(preferred) Dynamic LDS:** move `lds_meas` (and `lds_red_scratch`, below) out of static `extern` arrays into a single **dynamically-sized LDS block** passed via the AQL packet's `group_segment_size` (HsaLoadedKernel.group_segment_size, hsa_kernel_dispatch.h:28). The host sets `group_segment = base_static + round_up(total_meas_slots) + reduce_scratch_if_needed`. In the kernel, replace the fixed arrays with pointers into an `extern __attribute__((address_space(3))) u8 lds_dynamic[]` region indexed by computed offsets. This is the same mechanism HIP's `extern __shared__` uses; it decouples reservation from the 4096 cap.
- **(fallback, no ABI change) static shrink:** cut `V2_MAX_MEAS` (coop_interpreter.c:43) from 4096 to the true coop max (measurements per coop circuit are tens–low-hundreds per the static char §2). At 512 it saves 3.5 KB immediately with zero host change; guard with the existing `ins.a < V2_MAX_MEAS` checks (already present at :293/:355/:411).

### Step 3.2 — Reclaim `lds_red_scratch` (8 KB) [biggest single win]
It is used **only** by `OP_SWAP_MEAS_INTERFERE`'s strided fold (coop_interpreter.c:702-709). Every other coop circuit reserves 8 KB it never touches. Two paths:
- **In-place fold:** SWAP_MEAS's fold writes `scratch[idx]` then copies back `v[idx]=scratch[idx]` (:706-709). The reason for the temp is that the read pattern is strided (`base` weaves `from`/`to` bits) while the write is contiguous — a true scatter that can alias. If the permutation can be shown non-aliasing for the contiguous lower-half write (it maps `half=2^to` distinct sources to `idx∈[0,half)`), the copy can be done in place with one barrier, eliminating the 8 KB entirely.
- **Dynamic (safe) path:** fold `lds_red_scratch` into the same dynamic-LDS block as §3.1, allocated **only when the circuit contains a SWAP_MEAS opcode** (a host-side scan of `flat.instrs` for `OP_SWAP_MEAS_INTERFERE`). Circuits without it pay 0 KB.

### Step 3.3 — Occupancy result
LDS 25088 B → **~13 KB** (`lds_v` 8 KB + `lds_red0/1` 4 KB + ~1 KB classical + right-sized meas). `floor(64 KB / 13 KB) = 4 wg/CU` → **2 wg/CU → 4 wg/CU (2× occupancy)**, 16 waves/CU. The global tier (LDS 8704 B, surface_d9_t19.json:5) already runs at this occupancy and *wins*, which is the empirical proof this closes the coop gap. **Expected: qv10 1.31× → ~0.9-1.0×; surface coop 1.6× → ~1.0-1.2×.**

---

## 4. Barrier + tid0-serial reduction (P2/P3)

Static char §1 counts **~52 barrier sites** and **~34 `if(t==0)` regions**. Concrete, low-risk cuts:

- **Fold the `active_k++` barrier (P2, LOW risk).** EXPAND/EXPAND_T/EXPAND_ROT each use **2 barriers** (post-copy + post-`active_k++`, coop_interpreter.c:375-377, 389-391, 593-595). The `active_k++` is a single tid0 store that only needs to be visible before the *next* op reads it. Move the increment into the tail of the copy loop's owning thread and drop the second barrier → −1 barrier per expansion. **Byte-exact SAFE** (no arithmetic touched).
- **Batch consecutive FRAME_* ops (P2, MEDIUM).** Runs of FRAME_CNOT/CZ/H/S/SWAP (coop_interpreter.c:304-343) each end in a barrier though they only mutate `lds_px/lds_pz` under tid0. A peephole in `flatten_program` (host) can coalesce a maximal run of frame-only ops into a single `OP_FRAME_BLOCK` that tid0 executes then barriers **once**. Cuts barriers on frame-dense QEC circuits substantially. SAFE (frame algebra unchanged, order preserved).
- **Trim `coop_reduce2` from 3 barriers to 2 (P2, MEDIUM; verify).** The trailing barrier at :206 exists so all threads re-read the broadcast `lds_red0[0]`. If the broadcast is delivered via a `readfirstlane`/wave-uniform path instead of an LDS round-trip, the final barrier can drop. **Preserve the exact butterfly summation order (:190-201)** — that is the documented determinism invariant (:182-186). Only the *broadcast* mechanism changes, not the *accumulation*, so this is SAFE if the partials are bit-identical.
- **Two-pass measurement fusion (P3, MEDIUM).** MEAS_ACTIVE_DIAGONAL/INTERFERE do reduce (6 barriers incl. reduce2) → sample → fold (2 more). The fold *must* wait for the branch decision, but the norm-accumulation pre-pass and the fold both stream the same `v[i]/v[i+half]` pairs (:400-403 vs :447-451). Cache the loaded pairs in registers across the reduce so the fold does not re-load from LDS. Cuts LDS traffic (the 500× L2-miss driver) without changing barrier-ordered arithmetic. SAFE.

**Expected P2+P3 combined: 1.15-1.6× on frame/measurement-dense circuits** (surface codes, which are the coop 1.6× losers).

---

## 5. SAFE vs RISKY for byte-exactness

The kernel documents three hard invariants. Classify every optimization against them:

**Documented invariants:**
1. **f64 `cscale`** — scalar multiply in f64 then narrow to f32 (coop_interpreter.c:122-124). "Do NOT simplify to an f32 multiply; that desyncs measurement branches."
2. **`ds_bpermute` reduction summation order** — coop_reduce2's exact butterfly 32→1 then 4-warp 2→1 (:182-186). Diverges f64 rounding at branch points on rank-10 QEC if reordered.
3. **`-ffp-contract=off`** — forbids FMA fusion globally (ClifftAmdgcn.cmake); every `cmul` is 4 mul + 2 add, not fused.

| Optimization | SAFE / RISKY | Reason |
|---|---|---|
| P0 shot-packed register kernel | **SAFE** | identical opcode arithmetic per thread; barriers/reductions removed are no-ops at 1 thread/shot; rng_seed(shot_id) already per-shot |
| P1 LDS shrink / dynamic-size | **SAFE** | pure storage layout; no arithmetic, no order change |
| P2 fold `active_k++` barrier, batch frame ops | **SAFE** | scheduling/visibility only; frame algebra and order preserved |
| P2 reduce2 3→2 barriers | **SAFE only if** partials bit-identical | must keep the :190-201 accumulation order; change *broadcast* not *sum* |
| P3 measurement fold fusion | **SAFE** | fold is strictly post-branch; caching loaded pairs doesn't reorder the reduction |
| P6 U4 two-pass split | **SAFE** | preserves add associativity of the 4-term sums (:633-640); no FMA introduced |
| **P4 FMA restore** (`-ffp-contract=fast`) | **RISKY** | violates invariant #3; changes rounding of `cmul` chains. Gate: apply ONLY to U2/U4/H amplitude math (state, not probability) and **prove per-circuit** `passed_shots`+`observable_ones` match; keep OFF for cnorm/coop_reduce2/cscale/sample_branch |
| **P5 f64→f32** in `cscale`/folds | **RISKY** | violates invariant #1 directly. `cscale` feeds state renorm after a branch is chosen — arguably branch-independent, but the *next* measurement's reduction consumes that state, so an f32 fold can perturb a *downstream* branch. Gate hard: per-circuit exact-match sweep, revert on any mismatch |
| **P7 reduce2 shuffle packing / DPP** | **RISKY** | must reproduce invariant #2's exact summation order; DPP/permlane may change lane pairing |
| P8 XCD work-steal | **SAFE** | each shot processed once (v2_kernel.cc:876-879); result is order-independent atomic add |

**Rule for the RISKY set (P4/P5/P7):** never land without a full 22-circuit exact-match sweep. These attack VALU/shuffle count, but the profile shows V2 is **latency-bound, not compute-bound** (qv10 does fewer VALU than SVM yet loses) — so **do the SAFE structural work (P0/P1/P2/P3) first and re-measure before spending correctness risk on P4/P5/P7.** They may become unnecessary.

---

## 6. Validation plan

Reuse `V2_performance/tools/profile_sweep.sh` (SLURM, mi350x-es, 3 PMC passes already defined). For every optimization: (1) **correctness gate first** — `run_v2` vs `run_gpu --no-postselection`, same seed, exact `passed_shots`+`observable_ones` on all 22 circuits; (2) then the counter proof below.

| Opt | Prove-it circuits | rocprofv3 counter that must move | pass criterion |
|---|---|---|---|
| **P0** register kernel | frame_h (r0), circuit_d3 (r4) | kernel-trace **total_kernel_ns**; **SQ_WAVES** (frame_h: 80000→~316 like SVM); **LDS_Block_Size** (25088→~0) | V2/SVM → ≤ ~1.5× (from 28×/15×) |
| **P1** LDS reclamation | qv10, surface_d7_t15, surface_d9_t10 | **LDS_Block_Size** (25088→~13000); occupancy via waves-in-flight; **SQ_WAVE_CYCLES** (qv10 11.9B→↓); **TCC_MISS_sum** (17.4M→↓) | wave-cycles drop ≥20%; V2/SVM ≤ 1.1× |
| **P2** barrier/serial | frame-dense surface_d9_t10; qv10 | **SQ_WAIT_INST_LDS**; **GRBM_GUI_ACTIVE** (qv10 99.3M vs SVM 75.5M → toward SVM); total_kernel_ns | GUI_ACTIVE gap −30%+ |
| **P3** meas fusion | surface_d7_t15, surface_d9_t10 (meas-heavy) | **SQ_INSTS_LDS** (60.9M→↓); **TCC_MISS_sum** | LDS insts −15%+, no correctness regression |
| **P4** FMA restore | qv10 (U-gate-heavy) | **SQ_INSTS_VALU** (964M→~500-700M) | VALU −25%+ AND exact match |
| **P5** f64→f32 | qv10, surface_d9_t10 | SQ_INSTS_VALU, SQ_WAVE_CYCLES | speedup AND exact match on all 22 |
| **P6** U4 split | a U4-heavy circuit | **Accum_VGPR_Count** / `v_accvgpr` disasm traffic (currently nonzero despite MFMA=0, qv10.json/TOOLING_STATUS.md) | AGPR writes → 0 |
| **P7** reduce2 packing | surface_d9_t10 (reduce-heavy) | SQ_INSTS_LDS (ds_bpermute count) | shuffle count halved AND exact match |
| **P8** XCD steal | surface_d9_t19, surface_d11_t15 | total_kernel_ns; per-XCD balance | ≥ current 0.91×, no regression |

**Counter sources already collectable** (profile_sweep.sh:26-28): PMC_A (SQ_WAVES/VALU/MFMA/SALU/LDS), PMC_B (TCC_HIT/MISS), PMC_C (SQ_BUSY_CYCLES/GRBM_GUI_ACTIVE/SQ_WAIT_INST_LDS/SQ_WAVE_CYCLES). FETCH/WRITE_SIZE are **not** collectable on this gfx950 config (profile_sweep.sh:23-25) — use TCC L2 traffic as the memory proxy. Add frame_h + circuit_d3 to the P0 rows (already in the CIRCS list, profile_sweep.sh:31-32). Extend the 8-circuit sweep to the full 22 for the final correctness gate.

**Register/LDS per-iteration check:** lift `KernelForge/.../mcp_server/tools/registers.py:check_registers(binary_path=<.hsaco>)` (agentic_integration_guide.md §3) to read VGPR/AGPR/LDS/spill directly from the `.hsaco` after each build — this is the cheapest way to confirm P0 doesn't over-spill and P1 actually shrank `group_segment_size`.

---

## 7. Agentic-loop recommendation: **manual first, then wire KernelForge for P4/P5/P7 tuning.**

**Recommendation: do P0-P3 by hand; do NOT wire the forge-loop yet.** Justification from the data:

1. **The top wins are structural, not search-space tuning.** P0 (a new kernel + a host dispatch branch) and P1 (an LDS layout/dynamic-LDS change spanning kernel + `v2_kernel.cc` + the AQL `group_segment_size`) are *architectural edits across multiple files*. The KernelForge loop edits **one anchor file** (`--kernel <f>`, agentic_integration_guide.md §1.1) and keeps/reverts on a bench delta — it is built for local, single-file micro-optimization, not for adding a kernel and rerouting the host. It cannot express P0.
2. **The bottleneck is understood.** The value of an agentic loop is exploring an *unknown* optimization space. Here the profile already names the bottleneck precisely (occupancy/barriers/thread-waste, not compute) and the global tier already demonstrates the fix. Spending LLM-in-the-loop cost to rediscover "move data out of LDS" is wasteful.
3. **The correctness contract is exact-integer, and the loop's gate is a stdout regex** (`allclose: True`, agentic_integration_guide.md §1.2). That's fine, but the RISKY precision moves (P4/P5/P7) are *exactly* the class where an automated keep/revert loop with a per-circuit exact-match gate adds real value — a large, tedious search over which opcodes tolerate FMA/f32 without desyncing any of 22 circuits. **That is where the forge-loop earns its cost.**

**Concrete sequencing:**
- **Now (manual):** land P0, P1, P2, P3. Re-run the sweep. This is where 90% of the win is and it needs human architectural judgment.
- **Then (agentic):** once the structural refactor is stable and byte-exact, stand up KernelForge **Option A** (agentic_integration_guide.md §1.5): one `forge_driver.py` that (a) rebuilds `build-v2-nohip`, (b) correctness mode prints `allclose:` from the integer-tally comparison, (c) `--bench-mode` runs `rocprofv3 --kernel-trace` and prints `median_ms:`. Point `--kernel` at `coop_interpreter.c` and let it hunt the P4/P5/P7 precision-relaxation frontier under the 22-circuit exact-match gate. **Do the no-agent smoke test first** (§1.3 5-stage pipeline against our binaries) before paying for any LLM iterations.
- **Do not** invest in Apex (speedup measurement bound to Magpie, can't time our `.hsaco` — agentic_integration_guide.md §2.2/§2.3) or Accordo (CLI not installed, float-tolerance mismatch with exact-integer check).

---

## 8. Sequenced execution summary

1. **P0** shot-packed `clifft_v2_register` (kRegPackMax=4→6) + host route in `v2_kernel.cc:52-60,150`. → kills the 15-28× register catastrophe (~85% of suite). *SAFE.*
2. **P1** coop LDS reclamation (dynamic `lds_meas` + reclaim `lds_red_scratch`) → 25 KB→13 KB → 2→4 wg/CU. → closes/inverts the coop 1.3-1.65× gap. *SAFE.*
3. **P2/P3** barrier + tid0-serial + measurement-fusion cuts. → 1.15-1.6× on frame/meas-dense coop. *SAFE.*
4. Re-measure the full 22-circuit sweep. Protect the global-tier win (P8 optional).
5. **Only then**, if VALU still matters, wire KernelForge Option A to hunt **P4/P5/P7** under a 22-circuit exact-match gate. *RISKY — automated gate is exactly the right tool.*

Commit at every step (memory: "No code deletion"; annotate perf-significant commits per "Commit style").
