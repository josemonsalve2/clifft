# V2 GPU Optimization — Unified Roadmap (Codex + Claude, reconciled)

**Date:** 2026-07-25 · Node: mi350x-es (gfx950 / MI355X / CDNA4)
Two independent planning passes (Codex gpt-5.3 high-effort, Claude) analyzed the
V2_performance database separately. This merges them. **They agree on the entire
top of the priority stack** — that agreement is itself a high-confidence signal.

## Where the two plans AGREE (high confidence — act on these)

1. **Root cause is occupancy/latency, NOT compute.** Both cite the qv10 counters:
   V2 executes *fewer* VALU (964M vs 1128M) and LDS (60.9M vs 89.2M) instructions
   than SVM yet loses, burning 32% more wave-cycles and ~500× more L2 misses. The
   global tier wins *because* its LDS is 8.7KB (amps in HBM) → high occupancy.

2. **P0 = the low-rank catastrophe fix, top priority.** Both independently
   propose a **new 1-shot-per-thread register kernel + rank-based dispatch
   routing**. frame_h (rank 0) launches 80,000 waves for tid0 Pauli-bit work SVM
   does in 316 waves — a 255× thread-waste. Both reject "cooperative register
   kernel"; the problem is *cooperation itself* at low rank.

3. **P1 = coop LDS reclamation.** Both target the same fat structures:
   `lds_red_scratch[1024]`=8KB (used ONLY by SWAP_MEAS_INTERFERE) and
   `lds_meas[4096]`=4KB (worst-case cap). Goal: 25KB → ~13KB → 2wg/CU → 4wg/CU.
   Codex adds a sharp extra: **`lds_red0/1[256]→[4]`** (coop_reduce2 only touches
   warp IDs 0-3 / `t<4` — coop_interpreter.c:189/194/196), a free 4KB→32B cut.

4. **P2 = barrier + tid0-serial reduction.** Both: fold the `active_k++` second
   barrier in EXPAND family (lines 375-377/389-391/593-595), batch consecutive
   FRAME_* ops into one barrier, trim `coop_reduce2` 3→2 barriers (preserving
   summation order). Both flag the exact same barrier line numbers.

5. **Correctness invariant classification is identical.** SAFE: P0, P1, P2
   structural. RISKY (gate behind full 22-circuit exact-match): FMA-restore
   (`-ffp-contract=fast`), f64→f32 in cscale/folds, reduce2 shuffle repacking.
   Both note V2 is latency-bound so the RISKY VALU-cutting moves may be
   *unnecessary* — do structural first, re-measure.

6. **Agentic loop: manual P0-P2 first, THEN KernelForge for the RISKY frontier.**
   Both: the forge-loop edits one file + keep/reverts on a bench delta — it
   CANNOT express P0 (new kernel + host reroute across files). But the tedious
   per-opcode FMA/f32 search under a 22-circuit exact-match gate is *exactly*
   where an automated loop earns its cost. Both reject Apex (Magpie can't time
   our .hsaco) and Accordo (not installed, float-tolerance mismatch).

## Where they DIFFER (minor — resolved)

- **kRegPackMax start value.** Claude: start at 4 (matches SVM's
  kThreadMaxPeakRank=4, 16 amps), raise to 6 only if spill-free, measure VGPR via
  check_registers. Codex: start rank≤4, extend to ≤6. → **Resolved: start at 4,
  gate expansion on measured VGPR/scratch.** (Identical intent.)
- **lds_meas shrink mechanism.** Claude prefers dynamic-LDS via AQL
  `group_segment_size` (decouples from the 4096 cap, no static waste). Codex
  prefers a bitset pack (u64[64] for 4096 bits = 512B) + static shrink fallback.
  → **Resolved: do the cheap static shrink (V2_MAX_MEAS 4096→512) + bitset FIRST
  (zero host/ABI change, immediate ~3.5KB), then dynamic-LDS later if more needed.**
  Codex's bitset and Claude's dynamic-LDS are complementary, not competing.
- **Priority scoring.** Codex scores P0 at 1.80 (dominates), P1 at 0.44. Claude
  ranks P0 highest by suite-population (331/348 register circuits per the tier-
  classification memory). Same ordering, same conclusion.

## THE UNIFIED ROADMAP (execution order)

| Step | What | Expected | Risk | Correctness | Source |
|---|---|---|---|---|---|
| **P0** | New `clifft_v2_register` kernel: 1 shot/thread, statevector in registers/private (rank≤4, extend to 6), NO barriers/reduction. Route by rank in v2_kernel.cc:52-60. Share opcode semantics via a scalar build of execute_shot (`barrier()`→noop, `t==0`→true, strided loops→scalar). | frame_h 28×→~1.5×; d3 15×→~2×. ~85% of suite. | MED-HIGH | **SAFE** (same math per-thread; rng_seed(shot_id) already per-shot) | both |
| **P1** | Coop LDS reclaim: `lds_red0/1[256]→[4]` (free 4KB), `V2_MAX_MEAS 4096→512` + bitset (~3.5KB), `lds_red_scratch[1024]→[512]` or reclaim entirely (SWAP_MEAS-only). 25KB→~13KB → 4wg/CU. | qv10 1.31×→~0.95×; coop 1.6×→~1.1×. 11 coop circuits. | LOW-MED | **SAFE** (layout only) | both |
| **P2** | Barrier cuts: fold `active_k++` barrier (EXPAND family), batch FRAME_* runs (host peephole → OP_FRAME_BLOCK), reduce2 3→2 barriers (keep sum order). | 1.15-1.6× frame/meas-dense. | MED | **SAFE** if summation order preserved | both |
| **P3** | Two-pass measurement fusion: cache v[i]/v[i+half] pairs in registers across reduce→fold so the fold doesn't re-load LDS (cuts the L2-miss driver). | 1.1-1.3× meas-heavy. | MED | **SAFE** (fold is post-branch) | Claude(P3), Codex(P2) |
| **P4** | ARRAY_U4 AGPR/live-range: split 4×4 into two 2-row passes (kills v_accvgpr traffic seen with MFMA=0). | 1.05-1.2× U4-heavy. | MED | SAFE (associativity preserved) | both (Codex P3, Claude P6) |
| — RE-MEASURE full 22-circuit sweep here — | | | | | |
| **P5** | *(RISKY, agentic)* Selective FMA-restore on U2/U4/H amplitude math only; f64→f32 in branch-independent folds. | 1.2-1.4× U-gate-heavy. | HIGH | **RISKY** — 22-circuit exact-match gate, revert on any mismatch | both |
| **P6** | Global-tier: protect the win; optionally re-add XCD-interleaved work-steal (correct single-counter now). | 1.0-1.1× global. | LOW | SAFE | both |

## P0 concrete design (merged, most detailed)

- New kernel `clifft_v2_register`, `grid=ceil(shots/256)*256`, `block=256`, ONE
  shot per thread. Per-thread `CV2Complex vloc[1<<kRegPackMax]` in private/registers
  (kRegPackMax=4 → 16 amps → 128 f32). Per-thread px/pz/rng/meas locals. No LDS
  amplitude buffer, no `barrier()`, no `coop_reduce2` (scalar sum over ≤16 amps).
- **Single-source-of-truth:** wrap execute_shot's opcode switch so a `V2_SCALAR`
  build makes `barrier()` empty, the `t==0` guard always-true, and strided
  `for(i=t;i<N;i+=256)` collapse to `for(i=0;i<N;i++)`. Same arithmetic → byte-
  exact by construction. (Claude §2.2; preserves the coop_interpreter.c:262
  "written once" invariant.)
- Host: add `kRegPackMax=6` (start 4) tier band; REG uses coop kernarg prefix
  (v2_kernel.cc:151 offsetof trick already truncates — no global buffers).
- Risk: VGPR pressure at 16-64 amps. SVM's own register kernel runs VGPR=60,
  Scratch=4480 and is 15-28× faster than V2's coop path — a scratch-backed
  statevector is acceptable. Measure via `check_registers(<.hsaco>)` before raising
  kRegPackMax.

## Validation protocol (both plans agree; reuse profile_sweep.sh)

Every step: (1) correctness gate FIRST — `run_v2` vs `run_gpu --no-postselection`,
exact passed_shots+observable_ones, all 22 circuits, multi-seed for RISKY steps;
(2) counter proof:
- P0: SQ_WAVES (frame_h 80000→~316), LDS_Block_Size (→~0), total_kernel_ns → V2/SVM ≤1.5×.
- P1: LDS_Block_Size (25088→~13000), SQ_WAVE_CYCLES ↓≥20%, TCC_MISS_sum ↓, V2/SVM ≤1.1×.
- P2: SQ_WAIT_INST_LDS, GRBM_GUI_ACTIVE (qv10 99.3M→toward SVM 75.5M).
- P4: Accum_VGPR / v_accvgpr disasm traffic → 0.
- P5 (RISKY): SQ_INSTS_VALU 964M→500-700M AND exact match on all 22.
Counters FETCH/WRITE_SIZE are NOT collectable on this gfx950 — use TCC L2 as memory proxy.

## Bottom line

The data inverts the naive assumption: V2's interpreter isn't compute-slow, it's
*occupancy-starved and barrier-bound*, and catastrophically thread-wasteful at low
rank. The fix is structural and both models independently converged on it:
**(P0) stop launching 256 threads per tiny shot, (P1) reclaim LDS to double coop
occupancy, (P2/P3) cut barriers.** All SAFE for byte-exactness. Do these manually,
re-measure, and only then point KernelForge at the risky precision frontier. If
P0-P3 land as projected, V2 goes from 15-28× slower (low rank) / 1.3-1.65× slower
(coop) to roughly par-or-better across the whole suite, on top of the global-tier
wins it already has.
