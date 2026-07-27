# Verified-facts ledger

Every claim the report and deck make must appear here with a **source** that is
an artifact, not a sentence. Where a document in this tree asserts something the
data contradicts, the contradiction is recorded under **Stale claims** rather
than quietly dropped — several of those documents are still referenced by name
elsewhere, and a reader who finds them needs to know which parts died.

Ground rules used when auditing:

- A fact is *verified* only if it can be re-derived from a committed artifact
  (a CSV, a `summary.json`, a SLURM log, an `.hsaco`) or from a file+line in the
  source tree. Prose in a design doc is a *claim*, not a fact.
- Performance numbers are kernel time, never host wall time, and never compared
  across node types. `mi350x-es` is heterogeneous.
- Where a number could not be re-measured this cycle, it is marked
  **[unverified]** and the report must either omit it or attribute it.

---

## 0. INVALIDATION NOTICE — the 20260726T182433Z baseline dispatched stale kernels

> **Everything in §1–§3, §6 and §10 that comes from
> `runs/20260726T182433Z_report-final-postdust/` is under re-measurement.**
> Read this section before quoting any number below it.
>
> `compile_specialized()` keyed the specializer cache on the *generated C* plus
> toolchain paths, but **not on the device headers** that supply every
> `v2_op_*` body, `v2_barrier()`, and `V2_DUST_EPS`. A header change produced
> the same key, so the day-old `.hsaco` — and its `.gate` verdict — won.
> Fixed in `009df59`; the poisoned cache is quarantined (moved, not deleted) at
> `history/stale_spec_cache_20260725/`.
>
> Verified directly on the binaries with `llvm-objdump`, two independent markers:
>
> | marker | `build-v2-nohip/v2_spec_cache` (what job 50469 ran) | `scratch/*_cache` (fresh dirs) |
> |---|---|---|
> | `s_barrier` preceded by `lgkmcnt(0)` — the `150d09f` fence | **0 / 1509 (0.0 %)** | 1400 / 1509 (92.8 %) |
> | `V2_DUST_EPS` constant — the `2a015fd` fix | **`1e-18` (old)** | `1e-11` (fixed) |
>
> All 36 cached kernels are dated 2026-07-25; `150d09f` landed 07-26 06:33 and
> `2a015fd` landed 07-26 13:34. The run submitted specifically to prove the
> corpus reflected post-fix code in fact dispatched **pre-fix** code.
>
> **The two `coop_r10_n1720` gate failures in §6 are verdicts on the pre-fence
> binary.** Commit `435213e` already measured the post-fence gate *passing* on
> that exact shape (and the specializer then beating the interpreter 1.72–1.76×).
> §6's conclusion — "the specializer is still incorrect on this shape" — rests on
> stale zeros and is retracted pending the re-run.
>
> Re-measurement: **SLURM job 50505**, `runs/20260726T221403Z_postcachefix-headerkeyed/`.
> Facts not derived from that run (§4 spec_examples, §5 MLIR, §7 barrier, §8
> f32/f64, §9 compile time) are unaffected — they were measured from sources,
> fresh caches, or independently-built artifacts.

---

## 1. The measurement baseline

> ⚠️ **Invalid — see §0.** Superseded by job 50505. Retained verbatim because
> the invalidation argument in §0 refers to it.

| | |
|---|---|
| Run | `runs/20260726T182433Z_report-final-postdust/` |
| Node | `smci350-rck-g03-f13-21` (gfx950, MI350X) |
| SLURM job | 50469 |
| Commit | `f565075-dirty`, branch `mlir-v2` |
| Circuits | 26 (every fixture with compiled `peak_rank >= 5`, plus 4 register-tier references) |
| Source | `runs/20260726T182433Z_report-final-postdust/{summary.json,summary.md,node.json}` |

**Node identity is recorded in `node.json` for this run and only this run.**
Earlier runs in `runs/` did not capture which node they landed on. Cross-run
deltas in `analysis/TRENDS.md` that span those earlier runs therefore carry an
unquantified node-variance term. The `four_t` and `frame_h` deltas below are the
only ones large enough to survive that caveat, and both are corroborated by a
mechanism: commit `715f8d0`'s shot-packed register tier, which is a structural
change (1 shot per thread instead of 1 shot per workgroup), not a tuning delta.

Job 50469 was preempted at 12:04 elapsed by the user's own `spice100` jobs. All
26 circuits had complete `v2` and `svm` timing data at that point; the single
casualty is `qv24_L4_seed42/svm`, which lost its `pmcA/pmcB/pmcC` counter
passes. **Its timing ratio (0.880) is valid; its SVM counters are absent** and
appear as `-` in every counter table.

---

## 2. Headline result — V2/SVM kernel-time ratio (lower = V2 faster)

Source: `runs/20260726T182433Z_report-final-postdust/summary.md`.

| circuit | tier | ratio | V2 µs | SVM µs |
|---|---|---|---|---|
| surface_d11_t19 | global r14 | **0.255** | 20378 | 79794 |
| surface_d11_t15 | global r11 | **0.256** | 19419 | 75809 |
| surface_d9_t19 | global r13 | **0.262** | 11874 | 45266 |
| surface_d9_t15 | global r11 | **0.269** | 11167 | 41518 |
| surface_d7_t19 | global r12 | **0.298** | 6030 | 20232 |
| qv10 | coop r10 | **0.310** | 1361 | 4388 |
| surface_d11_t10 | coop r7 | 0.462 | 37822 | 81849 |
| surface_d9_t10 | coop r7 | 0.481 | 21228 | 44151 |
| four_t | register r0 | 0.497 | 10.1 | 20.4 |
| surface_d7_t15 | coop r10 | 0.505 | 11107 | 22014 |
| circuit_d3_p0.001 | register r4 | 0.525 | 224.8 | 428.0 |
| surface_d7_t10 | coop r7 | 0.529 | 10949 | 20695 |
| surface_d7_t5 | register r3 | 0.607 | 2044 | 3368 |
| frame_h | register r0 | 0.626 | 10.0 | 15.9 |
| qv20_L8_seed42 | global r20 | 0.723 | 1.56e6 | 2.16e6 |
| qv21_L8_seed42 | global r21 | 0.728 | 2.95e6 | 4.05e6 |
| qv20_seed42 | global r20 | 0.839 | 1.78e6 | 2.12e6 |
| qv24_L4_seed42 | global r24 | 0.880 | 7.24e6 | 8.23e6 |
| qv22_L6_seed42 | global r22 | 0.981 | 3.18e6 | 3.24e6 |
| qv23_L5_seed42 | global r23 | 0.990 | 6.08e6 | 6.14e6 |
| circuit_d5_p0.003 | coop r10 | **1.444** | 15217 | 10539 |
| cultivation_d5 | coop r10 | **1.443** | 30166 | 20899 |
| circuit_d5_p0.002 | coop r10 | **1.449** | 14983 | 10340 |
| circuit_d5_p0.005 | coop r10 | **1.450** | 15579 | 10748 |
| circuit_d5_p0.0005 | coop r10 | **1.458** | 14711 | 10091 |
| circuit_d5_p0.001 | coop r10 | **1.459** | 14873 | 10194 |

Three regimes, and the report should present them as three separate stories:

1. **V2 wins 3.9x** on large surface codes at global rank (0.255-0.298).
2. **V2 wins modestly or ties** on quantum-volume circuits (0.72-0.99), with the
   win shrinking monotonically as rank climbs past 21.
3. **V2 loses 1.44-1.46x** on exactly one family: the `circuit_d5` / `cultivation_d5`
   coop circuits — which are also **the only six circuits in the corpus that did
   not run the specializer at all**. Their specialization failed the correctness
   gate, so what these six rows measure is V2's *interpreter* against SVM's. See §6.

A column the summary table above does not show, and which changes its reading
entirely: **which kernel each row actually dispatched.** 20 of 26 ran
`clifft_v2_spec`; the 6 that ran `clifft_v2_coop` are exactly the 6 losses. §6
establishes this from `summary.json` and the on-disk `.gate` verdicts.

`MFMA = 0.0` on all 26 circuits. The workload is a butterfly reduction over
amplitudes, not a GEMM; the matrix cores are idle in both backends. Any claim
that MFMA is applicable here is unsupported by this corpus.

---

## 3. Where the time actually goes — hardware counters

Source: `summary.json`, counters `SQ_INSTS_VALU`, `SQ_INSTS_SALU`,
`SQ_INSTS_LDS`, `SQ_WAVES`, `TCC_HIT_sum`, `TCC_MISS_sum`.

| circuit | ratio | VALU V2/SVM | SALU V2/SVM | L2 hit V2 | L2 hit SVM |
|---|---|---|---|---|---|
| frame_h | 0.626 | 0.43 | 0.22 | 52.2% | 66.4% |
| four_t | 0.497 | 0.43 | 0.11 | 52.3% | 74.3% |
| qv10 | 0.310 | 0.44 | 0.20 | 99.6% | 99.8% |
| surface_d11_t15 | 0.256 | 0.48 | 0.14 | 98.7% | 98.4% |
| surface_d9_t19 | 0.262 | 0.48 | 0.14 | 97.4% | 89.8% |
| surface_d7_t5 | 0.607 | 0.74 | 0.17 | 99.2% | 98.9% |
| circuit_d3_p0.001 | 0.525 | 0.82 | 0.15 | 95.3% | 98.1% |
| qv20_seed42 | 0.839 | 0.82 | 0.34 | 69.9% | 69.9% |
| qv23_L5_seed42 | 0.990 | 0.86 | 0.36 | 68.0% | 68.1% |
| **circuit_d5_p0.001** | **1.459** | **2.29** | **1.62** | **71.5%** | **98.0%** |
| **cultivation_d5** | **1.443** | **2.23** | **1.62** | **73.5%** | **98.9%** |

**Verified:** SALU instruction count falls by 3x-9x on every circuit where V2
wins. This is the interpreter's `switch` dispatch disappearing — the specializer
turns `for(pc) switch(ins.opcode)` into straight-line calls, and the scalar unit
stops computing branch targets. It is the single most consistent signal in the
corpus.

> **Precision note (2026-07-27).** The `circuit_d5_p0.001` SALU ratio of
> **1.62** above is from the retracted `20260726T182433Z` run and does not
> reproduce. Two other runs (`20260726T011254Z_fullbench-3way` and
> `20260726T014859Z_all-tier5plus`) both give **1.468**. The SVM side is
> byte-identical across all three (`SQ_INSTS_SALU = 2,404,746,573`); only V2's
> count moves (3,531,312,912 → 3,891,138,310, +10.2 %), which is consistent
> with the stale-cache binary of §0 rather than with measurement noise. Use
> **1.47**, not 1.62. The qualitative claim — SALU *inverts* on this family —
> is unaffected, and so is the L2 collapse.
>
> Corrected win-side range across all 20 winners:
> **SALU ratio 0.110–0.361, i.e. a 2.8x–9.1x reduction** (the "3x-9x" above is
> right to one significant figure but `four_t` reaches 9.1x and `qv24` only
> 2.8x).

**Verified:** the `circuit_d5` family is the *only* one where V2's VALU count
goes **up** (2.2-2.3x) rather than down, the only one where SALU goes **up**
(1.47x vs 0.11-0.36 everywhere else), and the only one where V2's L2 hit rate
**collapses** (98% -> 71.5%). All three invert together — and they invert on
exactly the six circuits that ran `clifft_v2_coop`, the interpreter, because the
correctness gate rejected their specialization. The SALU inversion is the
tell: it is the *presence* of `switch` dispatch, which is precisely what the
other 20 circuits' specialized kernels removed. See §6.

**Read the last two rows of this table as "interpreter vs interpreter."** They
are the only rows in the corpus that do not compare V2's compiler against SVM's
interpreter.

---

## 4. Specialization — what is actually folded, per class

Source: `lowering/spec_examples/` (8 hand-built cases, interpreter form vs
specialized form, same harness, same flags, gfx950). Measured with
`lowering/spec_examples/build_examples.sh`; results in `gains.txt`, per-case
detail in `stats.csv`.

| # | class | instructions | VGPR | SGPR | VALU | branches |
|---|---|---|---|---|---|---|
| S1 | frame_cnot — frame operand folding | 136 -> 47 (**2.89x**) | 8 -> 3 | 14 -> 12 | 44 -> 8 (**5.50x**) | 6 -> 4 |
| S2 | meas_dormant — flag folding, dead-path deletion | 206 -> 70 (**2.94x**) | 25 -> 14 | 13 -> **26** | 86 -> 19 (**4.53x**) | 18 -> **0** |
| S3 | expand_rank — static rank tracking | 164 -> 83 (1.98x) | 18 -> 8 | 16 -> 10 | 67 -> 27 (2.48x) | 9 -> 6 |
| S4 | meas_active — rank-folded coop reduction | 331 -> 240 (1.38x) | 24 -> 14 | 30 -> 26 | 139 -> 96 (1.45x) | 15 -> 7 |
| S5 | array_cnot — `scatter_bits_2` index folding | 207 -> 81 (2.56x) | 16 -> 8 | 24 -> 10 | 84 -> 33 (2.55x) | 10 -> 8 |
| S6 | array_u2 — fused-matrix table lookup | 147 -> 86 (1.71x) | 24 -> 18 | 23 -> 16 | 54 -> 42 (1.29x) | 10 -> 2 |
| S7 | noise_block — runtime loop, stays a loop | 447 -> 407 (**1.10x**) | **56 -> 56** | 86 -> 85 | 214 -> 210 (**1.02x**) | 23 -> 20 |
| S8 | apply_pauli — mask index folds, contents don't | 185 -> 137 (1.35x) | 22 -> 21 | 44 -> 42 | 71 -> 63 (1.13x) | 3 -> 3 |

**Superseded numbers.** An earlier version of this table reported an *ISA line
count* (`wc -l` of the `.s`) rather than an instruction count, and read the
resource columns from the `; NumVgprs:` / `; NumSgprs:` comments. Three bugs,
all fixed 2026-07-25, none of which touched the 16 `.s` artifacts (verified
byte-identical before and after):
1. Line count includes 205–316 lines of directives, comments, labels and
   metadata per file — more than half of each — which diluted every ratio toward
   1.0. S1 read 1.37x where the instruction ratio is 2.89x.
2. LLVM prints `; NumVgprs:` as a symbolic expression
   (`max(56, amdgpu.max_num_vgpr)`) for any kernel calling an external function.
   S7 calls `__ocml_log_f64`, so a trailing-digit regex scraped **0** — hence the
   old `0 -> 0` VGPR entry. It is 56 -> 56.
3. `; NumSgprs:` is not emitted by this LLVM at all; that column was 0 for all
   16 files and is now read from `.set case_kernel.numbered_sgpr`.

**S2 is the cleanest demonstration**: 18 branches to zero. When the specializer
knows a measurement's flags at codegen time, the `FLAG_IDENTITY` path is not
predicated — it is *deleted*, and with it every branch that guarded it. S2 is
also the one case whose **SGPR count rises** (13 -> 26): the deleted vector work
and branches reappear as scalar selects and their literals. That is the
scalar-substitution mechanism in its clearest form — work moved, not merely
removed.

**S7 is the honest negative result and belongs in the report.** `noise_block`
gains 1.10x on instruction count and 1.02x on VALU — and, now that the column is
read correctly, **no VGPR relief whatsoever** (56 in both forms), which matters
because register pressure is exactly what straight-lining hundreds of these adds.
Its body is a
`while (st->next_noise >= start && st->next_noise < end)` loop whose trip count
depends on the PRNG stream. The specializer folds `start` and `count` to
immediates and can do nothing else. **This is the R5 principle working as
designed** — specialize the operand body, straight-line the sequence, never
unroll a data-dependent loop — and it is also precisely why the noise-heavy
circuits do not benefit (§6).

Quantified against the actual regressing circuit: `lowering/v2_src/coop_circuit_d5.c`
emits 1,720 `v2_op_*` calls, of which **329 (19.1 %)** are `noise` / `noise_block`
/ `readout_noise` — the one class measured at 1.02x — while 757 (44.0 %) are
frame ops that fold well but are individually cheap (136 interpreter-form
instructions each, against 447 for a noise block). The mix is weighted toward
the ops specialization cannot help, in exactly the dimension that costs time.

**Methodology note that materially changed the numbers.** The first version of
the harness declared the interpreter's instruction stream
`extern const volatile CV2Instr*`. `volatile` forced system-coherent
(`sc0 sc1`) loads and an `s_getpc_b64`/`@rel32` global-address sequence that the
real interpreter never issues, inflating S1's interpreter side to 444 ISA lines
/ 80 VALU / 128 B scratch. The committed harness passes
`const CV2Instr* instrs, u32 pc` as kernel arguments, matching
`coop_interpreter.c`'s actual `for(pc) switch` exactly, which brought S1's
interpreter side to 347 / 44 / 0. **The table above is the corrected, honest
comparison.** The inflated numbers appear nowhere in the report.

---

## 5. MLIR — what v2 actually does, and what v1 actually did

**Verified: V2 does not emit MLIR.** `v2_specializer.cc:127` emits a C
translation unit whose first line is
`#include "clifft/gpu/mlir/v2/v2_ops.h"`. The path from circuit to `.hsaco` is
C -> clang -> llvm-link -> opt -> llc -> ld.lld. The directory is still named
`mlir/v2/` for historical reasons; the name is the only MLIR left in V2.

**Verified: v1's MLIR was 100% `llvm` dialect.** Dialect census on
`lowering/v1/frame_h.1_emitted.mlir` and `circuit_d3.1_emitted.mlir`:
`func.func = 0`, `arith.* = 0`, `scf.* = 0`, `memref.* = 0`, `llvm.func = 6` and
`7` respectively. v1 used MLIR as a *text serialization format for LLVM IR*, not
as a multi-level IR.

**Verified: v1 authored zero custom passes.** `mlir_codegen.cc:65-67` runs
exactly three stock upstream passes: `--canonicalize --cse
--convert-func-to-llvm`.

**Verified: the third pass is a complete no-op.** Per-pass snapshots via
`mlir-opt --mlir-print-ir-tree-dir` (`lowering/v1_passes/frame_h/`) show the
diff between `1_cse.mlir` and `2_convert-func-to-llvm.mlir` is **exactly two
lines — both the "IR Dump After" banner comment**. Rendered at
`lowering/diffs/v1pass.frame_h.1_cse__2_convert-func-to-llvm.diff`. There is no
`func` dialect to convert, because the emitter never produced any.

Line counts through v1's MLIR stage (frame_h): emitted 947 -> canonicalize 736
-> cse 608 -> convert-func-to-llvm 608. Only `canonicalize` and `cse` did
anything, and what they did was fold duplicate constants
(`llvm.mlir.constant` 240 -> 59) and redundant `shl`/`and`/`xor`.

**Consequence for the report.** The goal asks for "explicit lowering diffs
showing what each of the MLIR passes we introduced do." The verified answer is
that *no passes were introduced*, and two of the three stock ones did trivial
cleanup while the third did nothing. The lowering-diff material is real and is
rendered (13 stage pairs, `lowering/diffs/`), but it documents **v1's stock
MLIR chain and v2's C->LLVM-IR->ISA chain**, and the report must say so plainly.

Rendered stage pairs, all in `lowering/diffs/` as `.diff` (hunk-scoped) and
`.html` (colorized side-by-side):

| pipeline | circuit | chain |
|---|---|---|
| v1 | frame_h, circuit_d3 | emitted.mlir -> opt.mlir -> translate.ll -> optO2.ll -> isa.s |
| v2 | circuit_d3 | emitted.c -> clangO0.ll -> clangO2.ll -> isa.s |
| v1pass | frame_h | canonicalize -> cse -> convert-func-to-llvm |

---

## 6. The `circuit_d5` regression — what actually ran

> **Correction (audit pass 2).** An earlier draft of this section attributed the
> 1.44-1.46x loss to `noinline` noise ops inside the *specialized* kernel. That
> attribution is **wrong**, and the data in this very run refutes it. The
> specialized kernel never executed on these circuits. What follows is the
> corrected account; the op-census and S7 evidence survive, but as an explanation
> of *why specialization would not have saved this family*, not of the measured
> gap.

### 6.1 The primary fact: these six circuits ran the interpreter

`summary.json` records, per circuit, the name of the kernel rocprofv3 actually
timed. Across the 26-circuit corpus:

| kernel actually dispatched | circuits |
|---|---|
| `clifft_v2_spec` (specialized) | 20 |
| `clifft_v2_coop` (interpreter) | 6 — `circuit_d5_p{0.0005,0.001,0.002,0.003,0.005}`, `cultivation_d5` |

Re-derive with:

```
python3 -c "import json;d=json.load(open('summary.json'));
print(sorted((c['circuit'],c['v2']['kernel_name']) for c in d))"
```

**Every circuit V2 wins on ran `clifft_v2_spec`. Every circuit V2 loses on ran
`clifft_v2_coop`.** The correlation in this corpus is perfect.

### 6.2 Why the interpreter ran: the correctness gate rejected the specialization

`v2_kernel.cc:340-405` validates each freshly compiled `.hsaco` against the
interpreter on a sample before it is allowed to run, and persists the verdict to
`<hsaco>.gate` so it is computed once ever. Verdicts on disk in
`build-v2-nohip/v2_spec_cache/`:

```
$ for f in build-v2-nohip/v2_spec_cache/*.gate; do echo "$(cat $f) $(basename $f)"; done | sort
0 coop_r10_n1720_977e1e830813621d.hsaco.gate
0 coop_r10_n1720_cadfdf1980b2e2e8.hsaco.gate
1 coop_r10_n140_4dfa2381ce3e045a.hsaco.gate
... (34 more, all 1)
```

**2 failures out of 36 verdicts, and both are `coop_r10_n1720`** — the compiled
shape of the entire `circuit_d5` / `cultivation_d5` family (rank 10, 1720
instructions). The gate did its job: it caught a wrong kernel and fell back to
the interpreter, which is why the numbers are still byte-exact against SVM
(§2's ratios are apples-to-apples on *results*, just not on *code paths*).

### 6.3 What the 1.44x therefore measures

The headline 1.44-1.46x is **V2's bytecode interpreter versus SVM's bytecode
interpreter**, on the one circuit family where V2's compiler is disqualified.
It is not a measurement of specialized code losing. Two consequences for the
report:

- V2's interpreter is genuinely ~1.44x slower than SVM's GPU interpreter on this
  shape. That is a real, reportable weakness of the V2 runtime path — V2 has
  optimized its interpreter far less than SVM has, because the interpreter is
  meant to be the fallback, not the product.

  > **Status update (2026-07-27):** this is still true *of the interpreter*, but
  > it is no longer the observed end-to-end result for these six circuits. The
  > gate failure that pinned them to the interpreter was the execution-only
  > `s_barrier` (§6/§11.2); with `150d09f` the `coop_r10_n1720` gate verdict
  > flips **0 → 1** on disk and the specializer is selected. Job 50389 measures
  > **1.65x-1.76x** faster than the interpreter arm, which against these same
  > SVM baselines implies ratios near **0.40** rather than 1.44. Quote the
  > 1.44x only as "V2 interpreter vs SVM interpreter", never as "V2 vs SVM".
- The counter inversion in §3 (VALU 2.2-2.3x, SALU 1.47x, L2 hit 71.5% vs 98.0%)
  is a property of `clifft_v2_coop`, **not** of `V2_NOISE_ATTR` call boundaries.
  The interpreter build compiles `v2_ops_body.inc` with `always_inline`
  throughout; there are no `noinline` ABI boundaries in the kernel that ran. The
  SALU ratio *above* 1.0 here — against 0.11-0.36 everywhere else — is the
  signature of `switch`-dispatch overhead that specialization removes and that
  this family never got to remove.

### 6.4 Why specializing this family would not have been a free win either

The op census and the S7 microbenchmark still bound the upside, and should be
reported as such.

**(a) Op census.** The specialized source `coop_r10_n1720_*.c` (1720 ops)
contains:

```
156  v2_op_noise          80  v2_op_noise_block      93  v2_op_readout_noise
```

329 of 1720 ops (19.1%) are noise ops — the densest in the corpus. `coop_r10_n140`
(qv10, ratio 0.310) has **zero**.

**(b) Specialization gain is ~1.0 on exactly these ops.** S7 (§4) measures
`noise_block` at 1.07x ISA / 1.02x VALU. The specializer cannot fold a
PRNG-trip-count loop; ~19% of this circuit is immune to the entire optimization,
and Amdahl bounds the rest.

**(c) But noise density alone does not explain the loss.** The whole surface
family is *also* ~16-17% noise ops and V2 wins 2-4x on it. What is unique to
`coop_r10_n1720` is the *combination*: `readout_noise = 93` (no other kernel in
the corpus has a single one), `noise_block = 80` (vs 35-53 elsewhere),
`swap_meas_interfere = 15`. Commit `9d9cc68` named exactly this combination as
the knife-edge case when the gate was introduced.

### 6.5 Structural facts that remain verified

v2's `-O2` output on a specialized noise-heavy circuit keeps exactly **5**
functions, down from 36 at `-O0` — `clifft_v2_spec`, `v2_op_noise_block`,
`v2_op_noise`, `v2_op_readout_noise`, `v2_op_swap_meas_interfere`. The survivors
are exactly the `V2_NOISE_ATTR` set plus the kernel; the attribute does what it
claims. This is a fact about the *compiled artifact*, and holds regardless of
whether that artifact was dispatched.

`v2_ops_body.inc:420-427` records the in-tree history: the `noinline` fence was
introduced on the theory that inlining reassociated FP and flipped a branch by
~1 ULP. **That theory was wrong** — the real bug was an execution-only
`v2_barrier()` (§7). The comment is explicit that the attribute is retained
"because it measurably helps register pressure and code size on large circuits,
NOT because it is load-bearing for correctness."

### 6.6 Open items

1. **Does `V2_SPEC_NOISE_INLINE=1` make `coop_r10_n1720` pass the gate?** The
   knob flips `V2_NOISE_ATTR` to `always_inline`. The question is now a
   *correctness* question, not a performance one: it would test whether the
   remaining disagreement is inlining-related at all. Not measured on current
   HEAD.
2. **What does the specialized `coop_r10_n1720` actually get wrong?** The gate
   reports pass/fail, not a diff. Nothing in the tree localizes the divergence to
   an opcode or an instruction index.
3. **Is V2's coop interpreter improvable to SVM parity?** Untested. It has never
   been an optimization target.

Until (1) and (2) are answered, the report must present this family as
*characterized* — we know exactly which kernel ran and why — but **not**
*resolved*: the specializer is still incorrect on this shape.

---

## 7. The barrier bug — what it really was

**Verified:** `s_barrier` on AMD is execution-only; LLVM models
`llvm.amdgcn.s.barrier` as `IntrNoMem`, so the compiler is free to sink stores
across it. The coop gate failure was a bare `s_barrier`, not FP rounding.

Two facts refute the FP story, both recorded at `v2_ops_body.inc:420-427`:
the build uses `-O2 -ffp-contract=off` with no fast-math, under which inlining
cannot legally change an FP result; and `V2_GATE_SELFTEST` showed the specialized
kernel disagreeing **with itself** across two runs on identical inputs, which no
rounding model explains.

Fix: `v2_barrier()` wraps `s_barrier` in
`__builtin_amdgcn_fence(release/acquire, "workgroup")`. Commit `150d09f`.

---

## 8. The f32/f64 gap — measured, not asserted

The claim in `docs/v2/pre_V2.md:66` ("f32 vs f64 accumulation differs across
backends") is prose. Here is the experiment.

**The mechanism.** `device_abi.h:31` states the GPU carries amplitudes in f32
with f64 used only in reductions; the CPU reference (`src/clifft/svm/svm.h:142`) uses
`std::complex<double>` throughout. Branch probabilities are *reduced* in f64 but
from f32 *inputs*, so `p0/total` is good to only ~1e-6 relative after summing
2^k terms. When `rand*total` lands within that window of `p0`, the two sides take
**different branches on the same PRNG stream**, and every subsequent draw
decorrelates. Prediction: divergence rate scales with amplitudes summed,
~sqrt(2^peak_rank) * 2^-24, so it should be invisible at register rank and
visible at coop rank 10.

**Confirmed, and then narrowed.** The decisive artifact is `scratch/dust_50444.log`
(job 50444, node `smci350-rck-g03-f13-21`), a two-arm A/B on `V2_DUST_EPS`:

| arm | `V2_DUST_EPS` | d5 same-stream shot 0 | d3 same-stream shot 0 |
|---|---|---|---|
| A | `1e-18` (fp64-calibrated) | gpu=[0] cpu=[1] **MISMATCH** | gpu=[0] cpu=[0] ok |
| B | `1e-11` (fp32-calibrated) | gpu=[1] cpu=[1] **match** | gpu=[0] cpu=[0] ok |

The rank-dependence prediction holds exactly: the register-tier circuit (`d3`,
rank 4) agreed under both thresholds; the coop-tier circuit (`d5`, rank 10)
disagreed under the fp64-calibrated threshold and agreed under the
fp32-calibrated one.

**Root cause of that particular divergence.** `sample_branch` returns *without
drawing a PRNG value* when a branch is dust. `V2_DUST_EPS` was calibrated at
`1e-18` against f64 amplitudes but is compared against f32 amplitudes, so the
clamp never fired on the GPU where it fired on the CPU — one side consumed a
PRNG draw the other did not, and the streams desynchronized permanently. Fixed
to `1e-11` in commit `2a015fd`.

**Post-fix verification.** `scratch/verify_50453.log` (job 50453) separates the
logical question from the statistical one:

- **Q1 — same-stream shot-for-shot, must be EXACT: 36/36 exact.** Twelve seeds x
  three circuits (`circuit_d5_p0.001`, `circuit_d3_p0.001`, `circuit_d5_p0.005`),
  GPU seed vs the CPU seed that reproduces the identical 256-bit xoshiro state.
  Every shot matches.
- **Q2 — statistical convergence, gap should shrink as 1/sqrt(shots):**

  | circuit | shots | cpu | gpu | rel gap | z |
  |---|---|---|---|---|---|
  | circuit_d5_p0.001 | 2000 | 709 | 661 | 6.77% | -1.60 |
  | | 10000 | 3395 | 3364 | 0.913% | -0.46 |
  | | 50000 | 17005 | 16862 | 0.841% | -0.96 |
  | | 200000 | 67376 | 66968 | 0.606% | -1.37 |
  | circuit_d3_p0.001 | 2000 | 52 | 63 | 21.15% | 1.04 |
  | | 10000 | 250 | 270 | 8.00% | 0.89 |
  | | 50000 | 1225 | 1224 | 0.082% | -0.02 |
  | | 200000 | 4811 | 4922 | 2.31% | 1.14 |

  Every |z| < 1.7. The residual gap is sampling noise between two *independent*
  streams, not a numerical defect: shot-for-shot agreement is exact (Q1), so the
  aggregate difference is only the difference between two valid samples of the
  same distribution.

**This is the "narrowing the gap" story, and it has a shape worth stating
plainly**: the gap was never really about f32 vs f64 precision in the
*arithmetic*. It was about a *threshold constant* that had been calibrated for
one precision and left in place when the storage format changed — and whose
effect was to make one side consume a random number the other side did not.

---

## 9. Compile time — v1's explosion is real, v2's is not

Source: `lowering/matrix.csv` (login node, gfx950 target, single-threaded).

| pipeline | circuit | IR lines | ISA lines | compile |
|---|---|---|---|---|
| v1 | glob_surface_d7_t19 | 239,803 | 160,777 | **221.48 s** |
| v1 | coop_surface_d7_t10 | 228,544 | 148,260 | 83.13 s |
| v1 | coop_qv10 | 336,988 | 133,797 | 76.02 s |
| v1 | coop_circuit_d5 | 118,493 | 80,257 | 36.16 s |
| v1 | reg_circuit_d3 | 23,002 | 17,802 | 3.58 s |
| v1 | reg_frame_h | 947 | 524 | 0.52 s |
| v2 | coop_circuit_d5 | **1,760** | 61,766 | **11.38 s** |
| v2 | reg_circuit_d3 | **383** | 10,398 | 2.04 s |

On the same circuit (`coop_circuit_d5`), v2's source IR is **67x smaller** than
v1's (1,760 vs 118,493 lines) and compiles **3.2x faster**. The ISA is only 1.3x
smaller, which is the point: v2 does not emit less *machine code*, it hands the
compiler a vastly smaller *problem*. v1 handed LLVM one giant already-unrolled
function; v2 hands it straight-line calls to shared bodies.

### 9.1 Source-IR density: the architectural invariant

Source: `lowering/ir_density.csv`, generated by `lowering/ir_density.sh`.
Lines of emitted source per bytecode instruction, same circuits, all three
per-circuit generators:

| circuit | instrs | v1 MLIR | lines/instr | Hybrid HIP | lines/instr | **v2 C** | **lines/instr** |
|---|---|---|---|---|---|---|---|
| reg_frame_h | 4 | 947 | 236.8 | — | — | 43 | 10.75 |
| reg_four_t | 11 | 1,384 | 125.8 | — | — | 49 | 4.45 |
| reg_circuit_d3 | 344 | 23,002 | 66.9 | 5,301 | 15.4 | 383 | **1.11** |
| coop_qv10 | 140 | 336,988 | **2407.1** | 7,805 | 55.8 | 179 | **1.27** |
| coop_circuit_d5 | 1,720 | 118,493 | 68.9 | 33,508 | 19.5 | 1,760 | **1.02** |
| coop_surface_d7_t10 | 4,134 | 228,544 | 55.3 | 75,409 | 18.2 | 4,174 | **1.00** |
| glob_surface_d7_t19 | 4,296 | 239,803 | 55.8 | 75,794 | 17.6 | 4,346 | **1.01** |

**This is the clearest single statement of what V2 changed.** V1 and Hybrid both
emit a per-instruction *block*; V2 emits a per-instruction *call*. V2's density
converges to exactly 1.00 as circuits grow — one `v2_op_*(...)` line per bytecode
instruction, plus a fixed ~40-line kernel wrapper. That fixed wrapper is the
entire explanation of the two register-tier outliers (10.75 and 4.45 lines/instr
on 4- and 11-instruction circuits); it is amortized to nothing by 344.

`coop_qv10` at **2407 lines/instr** is v1's worst case and shows the mechanism:
qv10 is dense in `OP_ARRAY_U2`/`U4`, which v1 unrolled per incoming-frame-state
branch (U4: up to 16 branches × a fully unrolled 4×4 complex matmul, *per
instruction*). 140 bytecode instructions became 336,988 lines of MLIR. V2 emits
179 lines for the same circuit — a **1883x** reduction in source size for
identical semantics.

Hybrid sits between them, and that placement is itself informative: Hybrid was
never the pathological case, it was merely *linear with a large constant*. V2 is
linear with a constant of 1.

**Note on compile-time numbers taken from `PERFORMANCE_OPTIONS.md`**
(surface_d5_r5 55 s, surface_d7_r7 193 s, cultivation_d7 224 s vs Hybrid 24 s):
these were measured on a different date and an unrecorded node, against v1.
They are directionally corroborated by `matrix.csv` above but are **not**
re-measured. Mark them **[unverified, v1-era]** if quoted.

---

## 10. Kernel resource footprints — who spills

Source: `lowering/kernel_resources.csv`, 89 kernels, read from AMDHSA metadata
notes via `llvm-readelf --notes` on every distinct `.hsaco`.

| tier | kernels with any spill |
|---|---|
| coop | **0 / 13** |
| register | 12 / 58 |
| global | 11 / 15 |
| interpreter (non-specialized) | 2 / 3 |

- **Coop tier never spills.** Zero of 13.
- **Global tier is where it hurts.** The three worst kernels in the entire
  corpus are `global_r22`/`r23`/`r24`: `sgpr_spill` 762 / 662 / 594,
  `vgpr_spill` 136 / 199 / 152. These are exactly `qv22`/`qv23`/`qv24`, whose
  ratios are 0.981 / 0.990 / 0.880 — the three circuits where V2's win
  vanishes. The correlation between heavy spilling and lost speedup is the
  cleanest scaling-limit story in the corpus and should be a figure.
- **Register tier's worst is `reg_r4_n344`**: `vgpr_spill 88`, `scratch 1024`.
  That is `circuit_d3_p0.001` — which still wins at 0.525, so spilling alone is
  not disqualifying at this size.

### 10a. Spill mechanism — what the data RULES OUT (2026-07-27)

Added while writing report §10.6. The natural explanation for the rank-22 break
— "the specializer emits one call per instruction, so longer circuits carry more
live scalars until the SGPR budget breaks" — is **refuted by this same CSV**.

Global-tier kernels sorted by emitted instruction count (`n` in the filename):

| rank | instrs | VGPR | AGPR | SGPR | vgpr_spill | sgpr_spill |
|---|---|---|---|---|---|---|
| 14 | 16,521 | 128 | 64 | 106 | 0 | 4 |
| 11 | 16,415 | 128 | 64 | 106 | 0 | 4 |
| 13 | 9,359 | 128 | 64 | 106 | 0 | 4 |
| 12 | 4,296 | 128 | 64 | 106 | 0 | 0–4 |
| 20 | 418 | 104 | 40 | 106 | 0 | 0 |
| 21 | 393 | 104 | 40 | 106 | 0 | 0 |
| 22 | 359 | 128 | 64 | 108 | 136 | 762 |
| 23 | 335 | 128 | 64 | 108 | 199 | 662 |
| 24 | 320 | 128 | 64 | 108 | 152 | 594 |

**The three spilling kernels are the three SHORTEST in the tier.** Rank 14 emits
46× more instructions than rank 22 and spills 4 SGPRs. The correlation with
instruction count runs backwards. Any claim of the form "qv24 does better
because it has fewer instructions" is unsupported — fewest instructions is the
norm among spillers.

Also ruled out: **it is not a source-level difference.** Diffing the emitted C
for `global_r21_n393` vs `global_r22_n359` with numeric literals normalized
(`sed -E 's/[0-9]+/N/g'`) shows an identical preprocessor preamble, identical
`spec_body` shape, and a near-identical opcode mix. The global tier wrapper is
rank-independent by construction (`v2_specializer.cc:196-224`). The only
differing constant is `amp_capacity` (2097152ull → 4194304ull).

Still open: crossing 2^22 coincides with the allocator moving from VGPR 104-106
/ AGPR 40-42 to the VGPR 128 / AGPR 64 cap. Since `SQ_INSTS_MFMA = 0` and CDNA
repurposes AGPRs as overflow when MFMA is unused, the pattern is *consistent
with* AGPR overflow exhausting at the cap and falling through to scratch — but
this is **inference from resource metadata, not a verified mechanism.** Do not
state it as cause. The next step is an LLVM allocation study at the 21/22
boundary.

Dedup note: the spec cache is content-hashed, so an identical `.hsaco` appears
under `build-v2-nohip/v2_spec_cache/` and under every per-run scratch cache. The
script dedupes by basename (`awk -F/ '!seen[$NF]++'`), one row per distinct
kernel.

**Cache-invalidation status of this section: UNAFFECTED.** The §11.4 stale-cache
bug does not touch these numbers. Verified directly: the pre-fence and
post-fence `.hsaco` for `global_r22/r23/r24` differ in 84 % of their bytes and
by +1,152 bytes, yet report byte-identical `vgpr_spill`, `sgpr_spill`,
`sgpr_count` and `private_segment_fixed_size`. Register pressure is a property
of what the specializer emits, not of the fences around it.

---

## 10b. gfx950 has 160 KB of LDS per CU, not 64 KB (2026-07-27)

Found while auditing report §9.3. **Every occupancy figure in the P1 commit
messages and in `plans/claude_plan.md` / `gpu/gpu_kernel_static_characterization.md`
is derived from a 64 KB LDS budget that is wrong for this chip.**

Measured by binary-searching the largest `address_space(3)` array the build
toolchain accepts (`clang 21.0.0git` from `LLVM_PREFIX`, the same one
`ClifftAmdgcn.cmake` uses):

| arch | max LDS/workgroup |
|---|---|
| gfx90a | 65,473 B |
| gfx942 | 65,473 B |
| **gfx950** | **163,681 B** |

Overshooting by one byte prints the constant: `error: local memory (163844)
exceeds limit (163840) in 'k'`.

LLVM's own `; Occupancy:` comment, 256-thread workgroup (so waves/SIMD == wg/CU):

| LDS bytes | gfx942 | gfx950 |
|---|---|---|
| 8,704 (global tier) | 7 | **8** |
| 13,312 (after P1b) | 4 | **8** |
| 16,896 (after P1a) | 3 | **8** |
| 25,088 (P0 baseline) | 2 | **6** |
| 32,768 | 2 | 5 |
| 65,536 | 1 | 2 |

The model is `min(8, 163840 / LDS)`, verified against LLVM at every step
boundary (first drop below 8 is at 20,992 B; `163840/20992 = 7`). Sweeping
`amdgpu_num_vgpr` from 32 to 256 changes nothing in the gfx950 column, so VGPRs
are not binding either.

**Consequences.**
- The claimed "2 → 3 → 4 wg/CU" is exactly right for **gfx942** and wrong for
  the node the benchmarks ran on. Both P1 steps moved inside the flat region:
  8 wg/CU before, 8 wg/CU after.
- The `LDS_Block_Size` measurements themselves (25,088 → 16,896 → 13,312,
  `tools/ldscheck_50017.log` / `ldscheck_50021.log`) are correct. Only the
  inference from them is wrong.
- The ~10 % gain on coop circuits in the `after-P0-P1` run is real but **not
  attributable** — that run also contains P0's topology change, and the two are
  not separable in the archive.
- Today's HEAD reports `.group_segment_fixed_size: 13064` in the coop kernel's
  ELF metadata, 248 B below the P1b figure.
- `gpu_kernel_static_characterization.md:77-78,148,151` state "64 KB LDS/CU" as
  a premise and conclude LDS is "the binding occupancy constraint". **On gfx950
  it is not.** Do not quote those lines without this correction.

---

## 11. Stale claims found during the audit

Recorded because these documents are still in-tree and still cited by name.

**`analysis/SYNTHESIS.md` is stale and must not be quoted.** It reports
`frame_h` at 28.3x and `circuit_d3` at 15.2x, and builds a thesis around "V2
loses badly at low rank." Latest measurements: `frame_h` 0.626, `circuit_d3`
0.525. Both are V2 *wins*. The document predates the register tier, which fixed
exactly the problem it describes. Its 28.258 and 15.193 figures do appear as the
`20260725T030000Z_baseline-preopt` column in `analysis/TRENDS.md`, where they
are correctly labelled as a superseded baseline — that is the only context in
which they should be shown.

**`docs/v2/reference_v1.md:263` — "115 KB/thread spill" does not reproduce.**
Measured on the artifacts: v1 `compiled_mlir_kernel` (circuit_d3) has
`ScratchSize 156`, occupancy 4. v1 coop (qv10) has `ScratchSize 0`, VGPR 53,
LDS 24664, occupancy 8. Whatever produced 115 KB/thread is not in the current
`ir_reference` corpus. **Do not repeat this number.** v1's real problem is
documented and large enough without it (§9: 221 s compile, 240 K lines of IR).

**`hsa_persistent_dispatch.h` per-op costs are asserted, not measured.** The
header claims `alloc_kernarg ~800 ns`, `allow_gpu_access ~1200 ns`,
`signal_create ~600 ns`, `signal_destroy ~400 ns`, `free_kernarg ~300 ns`,
"~3.3 µs saved of ~4.5 µs". No measurement in this tree produces those numbers.
`dispatch_bench/` exists to replace them with real gfx950 measurements
(naive / persistent / batched16 HSA modes vs HIP sync / stream_sync /
launch_only). **Until that job lands, these are [unverified] and the report must
not present them as measured.**

**`PERFORMANCE_OPTIONS.md` dispatch breakdown is v1-era.** HSA executable load
170-206 ms one-time, `PersistentDispatcher::dispatch` ~0.15 ms, total loop
0.26 ms, kernel 0.024 ms; SVM `sample_kernel` 20-27 µs; HIP copy kernels (x6)
16-24 µs. Node unrecorded, date differs, measured against v1. Directionally
useful, **[unverified]** as stated.

---

## 12. Open items

| item | status |
|---|---|
| **Re-measure the whole corpus on header-keyed cache** | **job 50505 running; §0. Every §1–§3/§6/§10 number is provisional until it lands** |
| HIP-vs-HSA dispatch cost, measured on gfx950 | SLURM job 50501 pending; replaces §11's asserted numbers |
| ~~`coop_r10_n1720` specialization is incorrect~~ | **retracted (§0)** — the failing verdicts were computed on the pre-fence binary; `435213e` measured the post-fence gate passing |
| `V2_SPEC_NOISE_INLINE=1` A/B on current HEAD | not run; now a *correctness* probe (does the gate pass?), not a performance one (§6.6) |
| V2 coop **interpreter** vs SVM interpreter parity | V2's is ~1.44x slower on the one shape where it has to run; never an optimization target (§6.3) |
| ~~Node identity for runs before `20260726T182433Z`~~ | **recovered via `sacct` (2026-07-27)**: runs 1–8 on `smci350-rck-g03-d13-21`, runs 9–12 on `smci350-rck-g03-f13-21`. Ratios are same-job so they stand; the column-8→9 *delta* carries a node change as well as a code change |
| Coop specializer gain over interpreter | commit `60d5728` claims 5.5×/7.6×; **the archive gives 3.54×/3.72×** and the commit figures do not reproduce. Use the archive |

---

## 13. Audit-pass-2 corrections to this document

This ledger is itself subject to its own ground rule. Corrections made after the
first commit, each traceable to data:

| § | claim as first written | correction | evidence |
|---|---|---|---|
| 6 | the 1.44x loss is caused by `noinline` noise-op ABI boundaries in the specialized kernel | the specialized kernel never ran; the gate rejected it and the interpreter ran instead | `summary.json` `v2.kernel_name = clifft_v2_coop` on all 6; `coop_r10_n1720_*.hsaco.gate = 0` |
| 3 | VALU/L2 inversion attributed to specialized-kernel call boundaries | inversion is a property of `clifft_v2_coop`; the interpreter build inlines `V2_NOISE_ATTR` throughout | same |
| 2 | three regimes described purely by ratio | the third regime is defined by *which kernel ran*, not by circuit shape | same |

The pattern to carry into the report: **`summary.json` records `v2.kernel_name`
per circuit. Any performance claim about "V2" must state which V2 it means.**

### 13.1 Audit-pass-3 (2026-07-27) — report §8/§9 against the archive

| claim | source | correction | evidence |
|---|---|---|---|
| P1 raised occupancy 2 → 4 wg/CU | `d873997`, `7802423`, both plans | 8 → 8 on gfx950; the figure is a gfx942 result | §10b |
| coop specializer is 5.5×/7.6× over the interpreter | `60d5728` | 3.54× / 3.72× | last-interpreter vs first-spec run, kernel identity from trace |
| the noise fence took `surface_d7_t15` 1.427 → 0.503 | report §9.5 | **1.793 → 0.503**; the 1.427 cell is gate-polluted and already partly specialized | VALU 4.71e9 / 3.49e9 / 1.15e9 across the three runs |
| `qv10`'s 0.252 → 0.310 unexplained | report §9.1 | the noise fence: VGPR 24 → 36 while **VALU falls** 5.10e8 → 5.02e8 | five runs, all `d13-21`, SVM side flat at 4317–4388 µs |
| P0 preceded P1 | report §9 ordering | **P1 shipped 25 min before P0** (04:46/04:50 vs 05:11) | `git log --reverse` |
| pre-fix specialized kernel had 1439/1509 (95.4 %) unfenced barriers | `150d09f` message, `d5_fence.sh:7-8`, report §11.2 | **1400 / 1509 (92.8 %)** | `llvm-objdump -d` on the archived pre-fix `.hsaco`; converges at scan windows 24/32/48/64/unbounded, and loosening `lgkmcnt(0)`→any `lgkmcnt` does not move it |
| pre-fix interpreter had 52/81 unfenced barriers | same | **50 / 80** | A/B rebuild of `coop_interpreter.c`, `-O2 -ffp-contract=off -mcpu=gfx950`; the archived 52/81 counted an `ocml`/`ockl`-linked binary, so one barrier is device-library code, not V2 |
| barrier fence costs +3.88 % static instructions | `150d09f` message | **+3.89 %** (5,629 → 5,848), and **0 %** on the register tier (byte-identical, 4,457) | same A/B rebuild |
| the dust fix landed 07-26 13:34 | report §11.4 | **13:31** (`2a015fd`) | `git log` |
| all 36 stale cached kernels are dated 07-25 | report §11.4 | **32 on 07-25, 4 on 07-26 00:00–01:48** — still all before the 06:33 barrier fix | `ls --time-style` on the archived cache |

**Method note that generalizes.** Every archived run kept its `rocprofv3`
kernel trace, and the kernel *name* plus *dispatch count* in that trace is a
stronger fact than any summary ratio: it says which code path produced the
number. `20260725T162524Z_noise-fenced-gated` is the only run in which any
circuit dispatched two different kernels — interpreter ×1 plus spec ×2 — which
identifies the gate-pollution artifact from the artifacts alone, without needing
the commit message that explains it. **Check `raw/<circ>/<engine>/kt/` before
attributing any step in a progression table.**

**Second method note: static ISA facts are cheap to re-derive, so re-derive
them.** The barrier counts above needed no GPU — amdgcn codegen runs on the
login node, and the whole A/B is `clang -emit-llvm` + `llc -filetype=asm` on the
same source with one header line changed. That makes every static claim in this
report (instruction counts, barrier counts, register pressure, LDS bytes)
independently checkable in seconds, with no queue wait. Note that the *linked*
`.hsaco` and the *unlinked* asm differ slightly — `ocml`/`ockl` contribute their
own instructions and, in this case, one extra barrier — so state which one a
count refers to. When only a ratio matters, the unlinked build is the cleaner
measurement because it contains only V2's own code.

**Third: a number quoted three times is still one measurement.** The 95.4 %
figure appeared in the commit message, in the script header comment, and in the
report, which reads like corroboration but is a single unreproducible source
copied forward. The counting code that produced it was never committed. Prefer
committing the *measurement script* over the *measured number*.

### 13.2 Audit-pass-4 (2026-07-27) — report §12 (the f32/f64 gap)

Everything in §12 reproduces except one under-specified table. Confirmed
exactly: the `CV2Complex` typedef (`device_abi.h:35`), the `sample_branch`
excerpt (`v2_ops.h`), `V2_DUST_EPS 1e-11` vs `kDustEpsilon = 1e-18`
(`svm_internal.h:46`), §12.3's A/B arms (`dust_50444.log`, node `f13-21`),
§12.4 Q1 at 36/36 exact over 12 seeds × 3 circuits, and the whole §12.4 Q2
convergence table (`verify_50453.log:47-59`).

| claim | source | correction | evidence |
|---|---|---|---|
| dust floor is 9.7e-15 / 1.42e-14 median, 1.2e-13 low-rank tail | `V2_DUST_EPS` comment prose, report §12.2 | **numbers stand**, but they are a *statistical envelope*, not the kernel's arithmetic; direct butterfly simulation puts the floor **~36× lower** (3.2e-16 median, ~5e-15 tail) | `tools/dust_floor.py`, both models |

The four rows reproduce to the digit under `--model residual` (9.730e-15 /
1.222e-13, 1.315e-14 / 5.048e-14, 1.422e-14 / 1.586e-14, 1.422e-14 / 1.476e-14),
which is what makes the provenance recoverable at all: "residual model" turned
out to be literal. It assumes each analytically-zero output carries a relative
error at full `eps` and that the errors never cancel, giving `eps²` times a
weighted mean of `Exp(1)`. `--model butterfly` instead simulates
`fl(fl(u·a) + fl(v·b))` over fp32 amplitudes that cancel exactly in fp64.

**Fourth method note: an estimator is part of the measurement.** This table had
the same defect as the 95.4 % figure — prose, no committed generator — but here
the numbers were *right*; what was missing was the model that produced them. A
value like "1.42e-14" carries no indication of whether it is an upper envelope
or a simulation, and the two differ by 1.5 decades. The decision is unaffected
(both models agree on rank-independence, on the tail being widest at *low* rank,
and on depth-insensitivity — `--depth 256` moves medians <5 % — so `1e-11`
clears the conservative floor by 2 decades and the simulated one by ~4), but
"the margin is 2 decades" and "the margin is 4 decades" are different claims.
**State the estimator alongside the number, or commit the script that encodes
it.**
