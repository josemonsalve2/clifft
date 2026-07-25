# V2_performance — clifft MLIR-V2 performance characterization database

Central database for the V2 (HIP-free/HSA) performance-improvement effort.
Populated by profiling jobs + characterization subagents; consumed by a
parallel Codex + Claude Code planning pass.

Node: mi350x-es (gfx950 / MI355X / CDNA4). Gold: GPU-SVM (f32) per R3.

## Layout

- `runs/`      — **one timestamped slot per benchmark run** (produced by the
                 `/benchmark-all` skill). Each `<UTC-stamp>_<label>/` holds
                 `raw/` (rocprofv3 CSVs), `gpu/<circuit>.json` (digested), plus
                 `summary.md`, `summary.json`, `manifest.json` (git provenance).
                 `runs/INDEX.md` lists all runs with mean V2/SVM ratio.
- `history/`   — `metrics.jsonl`: append-only longitudinal log, one row per
                 (run × circuit), tagged with git commit/branch. The progressive-
                 analysis source of truth.
- `analysis/`  — cross-run + cross-circuit synthesis. `TRENDS.md` (regenerated)
                 = circuit×run ratio matrix + latest-vs-previous delta.
                 `SYNTHESIS.md` = the static root-cause analysis.
- `gpu/` `cpu/` — STATIC characterization docs (kernel/CPU-path analysis, line-
                 cited). Not per-run; these describe the code, not a measurement.
- `plans/`     — optimization plans (Codex, Claude, merged UNIFIED_PLAN.md).
- `tools/`     — tooling status + one-off harness (superseded by the skill for
                 routine runs; kept for reference + the agentic-integration guide).

## Running a new benchmark (progressive)

Use the **`/benchmark-all [label]`** skill (~/.claude/skills/benchmark-all). It
mints a run slot, submits the SLURM sweep (V2+SVM, all tiers), digests to JSON,
appends to `history/metrics.jsonl`, and regenerates `runs/INDEX.md` +
`analysis/TRENDS.md`. Always pass a label tied to the change under test
(e.g. `after-P0-register-kernel`) so the trend columns are legible.

Metric of record: **V2/SVM kernel-time ratio** (rocprofv3 --kernel-trace, not
host wall). <1 = V2 faster. For low-rank circuits trust the absolute V2 µs
column (SVM denominators are timer-noise-small).

## Key measurement recipes (from ref_amd_profiling_tools + ref_gpu_opt)

- Clean kernel time:      rocprofv3 --kernel-trace --stats -- <app>
- HSA dispatch overhead:  rocprofv3 --hsa-core-trace -- <app>   (the piece the
                          host-wall baseline conflated; needed for a fair
                          V2-kernel vs SVM-kernel number)
- HW counters:            rocprofv3 -i pmc.txt -- <app>
    pmc: SQ_WAVES SQ_INSTS_VALU SQ_INSTS_MFMA TCC_HIT_sum TCC_MISS_sum FETCH_SIZE WRITE_SIZE
    (MFMA should be ~0 — our compute is 2x2/4x4 butterfly, NOT GEMM. Nonzero = mis-lowering.)
- SoL/roofline/occupancy: rocprof-compute profile -n <name> -- <app>; then analyze -p ...
- Registers/LDS/spill:    llvm-objdump on the .hsaco (occupancy step at VGPR 256;
                          LDS<=80KB => dual-occupancy on CDNA4)

## Bottleneck classification (KernelForge thresholds)

- COMPUTE>=60% & BW<40%          -> compute-bound
- BW>=50% & COMPUTE<40%          -> bandwidth-bound
- both<40% & occupancy<40%       -> occupancy-bound
- both<40% & occupancy>=40%      -> latency-bound
- else                           -> balanced
- LDS bank conflicts >= 20%      -> flag lds_bank_conflicts

## Circuit matrix (BENCHMARKS.md, ranks measured 2026-07-24)

reg(0-4):  frame_h(r0), circuit_d3(r4)
coop(5-10):qv10(r10), surface_d7_t15(r10), surface_d9_t10(r7), surface_d11_t10(r7)
global(11+):surface_d7_t19(r12), surface_d9_t19(r13), surface_d11_t15(r11),
            surface_d11_t19(r14, KNOWN DIVERGENCE), qv20(r20, over cap)
