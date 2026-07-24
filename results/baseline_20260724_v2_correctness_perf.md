# MLIR-V2 (HIP-free/HSA) Correctness + Perf Baseline — 2026-07-24

Branch: v2 reimplementation (coop/global tiers, plain-C amdgcn interpreter).
Node: mi350x-es (gfx950). Comparison gold: GPU-SVM (f32) per R3.

## Correctness (V2 vs GPU-SVM, byte-exact)

All at `--no-postselection` unless noted; V2 matches SVM EXACTLY:

- Incremental ladder (20/20): frame, expand/T, array-Clifford, active
  diagonal+interfere, noise, readout, apply-pauli, postselect.
- Coop tier: qv10 (rank 10), circuit_d3 (rank 4), circuit_d5 (rank 10,
  noise+swap-meas heavy), color_d3. Postselection matches on d3/color/rep/d5.
- Global tier: surface_d7_t19 (rank 12), d9_t19 (rank 13), d9_t15 (rank 13),
  d11_t15 (rank 11), d11_t10/t5 (coop). All EXACT across seeds 7/42/123.

### Byte-exactness invariants discovered
1. `cscale` must multiply in f64 then narrow to f32 (matches SVM).
2. `coop_reduce2` must use SVM's warp-butterfly order (ds_bpermute), not an
   LDS tree — else f64 rounding flips rare measurement branches.
3. amdgcn kernels MUST compile with `-ffp-contract=off` (SVM does); FMA
   fusion otherwise diverges cmul/cnorm.
4. ocml (`__ocml_log_f64`) linked for the noise hazard `log`.

### Known gaps
- surface_d11_t19 (rank 14): deterministic divergence vs SVM. Ruled OUT:
  work-steal race (WGS 1..512 identical), HBM overflow (over-alloc no change),
  high qubit index (d11_t5/t10 at maxq=273 match). Root cause unfound.
  Blast radius: this one circuit.
- qv20 (rank 20): exceeds kGlobalMaxPeakRank=19 — separate cap.

## Performance (directional; NOT methodologically clean)

CAVEAT: SVM column is `sample_seconds` (host wall incl. launch overhead);
V2 column is HSA-measured `kernel_seconds` (pure kernel). Not apples-to-apples
— a clean comparison needs SVM kernel-only timing. Treat as directional.

100k shots, best-of-3, warm cache, seed 1:

| circuit        | tier   | rank | SVM host(s) | V2 kernel(s) |
|----------------|--------|------|-------------|--------------|
| frame_h        | reg    | 0    | 0.0719      | 0.0027       |
| circuit_d3     | reg    | 4    | 0.0725      | 0.0326       |
| qv10           | coop   | 10   | 0.0906      | 0.0270       |
| surface_d7_t15 | coop   | 10   | 0.2835      | 0.3446       |
| surface_d7_t19 | global | 12   | 0.4556      | 0.3438       |
| surface_d9_t19 | global | 13   | 0.9121      | 0.7402       |

Observation: V2 (a pure runtime interpreter, zero specialization) is already
competitive-to-faster on host-wall terms at low rank, roughly par at high
rank. This is the O0 baseline; progressive-lowering specialization (per-circuit
MLIR codegen) is the intended path to the 2x/3x target and is NOT yet built.
