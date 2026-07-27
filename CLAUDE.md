# clifft GPU Performance Engineering

## Project Overview

clifft is a quantum circuit simulator using Pauli-frame tracking. The GPU backend
has four execution paths, in the order they were built:

| Path | What it is | Status |
|------|-----------|--------|
| **SVM** | Bytecode interpreter on the GPU | Baseline; still the correctness reference |
| **Hybrid** | Compiled HIP kernel, partial specialization | Superseded — 15–56 lines of IR per instruction |
| **V1 (MLIR)** | Textual MLIR LLVM dialect → mlir-opt → mlir-translate → opt → llc → lld → .hsaco | Failed — 55–2,407 lines of IR per instruction |
| **V2** | One `call` per instruction, direct C → amdgcn, HSA-only | Current. Wins 26/26, geomean 0.573 |

V1 unrolled the bytecode into one basic block per instruction, so IR grew with the
operand body. V2 emits one call per instruction — **1.00 lines/instr** — and moves
the specialization into the call's arguments instead of into its body.

**V2 does not emit MLIR.** The name `src/clifft/gpu/mlir/v2/` and the
`CLIFFT_ENABLE_MLIR_V2` option are historical. The pipeline is
`v2_kernel.c` → clang → amdgcn → `.hsaco` → HSA dispatch.

## Key Constraints

- **No HIP anywhere in the V2 path** — HSA runtime calls exclusively
- **`-ffp-contract=off`, no fast-math** — this is a correctness contract, not a perf knob
- **Never include compilation time in benchmarks** — always warm the cache first
- **Always report kernel_seconds vs host execution time separately**
- **Always verify correctness against CPU/SVM** (same seed → same passed_shots)
- **Never compare benchmarks across different node types** (MI300X vs MI350X), and
  not across nodes within `mi350x-es` either — the partition is heterogeneous
- **Commit at every step** — never lose code during refactoring
- **Ignore the `stale/` folder** unless explicitly asked

## Three GPU Tiers

| Tier | peak_rank | Amplitude Storage | `V2_STRIDE` | Barrier | `IS_OWNER` |
|------|----------|-------------------|-------------|---------|------------|
| Register | 0–4 | Stack alloca (≤16) | `1u` | empty | `1` |
| Coop | 5–10 | LDS addrspace(3) (≤1024) | `256u` | fenced | `(t == 0)` |
| Global | 11+ | HBM pointer | `256u` | fenced | `(t == 0)` |

The tier knob parameterises **cooperation, not arithmetic**. Register tier collapses
`v2_tid()` to 0 and `V2_REDUCE2` to plain assignment; the coop tiers keep the
`ds_bpermute_b32` reduction, whose **summation order is part of the ABI**
(`v2_ops.h:235-236`) — changing it changes results.

peak_rank = max active_k during circuit execution, determined at compile time by the
backend's `VirtualRegisterManager`. `StatevectorSqueezePass` minimizes peak_rank —
many circuits named "rank N" actually compile to rank 0–1. Of 353 fixtures, only 22
reach rank ≥ 5. **Read the measured LDS size, not the fixture name.**

## Benchmark Circuits

See `BENCHMARKS.md` for the full reference. Quick set spanning all 3 tiers:

- `tests/fixtures/incremental/01_frame_only/frame_h.stim` — rank 0, register
- `tests/fixtures/large/circuit_d3_p0.001.stim` — rank 4, register
- `tests/fixtures/qv10.stim` — rank 10, coop
- `tools/bench/fixtures/qv20_seed42.stim` — rank 20, global
- `tests/fixtures/large/surface_d7_t19.stim` — rank 12, global, 721 qubits

## Performance Tracking

- `V2_performance/runs/INDEX.md` — progressive run index, one row per benchmark run
- `V2_performance/runs/20260727T125310Z_report-final-allfixtures/` — **the canonical
  corpus**: job 50793, 26 circuits, wins 26/26, geomean 0.573. Every headline number
  in the report resolves here.
- `V2_performance/tools/bench_all.sh` — the full 26-circuit + rocprofv3 harness
- `PERFORMANCE_OPTIONS.md` — optimization roadmap (**stale**: pre-V2, rank-0/1 only)
- `PERF_ANALYSIS.md`, `BENCHMARKS.md` — counter data and circuit classification

Raw rocprofv3 CSVs under `runs/*/raw/` are gitignored — bulky and reproducible. The
digested `gpu/*.json` + `summary.md` are the durable record.

## Key Files

**V2 (current):**

- `src/clifft/gpu/mlir/v2/v2_kernel.cc` — device source, HBM pool sizing, tier select
- `src/clifft/gpu/mlir/v2/v2_ops.h` + `v2_ops_body.inc` — the operand library. One
  set of `static inline` bodies taking operands **and pre-op `active_k` by value**,
  compiled twice: the interpreter passes runtime values, the specializer passes
  literals. Byte-exactness by construction.
- `src/clifft/gpu/mlir/v2/v2_specializer.cc` — emits one call per instruction
- `src/clifft/gpu/mlir/v2/v2_compile_cache.cc` — cache keyed on toolchain, arch,
  bitcode dir and `device_header_ident()`
- `src/clifft/gpu/mlir/v2/device_abi.h` — the ABI the two builds must agree on

**Shared:**

- `src/clifft/gpu/runtime/hsa_runtime.cc` — HSA runtime wrapper
- `src/clifft/gpu/runtime/hsa_kernel_dispatch.h` — PersistentDispatcher
- `src/clifft/backend/backend.cc` — HIR lowering, peak_rank computation
- `src/clifft/optimizer/statevector_squeeze_pass.cc` — peak_rank minimization

**V1 (kept for comparison, not built by default):**

- `src/clifft/gpu/mlir/mlir_emit.cc`, `mlir_kernel_cache.cc`

## Build

```bash
# V2 — HSA only, no HIP
cmake -B build-v2-nohip -DCLIFFT_ENABLE_MLIR_V2=ON -DCLIFFT_ENABLE_HIP=OFF
cmake --build build-v2-nohip -j$(nproc) --target run_v2

# SVM / Hybrid / V1 reference build
cmake -B build-gpu-mlir -DCLIFFT_ENABLE_HIP=ON -DCLIFFT_ENABLE_MLIR=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx950   # gfx942 for MI300X
```

## Running

```bash
# V2
build-v2-nohip/run_v2 --circuit circuit.stim --shots 10000 --seed 1

# SVM (interpreter, the reference)
build-gpu-mlir/run_gpu --circuit circuit.stim --shots 10000 --seed 1 --no-postselection

# CPU reference (no GPU needed)
build-gpu-mlir/run_gpu --circuit circuit.stim --shots 10000 --seed 1 --cpu-reference
```

**`V2_SPECIALIZE=1` is opt-in** (`v2_kernel.cc:358`). Unset, a benchmark measures
V2's *interpreter* and reports what looks like a 2–3× regression. `bench_all.sh`
exports it; ad-hoc runs must set it themselves.

Other env knobs that matter: `V2_GATE_SELFTEST` / `V2_GATE_BISECT` (correctness
gating), `V2_DUST_EPS` (amplitude clamp — see below), `V2_GLOBAL_WGS` (override the
HBM pool grid), `V2_REGISTER` / `V2_NO_REGISTER` (force a tier).

## Two Bugs Worth Knowing About

**The barrier was execution-only.** A bare `s_barrier` orders execution but not
memory — it carries `IntrNoMem`. 92.8% of specialized barriers were unfenced. The
fix is a release/acquire pair in `v2_barrier()`. HIP's `__syncthreads()` already
expands to the fenced sequence, so dropping HIP made this our problem to get right.

**`V2_DUST_EPS` was calibrated against fp64.** At `1e-18` it sat four decades below
`fp32_eps² = 1.4e-14`, so on fp32 amplitudes the clamp never fired and
`sample_branch`'s early return desynced the PRNG. Now `1e-11`. A threshold that
never fires is a correctness bug, not dead code.

## SLURM

**radha is the login node — no GPU, no ROCm, no `/opt/rocm`.** All GPU builds and
runs go through SLURM.

- `mi350x-es` — MI350X (gfx950), 288 GB HBM3E, 256 CUs, 8 XCDs. **Heterogeneous**;
  nodes f13–21 have stale HIP headers. 2h max walltime. `--gpus=1` is required.
- `mi300x` — MI300X (gfx942), 192 GB HBM3
- `--cpu-reference` works on any partition (no GPU needed for compile-only tasks)

```bash
sbatch V2_performance/tools/bench_all.sh <label>
```
