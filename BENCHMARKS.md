# Benchmark Circuit Suite

Reference circuits for GPU kernel performance testing across all 3 tiers.
Circuits are classified by **compiled peak_rank** (post-StatevectorSqueezePass HIR optimization),
not by the circuit name or intended rank.

## Tier Classification

| Tier | Compiled Rank | Amplitude Array | Storage | Sync Mechanism |
|------|--------------|-----------------|---------|----------------|
| Register | 0-4 | ≤16 complex | Stack alloca | None |
| Coop | 5-10 | ≤1024 complex | LDS (addrspace 3) | s_barrier + fences |
| Global | 11-19 | ≤524K complex | HBM device memory | s_barrier + atomics + XCD work-stealing |

## Register Tier (rank 0-4) — 331 circuits

Representative benchmarks:

| Circuit | Rank | Qubits | Detectors | Instructions | Notes |
|---------|------|--------|-----------|-------------|-------|
| `tests/fixtures/incremental/01_frame_only/frame_h.stim` | 0 | 2 | 1 | 4 | Trivial, frame-only, sub-ms kernel |
| `tests/fixtures/incremental/02_single_expand/four_t.stim` | 0 | 5 | 1 | 11 | 4 T gates squeezed to rank 0 |
| `tests/fixtures/large/circuit_d3_p0.001.stim` | 4 | 21 | 20 | 344 | Surface code d=3 with T injection |
| `tests/fixtures/large/surface_d7_t5.stim` | 3 | 721 | 672 | 4151 | Large circuit, many qubits, register tier |
| `tests/fixtures/large/surface_d11_t5.stim` | 3 | 2761 | 2640 | 16432 | Very large, register tier |

All `rank_sweep/` circuits (56 total) compile to rank 0-1 due to StatevectorSqueezePass.
All `sweep/` circuits (188 total) compile to rank 0-1.
All `extreme_*` circuits compile to rank 1.
Color codes (all variants) compile to rank 0 (pure Clifford).

## Coop Tier (rank 5-10) — 11 circuits

| Circuit | Rank | Qubits | Detectors | Instructions | Notes |
|---------|------|--------|-----------|-------------|-------|
| `tests/fixtures/qv10.stim` | 10 | 10 | 0 | 140 | Quantum volume, compact, fast compile |
| `tests/fixtures/cultivation_d5.stim` | 10 | 112 | 107 | 1720 | Real QEC cultivation protocol |
| `tests/fixtures/large/circuit_d5_p0.0005.stim` | 10 | 112 | 107 | 1720 | Surface d=5 + T injection, low noise |
| `tests/fixtures/large/circuit_d5_p0.001.stim` | 10 | 112 | 107 | 1720 | Surface d=5 + T injection |
| `tests/fixtures/large/circuit_d5_p0.002.stim` | 10 | 112 | 107 | 1720 | Surface d=5 + T injection |
| `tests/fixtures/large/circuit_d5_p0.003.stim` | 10 | 112 | 107 | 1720 | Surface d=5 + T injection |
| `tests/fixtures/large/circuit_d5_p0.005.stim` | 10 | 112 | 107 | 1720 | Surface d=5 + T injection, high noise |
| `tests/fixtures/large/surface_d7_t10.stim` | 7 | 721 | 672 | 4134 | Surface d=7, 10 T injections |
| `tests/fixtures/large/surface_d7_t15.stim` | 10 | 721 | 672 | 4371 | Surface d=7, 15 T injections |
| `tests/fixtures/large/surface_d9_t10.stim` | 7 | 1521 | 1440 | 8859 | Surface d=9, 10 T injections |
| `tests/fixtures/large/surface_d11_t10.stim` | 7 | 2761 | 2640 | 16374 | Surface d=11, 10 T injections |

## Global Tier (rank 11+) — 6 circuits

| Circuit | Rank | Qubits | Detectors | Instructions | Notes |
|---------|------|--------|-----------|-------------|-------|
| `tools/bench/fixtures/qv20_seed42.stim` | 20 | 20 | 0 | 418 | Quantum volume 20, highest rank, compact |
| `tests/fixtures/large/surface_d7_t19.stim` | 12 | 721 | 672 | 4296 | Surface d=7, 19 T injections |
| `tests/fixtures/large/surface_d9_t15.stim` | 11 | 1521 | 1440 | 8952 | Surface d=9, 15 T injections |
| `tests/fixtures/large/surface_d9_t19.stim` | 13 | 1521 | 1440 | 9359 | Surface d=9, 19 T injections |
| `tests/fixtures/large/surface_d11_t15.stim` | 11 | 2761 | 2640 | 16415 | Surface d=11, 15 T injections |
| `tests/fixtures/large/surface_d11_t19.stim` | 14 | 2761 | 2640 | 16521 | Surface d=11, 19 T injections |

## Quick Benchmark Set

For fast iteration (< 5 min total), use these 5 circuits spanning all tiers:

```bash
QUICK_BENCH=(
    "tests/fixtures/incremental/01_frame_only/frame_h.stim"        # rank 0, register
    "tests/fixtures/large/circuit_d3_p0.001.stim"                  # rank 4, register
    "tests/fixtures/qv10.stim"                                     # rank 10, coop
    "tools/bench/fixtures/qv20_seed42.stim"                        # rank 20, global
    "tests/fixtures/large/surface_d7_t19.stim"                     # rank 12, global, large
)
```

## Full Benchmark Set

For comprehensive testing (30-60 min), add these:

```bash
FULL_BENCH=(
    # Register (rank 0-4)
    "tests/fixtures/incremental/01_frame_only/frame_h.stim"       # rank 0
    "tests/fixtures/incremental/02_single_expand/four_t.stim"     # rank 0
    "tests/fixtures/large/circuit_d3_p0.001.stim"                 # rank 4
    "tests/fixtures/large/surface_d7_t5.stim"                     # rank 3
    # Coop (rank 5-10)
    "tests/fixtures/qv10.stim"                                    # rank 10
    "tests/fixtures/cultivation_d5.stim"                          # rank 10
    "tests/fixtures/large/circuit_d5_p0.001.stim"                 # rank 10
    "tests/fixtures/large/surface_d7_t10.stim"                    # rank 7
    "tests/fixtures/large/surface_d7_t15.stim"                    # rank 10
    "tests/fixtures/large/surface_d9_t10.stim"                    # rank 7
    # Global (rank 11+)
    "tools/bench/fixtures/qv20_seed42.stim"                       # rank 20
    "tests/fixtures/large/surface_d7_t19.stim"                    # rank 12
    "tests/fixtures/large/surface_d9_t15.stim"                    # rank 11
    "tests/fixtures/large/surface_d9_t19.stim"                    # rank 13
    "tests/fixtures/large/surface_d11_t15.stim"                   # rank 11
    "tests/fixtures/large/surface_d11_t19.stim"                   # rank 14
    # Global, high rank. QV circuits do NOT squeeze, so peak_rank ==
    # num_qubits exactly; these are what exercise kGlobalMaxPeakRank
    # above the old cap of 19.
    "tests/fixtures/large/qv20_L8_seed42.stim"                    # rank 20
    "tests/fixtures/large/qv21_L8_seed42.stim"                    # rank 21
    "tests/fixtures/large/qv22_L6_seed42.stim"                    # rank 22
    "tests/fixtures/large/qv23_L5_seed42.stim"                    # rank 23
    "tests/fixtures/large/qv24_L4_seed42.stim"                    # rank 24
)
```

Ranks above are the **compiled** `peak_rank` (post-StatevectorSqueezePass, as
measured by `V2_performance/scratch/rank_probe`), not the qubit count. The
surface family squeezes hard — `surface_d11_t19` uses far more than 14 qubits.

## Fixture Rank Census (measured 2026-07-26)

`rank_probe` was run over **every** live fixture (353 files under
`tests/fixtures/`, `tools/bench/fixtures/`, `docs/guide/circuits/`). Full
results: `V2_performance/scratch/all_ranks.txt`. The distribution is lopsided:

| compiled rank | files | tier |
|---|---|---|
| 0-1 | 326 | register |
| 3-4 | 5 | register |
| 7-10 | 11 | coop |
| 11-14 | 5 | global |
| 20-24 | 6 | global |

**Only 22 of 353 fixtures reach rank >= 5**, i.e. only 22 exercise the coop or
global tier at all. The other 331 collapse to rank 0-1 under
StatevectorSqueezePass — including all 188 of `tests/fixtures/sweep/` and all
56 of `tests/fixtures/rank_sweep/`, whose *filenames* advertise ranks they do
not have (`rank_q17_r12_d1.stim` compiles to rank 1). Do not infer coverage
from a fixture's name or qubit count; probe it.

The benchmark matrix in `~/.claude/skills/benchmark-all/scripts/run_sweep.sh`
is exactly those 22, plus 4 register-tier circuits for the low end. That is
full coverage of every circuit in the tree that reaches coop or global.

Nothing in the tree exceeds rank 24, so the top of the raised
`kGlobalMaxPeakRank = 26` is not yet exercised by a fixture.

## StatevectorSqueezePass Impact

The compiler's `StatevectorSqueezePass` (enabled by default) minimizes peak_rank:

- **Sweep 1 (Eager Compaction)**: Pulls MEASURE operations leftward, enabling earlier amplitude folding
- **Sweep 2 (Lazy Expansion)**: Pushes T_GATE/PHASE_ROTATION rightward, deferring expansion

This means circuits must have T gates that **cannot be separated from their measurements** by
commutation. The `surface_d*_t*` circuits achieve this because T injections are interleaved with
stabilizer measurements that create classical data dependencies via DETECTORs.

Circuits where T gates on independent qubits are followed by a single measurement block
(like all `rank_sweep/` circuits) will be squeezed to rank 0-1.

## Generating High-Rank Circuits

Scripts in `scripts/`:
- `generate_high_rank_circuits.py` — rank_sweep (defeated by squeeze pass)
- `generate_large_circuits.py` — surface codes with T injection (survives squeeze pass)
- `generate_sweep_circuits.py` — parametric T-gate sweeps
- `generate_qv_circuits.py` — Quantum Volume; **exact** rank control

Two patterns survive the squeeze pass, with different guarantees:

**`surface_d*_t*`** — surface code + T injections interleaved with stabilizer
measurements. Survives, but the resulting rank is *emergent*: it depends on how
much the pass can still fold, so `surface_d11_t19` lands at rank 14, and asking
for a specific rank means guess-and-check.

**Quantum Volume** (`generate_qv_circuits.py`) — each layer applies a random
SU(4) to a fresh random pairing of *all* qubits, so nothing can be deferred or
folded and `peak_rank == num_qubits` **exactly**. This is the only reliable way
to hit a target rank on demand, and the only way to exercise the global tier
above rank ~14.

Verify any new fixture's real rank with `V2_performance/scratch/rank_probe`,
which runs the identical compile path as `run_v2` (trace → HIR passes →
reference syndrome → lower → bytecode passes) and prints `peak_rank`. Do not
infer rank from qubit count.

## Methodology

- Always cache MLIR compilation — never include compilation time in performance comparisons
- Report kernel_seconds and host overhead separately
- Verify correctness: MLIR passed_shots must match SVM passed_shots (same seed)
- Run on NUMA-bound nodes; never compare across different node types
- Use ≥3 runs; report median kernel time
- See PERFORMANCE_OPTIONS.md for optimization roadmap
