# Interesting Circuits for GPU Backend Benchmarking

Circuits that stress different aspects of the GPU sampler, organized by
the tier they exercise and what bottleneck they expose.

---

## Tier 1: Per-Thread (peak_rank ≤ 4, amplitudes in VGPRs)

| Circuit | rank | Instrs | Qubits | shots/s | Bottleneck |
|---------|------|--------|--------|---------|------------|
| `target_qec.stim` | 0 | 217 | 25 | 48M | Launch overhead |
| `surface_d3_r3.stim` | 0 | 212 | 33 | 19M | Instruction count |
| `surface_d5_r5.stim` | 0 | 997 | 145 | 18M | Instruction count |
| `surface_d7_r7.stim` | 0 | 2749 | 385 | 16M | Instruction count |
| `surface_d7_r14.stim` | 0 | 4298 | 721 | 15M | Instruction count + qubit width |
| `color_d7.stim` | 0 | 1470 | 163 | 18M | Instruction count |
| `circuit_d3_p0.001.stim` | 4 | 344 | 21 | 19M | **Only rank=4 circuit** |
| `hook_inject_d3_t_gate.stim` | 1 | 379 | 17 | 9.5M | T-gate (non-Clifford) |
| `rep_d5_r100.stim` | 0 | 1661 | 405 | 17M | Very deep (100 rounds) |

**Key circuit:** `circuit_d3_p0.001.stim` — the ONLY test circuit that reaches
rank=4 (per-thread tier boundary). All other circuits compile to rank ≤ 1.

## Tier 2: Shared-Coop (peak_rank 5-10, amplitudes in LDS)

| Circuit | rank | Instrs | Qubits | shots/s | Bottleneck |
|---------|------|--------|--------|---------|------------|
| `cultivation_d5.stim` | 10 | 1720 | 112 | 5.6M | **Sync barriers** (GEAK +40%) |
| `circuit_d5_p=0.001.stim` | 10 | 1720 | 112 | 11M | Same circuit, different noise |
| `qv10.stim` | 10 | 140 | 10 | ~4M | Short but high rank |

**Key circuit:** `cultivation_d5.stim` — the canonical QEC benchmark. Used for
all optimization comparisons. GEAK warp-shuffle gave +40% here.

## Tier 3: Global-Coop (peak_rank 11-19, amplitudes in HBM)

| Circuit | rank | Instrs | Qubits | shots/s | Bottleneck |
|---------|------|--------|--------|---------|------------|
| `circuit_d7_p0.0005.stim` | **19** | 5472 | 355 | **144K** | **HBM bandwidth** |

**Key circuit:** `circuit_d7_p0.0005.stim` — the ONLY circuit that reaches rank=19.
This is the ultimate stress test: 4MB amplitude array per shot, 5472 instructions,
entirely HBM-bound. All dispatch optimizations converge to ~144K shots/s here.

## Circuits That Stress GPU Memory

| Circuit | rank | Array Size/Shot | Max Concurrent Shots (192GB) |
|---------|------|-----------------|------------------------------|
| rank=0 | 0 | 8 bytes | ~24 billion |
| rank=4 | 4 | 128 bytes | ~1.5 billion |
| rank=10 | 10 | 8 KB (LDS) | N/A (LDS-resident) |
| rank=19 | 19 | **4 MB** (HBM) | **~48,000** |

## Circuits Blocked by 128-Qubit Limit

These circuits require the `kPauliWords=4` fix (256-qubit support):

| Circuit | Qubits | Status |
|---------|--------|--------|
| `surface_d9_r9.stim` | 188 | **Now works** with fix |
| `surface_d11_r11.stim` | 252 | **Now works** with fix |
| `surface_d13_r13.stim` | 376 | Needs kPauliWords=6 |
| `surface_d15_r15.stim` | 528 | Needs kPauliWords=9 |

## Where to Find More Circuits

1. **SOFT repo** (`haoliri0/SOFT`): d3 and d5 cultivation only
2. **Strilanc/magic-state-cultivation**: Can generate d7+ via Python tool
3. **Stim built-in generators**: `stim.Circuit.generated('surface_code:...')`
4. **clifft-amd**: `circuit_d5_p=0.001.stim` and `circuit_d7_p0.0005.stim`

## Key Observation

Clifft's `StatevectorSqueezePass` aggressively minimizes peak_rank. Synthetic
circuits with many T-gates still compile to rank ≤ 1. Only real QEC circuits
with genuinely entangled T-gate blocks (cultivation, distillation) achieve rank > 4.
This means **the per-thread tier handles nearly all workloads in practice**.
