# GPU Kernel Performance Analysis

Hardware counter data collected via `rocprof` on MI300X (gfx942, ROCm 7.2.3).
Same node, same job, interleaved OLD (clifft-amd) vs NEW (clifft-gpu).

## Tier 2: Shared-Coop Kernel (d5 p=0.001, peak_rank=10)

| Counter | OLD | NEW | Delta | Impact |
|---------|-----|-----|-------|--------|
| arch_vgpr | 72 | 128 | +78% | **Occupancy drops from 7 to 4 waves/SIMD** |
| sgpr | 96 | 112 | +17% | Minor |
| lds | 30,208 | 30,208 | same | Not a factor |
| scratch | 0 | 0 | same | No spilling |
| SQ_INSTS_VALU | 3,917M | 5,088M | +30% | More ALU work per shot |
| SQ_INSTS_SALU | 10,368M | 13,997M | +35% | More scalar work (switch dispatch) |
| SQ_WAIT_INST_ANY | 1,922M | 2,603M | +35% | More stalls from lower occupancy |

**Root cause:** 128 VGPRs in the new kernel vs 72 in the old. The extended
opcode handlers (U2, U4, EXP_VAL) are inlined into `execute_shot_coop`,
increasing register pressure. On MI300X, 128 VGPRs limits occupancy to 4
waves/SIMD (vs 7 at 72 VGPRs), reducing the GPU's ability to hide memory
latency.

## Tier 1: Per-Thread Kernel (hook-injection d3, peak_rank=0)

| Counter | OLD | NEW | Delta | Impact |
|---------|-----|-----|-------|--------|
| arch_vgpr | 68 | 84 | +24% | Moderate (still 6 waves/SIMD) |
| sgpr | 96 | 96 | same | — |
| scratch | 1,280 | 1,344 | +5% | Slight spill increase |
| SQ_INSTS_VALU | 173M | 178M | +2.6% | Minimal |
| SQ_INSTS_SALU | 249M | 250M | +0.6% | Minimal |
| SQ_WAIT_INST_ANY | 79M | 83M | +5.4% | Minor stall increase |

The `__noinline__` on extended opcode handlers keeps per-thread VGPR pressure
manageable (84 vs 68). Performance is within noise (~0-2%).

## After Optimization: __noinline__ on coop_u2_sweep / coop_u4_sweep

Extracted the array-sweep computation from coop_array_u2 and coop_array_u4
into separate `__noinline__` functions (`coop_u2_sweep`, `coop_u4_sweep`)
that take raw pointers instead of CoopShotState. The `__syncthreads()` barriers
remain in the wrapper functions.

| Counter | Before | After | Baseline |
|---------|--------|-------|----------|
| arch_vgpr | 128 | **108** | 72 |
| accum_vgpr | 0 | 4 | 0 |
| sgpr | 112 | **96** | **96** |

**Result:** SGPRs match baseline. VGPRs dropped 20 (128→108). Performance
now matches baseline across all 3 tiers:

| Tier | OLD shots/s | NEW shots/s | Delta |
|------|-------------|-------------|-------|
| T1: Per-thread (rank 0) | 11.58M | 11.75M | +1.4% |
| T2: Shared-coop (rank 10) | 1.16M | 1.17M | +1.0% |
| T3: Global-coop (rank 19) | 62.9K | 63.8K | +1.4% |

Tools used: `rocprof` PMC counters (SQ_WAVES, arch_vgpr, sgpr, SQ_INSTS_VALU,
SQ_INSTS_SALU, SQ_WAIT_INST_ANY) for measurement-driven optimization.
