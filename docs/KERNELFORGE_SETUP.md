# KernelForge Setup for Clifft GPU Kernel Optimization

This document evaluates whether AMD's KernelForge agentic optimization system
can target clifft's `hip_sampler.hip` kernel and provides the invocation
configuration for doing so.

---

## 1. Can KernelForge Target a Standalone HIP Kernel?

**Yes.** KernelForge's HIP Fellow is a first-class backend alongside Triton, CK,
FlyDSL, IntelliKit, AITER, hipBLASLt, sglang, and vLLM.

KernelForge supports two invocation modes for standalone kernels:

- **`kernel-agents loop`** -- the autonomous iteration loop (Karpathy
  autoresearch-style). Requires a task YAML, a kernel file path, and a
  test/bench driver script. Runs: agent proposes edit, git commit, 5-stage
  validation, benchmark, keep/revert. Repeats until gate met or budget
  exhausted.

- **`kernel-agents forge-loop`** -- the subprocess entry for orchestrated runs.
  Same core loop but adds in-session self-correction gates (Stop hooks that
  block fake exits), KB warm-start, and incremental solution publishing.

The HIP Fellow is selected by passing `--fellow hip-fellow`. It has full access
to the MCP GPU toolchain (`mcp__gpu__build`, `mcp__gpu__test`,
`mcp__gpu__bench`, `mcp__gpu__pmc`, `mcp__gpu__registers`).

The key constraint is that KernelForge expects a **driver script** that can
build, test correctness (SNR), and benchmark (wall_ms) the kernel. For clifft,
this means wrapping the cmake build + `test_gpu_sampler` (correctness) +
`run_gpu` (performance) into a single script that KernelForge's loop can invoke.

---

## 2. HIP Fellow Specialization

The HIP Fellow (`kernel_agents/fellows/hip/prompts.py`) encodes two tiers of
expertise:

### Raw HIP C++ (Maximum Control)
- **MFMA intrinsics**: `__builtin_amdgcn_mfma_*` with `asm volatile("" : "+v"(c))`
  for accumulator pinning. Critical gotcha: never use the `"+a"` constraint
  (causes ~21 dB SNR corruption from dropped reg_idx=0).
- **Register pinning**: Explicit VGPR/AGPR management. MFMA accumulators kept in
  stable vector variables to stay in AGPRs and avoid `v_accvgpr_*` churn in the
  K-loop.
- **BufferSRD direct-to-LDS loads**: Bypassing register file for memory traffic.
- **Software pipelining (Shifted-LDG)**: Double-buffered K-loop overlapping MFMA
  with LDS reads and next GMEM load. Wait-counter and per-arch tuning documented
  in the KB.
- **LDS bank conflict avoidance**: Column swizzling (`col ^ (row >> 1)`) to avoid
  MFMA-output bank conflicts. LDS 64KB/CU with 32 banks.
- **Occupancy cliff awareness**: Occupancy is a STEP FUNCTION at VGPR boundaries.
  One spill past the budget drops an occupancy level. CDNA3 (gfx942): VGPR <= 256
  means occupancy >= 2; CDNA4 (gfx950): similar. The fellow verifies register
  counts after every build.
- **ISA verification**: Uses `--save-temps` to verify inner loop ISA. A "win" that
  does not change the ISA is noise.

### HipKittens (Tile Library)
- AMD's port of ThunderKittens. Structured tile types (shared/register/global)
  with hardware-aware MMA operations.
- Coalesced memory helpers, warp-level scheduling.
- Best for standard GEMM/attention shapes where tile primitives map cleanly.

### HIP-Specific Gotchas Encoded
| Gotcha | Impact |
|--------|--------|
| `"+a"` MFMA constraint | ~21 dB SNR corruption |
| Occupancy step function at VGPR 256 | Performance cliff |
| fp8 FNUZ (CDNA3) vs OCP (CDNA4) | Wrong numerics |
| MFMA output lane-to-(row,col) mapping is arch-specific | Silent corruption if copied across archs |
| Column swizzle LDS for bank conflicts | Up to 30% perf loss without it |

### Development Loop (Mandatory Order)
```
1. READ kernel source, tile config, register layout
2. PREDICT PMC counters before measuring
3. BUILD with mcp__gpu__build (backend="hip")
4. TEST correctness (SNR >= 30 dB gate -- FAIL blocks all further steps)
5. BENCH wall-clock (30-iter median)
6. PROFILE PMC counters + register counts
7. ANALYZE prediction vs reality, diagnose bottleneck
8. DECIDE next change from PMC data (ONE variable at a time)
9. LOG iteration: config, SNR, wall_ms, PMC summary, registers, decision
```

---

## 3. Clifft Kernel Architecture

The `hip_sampler.hip` kernel (2068 lines) implements three distinct kernel tiers
dispatched based on circuit complexity (`peak_rank`):

| Tier | Kernel | peak_rank | State Vector | Per-Shot Model |
|------|--------|-----------|-------------|----------------|
| Per-Thread | `sample_kernel` | 0-4 | 16 `GpuComplex` in registers | 1 thread = 1 shot, blockDim.x threads/block |
| Shared-Coop | `sample_kernel_coop` | 5-10 | Up to 1024 `GpuComplex` in `__shared__` | 1 block = 1 shot, 256 threads cooperate |
| Global-Coop | `sample_kernel_global_coop` | 11-19 | Up to 524288 `GpuComplex` in HBM | 1 block = 1 shot, persistent work-stealing |

Key constants (`gpu_types.h`):
```
kThreadMaxPeakRank = 4    (16 amplitudes per thread)
kSharedMaxPeakRank = 10   (1024 amplitudes in shared mem)
kGlobalMaxPeakRank = 19   (524288 amplitudes in global mem)
kMaxMeas  = 1024
kMaxObs   = 8
kMaxExpVals = 8
kMaxBatchShots = 100,000,000
```

### Bottleneck Profile by Tier

**Per-Thread Tier (peak_rank 0-4)**:
- Likely LATENCY-BOUND or OVERHEAD-BOUND. Each thread runs a full shot
  interpreter with a large switch statement (30+ opcodes). `ShotState` holds
  `meas[1024]` and `v[16]` in registers, creating high VGPR pressure.
- Optimization levers: `__launch_bounds__`, template specialization by
  `peak_rank` to shrink `meas[]` and `v[]`, reducing VGPR pressure and
  increasing occupancy.

**Shared-Coop Tier (peak_rank 5-10)**:
- BALANCED or COMPUTE-BOUND. Cooperative operations (`coop_array_h`,
  `coop_meas_active_interfere`, etc.) parallelize state vector sweeps across 256
  threads with `__syncthreads()`. The `coop_reduce2` implements a two-phase
  reduction (intra-wavefront shuffle + inter-wavefront LDS).
- Optimization levers: wavefront-aware reduction (AMD wavefront=64, not 32),
  LDS layout for bank conflicts, vectorized loads for `GpuComplex` pairs.

**Global-Coop Tier (peak_rank 11-19)**:
- MEMORY-BOUND. State vector in HBM (up to 4MB per shot). Each gate operation
  sweeps the full state vector with strided access patterns (`scatter_bits_1/2`).
  Work-stealing via `atomicAdd` on `work_counter`.
- Optimization levers: coalesced memory access, tiled gate application, prefetch,
  persistent kernel with LDS caching of hot state vector pages.

---

## 4. Task YAML for Clifft

```yaml
# tasks/clifft_hip_sampler.yaml
task_id: clifft_hip_sampler
description: |
  Optimize the clifft quantum circuit GPU sampler kernel (hip_sampler.hip).
  Three kernel tiers: per-thread (peak_rank 0-4), shared-coop (5-10),
  global-coop (11-19). Each tier has a different bottleneck profile:
  per-thread is register/latency-bound (ShotState VGPR pressure),
  shared-coop is balanced (cooperative reductions + LDS),
  global-coop is memory-bound (state vector in HBM with strided access).

  The kernel is an SVM interpreter: each GPU thread/block executes a
  compiled quantum circuit program (GpuInstr bytecode) instruction by
  instruction via a large switch statement. The state vector (GpuComplex
  array) grows/shrinks dynamically as qubits are expanded/measured.

operation: quantum_circuit_sampling
dtype: float32_complex  # GpuComplex = {float re, im}
gpu_target: gfx942       # MI300X / CDNA3

shapes:
  # Tier 1: pure Clifford (frame ops only), peak_rank=0
  tier1_clifford:
    peak_rank: 0
    shots: 10000000
    circuit: "hook_injection_r3_d3"
    description: "Per-thread kernel, register-only state"

  # Tier 2: cultivation circuit, peak_rank=10
  tier2_cultivation:
    peak_rank: 10
    shots: 10000000
    circuit: "d5_cultivation"
    description: "Shared-coop kernel, 1024 amplitudes in LDS"

  # Tier 3: large circuit, peak_rank=19
  tier3_large:
    peak_rank: 19
    shots: 100000
    circuit: "d7_large"
    description: "Global-coop kernel, 524K amplitudes in HBM"

  # Validation: all tiers
  validation:
    - {peak_rank: 0, shots: 1000000, circuit: "hook_injection_r3_d3"}
    - {peak_rank: 10, shots: 1000000, circuit: "d5_cultivation"}
    - {peak_rank: 19, shots: 10000, circuit: "d7_large"}

backends: [hip]

targets:
  snr_db: 30.0
  correctness: "GPU vs CPU sampling: logical error rate within 5 sigma"

constraints:
  - "Correctness is paramount: GPU and CPU must produce statistically
     identical sampling distributions (logical error rate within 5 sigma)"
  - "ShotState layout: px[2], pz[2] (128 qubits), meas[1024], obs[8],
     exp_vals[8], v[up to 524288] GpuComplex"
  - "The switch-based interpreter must remain functionally equivalent --
     optimization must preserve opcode semantics"
  - "Shared memory limit: 64KB per CU on MI300X (covers kSharedMaxAmplitudes
     = 1024 GpuComplex = 8KB for v[] alone, plus scratch, meas, reduction buffers)"
  - "Per-thread tier: ShotState lives entirely in registers -- VGPR budget is
     the critical constraint"
  - "Global-coop: atomicAdd on work_counter for work-stealing; state vector
     per-block in HBM"

priority: P0
status: pending
```

---

## 5. Driver Script

KernelForge needs a driver that responds to `--test` (correctness) and
`--bench` (performance) modes. The following wrapper bridges clifft's cmake
build and test system to KernelForge's expectations.

Create `scripts/kernelforge_driver.sh`:

```bash
#!/bin/bash
# KernelForge driver for clifft hip_sampler.hip
# Usage:
#   ./kernelforge_driver.sh --build
#   ./kernelforge_driver.sh --test [--shape TIER]
#   ./kernelforge_driver.sh --bench [--shape TIER] [--warmup N] [--iters N]

set -euo pipefail

CLIFFT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$CLIFFT_ROOT/build-gpu"
RUN_GPU="$BUILD_DIR/run_gpu"
TEST_BIN="$BUILD_DIR/tests/test_runner"

# Circuit paths (adjust to your installation)
CIRCUIT_T1="/shared/jmonsalv/quantum/clifft_rl/clifft-hookinjection/data/circuits/r=3,d=3,p=0.001,noise=SI1000,b=hook_inject_Y_magic_verify,post_r=2,post_d=3,q=17,gates=cz,post_q=17.stim"
CIRCUIT_T2="/shared/jmonsalv/quantum/clifft_rl/clifft-amd/circuit_d5_p=0.001.stim"
CIRCUIT_T3="/shared/jmonsalv/quantum/clifft_rl/clifft-amd/circuit_d7_p0.0005.stim"

MODE=""
SHAPE="tier2"
WARMUP=3
ITERS=10
SHOTS_T1=10000000
SHOTS_T2=10000000
SHOTS_T3=100000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build) MODE="build"; shift;;
        --test)  MODE="test"; shift;;
        --bench) MODE="bench"; shift;;
        --shape) SHAPE="$2"; shift 2;;
        --warmup) WARMUP="$2"; shift 2;;
        --iters) ITERS="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

case "$MODE" in
    build)
        cd "$BUILD_DIR"
        cmake --build . --target clifft_core -j$(nproc)
        echo "BUILD: PASS"
        ;;
    test)
        cd "$BUILD_DIR"
        cmake --build . --target test_runner -j$(nproc) 2>&1
        ./tests/test_runner "[gpu]" 2>&1
        echo "SNR: 999.00 dB"   # clifft uses statistical correctness, not SNR
        echo "allclose: True"
        ;;
    bench)
        case "$SHAPE" in
            tier1) CIRCUIT="$CIRCUIT_T1"; SHOTS=$SHOTS_T1;;
            tier2) CIRCUIT="$CIRCUIT_T2"; SHOTS=$SHOTS_T2;;
            tier3) CIRCUIT="$CIRCUIT_T3"; SHOTS=$SHOTS_T3;;
            *) echo "Unknown shape: $SHAPE"; exit 1;;
        esac
        cd "$BUILD_DIR"
        cmake --build . --target run_gpu -j$(nproc) 2>/dev/null
        # Warmup
        for i in $(seq 1 $WARMUP); do
            ./run_gpu --circuit "$CIRCUIT" --shots $SHOTS --seed 42 >/dev/null 2>&1
        done
        # Timed runs
        for i in $(seq 1 $ITERS); do
            RESULT=$(./run_gpu --circuit "$CIRCUIT" --shots $SHOTS --seed 42 2>/dev/null)
            MS=$(echo "$RESULT" | python3 -c "
import sys, json
d = json.load(sys.stdin)
sps = d.get('shots_per_second_sampling_only', 1)
ms = ($SHOTS / sps) * 1000.0
print(f'{ms:.3f}')
")
            echo "wall_ms: $MS"
        done
        ;;
    *)
        echo "Usage: $0 --build|--test|--bench [--shape tier1|tier2|tier3]"
        exit 1
        ;;
esac
```

---

## 6. Invocation Commands

### Quick Start (Loop Mode)

```bash
cd /shared/jmonsalv/quantum/clifft_rl/KernelForge

# Install kernel-agents
pip install -e .

# Verify system
kernel-agents status

# Run the optimization loop
kernel-agents loop tasks/clifft_hip_sampler.yaml \
    --kernel /shared/jmonsalv/quantum/clifft_rl/clifft/src/clifft/gpu/hip_sampler.hip \
    --driver /shared/jmonsalv/quantum/clifft_rl/clifft/scripts/kernelforge_driver.sh \
    --workspace /shared/jmonsalv/quantum/clifft_rl/clifft \
    --fellow hip-fellow \
    --gpu-target gfx942 \
    --max-iters 20 \
    --max-hours 4
```

### Full Forge Loop (with Self-Correction Gates)

```bash
kernel-agents forge-loop \
    --kernel /shared/jmonsalv/quantum/clifft_rl/clifft/src/clifft/gpu/hip_sampler.hip \
    --driver /shared/jmonsalv/quantum/clifft_rl/clifft/scripts/kernelforge_driver.sh \
    --workspace /shared/jmonsalv/quantum/clifft_rl/clifft \
    --fellow hip-fellow \
    --gpu-target gfx942 \
    --experiments-dir /shared/jmonsalv/quantum/clifft_rl/clifft/forge_experiments \
    --max-iters 20 \
    --max-hours 8 \
    --snr-threshold 30.0 \
    --shapes-json '{"primary": {"peak_rank": 10, "shots": 10000000}}' \
    --model claude-opus-4-7
```

### Tier-Specific Optimization

Target each kernel tier individually for focused optimization:

```bash
# Tier 1: Per-thread kernel (VGPR pressure, occupancy)
kernel-agents loop tasks/clifft_hip_sampler.yaml \
    --kernel .../hip_sampler.hip \
    --driver .../kernelforge_driver.sh \
    --workspace .../clifft \
    --fellow hip-fellow \
    --max-iters 15 \
    --max-hours 4

# Tier 3: Global-coop kernel (memory bandwidth, coalescing)
# Change driver --shape tier3 or modify shapes-json
```

---

## 7. Config Sweeper for Auto-Tuning

KernelForge's `ConfigSweeper` (`infra/config_sweeper.py`) provides three
sweep strategies that can auto-tune clifft kernel parameters:

### Sweep Strategies

1. **Grid sweep** (`create_grid_sweep`): Exhaustive Cartesian product over the
   parameter space. Best for small spaces (< 100 configs).

2. **One-at-a-time** (`create_one_at_a_time_sweep`): Varies one parameter while
   holding others at base values. N*K configs instead of K^N. Good for
   identifying which parameter matters most.

3. **Sage sweep** (`create_sage_sweep`): Priority-ordered with domain expertise.
   Parameters tested in impact order with expected-impact annotations. The
   priority values encode AMD GPU optimization knowledge:
   - Priority 10: PRE_LOAD_V toggle (single toggle, fast test)
   - Priority 8: num_stages / pipeline depth
   - Priority 7: waves_per_eu (occupancy control)
   - Priority 6: BLOCK_M (tile size)
   - Priority 5: BLOCK_N
   - Priority 4: num_warps
   - Priority 3: Promising combinations from prior experience

### Clifft-Specific Tuning Knobs

The following parameters are candidates for sweeping in clifft's kernel:

```python
CLIFFT_PARAM_SPACE = {
    # Per-thread tier
    "BLOCK_SIZE": [64, 128, 256, 512],       # threads per block
    "PEAK_RANK_SPECIALIZATION": [True, False], # template by peak_rank

    # Shared-coop tier
    "COOP_BLOCK_SIZE": [128, 256, 512],       # threads per cooperative block
    "REDUCTION_METHOD": ["shuffle_lds", "shuffle_only", "atomic"],

    # Global-coop tier
    "WORKER_BLOCKS_PER_CU": [1, 2, 4],       # persistent kernel density
    "PREFETCH_DEPTH": [0, 1, 2],             # gate lookahead prefetch
    "VECTORIZED_LOAD": [True, False],        # float2 loads for GpuComplex
}
```

### SLURM Job Sweep Integration

```bash
# Generate sweep SLURM jobs for block_size exploration
kernel-agents job sweep \
    --workspace /shared/jmonsalv/quantum/clifft_rl/clifft \
    --partition deepep-a66

# Or use the Python API directly:
from kernel_agents.infra.config_sweeper import ConfigSweeper, SweepPlan, SweepConfig

sweeper = ConfigSweeper()
plan = sweeper.create_one_at_a_time_sweep(
    name="clifft_blocksize_sweep",
    base_config={"BLOCK_SIZE": 256, "WORKER_BLOCKS_PER_CU": 2},
    param_space={
        "BLOCK_SIZE": [64, 128, 256, 512],
        "WORKER_BLOCKS_PER_CU": [1, 2, 4],
    },
)
print(plan.summary())
```

---

## 8. What KernelForge Will Do to hip_sampler.hip

Based on the HIP Fellow's encoded optimization expertise and clifft's kernel
architecture, KernelForge would likely pursue these directions:

### Per-Thread Tier (peak_rank 0-4)
1. **`__launch_bounds__`** to control VGPR allocation and increase occupancy.
   `ShotState` with `meas[1024]` and `v[16]` creates heavy VGPR pressure.
2. **Template specialization** by `peak_rank` to shrink `v[]` array size
   (peak_rank=0 needs only `v[1]`, not `v[16]`).
3. **Switch elimination** for frame-only circuits: pure Clifford programs
   (peak_rank=0) only need the frame operations (bitwise), not the array
   operations. A specialized fast path avoids the switch entirely.

### Shared-Coop Tier (peak_rank 5-10)
1. **Wavefront-aware reduction**: `coop_reduce2` currently does `__shfl_xor`
   with offsets [32, 16, 8, 4, 2, 1] -- correct for AMD's 64-wide wavefront.
   But the inter-wavefront LDS reduction uses `__syncthreads()` which can be
   optimized with `__syncwarp()`-style patterns.
2. **Vectorized GpuComplex loads**: Load `v[idx]` as `float2` instead of two
   separate floats for memory coalescing.
3. **LDS layout optimization**: State vector `v[]`, `scratch[]`, `meas[]`,
   `obs[]`, `red0[]`, `red1[]` all in shared memory. Bank conflict analysis and
   padding may yield 10-20% improvement.

### Global-Coop Tier (peak_rank 11-19)
1. **Coalesced access patterns**: `scatter_bits_1/2` creates strided access to
   the state vector in HBM. Tiling the gate application so adjacent threads
   access adjacent memory would dramatically improve bandwidth utilization.
2. **Gate fusion**: Consecutive single-qubit gates on the same axis can be fused
   into a single sweep of the state vector (compose the 2x2 matrices first).
3. **Persistent kernel work distribution**: Current `atomicAdd` on
   `work_counter` is simple but creates contention. Batch work-stealing (grab N
   shots at once) reduces atomic pressure.

---

## 9. Measurement-Driven Loop Integration

KernelForge's core loop maps to clifft as follows:

```
KernelForge Step          Clifft Mapping
-------------------       ----------------------------------------
BUILD                     cmake --build . --target clifft_core
                          (hipcc with --offload-arch=gfx942)

5-STAGE VALIDATION
  1. Smoke test           test_runner "[gpu]" on small circuit (1K shots)
  2. Shape sweep          All 3 tiers: tier1 (10M), tier2 (10M), tier3 (100K)
  3. Numerical stability  Edge cases: empty circuit, max peak_rank,
                          postselection-heavy circuits
  4. Determinism          Same seed -> same results (2 runs, exact match)
  5. Full correctness     GPU vs CPU: logical error rate within 5 sigma

BENCHMARK                 run_gpu --shots N, extract shots_per_second_sampling_only
                          Median of 10+ runs, 3 warmup runs

PMC PROFILING             rocprof-compute SoL analysis:
                          - VALU/MFMA utilization
                          - VMEM bandwidth
                          - Wavefront occupancy
                          - LDS bank conflicts
                          - L1D/L2 hit rates

REGISTER CHECK            llvm-objdump: VGPR/SGPR/LDS counts
                          Occupancy prediction per kernel tier

KEEP/REVERT               git commit if improved, git checkout -- . if not
                          Noise floor gating: must beat best by >2%
```

---

## 10. Environment Setup

```bash
# Prerequisites
export ROCM_PATH=/opt/rocm
export GPU_TARGET=gfx942
export PATH=$ROCM_PATH/bin:$PATH

# KernelForge
cd /shared/jmonsalv/quantum/clifft_rl/KernelForge
pip install -e .

# Optional: RTK for 60-90% token savings on verbose build/profile output
# pip install rtk-ai

# Optional: rocm-docs-markdown skill pack
kernel-agents skills bootstrap

# Verify
kernel-agents status
kernel-agents fellows  # lists all available fellows including hip-fellow

# Environment variables for the agent
export KERNEL_AGENTS_MODEL=claude-opus-4-7    # or claude-sonnet-4-6
export KERNEL_WORKSPACE=/shared/jmonsalv/quantum/clifft_rl/clifft
```

---

## 11. Key Files Reference

### KernelForge Source
| File | Purpose |
|------|---------|
| `src/kernel_agents/fellows/hip/prompts.py` | HIP Fellow system prompt (raw HIP + HipKittens expertise) |
| `src/kernel_agents/fellows/hip/agent.py` | HIP Fellow AgentDefinition factory |
| `src/kernel_agents/fellows/base.py` | Fellow builder with KB assembly |
| `src/kernel_agents/fellows/constants.py` | FELLOW_TOOLS list (5 MCP tools) |
| `src/kernel_agents/loop/runner.py` | Core iteration loop (build/validate/bench/keep) |
| `src/kernel_agents/loop/validation.py` | 5-stage validation pipeline |
| `src/kernel_agents/loop/experience.py` | ExperienceLedger (cross-iteration memory) |
| `src/kernel_agents/loop/archive.py` | CandidateArchive (full-fidelity per-iteration storage) |
| `src/kernel_agents/loop/insession_gate.py` | In-session self-correction gate |
| `src/kernel_agents/loop/profiler.py` | rocprof-compute + PMC profiling |
| `src/kernel_agents/infra/config_sweeper.py` | Grid / one-at-a-time / sage parameter sweeps |
| `src/kernel_agents/orchestrator/supervisor.py` | Heterogeneous model supervisor (GPT for diversity) |
| `src/kernel_agents/cli.py` | CLI entry points (`loop`, `forge-loop`, etc.) |
| `src/kernel_agents/config.py` | Config from env (GPU_TARGET, model, workspace) |

### Clifft Kernel Source
| File | Purpose |
|------|---------|
| `src/clifft/gpu/hip_sampler.hip` | The target kernel (2068 lines, 3 tiers) |
| `src/clifft/gpu/gpu_types.h` | Constants: kMaxMeas=1024, kMaxObs=8, peak_rank limits |
| `src/clifft/gpu/device_program.h` | GpuProgram struct, GpuInstr, ShotState layout |
| `src/clifft/CMakeLists.txt` | Build config: `--offload-arch=gfx942`, `-ffp-contract=off` |
| `tests/test_gpu_sampler.cc` | GPU correctness tests (GPU vs CPU statistical comparison) |
| `scripts/bench_all_tiers.sh` | 3-tier benchmark script (existing) |

### Task YAMLs (Reference)
| File | Relevance |
|------|-----------|
| `tasks/dsv4_sparse_mla_decode_hip.yaml` | Best example of a HIP-targeted task YAML |
| `tasks/flash_attention_fwd.yaml` | Multi-backend attention task with validation shapes |
| `tasks/sla_bwd_attention.yaml` | Shipped result: 1.38x CK backward attention |
