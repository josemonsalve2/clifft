# Apex Integration for clifft GPU Kernel Optimization

## 1. Can Apex Optimize a Custom HIP Kernel?

**Yes.** Apex's `optimize-kernel` subcommand is explicitly designed to optimize
any standalone kernel -- not just vLLM/SGLang model kernels. The CLAUDE.md for
Apex documents this as a first-class workflow:

```
python3 workload_optimizer.py optimize-kernel \
  --kernel /path/to/baseline.hip \
  --kernel-type hip \
  --kernel-name <name> \
  --correctness-mode pytorch \
  --reference /path/to/ref.py \
  -r /path/to/results \
  --max-iterations 3 --max-turns 25 --target gfx942
```

The `optimize-kernel` subcommand bypasses the E2E LLM benchmark pipeline entirely.
It accepts `--kernel-type hip` alongside `triton` and `pytorch`, and supports YAML
spec files for full configuration. The agent receives baseline source code, MCP
tools, and a correctness definition, then iterates to produce an optimized solution.

The three correctness modes available are:
- **pytorch**: Compare optimized kernel output against a PyTorch reference
  implementation via `torch.allclose`.
- **library_test**: Run an existing test suite (e.g., `pytest tests/...`).
- **accordo**: HSA-level GPU buffer comparison for HIP/C++ kernels where
  source-level comparison is infeasible.

For clifft's `hip_sampler.hip`, the **accordo** mode is the most natural fit since
the kernel is a raw HIP C++ file, not a Python/Triton module. Alternatively,
**library_test** mode can invoke clifft's own test binary
(`test_gpu_sampler`) to validate correctness.


## 2. YAML Task Spec for clifft's hip_sampler.hip

### 2a. Using library_test mode (recommended -- uses existing C++ tests)

```yaml
# clifft_hip_sampler_spec.yaml
task_id: clifft_hip_sampler
kernel_path: /shared/jmonsalv/quantum/clifft_rl/clifft/src/clifft/gpu/hip_sampler.hip
kernel_type: hip
kernel_name: hip_sampler
description: >
  Optimize the clifft quantum circuit GPU sampler for MI300X.
  The kernel implements a stabilizer-frame SVM interpreter that simulates
  Clifford+T circuits shot-by-shot on GPU. Three dispatch tiers:
  - sample_kernel: one thread per shot (peak_rank <= 4, 16 amplitudes per thread)
  - sample_kernel_coop: one threadblock per shot (peak_rank 5-8, shared memory)
  - sample_kernel_global_coop: persistent blocks with work-stealing (peak_rank 9+)
  Key operations: complex amplitude butterfly (array_h, array_cnot, array_u2, array_u4),
  measurement with warp-shuffle reduction (coop_reduce2), Pauli frame tracking.
  GEAK already found +40% via warp-shuffle optimization on the coop reduction path.
gpu_arch: gfx942
framework: custom

ground_truth:
  mode: library_test
  unit_test_command: >
    cd /shared/jmonsalv/quantum/clifft_rl/clifft/build-gpu &&
    ctest --test-dir . -R gpu_sampler --output-on-failure
```

### 2b. Using accordo mode (HSA-level buffer comparison)

```yaml
# clifft_hip_sampler_accordo_spec.yaml
task_id: clifft_hip_sampler
kernel_path: /shared/jmonsalv/quantum/clifft_rl/clifft/src/clifft/gpu/hip_sampler.hip
kernel_type: hip
kernel_name: hip_sampler
description: >
  Optimize the clifft quantum circuit GPU sampler for MI300X.
  See description above.
gpu_arch: gfx942
framework: custom

ground_truth:
  mode: accordo
  accordo_config:
    correctness:
      backend: accordo
      accordo:
        kernel_name: sample_kernel_coop
        reference_binary: /shared/jmonsalv/quantum/clifft_rl/clifft/build-gpu/clifft_sampler
        tolerance: 0.0001
```

### 2c. CLI Invocation (no YAML)

```bash
cd /shared/jmonsalv/quantum/clifft_rl/Apex
source .venv/bin/activate
export MAGPIE_ROOT=$(cd ../Magpie && pwd)

python3 workload_optimizer.py optimize-kernel \
  --kernel /shared/jmonsalv/quantum/clifft_rl/clifft/src/clifft/gpu/hip_sampler.hip \
  --kernel-type hip \
  --kernel-name hip_sampler \
  --correctness-mode library_test \
  --test-cmd "cd /shared/jmonsalv/quantum/clifft_rl/clifft/build-gpu && ctest -R gpu_sampler --output-on-failure" \
  -r $HOME/clifft_apex_results \
  --max-iterations 3 --max-turns 25 \
  --agent-backend claude --target gfx942
```

### 2d. Grading an Existing Optimization (e.g., the GEAK warp-shuffle result)

```bash
python3 workload_optimizer.py grade-kernel \
  --kernel /shared/jmonsalv/quantum/clifft_rl/clifft/src/clifft/gpu/hip_sampler.hip \
  --solution /path/to/optimized_hip_sampler.hip \
  --kernel-type hip \
  --correctness-mode library_test \
  --test-cmd "cd /shared/jmonsalv/quantum/clifft_rl/clifft/build-gpu && ctest -R gpu_sampler --output-on-failure" \
  -r $HOME/clifft_apex_results --json
```


## 3. Apex gpu-info MCP Server: MI300X Architecture Data

The gpu-info MCP server (`tools/mcps/gpu_info/server.py`) contains a comprehensive
`ARCH_SPECS` database with full MI300X (gfx942) specifications:

| Spec | MI300X (gfx942) Value |
|------|----------------------|
| Architecture | CDNA3 |
| Compute Units | 304 |
| Stream Processors | 19456 |
| Wavefront Size | 64 |
| LDS per CU | 64 KB |
| L2 Cache | 256 MB |
| HBM Bandwidth | 5.3 TB/s |
| HBM Capacity | 192 GB |
| FP64 TFLOPS | 163.4 |
| FP32 TFLOPS | 163.4 |
| FP16 TFLOPS | 1307.4 |
| BF16 TFLOPS | 1307.4 |
| FP8 TFLOPS | 2614.9 |
| MFMA Support | Yes |
| Memory Coalescing | 128 bytes |
| Max Workgroup Size | 1024 |

**MFMA instructions for gfx942:**
- `mfma_f32_32x32x8_f16`
- `mfma_f32_16x16x16_f16`
- `mfma_f32_32x32x16_bf16`
- `mfma_f32_16x16x32_bf16`
- `mfma_f32_32x32x16_fp8`
- `mfma_f64_16x16x4_f64`

**Optimal tile sizes:**
- GEMM M: [32, 64, 128, 256]
- GEMM N: [32, 64, 128, 256]
- GEMM K: [8, 16, 32, 64]

**Optimization priorities (from Apex):**
1. Use MFMA instructions for matrix operations
2. Target 128-byte memory coalescing
3. Use LDS for data reuse (64 KB per CU)
4. Optimize for high HBM bandwidth (5.3 TB/s)
5. Consider FP8/BF16 for inference workloads

**Optimization hints by kernel type:** The `get_arch_optimization_hints` tool returns
per-type guidance for: gemm, reduction, elementwise, attention, moe, quantization,
and general kernels. The reduction hints are most relevant to clifft:
- Use warp/wave shuffle for intra-wavefront reduction (wavefront = 64)
- Use `__shfl_down` for reduction within a wavefront
- For large reductions, use two-stage: wave shuffle + shared memory
- Consider `__ballot` and `__popc` for boolean reductions

The gpu-info server auto-detects the local GPU via `rocminfo` and can be overridden
with `set_target_arch("gfx942")` for cross-compilation scenarios.

### Applicability to clifft

The MI300X specs directly inform several optimization opportunities in `hip_sampler.hip`:

1. **Wavefront-aware reduction**: The existing `coop_reduce2` function uses
   `__shfl_xor` with wavefront=64, which matches the gfx942 spec. The 4-wavefront
   inter-wavefront reduction via shared memory is correctly sized for blockDim=256.

2. **LDS usage**: `sample_kernel_coop` uses `__shared__` arrays for state vectors
   (`kSharedMaxAmplitudes` complex values), measurement arrays, Pauli frame bits,
   and reduction buffers. Total shared memory must fit in 64 KB per CU.

3. **Memory coalescing**: The `ShotState.v[]` array accesses use strided patterns
   via `scatter_bits_1/2`, which may not be 128-byte coalesced. This is a key
   optimization target.

4. **MFMA relevance**: The U4 gate application (`array_u4`/`coop_u4_sweep`) performs
   4x4 complex matrix-vector products. For larger peak_rank, these are effectively
   batched GEMMs that could benefit from MFMA instructions, though the current
   sizes (4x4) are too small for direct MFMA use.


## 4. Fusion Advisor Analysis for the SVM Kernel

### Can fusion-advisor detect fusion opportunities in hip_sampler.hip?

**Not directly in its current form.** The fusion-advisor MCP server is designed for
LLM inference kernel pipelines. Its 8 built-in fusion patterns target:
- elementwise_chain (add, relu, gelu, silu, ...)
- gemm_epilogue (matmul + bias + activation)
- attention_block (QKV projection + attention + output)
- norm_activation (layernorm/rmsnorm + silu/gelu)
- residual_norm (residual add + normalization)
- rotary_embedding (RoPE + attention)
- quantize_dequantize (quant/dequant + GEMM)
- moe_gating (topk + scatter + expert GEMM)

**None of these patterns match quantum circuit simulation operations.** The clifft
kernel implements:
- Pauli frame tracking (bitwise XOR on uint64_t words)
- Single-qubit gates (H, S, T, Rz, U2) -- butterfly-pattern amplitude updates
- Two-qubit gates (CNOT, CZ, SWAP, U4) -- paired butterfly-pattern updates
- Measurement with Born-rule sampling (norm computation + branch + collapse)
- Noise injection via hazard-based sampling

### Potential Quantum-Specific Fusion Opportunities (manual analysis)

While fusion-advisor cannot detect these automatically, the following fusion
opportunities exist within `hip_sampler.hip`:

#### 4a. Consecutive Frame Operations (already optimized)

The kernel already optimizes consecutive frame-only operations by skipping
`__syncthreads()` between them (see `is_frame_only_op()` check at lines
1378, 1389, 1400, 1410, 1419). Frame operations are pure bitwise updates on
`px[]`/`pz[]` arrays and are inherently fuseable since they only touch thread-0.

#### 4b. Gate-Measurement Fusion

When a gate (e.g., `array_h`) is immediately followed by a measurement on the
same qubit, the two operations could be fused: apply the gate and compute the
measurement probability in a single pass over the amplitude array, avoiding
the second `scatter_bits_1` traversal. This pattern is common in Clifford+T
circuits where the circuit ends with a Hadamard-basis measurement layer.

Current cost: 2 passes over `2^(active_k-1)` amplitudes.
Fused cost: 1 pass.
Estimated benefit: up to 2x for the gate+measurement pair.

#### 4c. Consecutive Array Gate Fusion (U2 chaining)

When two `OP_ARRAY_U2` instructions operate on the same qubit axis, the two
2x2 unitary matrices can be multiplied on the host before kernel launch, reducing
two amplitude sweeps to one. The compiler's `fused_u2` mechanism already does
this at the program compilation level, but there may be residual opportunities
that the current fusion pass misses (e.g., U2 gates separated by frame-only ops
on different qubits).

#### 4d. Expand + Gate Fusion

`OP_EXPAND` followed by a gate on the newly expanded qubit is a common pattern
(e.g., `expand_plain` then `array_h`). These could be fused: instead of copying
amplitudes to the upper half and then applying H across both halves, compute
the final amplitudes directly:
```
v[i]      = (v_orig[i] + v_orig[i]) * invSqrt2  // = v_orig[i] * sqrt(2) * invSqrt2
v[i+half] = (v_orig[i] - v_orig[i]) * invSqrt2  // = 0
```
This pattern already exists partially as `expand_t` (expand + T-gate fusion).
Generalizing it to arbitrary single-qubit gates would save one full amplitude
sweep per expand+gate pair.

#### 4e. Measurement Chain Optimization

When multiple qubits are measured consecutively (common at circuit end), each
measurement reduces `active_k` by 1 and does a full-array norm computation.
A batched measurement could compute all norms in parallel for qubits that are
measured in a fixed basis, reducing `N` full-array scans to fewer passes.


## 5. Adapting Fusion Advisor for Quantum Kernels

To use fusion-advisor for clifft, you would need to extend its pattern database.
Here is a sketch of quantum-specific patterns that could be added:

```python
QUANTUM_FUSION_PATTERNS = {
    "gate_chain_same_qubit": {
        "name": "Consecutive Gates on Same Qubit",
        "pattern": ["unitary_1q", "unitary_1q"],
        "benefit": "Multiply matrices on host, single amplitude sweep",
        "memory_saving": "medium",
        "complexity": "low",
    },
    "expand_gate_fusion": {
        "name": "Expand + Single-Qubit Gate",
        "pattern": ["expand", "unitary_1q"],
        "benefit": "Single pass instead of copy + sweep",
        "memory_saving": "medium",
        "complexity": "low",
    },
    "gate_measurement_fusion": {
        "name": "Gate + Measurement on Same Qubit",
        "pattern": ["unitary_1q", "measurement"],
        "benefit": "Combined gate application and Born-rule sampling",
        "memory_saving": "medium",
        "complexity": "medium",
    },
    "measurement_batch": {
        "name": "Consecutive Measurements (Fixed Basis)",
        "pattern": ["measurement", "measurement+"],
        "benefit": "Batched norm computation for multiple qubits",
        "memory_saving": "high",
        "complexity": "high",
    },
}
```

This extension would require modifying `tools/mcps/fusion_advisor/server.py` to
add quantum kernel categories to `KERNEL_TYPE_MAP` and the new patterns to
`FUSION_PATTERNS`. However, per Apex's read-only codebase policy, this should be
done in a fork or via configuration injection rather than modifying Apex source.


## 6. Practical Integration Steps

### Step 1: Environment Setup

```bash
# Ensure Apex is set up
cd /shared/jmonsalv/quantum/clifft_rl/Apex
bash setup.sh  # Interactive: choose Claude backend
source .venv/bin/activate
export MAGPIE_ROOT=$(cd ../Magpie && pwd)

# Verify GPU
rocm-smi --showproductname  # Should show MI300X / gfx942
```

### Step 2: Build clifft with GPU Support

```bash
cd /shared/jmonsalv/quantum/clifft_rl/clifft
cmake -B build-gpu -DCLIFFT_GPU=ON -DCMAKE_HIP_ARCHITECTURES=gfx942
cmake --build build-gpu -j$(nproc)
```

### Step 3: Verify Baseline Correctness

```bash
cd /shared/jmonsalv/quantum/clifft_rl/clifft/build-gpu
ctest -R gpu_sampler --output-on-failure
```

### Step 4: Run Apex Optimization

```bash
cd /shared/jmonsalv/quantum/clifft_rl/Apex

python3 workload_optimizer.py optimize-kernel \
  --kernel-spec /shared/jmonsalv/quantum/clifft_rl/clifft/docs/clifft_hip_sampler_spec.yaml \
  -r $HOME/clifft_apex_results \
  --max-iterations 3 --max-turns 25 \
  --agent-backend claude --target gfx942
```

### Step 5: Review Results

```bash
cat $HOME/clifft_apex_results/standalone_result.json
# Check: compiled=true, correct=true, speedup > 1.0
```

### Step 6: Profile with rocprof-compute (Independent of Apex)

Apex's reflector and bottleneck classification can also be used independently
for profiling guidance:

```bash
# Profile the hip_sampler kernel
rocprof-compute profile \
  --kernel-name sample_kernel_coop \
  -- /shared/jmonsalv/quantum/clifft_rl/clifft/build-gpu/clifft_sampler \
       --backend gpu --shots 1000000 circuit.stim

# Classify bottleneck using Apex's categories:
# VALU > 60%, VMEM < 40%  -> COMPUTE-BOUND
# VMEM > 60%, VALU < 40%  -> MEMORY-BOUND
# Both < 40%              -> LATENCY-BOUND
# LDS > 50%               -> LDS-BOUND
```


## 7. Scoring Model

Apex scores kernel optimizations as follows:

| Component | Points | Condition |
|-----------|--------|-----------|
| Compiled | +20 | Solution compiles and defines expected function |
| Correct | +100 | Passes all unit tests against baseline |
| Speedup | +100 * speedup_score | `speedup_score = 100 + (speedup - 1) * 200` for speedup >= 1.0 |

Reference scores:
- Compile only: 20
- Compile + correct + 1.0x (no speedup): 220
- Compile + correct + 1.2x: 260
- Compile + correct + 1.4x (like GEAK result): 300
- Compile + correct + 2.0x: 420

The RL reward function adds gating: performance reward only unlocks when
`correctness_score > 0.95`, preventing fast-but-wrong optimizations from
earning positive reward.


## 8. Caveats and Limitations

1. **HIP kernel integration path**: Apex's hot-patching mechanism replaces `.py`
   files (Triton auto-recompiles) or standalone `.so` files. For clifft's
   `hip_sampler.hip`, the kernel is compiled into `libclifft_core.so` via CMake,
   not as a standalone `.so`. The optimize-kernel workflow can still optimize
   the source code, but integration requires rebuilding clifft rather than
   Apex's file-replacement mechanism.

2. **Correctness challenge**: clifft's kernel correctness is statistical -- it
   produces Monte Carlo samples of quantum circuit outcomes. Exact output
   comparison is not meaningful; correctness is validated by statistical tests
   (error rates within expected bounds). The library_test mode using `ctest`
   handles this, but the pytorch mode with `torch.allclose` does not apply.

3. **No Magpie integration**: Magpie is designed for LLM inference benchmarking
   (serving throughput, latency percentiles). It cannot benchmark clifft directly.
   The optimize-kernel pathway's built-in benchmark (compile + measure latency)
   is the appropriate mechanism.

4. **Single-file scope**: Apex's agent optimizes a single kernel file. The clifft
   GPU backend spans multiple files (`hip_sampler.hip`, `device_program.h`,
   `gpu_sampler.h`, `gpu_types.h`). The agent would need all related headers
   available and understanding of the cross-file dependencies.

5. **Build system**: Apex expects to compile HIP kernels with `hipcc`. clifft uses
   CMake with `CMAKE_HIP_ARCHITECTURES`. A wrapper script or custom build command
   in the YAML spec may be needed.
