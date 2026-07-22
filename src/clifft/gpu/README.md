# GPU Backend

The `gpu/` directory implements Clifft's GPU-accelerated sampling backend for AMD GPUs (MI300X, MI350X) via HIP and HSA.

## Directory Structure

```
gpu/
├── gpu_types.h          Shared constants, types (GpuComplex, GpuInstr, BlockCounts, etc.)
├── gpu_sampler.h        Public API: gpu_sample_survivors(), GpuSamplerOptions
├── device_program.h/cc  FlattenedProgram: bytecode → GPU instruction flattening
│
├── runtime/             HSA/HIP dispatch and device management
│   ├── hsa_runtime      HSA agent, memory pool, timing wrappers
│   ├── hsa_kernel_dispatch   AQL packet submission (one-shot dispatch)
│   └── hsa_persistent_dispatch   PersistentDispatcher: pre-allocated HSA resources
│
├── sampler/             SVM interpreter kernels (data-driven dispatch)
│   ├── hip_sampler.hip  Three __global__ kernels (register, coop, global-coop)
│   │                    + device functions + host orchestrator
│   └── hip_sampler_stub.cc   CPU-only build stub
│
├── codegen/             Compiled megakernel path (HIP text emission)
│   ├── kernel_codegen.h Public API: generate_compiled_kernel{,_coop,_global}()
│   ├── kernel_codegen.cc   UsedFunctions analysis, pipeline analysis
│   ├── codegen_types.h  Internal declarations shared across codegen .cc files
│   ├── emit_preamble    Preamble, constant pool, device function emission
│   ├── emit_instructions   Straight-line instruction emitters (register + coop)
│   ├── emit_register_kernel   Register-tier kernel generator (rank ≤ 4)
│   ├── emit_coop_kernel       LDS/coop-tier kernel generator (rank 5-10)
│   ├── emit_global_kernel     Global/HBM-tier kernel generator (rank 11-19)
│   ├── kernel_cache     clang++ compilation pipeline + disk cache
│   └── ops/             Per-category device function string literals (.inc)
│       ├── frame_ops.inc        Pauli frame tracking (CNOT, CZ, H, S, SWAP)
│       ├── array_ops.inc        Single-qubit amplitude sweeps
│       ├── multi_qubit_ops.inc  Multi-controlled gates
│       ├── expand_ops.inc       Qubit expansion operations
│       ├── measurement_ops.inc  Dormant/active measurements
│       ├── noise_ops.inc        Noise channels, readout noise
│       ├── unitary_ops.inc      Fused U2/U4 unitaries
│       ├── exp_val_ops.inc      Expectation value computation
│       └── reduction_ops.inc    Warp-shuffle reductions, coop state
│
└── mlir/                MLIR→LLVM-IR alternative codegen (experimental)
    ├── mlir_codegen      Textual MLIR emission (LLVM dialect)
    └── mlir_kernel_cache llc/lld compilation + disk cache
```

## Two Execution Paths

### SVM Interpreter (`sampler/`)
The default path. `hip_sampler.hip` contains three `__global__` kernels that interpret `GpuInstr` bytecode at runtime via a switch-dispatch loop. Each operation is a real `__device__` HIP function. Supports all operations but pays interpretation overhead.

### Compiled Megakernel (`codegen/`)
The `--hybrid` path. `kernel_codegen.cc` walks the flattened bytecode and emits a complete, self-contained HIP C++ source file with straight-line calls — no switch dispatch. The source is compiled by `clang++` at runtime and loaded via HSA. Three tiers handle different amplitude array sizes:
- **Register tier** (rank ≤ 4): amplitudes in thread-private registers
- **LDS/Coop tier** (rank 5-10): amplitudes in LDS, 256 threads cooperate per shot
- **Global/HBM tier** (rank 11-19): amplitudes in HBM, per-XCD work stealing

## Adding a New Operation

1. Add the `__device__` function to `sampler/hip_sampler.hip` (interpreter path)
2. Add the `static const char*` string literal to the appropriate `codegen/ops/*.inc` file
3. Add emission logic to `codegen/emit_preamble.cc` (conditional include) and `codegen/emit_instructions.cc` (instruction emission for both register and coop paths)
4. If the operation needs new constant pool data, extend `emit_constant_pool()` in `emit_preamble.cc`
5. Update `UsedFunctions` in `kernel_codegen.h` and `compute_derived()` in `kernel_codegen.cc`

## Build Flags

- `CLIFFT_ENABLE_HIP=ON` — Build GPU backend (requires ROCm)
- `CLIFFT_ENABLE_MLIR=ON` — Build experimental MLIR codegen path (requires LLVM with MLIR)
- `CMAKE_HIP_ARCHITECTURES` — Target GPU architecture (default: gfx942)
