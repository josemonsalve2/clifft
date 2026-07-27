# MLIR Kernel Performance Optimization Roadmap

## Current State (Jul 23, 2026)

### Circuit Tier Classification (compiled peak_rank, post-StatevectorSqueezePass)

**Important**: The compiler's `StatevectorSqueezePass` minimizes peak_rank by pulling
measurements forward and deferring T gates. Circuits named "rank_q33_r19" compile to
peak_rank=1 because independent T gates on separate qubits are serialized. Only circuits
with T gates on entangled qubits that cannot be squeezed achieve high compiled rank.

| Tier | Compiled Rank | Circuits Available | Representative Benchmarks |
|------|--------------|-------------------|--------------------------|
| Register | 0-4 | 331 | `frame_h` (r0), `circuit_d3_p0.001` (r4), `surface_d7_t5` (r3, 721q) |
| Coop | 5-10 | 11 | `qv10` (r10, 10q, 140i), `cultivation_d5` (r10, 112q, 1720i), `surface_d7_t10` (r7, 721q) |
| Global | 11+ | 6 | `qv20_seed42` (r20, 20q, 418i), `surface_d7_t19` (r12, 721q), `surface_d11_t19` (r14, 2761q) |

**Real QEC circuits with genuine high rank:**
- `circuit_d5_p*` (5 variants) — rank 10, 112 qubits, 1720 instructions (coop)
- `surface_d*_t10/t15/t19` — rank 7-14, 721-2761 qubits (coop/global)
- `qv20_seed42` — rank 20, 20 qubits, 418 instructions (global, compact)

### Measured Performance (MI350X gfx950, Jul 23 2026)

**Per-process cold start (10k shots, rank_q17_r4_d1 — REGISTER TIER ONLY, rank=1):**

| Path | sample_seconds | buffer_alloc | dispatch_loop | kernel_ms |
|------|---------------|-------------|--------------|-----------|
| SVM (interpreter) | 59 ms | 52 ms | 8 ms | 0.67 |
| Hybrid (compiled) | 49 ms | 48 ms | 2.5 ms | 0.16 |
| MLIR (pure HSA) | 259 ms | 47 ms | 0.26 ms | 0.024 |
| MLIR first run (w/compile) | 1464 ms | 47 ms | 0.26 ms | 0.024 |

**WARNING**: All prior benchmarks were on rank-0/1 circuits where GPU compute is
negligible. Coop and global tier circuits have NOT been benchmarked yet.

**MLIR dispatch loop breakdown (warm kernel, warm pool):**

| Component | Time | Notes |
|-----------|------|-------|
| HSA executable loading | 170-206 ms | One-time per process (static) |
| Buffer allocation (14× device_malloc) | 47-59 ms | One-time per process (persistent pool) |
| hsa_amd_memory_fill (memset) | ~0.05 ms | Per-batch |
| PersistentDispatcher::dispatch() | ~0.15 ms | memcpy kernargs + AQL write + wait |
| memcpy_d2h (result readback) | ~0.05 ms | Per-batch, direct CPU read |
| **Total dispatch loop** | **0.26 ms** | **Per-sample call (amortized)** |
| **GPU kernel execution** | **0.024 ms** | **Sub-millisecond** |

**Correctness: SVM=MLIR on 14/15 rank_sweep circuits (surface_d5_r5 has a bug)**

**Large circuit compilation times (first run, cached after):**

| Circuit | Qubits | Lines | MLIR compile | Hybrid compile |
|---------|--------|-------|-------------|----------------|
| rank_q17_r4_d1 | 17 | ~100 | 220 ms | ~270 ms |
| rank_q33_r15_d1 | 33 | ~200 | 427 ms | ~400 ms |
| surface_d5_r5 | 49 | 153 | 55 s | 0.06 s (AOT) |
| surface_d7_r7 | 97 | 239 | 193 s | 0.06 s (AOT) |
| cultivation_d7 | 80 | 1655 | 224 s | 24 s |

### GPU Kernel Time (rocprofv3 via GEAK profile_kernel.sh)

| Kernel | GPU Duration | Platform |
|--------|-------------|----------|
| SVM sample_kernel | 20-27 µs | MI350X/MI300X |
| MLIR compiled_mlir_kernel | ~30 µs (est.) | MI350X |
| HIP copy kernels (×6) | 16-24 µs | Both |

---

## P0: Host Dispatch Overhead (Dominant Bottleneck)

### [DONE] P0.1: Pre-load MLIR kernel outside dispatch loop
- **Impact**: 175 ms → 2.4 ms dispatch loop (73× improvement)
- **Root cause**: `compile_or_load_mlir_kernel()` called lazily inside the dispatch loop. HSA executable loading (`hsa_executable_load_agent_code_object`) takes ~170ms per `.hsaco`.
- **Fix**: Moved to `static` pre-load before the dispatch loop. One-time cost per process, amortized to 0 on subsequent calls.
- **File**: `hip_sampler.hip` lines 2218-2240

### [IMPLEMENTING] P0.2: Persistent device buffer pool
- **Impact**: 49 ms → ~0.2 ms buffer allocation (245× improvement expected)
- **Root cause**: 14× `device_malloc` + `allow_access` each take ~3ms = 42ms. Plus 14× `memcpy` ~7ms.
- **Fix**: Single `device_malloc` for the entire pool, pack all data with 256-byte alignment. Re-use across calls. Only `memcpy` changes per circuit.
- **File**: `hip_sampler.hip` buffer allocation section

### [DONE] P0.3: Eliminate HIP from hot path
- **Impact**: Removes HIP runtime overhead from per-dispatch operations
- **What was removed**: `hipMemset` → `hsa_amd_memory_fill`, `hipMemcpy D2H` → `std::memcpy` (CPU reads device memory directly), HIP events → HSA timestamps, `hipGetDeviceProperties` → `hsa_runtime().device().cu_count`
- **File**: `hip_sampler.hip` dispatch sections, `hsa_runtime.cc` memcpy functions

### [DONE] P0.4: Persistent copy signal
- **Impact**: ~1µs per copy (signal create/destroy) eliminated
- **Root cause**: `memcpy_h2d`/`memcpy_d2h` created and destroyed an HSA signal per call
- **Fix**: Single persistent signal created at init, reset via `hsa_signal_store_relaxed` per copy. Matches PersistentDispatcher pattern.
- **File**: `hsa_runtime.cc` (copy_signal_ member), now uses direct `std::memcpy` instead

### [DONE] P0.5: Direct memcpy for H2D/D2H
- **Impact**: Eliminates SDMA engine overhead for small buffers
- **Root cause**: `hsa_amd_memory_async_copy` involves SDMA engine setup + signal + wait. For buffers < 64KB, direct CPU `std::memcpy` is faster.
- **Fix**: Grant CPU+GPU access at `device_malloc` time. Subsequent H2D/D2H use `std::memcpy`.
- **File**: `hsa_runtime.cc` memcpy_h2d/memcpy_d2h

### [TODO] P0.6: Pre-allocate global-tier buffers in persistent pool
- **Impact**: Additional ~50ms saved for global-tier circuits
- **Root cause**: `global_v`, `global_scratch`, `work_counter` are separate allocations
- **Fix**: Include in the persistent pool or keep as separate persistent statics

---

## P1: MLIR Kernel Codegen Quality

### [DONE] P1.1: `amdgpu-flat-work-group-size="256,256"`
- **Impact**: `uses_dynamic_stack: false`, `private_segment: 0` (was non-zero)
- **Root cause**: Without this attribute, LLVM assumed max workgroup=1024, conservatively allocating scratch
- **Source**: IREE `ROCDLAnnotateKernelForTranslation.cpp`, Triton AMD backend
- **File**: `mlir_emit.cc` all three kernel declarations

### [DONE] P1.2: `denormal-fp-math-f32="preserve-sign"`
- **Impact**: 1.2-2× on transcendental ops (exp2, log)
- **Root cause**: Without PreserveSign, the AMDGPU backend inserts extra instructions around FP ops to handle denormals
- **Source**: rocMLIR `RockPrepareLLVM.cpp`, IREE
- **File**: `mlir_emit.cc` kernel attributes

### [DONE] P1.3: `uniform-work-group-size="true"`
- **Impact**: Eliminates partial workgroup checks
- **Source**: All reference projects (IREE, Triton, OpenMP)

### [DONE] P1.4: `amdgpu-no-implicitarg-ptr`
- **Impact**: Eliminates 256-byte implicit args overhead, saves 1 SGPR
- **Source**: IREE recommendation: "We really should try to force amdgpu-no-implicitarg-ptr"
- **Caveat**: Only safe if kernel never calls device library functions that read implicit args (no printf, no `get_global_id()`)

### [DONE] P1.5: Compile-time `active_k` constant folding
- **Impact**: All loop trip counts become compile-time constants, enabling LLVM loop unrolling and dead code elimination
- **Root cause**: `active_k` was loaded from a runtime pointer for each gate operation. Since we compile a specialized kernel per circuit, `active_k` at each instruction is known at emit time.
- **File**: `mlir_emit.cc` — `g_emit_ak` global tracks active_k, all gate emitters use it

### [DONE] P1.6: `opt -O3` for coop/global tiers
- **Impact**: SROA, GVN, LICM, instcombine, loop simplification, vectorization
- **Root cause**: Only register tier ran `opt -O3` before `llc`. Coop/global went directly to `llc -O3`.
- **File**: `mlir_kernel_cache.cc` — added opt -O3 to coop/global compile functions

### [DONE] P1.7: Wavefront reduction offset-1 fix
- **Impact**: Correctness — the final pair of lanes in the 64-wide reduction was never folded
- **Root cause**: Reduction loop used `{32,16,8,4,2}` instead of `{32,16,8,4,2,1}`
- **File**: `mlir_emit.cc` line ~538

### [TODO] P1.8: Per-argument `noalias` + alias scope metadata
- **Impact**: 1.5-3× from instruction scheduling (rocMLIR finding)
- **Root cause**: AMDGPU backend discards function-level `noalias` during kernel entry rewriting. Must use per-instruction alias scope metadata.
- **Pattern**: Create one `AliasScopeDomainAttr` for the kernel, one `AliasScopeAttr` per pointer arg. Each load/store gets `alias_scopes` (its own arg) and `noalias_scopes` (all other args).
- **Source**: rocMLIR `RockPrepareLLVM.cpp`
- **MLIR syntax**:
  ```
  %val = llvm.load %ptr {alias_scopes = [@scope_arg0], noalias_scopes = [@scope_arg1, @scope_arg2]} : !llvm.ptr -> i64
  ```
- **Difficulty**: Medium — requires tracking pointer provenance through GEPs

### [REVERTED] P1.9: Load/store alignment annotations
- **Result**: Syntax was correct but caused massive register spilling (VGPRs: 16→128, SGPRs: 18→106, 304 spills). Needs investigation — alignment may change LLVM's vectorization decisions. Try selectively on amplitude ops only.
- **Impact**: Up to 4× instruction count difference for vector loads
- **Root cause**: Without explicit alignment, LLVM defaults to 1-byte alignment, preventing `dwordx2`/`dwordx4` coalescing
- **Pattern**: Set alignment = min(16, natural_width) on every load/store
- **Source**: rocMLIR `RockPrepareLLVM.cpp`
- **MLIR syntax**: Add `{alignment = 8}` to complex struct loads/stores
- **Difficulty**: Easy — walk all load/store ops in emitter

### [TODO] P1.10: `invariant.load` on readonly kernel arguments
- **Impact**: 1.1-1.5× from LICM + scalar promotion
- **Root cause**: Loads from noise_hazards, noise_sites, detector_offsets etc. are readonly but LLVM doesn't know this
- **Pattern**: Mark readonly kernel args with `LLVM::LLVMDialect::getReadonlyAttrName()`. Walk loads tracing to readonly args and set `invariant=true`.
- **Source**: rocMLIR `RockPrepareLLVM.cpp`

### [DONE] P1.11: `inreg` on kernel arguments (SGPR preloading)
- **Result**: Saved 3 SGPRs (18 → 15). No measurable runtime impact (host overhead dominated). Correctness preserved.
- **Impact**: Avoids `s_load_dword` at kernel entry for each arg
- **Scope**: gfx942 (MI300X) and gfx950 (MI350X) only
- **Source**: IREE, Triton — both mark ALL kernel args `inreg` on gfx940+
- **Implementation**: Add `llvm.inreg` to each arg in the MLIR function signature
- **Difficulty**: Easy — single attribute per arg in the emitter

### [TODO] P1.12: GEP `inbounds` flag
- **Impact**: 1.1-1.3× from better addressing modes
- **Source**: rocMLIR — sets `inbounds` on all non-addrspace-7 GEPs
- **Current state**: Our emitter already uses `inbounds` on most GEPs

### [DONE] P1.13: Atomic relaxation (syncscope)
- **Result**: Applied `syncscope("agent")` with correct inline syntax. No regression. Correctness preserved.
- **Impact**: Critical if using atomics (we use flat_atomic_add for result aggregation)
- **Pattern**: `syncscope("agent-one-as")`, `ordering=monotonic`, `amdgpu.no_remote_memory`, `amdgpu.no_fine_grained_memory`
- **Source**: rocMLIR `RockPrepareLLVM.cpp`
- **Current state**: Our atomics use default ordering (seq_cst) → cache flushes around every atomic

---

## P2: Architectural Improvements

### [TODO] P2.1: Wavefront cooperative reduction for MLIR
- **Impact**: 64× fewer atomic operations for result aggregation
- **Root cause**: MLIR uses `flat_atomic_add_x2` per thread (64 atomics per wavefront). SVM uses `ds_bpermute` butterfly reduction (5 rounds) + LDS accumulation → 1 atomic per wavefront.
- **Pattern**: Emit the same ds_bpermute reduction tree that the SVM codegen uses
- **Source**: SVM kernel ISA shows 40× ds_bpermute + 3 barriers + 6 ds_read/write for wavefront reduction

### [TODO] P2.2: Buffer descriptor loads (Triton pattern)
- **Impact**: Zero-branch OOB masking, hardware SRD-based addressing
- **Pattern**: `llvm.amdgcn.make.buffer.rsrc` with SRD flags (0x7000 for gfx942) + `llvm.amdgcn.raw.ptr.buffer.load.v4f32`
- **Source**: Triton `BufferOpsEmitter.cpp`
- **Difficulty**: High — requires restructuring all memory access patterns

### [TODO] P2.3: LDS bank conflict padding
- **Impact**: Eliminates bank conflicts in coop/global tier LDS access
- **Pattern**: Pad innermost LDS dimension by 128/bitwidth elements
- **Source**: IREE `GPUReduceBankConflicts.cpp` — pads + creates subview

### [TODO] P2.4: Pre-allocate ALL buffers once at init
- **Impact**: Eliminates 49ms buffer_alloc cost entirely
- **Root cause**: Every `gpu_sample_survivors()` call allocates 14+ device buffers
- **Fix**: Allocate once at process start (or first call), keep alive for lifetime, re-upload data per circuit via `memcpy`. Only re-allocate if buffer sizes change.

---

## P3: Compilation Pipeline

### [TODO] P3.1: Reduce `opt -O3` compilation time
- **Impact**: First-run MLIR compilation from ~1.2s to ~0.3s
- **Root cause**: `opt -O3` runs all LLVM passes on the generated IR. For small kernels this is overkill.
- **Options**: Use `-O2`, or use `--passes=` to select only the passes that matter (SROA, GVN, LICM, instcombine)

### [TODO] P3.2: Cache HSA executable in-memory
- **Impact**: Eliminates 170ms HSA executable load on first call
- **Root cause**: `hsa_load_kernel` reads .hsaco from disk, creates executable, loads code object, freezes, validates, extracts symbol — every time the process starts
- **Fix**: Keep the HSA executable alive in a static. Already implemented via `static HsaLoadedKernel` + pre-load, but only for single-circuit workloads.

---

## Tools & Methodology

### Available Profiling Tools

| Tool | Status | Capability |
|------|--------|-----------|
| **rocprofv3** (via GEAK `profile_kernel.sh`) | ✅ Working (SVM) | Kernel trace, per-dispatch timing, kernel stats CSV |
| **rocprof-compute** (omniperf) | ❌ Missing Python deps | Roofline, occupancy, cache analysis, A/B comparison |
| **Magpie** (AMD-AGI) | Not tested | Kernel comparison, roofline, weighted scoring |
| **KernelForge** (AMD-AGI) | Not tested | Static .hsaco register analysis, PMC counters |
| **GEAK** (AMD-AGI) | ✅ profile_kernel.sh works | Profiler auto-detection, raw output collection |
| **APEX** (AMD-AGI) | Explored, not applicable | RL kernel optimization pipeline (source-level only) |
| **Hyperloom** (AMD-AGI) | Explored, not applicable | Agentic orchestrator for LLM inference optimization |
| **Kerncap** (IntelliKit) | Not installed | HSA-native kernel capture + replay — most relevant for our dispatch path |

### Key Profiling Data

**GEAK rocprofv3 kernel_stats.csv (SVM, MI350X gfx950, 100k shots):**
```
"Name","Calls","TotalDurationNs","AverageNs","Percentage"
"sample_kernel",1,19920,19920,56.08
"__amd_rocclr_copyBuffer",6,15600,2600,43.92
```

**Timing instrumentation breakdown (SVM, MI350X, 100k shots):**
```
flatten_program:  0.003 ms
buffer_alloc:    58.1  ms  ← 14× hipMalloc + hipMemcpy
dispatch_loop:    8.3  ms  ← hipMemset + <<<>>> + hipMemcpy D2H
  kernel_seconds: 0.67 ms
  host_in_loop:   7.4  ms
```

**Timing instrumentation (MLIR, MI350X, 100k shots, post-preload fix):**
```
flatten_program:  0.003 ms
buffer_alloc:    48.6  ms  ← 14× device_malloc + memcpy (→ 0.2ms with pool)
mlir_preload:   169.8  ms  ← HSA executable load (one-time)
dispatch_loop:    2.4  ms  ← hsa_memory_fill + PersistentDispatcher + memcpy D2H
  kernel_seconds: 0.03 ms
  host_in_loop:   2.4  ms
```

### ISA Comparison (gfx950, frame_h circuit)

| Metric | MLIR | SVM/Hybrid |
|--------|------|-----------|
| .hsaco size | 6,096 B | 9,072 B |
| ISA lines | 492 | 1,080 |
| VGPRs | 16 | 54 |
| SGPRs | 18 | 42 |
| private_segment | 0 | 1,280 B |
| FP compute ops | 1 | 9 |
| LDS ops | 0 | 46 |
| Barriers | 1 | 3 |

---

## Reference Project Insights

### rocMLIR `RockPrepareLLVM.cpp`
**Repo**: https://github.com/ROCm/rocMLIR
**Key file**: `mlir/lib/Dialect/Rock/Transforms/RockPrepareLLVM.cpp`
**Also**: `CheckResidency.cpp` (occupancy), `ReuseLDS.cpp` (graph coloring), `RockPipeline.cpp` (software pipelining)

Six annotations in priority order:
1. Load/store alignment (4× instruction count) — `getAlign()` function, line ~50
2. Per-argument alias scope metadata (1.5-3× scheduling) — `AliasScopeDomainAttr` + `AliasScopeAttr`, lines ~100-200
3. Denormal FP mode (1.2-2× transcendentals) — `setDenormalFpenvAttr()` — **DONE**
4. GEP inbounds (1.1-1.3× addressing) — `setNoWrapFlags(GEPNoWrapFlags::inbounds)` — partially done
5. invariant.load on readonly (1.1-1.5× LICM) — `load.setInvariant(isReadonly[argNo])`
6. Atomic relaxation — `setSyncscope("agent-one-as")`, `setOrdering(monotonic)`, `noRemoteMemHelper`, `noFineMemHelper`

### IREE `ROCDLAnnotateKernelForTranslation.cpp`
**Repo**: https://github.com/iree-org/iree
**Key files**:
- `compiler/src/iree/compiler/Codegen/LLVMGPU/ROCDLAnnotateKernelForTranslation.cpp` — kernel attributes
- `compiler/src/iree/compiler/Codegen/Common/GPU/GPUReduceBankConflicts.cpp` — LDS padding
- `runtime/src/iree/hal/drivers/amdgpu/util/aql_emitter.h` — direct AQL dispatch
- `runtime/src/iree/hal/drivers/amdgpu/util/pm4_dispatch.c` — PM4 dispatch (fastest)
- `runtime/src/iree/hal/drivers/amdgpu/abi/kernel_args.h` — implicit args (256B)

Patterns:
- `inreg` on ALL args for gfx940+ (chipset >= 9,4,0) — SGPR preloading
- `amdgpu-no-implicitarg-ptr` — **DONE**
- `amdgpu-waves-per-eu` — register allocation tuning (set "N,N" for exact)
- `amdgpu-no-workgroup-id-y/z` — for 1D dispatches, saves 2 SGPRs
- PM4 COMPUTE_DISPATCH_DIRECT — bypasses AQL entirely, pre-computes COMPUTE_PGM_RSRC1/2/3

### Triton AMD Backend
**Repo**: https://github.com/triton-lang/triton
**Key files**:
- `third_party/amd/backend/compiler.py` — kernel attribute setting (post-MLIR, on raw LLVM-IR)
- `third_party/amd/lib/TritonAMDGPUToLLVM/BufferOpsEmitter.cpp` — buffer descriptor ops
- `third_party/amd/lib/TritonAMDGPUToLLVM/AllocateSharedMemory.cpp` — static LDS
- `third_party/amd/lib/TritonAMDGPUTransforms/ConvertToBufferOps.cpp` — raw ptr → SRD conversion
- `third_party/amd/lib/Dialect/TritonAMDGPU/IR/TargetFeatures.cpp` — gfx942 vs gfx950

Patterns:
- Buffer descriptor ops: `llvm.amdgcn.make.buffer.rsrc(ptr, stride=0, num_records=INT_MAX-1, flags=0x7000)`
- OOB masking: `select(valid, byte_offset, 0x80000000)` — zero-branch, hardware returns 0
- Cache policy via `aux` field: 0=default, 3=cache_global(.cg), 17=cache_volatile(.cv)
- gfx950 features: 160KB LDS, 64 banks, async direct-to-LDS (`buffer_load_async_lds`), permlane_swap

### OpenMP AMDGPU Plugin
**Repo**: https://github.com/llvm/llvm-project
**Key files**:
- `offload/plugins-nextgen/amdgpu/src/rtl.cpp` — AQL packet construction, queue management
- `offload/plugins-nextgen/amdgpu/utils/UtilitiesRTL.h` — AMDGPUImplicitArgsTy (256B struct)
- `openmp/libomptarget/DeviceRTL/src/Kernel.cpp` — generic state machine (persistent kernel)

Patterns:
- 64 pre-pooled HSA signals via `hsa_amd_signal_create` + `GenericDeviceResourceManagerTy`
- 4 HSA queues per device with busy-tracking (`NumUsers` per queue)
- Queue: `HSA_QUEUE_TYPE_MULTI`, 512 packets, with `std::mutex` for serial publish
- AQL publish: fill all fields, then `__atomic_store_n(header, __ATOMIC_RELEASE)`, then doorbell
- Fence scopes: `HSA_FENCE_SCOPE_SYSTEM` for both acquire/release (safe default)
- Active wait with 2s timeout, fallback to `HSA_WAIT_STATE_BLOCKED`

### APEX (AMD-AGI)
**Repo**: https://github.com/AMD-AGI/Apex
**Relevant files**:
- `pipeline/reflector.py` — rocprof metric parsing (`_parse_rocprof_metrics()`)
- `tools/skills/rocprof-compute/SKILL.md` — profiling commands
- `tools/skills/hip-kernel-optimization/SKILL.md` — optimization phases (low-hanging → targeted → complex)
- `tools/mcps/gpu_info/server.py` — MI300X/MI350X hardware specs database

**Not directly applicable** (RL pipeline for source-level kernel rewriting), but methodology is adoptable:
- Gated reward: performance reward requires correctness > 0.95
- Scoring: `score = compiled * 20 + correct * 100 + speedup_score(S)`
- Phased optimization: low-hanging-fruit → targeted → complex

### Hyperloom (AMD-AGI)
**Repo**: https://github.com/AMD-AGI/Hyperloom
**Key finding**: Kerncap (IntelliKit sub-tool) does VA-faithful HSA-native kernel replay.
**Relevant**: `src/hyperloom/agents/kernel/tools/bypass_trace_analysis.py` — standalone analytical roofline (TraceLens-free).
**Not directly applicable** (orchestrator for LLM inference optimization on vLLM/SGLang)
