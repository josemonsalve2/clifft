# MI300X NUMA-Like HBM Topology and Persistent Kernel Design for Clifft

## 1. MI300X Physical Topology

### 1.1 Architecture Diagram

```
                           MI300X Package (Top View)
    ======================================================================

    Layer 2 (top): 8 XCDs (Accelerator Complex Dies) -- TSMC 5nm
    Layer 1 (bottom): 4 IODs (I/O Dies) -- TSMC 6nm + 8 HBM3 Stacks

         XCD 0     XCD 1       XCD 2     XCD 3       XCD 4     XCD 5       XCD 6     XCD 7
        [38 CU]   [38 CU]    [38 CU]   [38 CU]    [38 CU]   [38 CU]    [38 CU]   [38 CU]
        [4MB L2]  [4MB L2]   [4MB L2]  [4MB L2]   [4MB L2]  [4MB L2]   [4MB L2]  [4MB L2]
          |   |     |   |      |   |     |   |      |   |     |   |      |   |     |   |
          +---+-----+---+      +---+-----+---+      +---+-----+---+      +---+-----+---+
              |   3D   |           |   3D   |            |   3D   |           |   3D   |
              | Stack  |           | Stack  |            | Stack  |           | Stack  |
    +---------+---------+---------+---------+--+---------+---------+---------+---------+
    |       IOD 0       |       IOD 1       |  |       IOD 2       |       IOD 3       |
    |  [64MB LLC slice] |  [64MB LLC slice] |  |  [64MB LLC slice] |  [64MB LLC slice] |
    |  [Mem Ctrl x2]    |  [Mem Ctrl x2]    |  |  [Mem Ctrl x2]    |  [Mem Ctrl x2]    |
    |  [IF Links]       |  [IF Links]       |  |  [IF Links]       |  [IF Links]       |
    +----+--------+-----+----+--------+-----+--+----+--------+-----+----+--------+-----+
         |        |          |        |              |        |          |        |
       HBM 0   HBM 1      HBM 2   HBM 3          HBM 4   HBM 5      HBM 6   HBM 7
       24 GB   24 GB       24 GB   24 GB           24 GB   24 GB       24 GB   24 GB
      662GB/s  662GB/s    662GB/s  662GB/s        662GB/s  662GB/s    662GB/s  662GB/s

    <----------- Infinity Fabric on-package mesh (all IODs interconnected) ----------->

    Legend:
      XCD = Accelerator Complex Die (compute chiplet)
      IOD = I/O Die (memory controller + interconnect)
      LLC = Last-Level Cache (Infinity Cache), 256 MB total
      IF  = Infinity Fabric
      CU  = Compute Unit (64 stream processors each)
```

### 1.2 Component Summary

| Component       | Count | Per Unit                                | Total       |
|-----------------|-------|-----------------------------------------|-------------|
| XCDs            | 8     | 38 CUs, 4 MB L2 cache                  | 304 CUs     |
| IODs            | 4     | 2 XCDs stacked, 2 HBM stacks, 64 MB LLC| 256 MB LLC  |
| HBM3 Stacks     | 8     | 24 GB, ~662 GB/s                        | 192 GB, 5.3 TB/s |
| CUs             | 304   | 64 stream processors, 4 SIMDs           | 19,456 SPs  |
| Wavefront size  | --    | 64 threads                              | --          |
| LDS per CU      | 304   | 64 KB                                   | 19.5 MB     |

### 1.3 XCD-to-IOD-to-HBM Mapping

Each IOD is the parent of exactly 2 XCDs and 2 HBM stacks:

| IOD | XCDs   | HBM Stacks | Local HBM | Local LLC Slice |
|-----|--------|------------|-----------|-----------------|
| 0   | 0, 1   | 0, 1       | 48 GB     | 64 MB           |
| 1   | 2, 3   | 2, 3       | 48 GB     | 64 MB           |
| 2   | 4, 5   | 4, 5       | 48 GB     | 64 MB           |
| 3   | 6, 7   | 6, 7       | 48 GB     | 64 MB           |

Two XCDs on the same IOD share access to the same HBM stacks and LLC
slice with minimal latency. Access to HBM on a remote IOD requires
traversal of the Infinity Fabric on-package mesh.

---

## 2. Memory Access Latency Hierarchy

### 2.1 Latency Table

| Access Path                                   | Latency       | Notes                                      |
|-----------------------------------------------|---------------|--------------------------------------------|
| LDS (Local Data Share)                        | ~1 cycle      | 64 KB per CU, intra-workgroup only         |
| L1 scalar cache (hit)                         | ~12 cycles    | Read-only, per-CU                          |
| L2 cache (same XCD, hit)                      | ~100 ns       | 4 MB per XCD, private                      |
| Infinity Cache / LLC (same IOD)               | ~218 ns       | 64 MB per IOD                              |
| HBM3 (same IOD, L2+LLC miss)                  | ~300 ns       | Direct path: XCD -> L2 -> LLC -> HBM       |
| HBM3 (remote IOD, L2+LLC miss)                | ~340-370 ns   | Extra IF hop: XCD -> L2 -> IF -> LLC -> HBM |
| Global atomic (same XCD)                      | ~116 ns       | Device-scope atomic on L2-cached address   |
| Global atomic (cross-XCD, same IOD)           | ~140-160 ns   | Estimated, shares LLC but separate L2      |
| Global atomic (cross-IOD)                     | ~200-202 ns   | Measured by Chips and Cheese               |
| System-scope atomic                           | ~250-400 ns   | Forces LLC coherence across all XCDs       |
| TLB miss penalty                              | ~47 ns        | Additional penalty on top of access        |

Source: Chips and Cheese MI300X testing, AMD HotChips 2024 presentation.

The 116-202 ns range for device-scope atomics represents a 1.75x latency
variation depending on XCD placement -- a significant NUMA effect.

### 2.2 Three-Tier Bandwidth Hierarchy

```
Tier 1: XCD-local (same L2)
  - L2 bandwidth: ~4 TB/s per XCD (internal crossbar)
  - Optimal: data fits in 4 MB L2

Tier 2: Same-IOD (shared LLC + local HBM)
  - LLC bandwidth: ~3 TB/s per IOD (11.9 TB/s total measured, 17 TB/s theoretical)
  - HBM bandwidth: ~1.3 TB/s per IOD (2 stacks, 5.3 TB/s total)
  - Data path: XCD -> L2 miss -> LLC -> HBM (direct)

Tier 3: Cross-IOD (remote HBM)
  - IF bandwidth: ~0.9 TB/s per link (estimated)
  - Data path: XCD -> L2 miss -> IF -> remote LLC -> remote HBM
  - 10-15% additional latency over same-IOD access
```

### 2.3 NPS1 vs NPS4 Memory Mode Impact

In default NPS1/SPX mode, HBM addresses are **interleaved** across all
8 stacks. A contiguous allocation from `hipMalloc` is spread across all
4 IODs at page granularity. This flattens bandwidth (all XCDs get equal
access) but means no allocation is "local" to any XCD.

| Mode   | NUMA Domains | Stream Bandwidth | Notes                              |
|--------|-------------|------------------|------------------------------------|
| NPS1   | 1           | ~4,017 GB/s      | All HBM interleaved; uniform access |
| NPS4   | 4           | ~4,210 GB/s      | Per-IOD HBM; 5% higher peak BW     |
| CPX    | 8 devices   | Per-XCD local    | Each XCD = separate device          |

NPS4 achieves 5-10% higher bandwidth because accesses are localized to
the IOD's HBM stacks, eliminating cross-IOD traffic.

### 2.4 Impact of Naive Scheduling on L2 Hit Rates

Without chiplet-aware workgroup scheduling, L2 cache performance
degrades dramatically:

| Scheduling Pattern         | L2 Hit Rate | Source                  |
|---------------------------|-------------|-------------------------|
| Naive block-first         | ~1%         | NUMA-Aware Attention    |
| Default round-robin       | ~12%        | Fleet baseline          |
| Swizzled head-first       | 80-97%      | NUMA-Aware Attention    |
| Fleet chiplet-task (bs=32)| 51.0%       | Fleet paper             |
| Fleet chiplet-task (bs=64)| 61.4%       | Fleet paper             |
| SwizzlePerf LLM-guided   | +23.9% avg  | SwizzlePerf paper       |

The fundamental problem: in default SPX round-robin, workgroups sharing
data scatter across 8 XCDs. Each XCD's 4 MB L2 cache independently
fetches the same data from HBM, wasting bandwidth. Chiplet-aware
scheduling clusters related workgroups on the same XCD to share L2
cache lines.

**SwizzlePerf** (arXiv:2508.20258) demonstrated that LLM-guided swizzle
pattern generation can achieve expert-equivalent solutions automatically,
with an average 1.29x speedup across 10 kernels and peak 2.06x on
transpose operations.

---

## 3. Cross-XCD Synchronization Mechanisms

### 3.1 Scope-Controlled Memory Operations

MI300X uses the SC-HRF (Scope-Controlled Heterogeneous Race-Free)
memory model. Cache behavior is controlled by scope bits:

| Scope   | L1 Cache | L2 Cache | LLC         | Use Case                           |
|---------|----------|----------|-------------|-------------------------------------|
| SC0 (CU)| Cached   | Cached   | Cached      | Thread-local data                   |
| SC1 (GPU)| Bypass  | Coherent | Coherent    | Cross-CU/XCD device-scope sync      |
| SC2 (Sys)| Bypass  | Bypass   | Write-through| Cross-GPU system-scope sync         |

For cross-XCD synchronization within a single GPU, `sc1` (device scope)
is the correct and cheapest option. This skips the non-coherent L1 and
uses scoped L2 coherence. The Iris framework exposes this as:

```python
# Store with write-through (bypasses L1 + L2 entirely)
tl.store(ptr, value, cache_modifier=".wt")

# Load with cache-volatile (bypasses all GPU caches)
tl.load(ptr, cache_modifier=".cv", volatile=True)

# Atomic with device scope (L2-coherent)
tl.atomic_xchg(locks_ptr, 1, sem="release", scope="gpu")
tl.atomic_cas(locks_ptr, 1, 0, sem="acquire", scope="gpu")
```

### 3.2 Synchronization Pattern Comparison

| Pattern                    | Latency    | Mechanism                                          | Source    |
|---------------------------|------------|-----------------------------------------------------|-----------|
| Sentinel-based (Kog AI)   | 0.8-0.9 us | NaN init, poll data locations, sc1 scope            | Kog blog  |
| Naive counter barriers    | 7.6-7.9 us | atomicAdd arrival + __threadfence + epoch poll      | Kog blog  |
| Iris WG specialization    | ~1-2 us    | volatile load (.cv) + acquire CAS, write-through    | Iris repo |
| Fleet hierarchical events | ~0.5-1 us  | Per-XCD L2-local counters + single global fence     | Fleet paper|

**Sentinel-based synchronization** (Kog AI pattern) is 8-9x faster
than naive counter-based barriers because:

1. No `__threadfence()` (which lowers to `buffer_wbl2` + `buffer_inv`,
   flushing the entire L2)
2. Polling is on the actual data buffer, not a separate flag
3. The sentinel (NaN) encodes readiness in the data path itself
4. Multiple buffers allow asynchronous reset without blocking

### 3.3 Fleet's Hierarchical Event Model

Fleet (AMD RAD, arXiv 2604.15379) introduces a two-level synchronization
scheme that amortizes cross-XCD fence costs:

```
Level 1: XCD-local workers -> XCD-local counter
  - Each worker atomically increments a per-XCD counter (L2-local, ~116 ns)
  - Only the LAST worker to complete on each XCD:

Level 2: Last XCD worker -> global event counter
  - Issues buffer_wbl2 fence (expensive, but only 1 per XCD)
  - Updates global event counter via GPU-scope atomic
  - Total: 8 fences per linear event (one per XCD) instead of 608

Scheduler -> next wave
  - Per-XCD scheduler polls global event counter
  - Dispatches next chiplet-task when dependency is satisfied
```

This reduces cross-XCD synchronization cost by a factor of ~76x
(608 workers / 8 XCDs).

---

## 4. Chiplet-Aware Work Mapping Patterns

### 4.1 Workgroup-to-XCD Assignment in SPX Mode

In SPX mode, the hardware dispatches workgroups to XCDs using a
**chunked round-robin** policy with chunk size 1 (confirmed by the
ARCAS/attention optimization paper). This means:

```
blockIdx.x = 0 -> XCD 0
blockIdx.x = 1 -> XCD 1
blockIdx.x = 2 -> XCD 2
...
blockIdx.x = 7 -> XCD 7
blockIdx.x = 8 -> XCD 0  (wraps around)
blockIdx.x = 9 -> XCD 1
...
```

**Therefore `xcd_id = blockIdx.x % 8` is correct for SPX mode.**

**Important caveat:** The round-robin mapping is driver-managed and
AMD documents it as "subject to change across GPU generations." Kernels
should prefer reading `HW_REG_XCC_ID` at runtime (section 4.2) for
definitive XCD identity rather than relying on `blockIdx.x % 8`. The
Iris and Fleet codebases both use the hardware register approach.

This is confirmed by the Iris source code (`iris/ccl/utils.py`):

```python
@triton.jit()
def chiplet_transform_chunked(pid, num_workgroups, num_xcds, chunk_size):
    local_pid = pid // num_xcds         # which "round" of XCD assignment
    xcd = pid % num_xcds                # which XCD this workgroup runs on
    new_pid = chunk_idx * num_xcds * chunk_size + xcd * chunk_size + pos_in_chunk
    return new_pid
```

And by the simpler form used in Iris GEMM examples:

```python
if NUM_XCDS != 1:
    pid = (pid % NUM_XCDS) * (NUM_SMS // NUM_XCDS) + (pid // NUM_XCDS)
```

### 4.2 XCD Identity at Runtime

Iris provides a `get_xcc_id()` intrinsic that reads the hardware register
directly:

```python
@triton.jit
def get_xcc_id():
    return tl.inline_asm_elementwise(
        asm="s_getreg_b32 $0, hwreg(HW_REG_XCC_ID, 0, 16)",
        constraints=("=s"), args=[], dtype=tl.int32,
        is_pure=False, pack=1)
```

In HIP C++ (for clifft's use):

```cpp
__device__ int get_xcd_id() {
    int xcd_id;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(xcd_id));
    return xcd_id;
}
```

This hardware register gives the definitive XCD assignment, independent
of any software scheduling assumptions about `blockIdx.x`.

### 4.3 IOD Identity from XCD

Since each IOD hosts exactly 2 XCDs:

```cpp
__device__ int get_iod_id() {
    return get_xcd_id() / 2;
}
```

### 4.4 Fleet's Chiplet-Aware Scheduling

Fleet (AMD RAD) demonstrates the production pattern for chiplet-aware
persistent kernels:

```
Grid: 304 workgroups (1 per CU on MI300X)
  - 8 schedulers (1 per XCD, reading HW_REG_XCC_ID)
  - 296 workers (37 per XCD)

Per-XCD scheduler:
  1. Read task graph event dependencies
  2. Dispatch chiplet-tasks to local workers' task queues
  3. Each chiplet-task operates on XCD-local data partition

Weight partitioning (GEMM example):
  - [M, K] x [K, N] -> each XCD computes [M, K] x [K, N/8]
  - Output tile assignment: task_k gets base_ptr + k*(N/8)*K*sizeof(bf16)

L2 hit rate improvement (Qwen3-8B):
  - Baseline (Mirage, no XCD awareness): 38.9%
  - Fleet (chiplet-aware M-tile): 51.0%  (+31% relative)
  - Fleet at batch=64: 61.4% hit rate, 37% less HBM traffic
```

---

## 5. Application to Clifft's Global-Coop Kernel

### 5.1 Current Architecture

The global-coop kernel (rank 11-19) in `hip_sampler.hip` is already a
persistent kernel with work-stealing:

```cpp
// Current: 608 blocks, each with a dedicated HBM slot
global_worker_blocks = prop.multiProcessorCount * 2;  // 304 * 2 = 608

// Allocation: contiguous, address-interleaved across all IODs (NPS1)
hipMalloc(&global_v, 608 * kGlobalMaxAmplitudes * sizeof(GpuComplex));

// Work-stealing: single global counter
atomicAdd(work_counter, 1);  // Cross-XCD atomic every shot claim
```

### 5.2 Memory Footprint Analysis at Rank=19

| Item                     | Per Shot    | 608 Blocks Total | Per XCD (76 blocks) |
|--------------------------|-------------|-------------------|---------------------|
| Amplitude array `v[]`    | 4 MiB       | 2.4 GiB           | 304 MiB             |
| Scratch buffer           | 2 MiB       | 1.2 GiB           | 152 MiB             |
| **Total**                | **6 MiB**   | **3.6 GiB**       | **456 MiB**         |
| L2 cache per XCD         | --          | --                 | 4 MiB               |
| HBM per IOD              | --          | --                 | 48 GiB              |

At rank=19, each shot's 4 MiB amplitude array exactly equals one XCD's
4 MiB L2 cache. This means:

- A single shot's amplitude sweep at rank=19 **completely thrashes** the
  L2 cache on its processing XCD.
- There is zero opportunity for L2 reuse across shots, since each shot
  has unique amplitude values.
- The workload is **purely HBM bandwidth-bound** at rank=19.

### 5.3 Current Problems with NPS1 Interleaving

With NPS1 (default), the `hipMalloc`-allocated `global_v` buffer is
interleaved across all 8 HBM stacks at page granularity (typically
4 KB or 2 MB huge pages). This means:

1. **Every amplitude sweep generates cross-IOD traffic.** A rank=19
   sweep touches 4 MiB = 1024 pages. With 8-way interleaving, ~75%
   of pages are on remote IODs, incurring 10-15% additional latency
   per access.

2. **The work-stealing atomic is a cross-XCD bottleneck.** Each of
   the 608 blocks issues `atomicAdd(work_counter, 1)` after every
   shot. This single counter lives on one HBM stack; 87.5% of blocks
   access it cross-IOD at ~200 ns instead of ~116 ns.

3. **No L2 cache benefit for amplitude data.** Even if one block's
   amplitude pages happen to reside on its local HBM, the L2 cannot
   cache the full 4 MiB array (L2 = 4 MiB, but needs room for
   instructions, stack, shared data, bytecode, etc.).

### 5.4 Proposed Optimizations

#### Optimization A: Per-XCD Work Counters

Replace the single `work_counter` with 8 per-XCD counters, each on
its local HBM partition. Two-level stealing protocol:

```cpp
__device__ uint64_t claim_shot(uint64_t* xcd_counters, uint64_t total_shots) {
    int xcd = get_xcd_id();
    // Try local counter first (same-XCD atomic: ~116 ns)
    uint64_t shot = atomicAdd(&xcd_counters[xcd], 1);
    if (shot < total_shots / 8) {
        return xcd * (total_shots / 8) + shot;
    }
    // Local work exhausted; steal from global fallback counter
    return atomicAdd(&xcd_counters[8], 1) + total_shots;
    // Alternatively: round-robin steal from other XCDs
}
```

**Expected impact:** Reduces work-claiming atomic latency by ~1.75x
(116 ns vs 200 ns) for the common case. For 143K shots/s, each shot
claim saves ~84 ns, totaling ~12 ms/s saved.

#### Optimization B: XCD-Affine HBM Allocation

Instead of one contiguous `hipMalloc`, allocate per-XCD buffers using
NPS4-aware placement or `hipExtMallocWithFlags` (if available in ROCm):

```cpp
// Conceptual: place each XCD's amplitude buffers on its IOD's HBM
for (int xcd = 0; xcd < 8; xcd++) {
    int iod = xcd / 2;
    size_t per_xcd_slots = global_worker_blocks / 8;  // 76 slots
    size_t per_xcd_bytes = per_xcd_slots * kGlobalMaxAmplitudes * sizeof(GpuComplex);
    // hipMallocOnIOD(&xcd_v[xcd], per_xcd_bytes, iod);  // hypothetical API
}
```

**Current ROCm limitation:** As of ROCm 7.x, there is no public API
for IOD-affine allocation in NPS1 mode. Options:

1. **Switch to NPS4/CPX mode:** Each XCD becomes a separate HIP device
   with its own local HBM partition. Allocations via `hipSetDevice(xcd);
   hipMalloc(...)` are automatically local. Requires restructuring the
   launcher to use 8 separate GPU devices.

2. **Reverse-engineer physical-to-IOD mapping:** The Kog AI team did
   this for their monokernel, recovering physical addresses from virtual
   addresses to determine IOD location. Undocumented and fragile.

3. **First-touch placement with manual touching:** Allocate with
   `hipMalloc`, then have each XCD's first block touch its pages from
   local CUs to trigger NUMA first-touch placement. May not work with
   NPS1 interleaving (which forces round-robin regardless of touch).

4. **Accept NPS1 interleaving and optimize other aspects.** The 10-15%
   cross-IOD latency penalty may be acceptable given the other costs.

#### Optimization C: Sentinel-Based Synchronization (Kog Pattern)

For future split/persistent hybrid kernels where inter-segment
synchronization is needed, use sentinel-based sync instead of counters:

```cpp
// Initialize handoff buffer to NaN sentinel
__device__ void init_handoff(GpuComplex* buf, uint32_t n) {
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x)
        buf[i] = {NAN, NAN};
}

// Consumer polls until non-NaN data appears
__device__ GpuComplex poll_handoff(GpuComplex* buf, uint32_t idx) {
    GpuComplex val;
    do {
        // sc1 scope: skip L1, use L2 coherence
        asm volatile("global_load_dwordx2 %0, %1, off sc1"
                     : "=v"(val) : "v"(buf + idx));
    } while (__builtin_isnan(val.re));
    return val;
}
```

This would reduce inter-segment synchronization from ~7.6 us (counter-
based) to ~0.8 us (sentinel-based), an ~8x improvement.

#### Optimization D: IOD-Local Attention Grouping (Kog Pattern)

For future multi-block cooperative operations (e.g., reduction across
blocks for measurement), group blocks by IOD:

```cpp
// 76 blocks per XCD, 2 XCDs per IOD = 152 blocks per IOD
int iod = get_xcd_id() / 2;
int iod_local_id = blockIdx.x / 2 - iod * (num_blocks / 4);

// Phase 1: IOD-local reduction (no cross-IOD traffic)
iod_local_reduce(partial_results, iod_local_id);

// Phase 2: Cross-IOD reduction (only 4 values, negligible traffic)
if (iod_local_id == 0) {
    global_reduce_4(partial_results, iod);
}
```

---

## 6. Quantitative Impact Assessment

### 6.1 Bottleneck Analysis at Rank=19 (143K shots/s on d7)

The d7 benchmark processes 143,624 shots/s with:
- 5472 bytecode instructions per shot
- Peak rank = 19 (524,288 complex amplitudes per shot)
- Block size = 256 threads
- Grid = 608 persistent blocks

**Time per shot per block:** 1 / (143624 / 608) = ~4.2 ms per shot

At rank=19, each 1-qubit gate sweeps 262,144 elements (2^18 pairs),
each 2-qubit gate sweeps 131,072 quadruples (2^17). The amplitude array
is 4 MiB. A single sweep at 5.3 TB/s theoretical bandwidth takes:

```
4 MiB / 5.3 TB/s = 0.76 us per sweep (theoretical)
4 MiB / ~3.5 TB/s = 1.14 us per sweep (achievable ~66% efficiency)
```

With ~10-15% of instructions being array sweeps (per SPLIT_HEURISTIC.md
analysis), approximately 500-800 sweeps per shot:

```
600 sweeps * 1.14 us = 684 us per shot in sweeps
Total shot time: ~4200 us
Sweep fraction: ~16% of total time
```

The remaining ~84% of shot time is dominated by:
- Frame operations (branch-divergent switch dispatch)
- Measurement reductions (cooperative `__shfl_xor`)
- Address computation (scatter_bits SALU overhead)
- Synchronization barriers (`__syncthreads` between ops)

### 6.2 Estimated Impact of NUMA-Aware Optimizations

| Optimization | Mechanism | Expected Speedup | Confidence |
|-------------|-----------|-----------------|------------|
| Per-XCD work counters | Local atomic 116ns vs 200ns | +1-2% | High |
| XCD-affine HBM (NPS4) | Eliminate cross-IOD traffic | +5-10% on sweep | Medium |
| Sentinel sync (future) | 0.8us vs 7.6us barriers | +5-15% on hybrid kernel | High |
| IOD-local reductions | Avoid cross-IOD in measurements | +2-5% on meas ops | Medium |
| **Combined**          | All above                     | **+8-20%**        | Medium |

### 6.3 Why the Impact Is Moderate

The NUMA-aware optimizations primarily affect the **memory-bound sweep
fraction** of the workload, which is only ~16% of total shot time at
rank=19. The dominant costs are:

1. **SALU address computation** (~30% of cycles): `scatter_bits_1`,
   `insert_zero_bit`, bit manipulation for butterfly addressing. This
   is purely compute-bound and unaffected by NUMA placement.

2. **Synchronization barriers** (~25% of cycles): `__syncthreads` for
   cooperative reduction in measurements. The warp-shuffle optimization
   already reduced this from 40% to ~25%.

3. **Measurement probability computation** (~20% of cycles): Two-phase
   sum reduction, probability comparison, amplitude rescaling. This is
   a mix of compute and LDS traffic.

4. **HBM bandwidth for sweeps** (~16%): This is the fraction that
   benefits from NUMA-aware placement.

5. **Overhead** (~9%): Switch dispatch, instruction fetch, RNG, noise
   handling.

### 6.4 When NUMA Optimization IS High-Impact

The NUMA optimizations become critical for:

1. **Higher ranks (rank > 19):** If clifft ever supports rank > 19,
   the sweep fraction grows exponentially. At rank=22 (16 MiB/shot),
   sweeps would dominate and NUMA placement would be essential.

2. **Multi-block cooperative operations:** If a future kernel design
   uses cross-block reductions (Fleet-style chiplet-tasks), the
   cross-XCD atomic latency (200 ns vs 116 ns) becomes the critical
   path.

3. **NPS4/CPX deployment:** If the entire clifft GPU pipeline is
   restructured for CPX mode (each XCD = separate device), then
   per-XCD allocation is automatic and the full 5-10% bandwidth
   improvement is realized.

---

## 7. Specific Code Patterns for Implementation

### 7.1 Reading XCD ID in HIP C++

```cpp
// CDNA3 (gfx942/MI300X) and CDNA4 (gfx950/MI350X)
__device__ __forceinline__ int get_xcd_id() {
    int xcd_id;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(xcd_id));
    return xcd_id;
}

__device__ __forceinline__ int get_iod_id() {
    return get_xcd_id() / 2;
}

__device__ __forceinline__ int get_cu_id() {
    int cu_id;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 8, 8)" : "=s"(cu_id));
    return cu_id;
}
```

### 7.2 Per-XCD Work Counter

```cpp
// Device: claim shots from XCD-local counter, fall back to global
__device__ uint64_t claim_shot_xcd_aware(
    uint64_t* xcd_counters,  // [9]: indices 0-7 = per-XCD, 8 = overflow
    uint64_t per_xcd_budget,
    uint64_t total_shots)
{
    int xcd = get_xcd_id();

    // Fast path: claim from local XCD counter (~116 ns atomic)
    uint64_t local_shot = atomicAdd(
        reinterpret_cast<unsigned long long*>(&xcd_counters[xcd]), 1ULL);

    if (local_shot < per_xcd_budget) {
        return xcd * per_xcd_budget + local_shot;
    }

    // Slow path: overflow counter (~200 ns cross-XCD atomic)
    uint64_t overflow_shot = atomicAdd(
        reinterpret_cast<unsigned long long*>(&xcd_counters[8]), 1ULL);
    uint64_t global_id = 8 * per_xcd_budget + overflow_shot;

    return (global_id < total_shots) ? global_id : UINT64_MAX;
}

// Host: allocate and initialize counters
uint64_t* xcd_counters;
hipMalloc(&xcd_counters, 9 * sizeof(uint64_t));
hipMemset(xcd_counters, 0, 9 * sizeof(uint64_t));
uint64_t per_xcd_budget = (total_shots + 7) / 8;
```

### 7.3 XCD-Aware Buffer Mapping (Software Approach for NPS1)

Since NPS1 interleaves addresses, we cannot control physical placement.
Instead, we can ensure each block accesses its slot consistently:

```cpp
// Map blockIdx.x to an XCD-local slot index
__device__ void setup_xcd_local_buffers(
    GpuComplex* global_v,           // contiguous allocation
    GpuComplex* global_scratch,
    GpuComplex*& my_v,
    GpuComplex*& my_scratch)
{
    int xcd = get_xcd_id();
    int blocks_per_xcd = gridDim.x / 8;  // 76
    int local_id = blockIdx.x / 8;        // round-robin local index

    // Remap: blocks on same XCD use contiguous memory
    int remapped_slot = xcd * blocks_per_xcd + local_id;
    my_v = global_v + (size_t)remapped_slot * kGlobalMaxAmplitudes;
    my_scratch = global_scratch + (size_t)remapped_slot * (kGlobalMaxAmplitudes / 2);
}
```

This doesn't change physical placement under NPS1, but it does ensure
that blocks on the same XCD access a contiguous region of virtual
addresses, which may improve TLB behavior.

### 7.4 Cache Modifier Usage for Cross-XCD Communication

```cpp
// Store with write-through: visible to all XCDs immediately
// Use when producing data consumed by a block on a different XCD
__device__ void store_wt(GpuComplex* ptr, GpuComplex val) {
    asm volatile("global_store_dwordx2 %0, %1, off sc1 nt"
                 : : "v"(ptr), "v"(val) : "memory");
}

// Load with cache-volatile: bypass all GPU caches
// Use in spin-wait loops polling for cross-XCD data
__device__ GpuComplex load_cv(const GpuComplex* ptr) {
    GpuComplex val;
    asm volatile("global_load_dwordx2 %0, %1, off sc1"
                 : "=v"(val) : "v"(ptr));
    return val;
}
```

---

## 8. Recommendation: Is NUMA Optimization Worth Implementing?

### 8.1 For the Current Global-Coop Kernel at Rank=19

**Not as a priority.** The expected 8-20% improvement is real but
modest. The dominant bottlenecks (SALU address computation, sync
barriers, measurement reductions) are not affected by NUMA placement.
The easiest win -- per-XCD work counters -- can be implemented in
~30 lines of code and is worth doing regardless.

The current 143K shots/s on d7 is already within range of theoretical
limits for a bytecode-interpreting kernel at rank=19. Further speedup
requires algorithmic changes (compiled kernel, split heuristic) rather
than memory placement tuning.

### 8.2 For the Future Split/Persistent Hybrid Kernel

**Moderately important.** When the split heuristic (SPLIT_HEURISTIC.md)
is implemented, the hybrid kernel will have inter-segment synchronization
points. Sentinel-based sync (0.8 us vs 7.6 us) will be valuable here.
IOD-local measurement reductions will also help.

### 8.3 For a Future Compiled/Megakernel Architecture

**Essential.** A compiled megakernel (Stanford/Fleet-style) with
chiplet-task scheduling would make NUMA-aware placement the key
differentiator. Fleet demonstrates 1.3-1.5x speedup from chiplet-aware
scheduling on LLM workloads. The quantum analog -- assigning shots to
XCD-local memory and scheduling gate operations as chiplet-tasks --
would see similar benefits.

### 8.4 Implementation Priority Ranking

| Priority | Optimization                          | Effort   | Expected Impact |
|----------|---------------------------------------|----------|-----------------|
| 1 (now)  | Per-XCD work counters                 | Trivial  | +1-2%           |
| 2 (now)  | `get_xcd_id()` intrinsic in codebase | Trivial  | Diagnostic value |
| 3 (soon) | XCD-aware buffer slot remapping       | Low      | +1-3% (TLB)     |
| 4 (later)| Sentinel-based sync for hybrid kernel | Medium   | +5-15%          |
| 5 (later)| IOD-local measurement reductions      | Medium   | +2-5%           |
| 6 (future)| NPS4/CPX deployment mode             | High     | +5-10% BW       |
| 7 (future)| Fleet-style chiplet-task scheduler   | Very High| +30-50%         |

---

## 9. Sources

### Primary Sources (Analyzed)

- **Iris source code:** `/shared/jmonsalv/quantum/clifft_rl/rocm-iris/`
  - XCD detection: `iris/host/platform/hip.py:218-232`
  - Chiplet transform: `iris/ccl/utils.py:16-41`
  - XCC ID intrinsic: `iris/mem/utils.py:get_xcc_id()`
  - Synchronization patterns: `iris/mem/triton/ops.py`

- **Clifft GPU kernel:** `/shared/jmonsalv/quantum/clifft_rl/clifft/src/clifft/gpu/hip_sampler.hip`
  - Global-coop kernel: lines 1913-1995
  - Allocation: lines 2051-2064
  - Tier thresholds: `gpu_types.h`

### Published Research

- **Fleet:** Hierarchical Task-based Abstraction for Megakernels on
  Multi-Die GPUs. arXiv:2604.15379.
  - Chiplet-task scheduling, L2 affinity (38.9% -> 51.0% hit rate)
  - Per-XCD scheduler with HW_REG_XCC_ID discovery

- **NUMA-Aware Attention:** Optimizing Attention on GPUs by Exploiting
  GPU Architectural NUMA Effects. arXiv:2511.02132.
  - Swizzled head-first mapping, 90-96% L2 hit rates
  - blockIdx % NUM_XCD for XCD identification

- **Iris:** First-Class Multi-GPU Programming Experience in Triton.
  arXiv:2511.12500.
  - Symmetric heap, pointer translation, cache modifiers

- **MI300X Partition Modes:** Muhammad Osama et al., ROCm Blogs.
  https://rocm.blogs.amd.com/software-tools-optimization/compute-memory-modes/README.html
  - XCD-IOD-HBM topology, NPS1 vs NPS4 bandwidth

- **MI300X Testing:** Chips and Cheese.
  https://chipsandcheese.com/p/testing-amds-giant-mi300x
  - Cross-XCD atomic latency: 116 ns (same-XCD) to 202.5 ns (cross-XCD)
  - Infinity Cache latency: ~218 ns

- **Kog AI Monokernel:**
  https://blog.kog.ai/building-a-single-kernel-latency-optimized-llm-inference-engine-on-amd-mi300x-gpus/
  - Sentinel sync: 0.80-0.93 us vs 7.59-7.88 us (counter-based)
  - IOD-aware tensor replication
  - sc1 scope for cross-chiplet communication

- **Clifft benchmarks and lessons learned:**
  - `docs/LESSONS_LEARNED.md`: rank=19 at 143K shots/s
  - `docs/SPLIT_HEURISTIC.md`: 60-70% frame-only instructions
  - `docs/K_PROFILE_ANALYSIS.md`: persistent hybrid kernel design
