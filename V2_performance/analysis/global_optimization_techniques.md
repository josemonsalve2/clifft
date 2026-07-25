# Global-tier optimization techniques — extracted from AMD agentic frameworks

Target kernel: `clifft_v2_global` (src/clifft/gpu/mlir/v2/coop_interpreter.c:139), high-rank
(11-19) statevector simulation on **gfx950 / MI355X / CDNA4**. Statevector = 2^rank complex
amplitudes (4096-524288) in HBM. 1 workgroup = 256 threads cooperate on ONE shot; amplitudes
touched via strided gather/scatter (`scatter_bits_1/2`, v2_ops.h:151-160). Persistent pool of
~2048 workgroups drains shots via a global atomic work-steal counter.

Bottleneck (rocprofv3, V2 vs GPU-SVM on surface_d7_t19 rank12, ~equal wall time):
V2 = 2x VALU (2.34B vs 1.18B), 25x L2 misses (TCC_MISS 8.2M vs 0.33M), 83% L2 hit rate,
4x waves (8192 vs 2048). Memory-bound with poor L2 locality from the strided cooperative
access pattern. **Hard constraint: FP op ORDER must be unchanged (byte-exact vs GPU-SVM).**
Layout / caching / index-precompute changes are SAFE; FP reassociation is NOT.

This report extracts the concrete techniques these frameworks encode and ranks them for THIS
kernel. Sources cited by file:function/line.

---

## 1. The frameworks' memory-bound playbook (verbatim priority lists)

### Hyperloom — `kernel_optimization.py:_PRIORITY_BULLETS["memory"]` (lines 1024-1050)
The lever order Hyperloom injects into every memory-bound kernel prompt:
1. **Memory traffic reduction** (primary): improve coalescing / vectorization, fuse with
   neighbouring ops to amortize global loads, reduce intermediate writes, avoid extra
   global-memory round trips.
2. **Shape-aware tuning**: specialize block sizes and grid indexing for the dominant shape.
   "Memory-bound kernels are especially sensitive to load-coalescing alignment."
3. **Launch amortization** for tiny high-count shapes: persistent / batched handling.
4. **Structural simplification**: hoist loop-invariant computations, **remove redundant
   address arithmetic**, collapse dual-pass logic.
5. **Compute utilization** (rarely the bottleneck): MFMA tile, occupancy, register balance.

Lever 1 (coalescing) and lever 4 (remove redundant address arithmetic) map DIRECTLY onto our
two symptoms (25x L2 misses and 2x VALU respectively).

### KernelForge — `rocpc.py:_classify` (lines 471-574) + bottleneck hints
KernelForge classifies from empirical roofline (achieved-vs-empirical-peak). Our profile
(nothing near a compute roof, high L2 miss, 4x waves) lands as **BANDWIDTH-BOUND** or
**OCCUPANCY/LATENCY-BOUND** — its hint for bandwidth is "memory traffic is near the HBM/L2 roof
… cut or better reuse data movement"; for latency "raise in-flight work (ILP, prefetch,
pipelining)." Note `_SUMMARY_METRICS` (lines 63-76) explicitly track **vL1D hit rate, L2 hit
rate, L2-fabric read/write BW, L2-fabric read latency** — L2 locality is a first-class signal.

### IntelliKit — `fellows/intellikit/prompts.py:181-190` optimization priority TABLE
| Optimization | Typical gain | Effort |
|---|---|---|
| MFMA opcode upgrade | 20-80% | Low | (N/A — no GEMM here, MFMA=0) |
| **Direct-to-LDS (`buffer_load_lds`)** | **10-17%** | Medium |
| **Register pressure → cross occupancy threshold** | **3-33%** | High |
| **Software pipelining (double/triple buffer)** | **10-30%** | High |
| NOP scheduling | 2-5% | Medium |
| `s_setprio 3` around MFMA | 0.5-1% | Low | (N/A) |
| Barrier elimination (wave-0-only spin) | 3-6% | Medium |

### Apex — `gpu_info/server.py` gfx950 `optimization_priorities` (lines 61-68) + elementwise hints
- "Target **128-byte** memory coalescing" (`memory_coalescing_bytes: 128`, gfx950).
- "Use LDS for data reuse (64 KB per CU)"; "Leverage high HBM3e bandwidth (8.0 TB/s)".
- Elementwise (memory-bound) hints (lines 545-550): **vectorized loads (float4)**, **fuse
  multiple ops into one kernel**, **grid-stride loops**. Our op loops already are grid-stride
  (`for i=t; i<iters; i+=V2_STRIDE=256`) — good — but they are NOT vectorized or coalesced.

---

## 2. L2 locality / cache-thrashing remedies (the 25x-miss angle)

### KernelForge `xcd_l2_locality.md` — the single most relevant card
CDNA4 (MI355X) keeps the **multi-XCD chiplet design: each XCD has its OWN partitioned L2**
(the card notes MI300X = 8 XCDs / partitioned L2; CDNA4/MI350X keeps multi-XCD with a
different CU/XCD count). Key mechanics for our kernel:
- **"L2 is NOT unified across XCDs — a miss serviced from another XCD's L2 or HBM costs more."**
- **"Round-robin block dispatch"**: hardware assigns workgroup-ids to XCDs round-robin; the
  *default linear* workgroup→XCD mapping **scatters data-sharing blocks across all dies,
  defeating L2 reuse.** This is EXACTLY our failure: 2048 persistent workgroups each stream a
  4KB-4MB HBM slice with no XCD affinity → cross-die L2 traffic.
- Levers: **≥1024 workgroups** (we have 2048 ✓), **tile counts a multiple of 8** (so
  round-robin balances across XCDs with no straggler die), and a **swizzled / XCD-aware
  workgroup→data mapping** so blocks reusing the same data co-locate on one XCD's L2. The card
  calls out **persistent kernels** as the enabling structure: "a persistent grid lets you
  assign the tile→XCD mapping *explicitly* instead of trusting the dispatcher." **Our kernel is
  already persistent (work-steal pool)** — we can pin which XCD drains which shots.

  Concrete for us: our per-shot buffer is `global_v + slot*amp_capacity`
  (coop_interpreter.c:162). With 2048 slots each 2^rank complex64, a single shot's statevector
  is far larger than one XCD's L2 slice, so cross-shot L2 reuse is nil regardless. The win is
  **intra-shot**: keep one shot's *hot working set* (the amplitudes a workgroup revisits across
  consecutive ops) resident in ONE XCD's L2 by ensuring a workgroup stays on its XCD and its
  buffer is placed to hit that XCD's L2. `SYNTHESIS.md` already flags "XCD-interleaved
  work-steal (deferred for correctness)" as a global-tier lever — this card is the recipe.

- Verify (from the card): "Omniperf: L2 hit rate per channel / cross-XCD traffic, HBM read BW;
  XCD-aware order should RAISE L2 hits and LOWER HBM reads at equal FLOPs." That is precisely
  our TCC_MISS / L2-hit-rate target.

### Tiling to fit L2 + read-only reuse
The `xcd_l2_locality` and `vectorization_and_coalescing` cards both say **stage strided data
through LDS and transpose/reuse there rather than re-reading from strided global**. Our global
tier deliberately put amplitudes in HBM (LDS pressure was the coop-tier problem), but the
*hot sub-slice* an op sweeps could still be staged: for a low-axis gate, the pairs
`v[base]`/`v[base|axis_bit]` are close together and reused — an LDS-staged tile of the active
window would convert strided HBM re-reads into LDS hits without changing FP order.

---

## 3. The 2x-VALU angle: index-precompute / scatter LUT

**Root cause, confirmed in source.** Every global op body recomputes the gather index per
amplitude per thread. E.g. `v2_op_array_cz` (v2_ops_body.inc:185):
```c
for (u64 i = t; i < iters; i += V2_STRIDE)      // 256-strided grid loop
    u64 idx = scatter_bits_2(i, a, b) | both;    // <-- recomputed EVERY iteration
```
`scatter_bits_2` → two `insert_zero_bit` calls (v2_ops.h:151-160), each a variable shift + two
mask/AND/OR + a shifted-mask — i.e. ~8-12 VALU ops of pure address arithmetic per amplitude,
per thread, for EVERY op in the circuit. That is the "2x VALU / recompute index math per
amplitude" the profile shows, and it is Hyperloom's memory-bound **lever 4: "remove redundant
address arithmetic."**

**The frameworks' pattern AND our own SVM reference already encode the fix.** The GPU-SVM
reference precomputes a **scatter LUT** and reads it instead of recomputing:
`hip_sampler.hip:878-931` — `scatter_lut` (a `uint32_t*` in LDS), `precompute_scatter_1/2`
fill `lut[i] = insert_zero_bit(i, axis)` once (cooperatively, grid-strided), then op bodies do
`base = st.scatter_lut ? st.scatter_lut[i] : scatter_bits_2(...)` (e.g. lines 1001, 1024, 1048)
with a **cache** (`cached_axis1/2/k/mode`, `ensure_scatter_1/2`) that skips recompute when the
(active_k, axis) is unchanged across consecutive same-axis ops.

**Critical gap this uncovers:** the SVM LUT is `nullptr for global-coop` (comment,
hip_sampler.hip:878) — **even the SVM reference does NOT LUT the global tier**, yet the V2
global kernel recomputes the index every amplitude. So V2 pays index math that SVM's
1-shot-per-thread model amortizes trivially (SVM computes `scatter_bits(i,...)` once per its own
`i`, not 256-way). Porting the LUT-in-LDS pattern to V2 global is a NET-NEW win available to
V2 that even SVM leaves on the table.

**Why it's byte-exact-SAFE:** a LUT stores the *same integer index* the arithmetic would
produce (`lut[i] == insert_zero_bit(i, axis)`, exactly). It changes NO floating-point value and
NO summation order — it only replaces recomputed integer addresses with a table read. This is
the strongest candidate: it directly attacks the 2x-VALU with zero numerical risk.

**Sizing for global:** at rank 12, `iters` up to 2^11 = 2048 entries × 4 B = 8 KB LDS per
active axis. Our global kernel currently uses only LDS=8704 B (SYNTHESIS.md) and has occupancy
headroom, so an 8-16 KB scatter LUT in LDS is affordable. The LUT is shared by all 256 threads
(cooperative fill, grid-strided, one `s_barrier`), amortized across every amplitude the op
sweeps AND across consecutive same-axis ops via the cache.

---

## 4. CDNA4 / gfx950-specific memory tips

From Apex `gpu_info/server.py` (lines 31-69), Hyperloom `_GPU_HW["mi355x"]` (lines 554-561),
IntelliKit `_ROOFLINE["gfx950"]` (prompts.py:11), and KernelForge `memory_pipelining.md`:
- **HBM3E ~8.0 TB/s**, 288 GB, **256 MB L2**, 64 KB LDS/CU, 256 CUs, wave64.
- **128-byte coalescing granularity** (`memory_coalescing_bytes: 128`). A wave of 64 lanes
  should touch a contiguous 128B-aligned window. Our complex64 amplitude = 8 B, so 16
  contiguous amplitudes = one 128B line. Strided `scatter_bits` indices break this: for a
  gate on a low axis, lane i → `v[base_i]` and lane i+1 → `v[base_{i+1}]` are NOT adjacent, so
  each lane can miss its own line → the 25x L2 misses.
- **`global_load_dwordx4` (128-bit)** is the widest load; complex64 pairs pack into one
  `dwordx4` (`vectorization_and_coalescing.md`). Where an op reads `v[base]` and
  `v[base|axis_bit]` for a HIGH axis (the two are far apart) vectorization can't help; for a
  LOW axis (adjacent) a `float4`/`dwordx4` load of the pair is a free win.
- **CDNA4 direct-to-LDS widened to 128-bit `GLOBAL_LOAD_LDS` + a direct L1→LDS path**
  (`memory_pipelining.md` CDNA3-vs-CDNA4 table). This is the async-copy analogue: stage the
  active amplitude window HBM→LDS without a VGPR round-trip, freeing registers and overlapping
  load with compute. Relevant if we stage tiles (§2).
- CDNA4 LDS is larger per the pipelining card (160 KB cited there vs Apex's 64 KB — re-verify
  on-device via `rocminfo`; the frameworks disagree, so confirm before sizing big LDS tiles).
- **fp8 is OCP on CDNA4** (not FNUZ) — irrelevant here (we're complex64/f64), noted only so a
  future quantization idea isn't mis-applied.

---

## 5. Ranked techniques for THIS kernel (impact × effort × byte-exact risk)

Ranking axis: expected impact on THIS kernel's L2-thrash + 2x-VALU × implementation effort ×
risk to byte-exactness. All top picks are FP-order-preserving.

| # | Technique | Attacks | Impact | Effort | Byte-exact risk | Source |
|---|---|---|---|---|---|---|
| **1** | **Scatter-index LUT in LDS** (port SVM's `precompute_scatter_1/2` + `ensure_*` cache to global; op bodies read `lut[i]` instead of recomputing `insert_zero_bit`) | **2x VALU** | **High** | **Low-Med** | **None** (same integer index) | hip_sampler.hip:878-931; Hyperloom lever 4 |
| **2** | **XCD-aware work-steal / buffer placement** (pin a workgroup's drained shots + its `global_v` slice to one XCD's L2; keep tile/slot count a multiple of 8) | **25x L2 miss** | **High** | **Med-High** | **None** (mapping only) | xcd_l2_locality.md; SYNTHESIS "XCD-interleaved work-steal (deferred)" |
| **3** | **Vectorize adjacent-amplitude access** (`dwordx4`/`float4` load-store of `v[base]`+`v[base|low_bit]` pairs; align `global_v` slices to 128 B) | **L2 miss + VALU** | **Med** | **Med** | **None** if same values loaded/stored | vectorization_and_coalescing.md; Apex elementwise hints |
| **4** | **LDS-stage the active amplitude window** via CDNA4 128-bit `GLOBAL_LOAD_LDS` (convert strided HBM re-reads across consecutive same-axis ops into LDS hits) | **L2 miss** | **Med** | **High** | **Low** (staging is a copy; watch barrier discipline) | memory_pipelining.md |
| 5 | **Occupancy: raise waves-per-CU** by shrinking any remaining per-shot LDS so more shots resident to hide HBM latency (V2 already 4x waves — verify this isn't already saturating; occupancy may be fine and latency the real limit) | latency-hiding | Med | Med | None | rocpc occupancy/latency class; IntelliKit reg-pressure row |
| 6 | **Barrier elimination** (wave-0-only spin, fuse the per-op trailing `v2_barrier`) — 3-6% per IntelliKit; smaller lever, but stacks | latency | Low | Med | Low (must preserve cross-thread ordering that feeds FP reductions) | intellikit prompts.py:189 |

**Techniques explicitly EXCLUDED as byte-exact-UNSAFE:** any op fusion that merges two gates'
FP math (changes `cmul`/`cadd` order); any reassociation of the cooperative reduction
(`coop_reduce2`, v2_ops.h:175 — comment: "MUST reproduce SVM coop_reduce2's exact summation
order or f64 rounding diverges at measurement branch points"); wider-but-reordered reductions;
fast-math. These are ruled out by the byte-exact constraint and the ULP-sensitive measurement
branches.

---

## Bottom line (top 3-4 for the implementation pass toward 2-3x)

1. **Scatter LUT in LDS (do this first).** Lowest risk (zero — it's the identical integer
   index), directly kills the 2x-VALU, and the exact code already exists in the SVM reference
   (`precompute_scatter_1/2` + the `ensure_*`/`cached_*` skip-cache, hip_sampler.hip:878-931) —
   port it to `clifft_v2_global` and have the `v2_ops_body.inc` loops read `lut[i]` instead of
   calling `scatter_bits_*`. The reference tellingly leaves this OFF for global
   (`nullptr for global-coop`), so V2 has been paying recompute SVM never does 256-way — a
   net-new win. ~8-16 KB LDS at rank 12, which the global tier can afford.
2. **XCD-aware work-steal + 128B-aligned buffer placement (biggest L2 win).** CDNA4's L2 is
   partitioned per XCD; the default round-robin workgroup dispatch scatters data-sharing blocks
   across dies and defeats reuse (`xcd_l2_locality.md`). Because the kernel is already a
   persistent work-steal pool, we can *explicitly* pin which XCD drains which shots and place
   each shot's `global_v` slice to hit that XCD's L2 — the card's prescribed use of persistent
   grids. Keep slot count a multiple of 8. This is the direct lever on the 25x TCC_MISS.
3. **Vectorize + coalesce adjacent-amplitude pairs (`dwordx4`, 128B alignment).** For low-axis
   gates the `v[base]`/`v[base|low_bit]` pair is contiguous — a `float4`/`dwordx4` load-store
   halves transaction count and improves coalescing at the 128B granularity gfx950 wants. Free
   where applicable, byte-exact (same values moved).
4. **(If 1-3 leave HBM latency exposed) LDS-stage the active window via CDNA4 128-bit
   `GLOBAL_LOAD_LDS`.** Converts strided cross-op HBM re-reads into LDS hits using CDNA4's
   direct L1→LDS path; higher effort, do after profiling confirms residual latency.

Verify each with rocprofv3/rocprof-compute the frameworks' way: XCD/LUT changes should RAISE
L2 hit rate and LOWER TCC_MISS at equal FLOPs; the LUT should drop SQ_INSTS_VALU toward SVM's
~1.18B; confirm byte-exact `passed_shots` vs `--cpu-reference`/SVM at the same seed after every
change (per clifft CLAUDE.md).
