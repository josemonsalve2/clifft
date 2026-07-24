# The Tile / Fine-Grained-Task Model, Mapped to the V2 MLIR Backend

**Audience:** V2 MLIR-backend ultrathink. **Target HW:** AMD MI350X (gfx950, CDNA4).
**Date:** 2026-07-24.
**Supersedes context in** `docs/v2/runtime_patterns_iris_hipkittens.md` (persistence / memory-model / IRIS
angle). That brief predates two hard constraints: **R6 — no HIP/CUDA C++ may be linked or vendored**, and
**R7 — V2 must exploit tiles-as-data + an explicit memory↔compute split**, not flat SPMD. This document is
about the *programming MODEL* of four tile libraries (ThunderKittens, HipKittens, Composable Kernel/`ck_tile`,
cuTile, + Triton as an IR reference), extracted so it can be **reimplemented in an MLIR `quantum` dialect**.
None of these libraries can be vendored: TK/HK/CK/cuTile are all HIP/CUDA C++ template headers. We steal
concepts, not code.

The reader must keep one fact in front of them at all times: **our compute is a 2×2 / 4×4 complex butterfly,
not an MMA.** Every one of these libraries is architected around feeding a matrix core with a GEMM. That is
the single largest source of "looks relevant, is actually dead weight" risk in this brief, and it is
flagged repeatedly below.

---

## 1. Tile as a DATA / COOPERATION abstraction — the part that transfers

This is the half of the tile model that maps cleanly onto our **measurement reduction** and our **coop-tier
cooperation**, and it is where the four projects agree most.

**ThunderKittens** (arXiv:2410.20399, §3.1) makes a tile a *templated datatype*: register tiles (`rt_bf<16,64>`),
shared tiles (`st_bf<64,64>`), plus 1-D register/shared vectors, parameterized by dtype, rows, cols, and (for
register tiles) a **layout** (`row`/`col`, `ducks::rt_layout`). A **warp owns a tile with its contents
distributed across the 32 threads' registers**; a `group<4>` warpgroup (128 threads) owns a larger tile. HBM
is addressed through a `gl<...>` global-layout descriptor so the kernel "pretends everything is a PyTorch
tensor." Bank conflicts are handled by three compile-time **swizzled** shared layouts at 32/64/128-byte
granularity (§ App. C: Naive vs Padded vs Swizzled). Reductions are **layout-aware bulk operators** —
`row_max`, `row_sum`, `col_reduce`, `sub_row`, `div_row` — that pick a cross-lane shuffle strategy matching
the tile's register distribution (used directly in the flash-attention softmax, Fig. 2).

**HipKittens** is TK's CDNA3/CDNA4 port and is the most directly relevant because it is gfx950-native. Same
tile datatype on **64-wide waves**, register vs shared (LDS) backing, bank-conflict-free swizzles (§3.2.2),
and layout-aware reductions. Its distinctive contribution is **developer register pinning** (§3.2.1): because
HIPCC refuses to use AGPRs as matrix inputs and mis-allocates under pressure, HK lets the developer *pin each
tile's registers explicitly*, worth 855→1024 TFLOPS on MHA-backward (Table 1). It also uses **direct async
HBM→LDS buffer loads** (§3.2.2) and notes (App. B.2) that "as of September 2025, buffer loads are not the
default for Triton loads/stores on AMD" — a concrete reason hand-tuned AMD kernels still beat Triton.

**Composable Kernel / `ck_tile`** is the most *structured* and the most interesting for an IR designer,
because it **separates three concerns that the C++ libraries fuse**:
- a **tensor descriptor / tensor view** — *what data the tile spans*, built from composable coordinate
  transforms (`MergeTransform`, `UnmergeTransform`, `EmbedTransform`, `PadTransform`, `OffsetTransform`)
  chained through a `TensorAdaptor`. Layout is *algebra resolved at compile time.*
- a **`tile_distribution`** — *which lane owns which element* — a static encoding over coordinate spaces
  X (physical), Y (per-thread traversal), P = (warp_id, lane_id), R (replication), D (register slot). The
  core map is **P + Y → X**. The H-levels are named **Repeat / WarpPerBlock / ThreadPerWarp / Vector**. Every
  lane computes its own physical coordinates from *the same static encoding*, which is precisely how a
  workgroup coalesces without hand-written index math.
- a **`tile_window`** (`make_tile_window(view, lengths, origin, distribution)`) — the access gateway with
  `.load()`/`.store()`/`.set_window_origin()`, whose `LoadStoreTraits` picks the vector dimension,
  `scalar_per_vector`, and a **space-filling (snake) curve** for cache locality. `sweep_tile` iterates a
  thread's owned elements; `block_reduce` / `block_reduce2d` do cross-lane-then-cross-warp-via-LDS reductions.

**Triton / cuTile** express the same idea at the IR level (§3 below): a tile is a *typed SSA value*; the
compiler chooses the intra-block thread layout.

**What transfers to us.** Our **coop tier** (256 threads cooperating in LDS on one shot's statevector) *is*
a shared tile with a distribution. Our **measurement reduction** (`sum |v[i]|²` over a half-space, then a
sampled-branch broadcast) is exactly the `row_reduce` / `block_reduce2d` primitive that all four encapsulate
*because hand-rolling the cross-lane shuffle is where bugs live* — which is precisely our current
`ds_bpermute` bug source. The single most valuable idea to steal is **CK's `tile_distribution`**: an explicit,
compile-time lane↔amplitude map from which coalesced addresses and correct reduction trees are *derived*
rather than hand-written. That is a dialect concept, not a C++ artifact, so R6 does not block it.

---

## 2. Tile as a COMPUTE primitive (MMA feeder) — mostly DEAD WEIGHT for us

Be blunt: every library's compute hero is a matrix-core GEMM.

- **TK:** `mma_AB`, `mma_ABt`, warpgroup `mma_async` lowering to WGMMA/TCGEN05; the 16×16 base tile is chosen
  "to maximize compatibility with tensor cores" (§3.1). The tile shape *is* the tensor-core fragment shape.
- **HK:** MFMA-wrapped warp GEMM; the whole scheduling apparatus assumes an MFMA compute stream.
- **CK:** `warp_gemm_attribute_mfma` (32×32×8, 16×16×16), `block_gemm_areg_bsmem_creg`, `block_universal_gemm`.
  The register-fragment layout fed to MFMA *is* what `tile_distribution` produces.
- **cuTile:** `ct.mma`, "the compiler generates optimal Tensor Core instructions when shapes qualify."

Our gate is a **2×2 (U2) or 4×4 (U4) complex mat-vec applied to amplitude planes** — `k`≤19, so the *batch*
of independent butterflies is huge (2^(k-2) groups for a U4) but each contraction is trivially small. **None
of the `mma`/MFMA operators apply.** The MI350X matrix core's smallest MFMA is 16×16×16 — two orders of
magnitude larger than a 4×4, and it wants a shared contraction dimension we do not have.

**The tempting trap (evaluate, don't adopt blindly):** batch many independent U4 butterflies into a
structured small matmul — stack N amplitude-groups as columns and multiply by the (block-diagonal, per-group
*different*) 4×4. This fails cleanly: MMA contracts a *shared* operand across the batch; our 4×4 unitary is
**the same within one gate but the amplitudes differ per group** — so it is actually `C[g] = U · X[g]`, a
batched mat-vec with a *shared small matrix* and *many vectors*. That is a legitimate GEMM shape only if you
reshape to `U (4×4) · X (4×G)` = `Y (4×G)`, i.e. a **4×4 · 4×G matmul with contraction dim K=4**. K=4 is far
below the MFMA's native K (8/16/32); you would pad K to 16 and waste 75%+ of the matrix core, plus pay a
transpose/pack cost to get amplitudes into MFMA fragment layout, plus the complex-arithmetic expansion (a
complex 4×4 becomes a real 8×8 with sign structure). Verdict: **matrix cores are a trap for the gate sweep.**
The butterfly is memory-bound and ALU-light; a well-vectorized strided loop (`v_pk_*` packed-f32 FMAs)
saturates the same VGPR/LDS bandwidth without the pack/transpose tax. Keep MMA operators entirely out of the
`quantum` dialect. (Open question 6.1 revisits whether the *coop tier specifically*, where a whole shot lives
in LDS and G is small, could ever justify it — provisional answer: no.)

---

## 3. The memory↔compute SPLIT and async pipeline — help the mem/gate overlap? Be skeptical.

R7 asks for producer(memory)/consumer(compute) wave specialization with async pipelines. Here the sources
**disagree in a way that matters for us**, and the AMD-native one is the most sceptical.

**TK (NVIDIA)** makes this its centrepiece: the **Load-Compute-Store-Finish (LCSF)** template (§3.2) is
explicit producer/consumer warp specialization; producers `decrease_registers<40>()`, consumers
`increase_registers<232>()`; async copies via `cp.async` or Hopper **TMA**; an N-stage shared-memory ring
(`INPUT_PIPE_STAGES=4`, `tic`/`toc` swap) drives 260→760 TFLOPS as stages go 1→4 (Table 1). This works
**because Hopper has hardware for it**: TMA, mbarriers, and register reallocation.

**HipKittens is the crucial counter-evidence, and it is on our exact silicon.** §3.3.1: "AMD hardware
statically divides registers across all waves, meaning producers consume registers without contributing to
output computation." AMD **lacks TMA, WGMMA-style async matrix ops, register reallocation, and mbarriers**.
Table 2 shows wave-specialization performance *degrading as producer count rises* on MI355X — the opposite of
B200. HK therefore *abandons producer/consumer specialization on CDNA* and uses two schedules that keep every
wave doing both jobs:
- **8-wave ping-pong** (§3.3.2): 8 waves, 2 per SIMD, split into two groups of 4 that alternate "issue only
  compute" ↔ "issue only memory" and swap. Best for **balanced** compute/memory workloads.
- **4-wave interleave** (§3.3.2): one wave per SIMD, each issuing compute+memory in a **staggered** sequence.
  Best for **imbalanced** (compute-heavy *or* memory-heavy) workloads.

**CK** takes the same non-specialized road via its `gemm_pipeline_ag_bg_cr_*` families: `GlobalPrefetchAsync`
(vector buffer-load HBM→LDS, skipping the register hop), `LocalPrefetch` (LDS→register), `PrefetchStages=3`,
double-buffered LDS + register ping-pong, and an **intrawave vs interwave** knob — interwave for memory-bound
(K chunked, waves lockstep, better cache hits), intrawave for compute-bound. No producer/consumer wave roles.

**Triton** derives the pipeline *automatically*: the prefetching pass (MAPL'19 §5.1.1, Listing 7) hoists
loop-carried tile loads into the prologue and rewrites with `phi` nodes — software pipelining synthesized by
the compiler, driving `async_copy` on the backend.

**What transfers.** Our workload is **imbalanced and memory-bound** (gate sweep streams 4 amplitude planes
with a few FMAs; measurement is a pure bandwidth reduction). By HK's own taxonomy that points at **4-wave
interleave**, *not* ping-pong, and **explicitly away from producer/consumer wave specialization** — which HK
measured to underperform on CDNA4. The async **buffer_load HBM→LDS with double buffering** (HK §3.2.2 / CK
`GlobalPrefetchAsync`) is the one pipeline piece worth stealing for the **global tier** statevector stream.
But see Open Question 6.2: with 10,000 embarrassingly-parallel shots, occupancy may *already* hide the
latency the pipeline is meant to hide, making the whole split a complexity tax with no payoff.

---

## 4. How the MLIR `quantum` dialect should express this

The synthesis: adopt the **IR-level tile concepts from CK / Triton / cuTile** (they are already IR-shaped),
not the C++ operator libraries. Concretely:

**Types.**
- `!quantum.statevector<tier, k>` — a tile of 2^k complex-f32 amplitudes, `tier ∈ {register, coop, global}`
  selecting VGPR / LDS(addrspace 3) / HBM backing. This is TK/HK's "same tile interface, different memory
  backing," expressed as a *type parameter* instead of a C++ template instantiation.
- `!quantum.tile_dist` — a **CK-style static distribution** (lane↔amplitude map with H-levels
  Repeat/WavePerBlock/ThreadPerWave/Vector). This is the load-bearing IR concept: coalesced addresses and
  correct reduction trees are *derived* from it.

**Ops (carry the tile, not raw pointers — Triton's `make_block_ptr` lesson):**
- `quantum.gate_sweep %sv, %U : (statevector, matrix) ` — apply a U2/U4 to the tile. Attributes: target
  qubit(s), and the 2×2/4×4 constants when known at compile time (see specialization).
- `quantum.measure_reduce %sv -> f32` — cooperative `sum|v|²` over a half-space. Lowers to the
  distribution-derived cross-lane reduction (replaces hand-rolled `ds_bpermute`). This op *is* CK's
  `block_reduce2d` reified for our layout.
- `quantum.sample_broadcast %prob, %rng -> i1` — sampled-branch decision broadcast to all lanes.
- `quantum.compact %sv, %branch` — fold/compact the statevector after measurement (the fold half of measure).
- `quantum.frame_xor %frame, %mask` — the tiny Pauli-frame bookkeeping; scalar, no tile.

**Lowering passes.**
1. **tile-load → tile-compute → tile-store** materialization: `gate_sweep` expands to (buffer_load planes →
   4×4 complex mat-vec in registers → store planes) over the distribution's owned groups; global-tier
   load/store become async `buffer_load`/`buffer_store` to/from LDS with double-buffering.
2. **mem/compute scheduling pass** (the R7 request, done at IR level like Triton §5.1.1): interleave
   `measure_reduce` memory traffic with adjacent `gate_sweep` ALU — but emit **4-wave-interleave** staggering,
   **not** producer/consumer wave roles (HK §3.3.1). This pass should be *optional and measured*, gated behind
   Open Question 6.2.
3. **specialization by burning constants:** when the gate's 4×4 unitary and the active rank `k` are known at
   compile time (they are, from the circuit), specialize the sweep — unroll the plane addressing by `k`, fold
   the matrix constants into FMAs, eliminate zero terms (Clifford gates have many). This is Triton's
   "shape is a compile-time constant, autotuner instantiates" applied to *rank and matrix*.

**What to steal at the IR level specifically:** from **Triton** — the tile as a *typed SSA value* over which
ordinary elementwise ops are reinterpreted, plus the pass-derived thread layout (MAPL'19 §4.2.2, §5.2.2);
from **cuTile** — the `tile<...>` type living in an MLIR-style dialect (`cuda_tile.module`, virtual-ISA
"Tile IR" boundary, `latency=` load hints as scheduling hints) which validates that *this exact design
lands as an MLIR dialect*; from **CK** — `tile_distribution` and coordinate-transform layout algebra as the
representation of "who owns what." These three are model, not code, and clear R6.

---

## 5. MODEL (reimplement) vs CODE (ignore) — ranked

**Adopt as MODEL, high confidence (reimplement in MLIR/IR):**
1. **CK `tile_distribution`** — static lane↔amplitude map; foundation for coalescing *and* correct
   reductions. The single highest-value transfer.
2. **Layout-aware cooperative reduction / broadcast** (TK `row_reduce`, CK `block_reduce2d`, HK's tuned
   cross-lane) for `measure_reduce` + `sample_broadcast` — kills the `ds_bpermute` bug class.
3. **"Same tile op, three memory backings" via a tier type param** (TK/HK interface-vs-implementation split)
   — one `gate_sweep` lowering, instantiated register/coop/global.
4. **Triton/cuTile IR-level tile type + compiler-chosen intra-block layout** — the dialect's spine.
5. **Register pinning intent** (HK §3.2.1) — for the register tier, guarantee the statevector stays in VGPRs.
   In our path this is an `llc`/allocator concern, not a C++ API, but the *intent* (pin the working tile,
   don't let the allocator spill) is worth encoding.
6. **Async `buffer_load` HBM→LDS with double buffering** (HK §3.2.2 / CK `GlobalPrefetchAsync`) — global-tier
   streaming only; *measure before committing* (6.2).

**Adopt with caution / template only:**
7. **4-wave interleave** (HK §3.3.2) for the imbalanced, memory-bound sweep — inspiration for the scheduling
   pass, not a drop-in.
8. **CK intrawave/interwave knob** — interwave maps to our memory-bound case; a scheduling-pass parameter.

**Explicitly does NOT transfer:**
- **All MMA/MFMA/WGMMA compute operators** (TK `mma_*`, HK MFMA GEMM, CK `warp_gemm_attribute_mfma` +
  `block_universal_gemm`, cuTile `ct.mma`). Our butterfly is 2×2/4×4; the batched-matmul reshape is a trap
  (§2). Matrix cores stay out of the dialect.
- **Producer/consumer wave specialization + 8-wave ping-pong** (TK LCSF, register reallocation). HK §3.3.1
  measures this *underperforming on CDNA4* — no TMA, no mbarriers, no register reallocation; statically-split
  registers make producers pure overhead. Do not build wave roles.
- **Any C++ runtime / header** — TK/HK/CK/cuTile are HIP/CUDA C++ (R6). Zero lines vendored. TK's `gl<>`
  descriptors, CK's `tile_window` class, cuTile's `ct.*` host API — all reimplemented or replaced by HSA.
- **Hopper/Blackwell-only mechanisms** — TMA, TCGEN05, mbarriers, `cp.async` (that is NVIDIA; our async is
  AMD `buffer_load`/`global_load_lds`).

---

## 6. Open questions for the V2 ultrathink

1. **Butterfly-as-tile vs plain strided loop.** Is wrapping the U4 sweep in a tile/distribution abstraction
   worth it, or does a plain strided `for` loop over 2^(k-2) groups — which `llc` already auto-vectorizes into
   packed-f32 FMAs — get the same code with less IR machinery? The tile buys us (a) a *shared* reduction
   substrate with measurement and (b) coalescing guarantees; if those are the only wins, the sweep itself may
   not need the tile — only `measure_reduce` does. **Hypothesis to test: use the distribution abstraction for
   measurement/coop cooperation, keep the gate sweep a plain vectorized loop.** This is the highest-leverage
   design decision in the brief.

2. **Is the mem/compute overlap worth it given 10,000 shots?** Shots are embarrassingly parallel and already
   provide massive occupancy; the GPU may already hide global-load latency behind other shots' compute,
   making the async pipeline / 4-wave interleave a pure complexity tax with ~0 payoff. **Measure occupancy and
   memory-latency stalls (rocprof-compute) on the global tier *before* building the scheduling pass.** HK/CK
   pipelines exist because a *single* GEMM has no other work to hide behind; we do. This likely *deletes* R7's
   pipeline half for the shot-parallel tiers.

3. **Does `tile_distribution` survive our layout?** CK's encoding assumes MFMA-friendly fragment layouts. Our
   amplitude layout is qubit-index-natural. Do we adopt an MFMA-agnostic distribution (we have no MFMA, so we
   are free to), and is a *complex-f32* interleaving (SoA real/imag vs AoS) the right Vector-level choice for
   coalesced `buffer_load`?

4. **Coop-tier reduction correctness under divergence.** The `measure_reduce` cross-lane tree must converge
   all 256 lanes (the divergent-barrier bug from V1). Encoding the reduction as a distribution-derived op
   *should* make convergence structural — verify the lowering emits the barrier outside per-lane branches.

5. **Specialization blow-up.** Burning rank + matrix constants per gate is Triton-style monomorphization; with
   16k operands does this reintroduce the V1 IR-size explosion? The runtime-interpreter direction (prior
   brief) says *don't* fully specialize; the tile direction says *do* specialize the hot sweep. Reconcile:
   specialize the *operand-kind* device functions once, dispatch at runtime — specialization of the tile
   shape (by `k`), not of every gate instance.

6. **Global-tier distribution vs work-stealing granularity.** The prior brief's work-stealer assigns
   shots/tiles to workgroups; the distribution assigns amplitudes to lanes. These compose (workgroup steals a
   shot-tile, distribution splits it across lanes) but the two-level granularity for heterogeneous circuit
   tails is unresolved.

---

## Sources

- ThunderKittens paper (arXiv:2410.20399), full HTML: https://arxiv.org/html/2410.20399v1 — §3.1 tile
  datatypes/layouts/swizzles, §3.2 LCSF + TMA pipeline, Fig. 2/4/5, Table 1, App. B/C. Abstract:
  https://arxiv.org/abs/2410.20399
- ThunderKittens repo: https://github.com/HazyResearch/ThunderKittens (`include/{types,ops,prototype}` layout);
  blog https://hazyresearch.stanford.edu/blog/2024-05-12-tk and https://hazyresearch.stanford.edu/blog/2024-10-29-tk2
- HipKittens paper (arXiv:2511.08083), HTML: https://arxiv.org/html/2511.08083v1 — §3.2.1 register pinning
  (Table 1, 855→1024 TFLOPS), §3.2.2 buffer_load/swizzle + Triton comparison (App. B.2), §3.3.1 wave
  specialization underperforms on CDNA (Table 2), §3.3.2 8-wave ping-pong vs 4-wave interleave. Abstract:
  https://arxiv.org/abs/2511.08083
- Composable Kernel / `ck_tile`: https://github.com/ROCm/composable_kernel and develop-branch trees under
  `include/ck_tile/{core/tensor, ops/gemm/{warp,block,pipeline}, ops/reduce}`; ROCm conceptual docs —
  https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/CK-tile-index.html ,
  .../tile_window.html , .../tile_distribution.html , and .../CK-Tile-intra-inter-wave.html (intrawave vs
  interwave). Async pipeline header: `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async.hpp`.
- cuTile / Tile IR (NVIDIA dev blog): https://developer.nvidia.com/blog/simplify-gpu-programming-with-nvidia-cuda-tile-in-python/
  (Tile IR as virtual ISA; `ct.bid/ct.load/ct.store`), .../how-to-write-high-performance-matrix-multiply-in-nvidia-cuda-tile/
  (`ct.mma`, per-tile-size specialization), .../cutile-jl-brings-nvidia-cuda-tile-based-programming-to-julia/
  (`cuda_tile.module` textual MLIR-style IR, `tileiras` assembler), .../tuning-flash-attention-for-peak-performance-in-nvidia-cuda-tile/
  (`latency=` load hints).
- Triton (Tillet, Kung, Cox, MAPL'19): §4 Triton-IR (tile as typed value, elementwise reinterpretation,
  `dot`/`reshape`/`broadcast`, Listing 5), §5 passes (5.1.1 prefetching/pipelining Listing 7, 5.2.1
  hierarchical tiling, 5.2.2 coalescing, 5.2.3 shared-mem alloc, 5.2.4 sync). Docs:
  https://triton-lang.org/main/programming-guide/chapter-1/introduction.html and
  https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
- Prior brief (read, not re-derived): `docs/v2/runtime_patterns_iris_hipkittens.md` (IRIS persistence,
  SC-HRF memory model, Three Taxes).
