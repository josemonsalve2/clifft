# CPU Reference Path — Static Characterization (clifft SVM)

Static analysis only; no code was run. Line numbers verified against source dated
2026-06-18 (svm/*), current as of this reading. The CPU path is the f64 "gold"
reference used to validate GPU backends.

Entry point: `clifft::sample_survivors()` at
`src/clifft/svm/svm.cc:266`. It calls `execute()` (`src/clifft/svm/svm.cc:158`)
which dispatches to `scalar::/avx2::/avx512::execute_internal()` — one body,
compiled 3× with different `-m` flags (`src/clifft/svm/svm_scalar.cc`,
`svm_avx2.cc`, `svm_avx512.cc`, all including `svm_kernels.inl`). Runtime CPUID
dispatch resolves the highest kernel the host can run (`svm.cc:112-145`).

---

## 1. Execution structure

**Shot loop (SERIAL):** `src/clifft/svm/svm.cc:299-342`. A plain
`for (uint32_t shot = 0; shot < shots; ++shot)` over one reused
`SchrodingerState`. There is **no per-shot parallelism** — no `omp parallel`,
`std::thread`, or `std::async` wraps the shot loop. Each shot:
`state.reset()` (svm.cc:300-302) → seed next noise gap (svm.cc:304-305,
`draw_next_noise`) → `execute(program, state)` (svm.cc:307) → survivor
bookkeeping (svm.cc:309-341). `sample()`, `sample_k()`, `sample_k_survivors()`
share the identical serial structure.

**Dispatch loop (per shot):** `src/clifft/svm/svm_kernels.inl:2183`
`execute_internal()`. It is a **computed-goto threaded interpreter**: a
`static const void* dispatch_table[256]` of label addresses
(svm_kernels.inl:2196-2255) indexed by `pc->opcode`; the `DISPATCH()` macro
(svm_kernels.inl:2260-2265) does `++pc; goto *dispatch_table[pc->opcode]`. Each
opcode label calls an `exec_*` handler then `DISPATCH()`. This is a tight,
branch-predictor-friendly loop with per-opcode indirect-branch history.

**Parallelism that DOES exist — intra-op, gated by rank.** OpenMP `#pragma omp
parallel for` lives *inside* the amplitude sweeps via helpers in
`src/clifft/svm/svm_internal.h:89-267` (`parallel_for`, `parallel_reduce`,
`parallel_flat_loop`, `parallel_stride_loop`, `parallel_3d_stride_loop`). All are
guarded by `if (active_k >= kMinRankForThreads)` where
`kMinRankForThreads = 18` (svm_internal.h:60). **Threads only engage at
active_k ≥ 18** (statevector ≥ 4 MB); below that everything runs single-threaded
to avoid thread-wakeup overhead. State allocation zero-fill is likewise threaded
only at peak_rank ≥ 18 (`svm_state.cc:114`).

Net: for the vast majority of benchmark circuits (register rank 0-4, coop rank
5-10, most "global" ≤17), **the CPU path is effectively single-threaded**;
throughput is one core running the computed-goto loop `shots` times.

---

## 2. Per-shot cost model (in active_k; statevector dim = 2^active_k)

Two opcode families with sharply different cost:

**Frame ops — O(1)** (`svm_kernels.inl:60-116`): `OP_FRAME_{CNOT,CZ,H,S,S_DAG,
SWAP}` touch only the Pauli-frame bitsets `p_x/p_z` (a few `bit_get/bit_xor`).
They **never read the amplitude array**. This is the whole point of Pauli-frame
tracking: Clifford gates on dormant/frame qubits are free. `OP_APPLY_PAULI`
(svm_kernels.inl:1869) is O(num_qubits/64) word XORs — still frame-only.

**Array / amplitude ops — O(2^active_k)** (each is one or more linear passes
over `v_`):
- `OP_ARRAY_{CNOT,CZ,SWAP,MULTI_CNOT,MULTI_CZ}` (svm_kernels.inl:124-638):
  permutation sweeps over 2^k complex entries (structured stride / bit-weave).
- `OP_ARRAY_H` (svm_kernels.inl:639): butterfly over 2^k.
- `OP_ARRAY_{S,S_DAG,T,T_DAG,ROT}` (svm_kernels.inl:854-1086): diagonal
  `apply_phase_waterfall` (svm_kernels.inl:764) — one pass over 2^k, phasing the
  half where bit v is set.
- `OP_ARRAY_U2` (svm_kernels.inl:1091), `OP_ARRAY_U4` (svm_kernels.inl:1310):
  fused 2×2 / 4×4 unitary, one pass over 2^k with 4 / 16 complex-mul per group.
- `OP_EXPAND` / `OP_EXPAND_{T,T_DAG,ROT}` (svm_kernels.inl:876, 995-1043):
  duplicate lower half into upper half → **grows k by 1, cost O(2^k)** for the
  copy, then `scale_magnitude(1/√2)`. Expand is where the working set doubles.
- Measurements on the active axis:
  `OP_MEAS_ACTIVE_DIAGONAL` (svm_kernels.inl:1573) and
  `OP_MEAS_ACTIVE_INTERFERE` (svm_kernels.inl:1634): **two O(2^k) passes** — a
  `parallel_reduce` to compute branch norms, then a `parallel_for` fold/compact
  — and **shrink k by 1**. `OP_SWAP_MEAS_INTERFERE` (svm_kernels.inl:1728) fuses
  a swap into the fold (still O(2^k), one pass).
- `OP_EXP_VAL` (svm_kernels.inl:~2007+): `parallel_reduce` over 2^k computing
  `<P>` numerator/denominator — O(2^k), read-only.

**Cheap / classical — O(1) or O(list):**
- `OP_MEAS_DORMANT_STATIC/RANDOM` (svm_kernels.inl:1540, 1551): dormant qubit is
  frame-only, **no array touch**, O(1) (+ a PRNG roll for RANDOM).
- `OP_NOISE` (svm_kernels.inl:1886): frame-only Pauli application, O(channels);
  gap-skipped so most sites are O(1) `site_idx != next_noise_idx` early-outs.
  `OP_NOISE_BLOCK` (svm_kernels.inl:1926) loops firing sites only.
- `OP_READOUT_NOISE`, `OP_DETECTOR`, `OP_POSTSELECT`, `OP_OBSERVABLE`
  (svm_kernels.inl:1937-2007): classical-record XOR/parity over a target list,
  O(list length), no array touch.

**Consequence:** per-shot cost ≈ Σ over array/expand/active-meas opcodes of
2^(active_k at that op). `scale_magnitude` (svm.h:177) can add an extra O(2^k)
renormalization pass but only when |gamma| drifts past 1e±100 (rare). The
`StatevectorSqueezePass` (per CLAUDE.md) minimizes peak_rank, so many nominally
"high-rank" circuits execute mostly frame-only and stay cheap.

---

## 3. Data layout & memory

**Statevector `v_`** (`svm.h:257`, allocated in `svm_state.cc:36-126`): a raw
`std::complex<double>*` — **f64, interleaved re/im**, 16 bytes/amplitude. Sized
to `array_size_ = 2^peak_rank` (the AOT compile-time max), allocated **once per
`SchrodingerState`** (i.e. once per `sample_survivors` call, not per shot).

Alignment / backing:
- ≥ 2 MB: tries `mmap(MAP_HUGETLB)` 2 MB huge pages (svm_state.cc:72-81), else
  `aligned_alloc` at 2 MB alignment + `madvise(MADV_HUGEPAGE)` (svm_state.cc:83-98).
- < 2 MB: `aligned_alloc(4096, …)` (svm_state.cc:100-105).
- Zero-fill is threaded only at peak_rank ≥ 18 for NUMA first-touch
  (svm_state.cc:112-124); `v_[0] = 1`.

**Working-set size (2^peak_rank × 16 B):**
- rank 4: 256 B (fits in L1)
- rank 10 (coop): 16 KB (fits in L1/L2)
- rank 14: 256 KB (L2)
- rank 18: 4 MB (spills L3 — the threading threshold)
- rank 19: 8 MB; rank 20: 16 MB (main-memory bound)

Plus small per-state vectors: `p_x/p_z` = 2×⌈num_qubits/64⌉ words;
`meas_record/det_record/obs_record` (uint8) and `exp_vals` (double), all sized to
program counts. **No malloc in the hot path** — `state.reset()`
(svm_state.cc:220) only rezeros the active `2^active_k` prefix, the frame words,
and records; it reuses the existing allocation. The only per-shot heap activity
is in survivor bookkeeping when `keep_records=true` (`vector::insert`/`push_back`
into result arrays, svm.cc:325-341) and in the *forced-fault* sampler
(`sample_k`), whose persistent pool / DP table are hoisted out of the loop
(svm.cc:577-660) — `dp_sample_indices`/`uniform_sample_indices` push into
pre-cleared vectors, so allocations amortize to zero after warmup.

---

## 4. Hotspot hypotheses

**(a) rank-4 register circuit** — 256 B statevector lives in L1; single core.
Time is dominated by **interpreter dispatch overhead** (the computed-goto loop,
`pc` walk, per-opcode indirect branch) and per-shot fixed costs
(`state.reset()`, `draw_next_noise`, survivor bookkeeping), not by amplitude
math. Array sweeps of 16 amplitudes are near-free. Expected hotspots:
`execute_internal` dispatch + `reset()` + RNG.

**(b) rank-10 coop circuit** — 16 KB statevector (L1/L2), still single-threaded
(< kMinRankForThreads=18). Array/expand/active-measurement passes over 1024
amplitudes now dominate. Likely hotspots: `apply_phase_waterfall`
(T/S/ROT phasing), `exec_array_h`, `exec_expand*`, and the
`parallel_reduce`/`parallel_for` bodies inside
`exec_meas_active_diagonal`/`_interfere` (svm_kernels.inl:1573/1634) — each
active measurement is two full 2^k passes. AVX-512 kernel processes 4 complex/vec
but it is still one core.

**(c) rank-14+ global circuit** — 256 KB–8 MB, **memory-bandwidth bound**.
Below rank 18 still single-threaded, so one core streams the whole array per
array op — DRAM/L3 bandwidth is the ceiling. At rank ≥ 18 OpenMP finally engages
and it becomes multi-core bandwidth bound. Dominant hotspots: `exec_expand*`
(doubles the array), `exec_array_{cnot,cz,swap}` strided permutations (poor
locality at large stride → TLB/prefetch pressure, mitigated by huge pages), and
active-measurement reduce+fold passes. `scale_magnitude` renormalization adds
occasional extra full passes.

---

## 5. RNG

**`Xoshiro256PlusPlus`** — xoshiro256++ seeded by SplitMix64
(`svm.h:40-90`). State is 32 bytes, embedded **per `SchrodingerState`** (member
`rng_`, svm.h:262). `random_double()` (svm.h:150) is
`(rng_() >> 11) * 2^-53` — deliberately **not** `std::uniform_real_distribution`
(for cross-compiler determinism).

**Serial dependency, per-op (not per-shot):** the RNG is **seeded once per batch
and streams forward across shots** — `reset()` explicitly does *not* reseed
(svm_state.cc:266; svm.h:134-138). Within a shot the stream is advanced on every
random draw: measurement branch sampling (`sample_branch`, svm_internal.h:77),
noise-gap draws (`draw_next_noise`, svm.h:244), channel selection
(`exec_noise`, svm_kernels.inl:1902), readout noise (svm_kernels.inl:1953), and
forced-fault index sampling. Each `rng_()` is a strict scalar
state-carried-dependency (rotate/xor chain, ~handful of cycles) — **inherently
serial and unvectorizable**, but tiny relative to array sweeps. Because the
single stream threads through all shots in order, the serial shot loop is also a
correctness constraint, not just a perf choice: parallelizing shots would need
per-stream jump-ahead / substream splitting.

---

## 6. Vectorization potential

**Already hand-vectorized.** The hot array kernels ship explicit AVX-512 and
AVX2 intrinsic paths with a scalar fallback, selected at compile time
(`#if defined(__AVX512F__) …` / `#if defined(__AVX2__)`), and dispatched at
runtime by CPUID (svm.cc:112-145). Examples: `apply_phase_waterfall`
(svm_kernels.inl:764, `_mm512_*` complex-mul + blend), `expand_with_phase`
(svm_kernels.inl:906, `cmul_m512d`/`cmul_m256d`), `exec_array_cnot`
(svm_kernels.inl:129, `_mm512_permutexvar_pd` / masked blend). `-ffast-math` and
`-march=native`/`x86-64-v3` are on globally (CMakeLists.txt:109-130).

**Auto-vectorizability of the scalar/reduce loops:**
- Pointers are `__restrict` (`auto* __restrict arr`, e.g. svm_kernels.inl:881,
  1578, 1639), removing aliasing barriers.
- Measurement branches are **hoisted outside the inner loop** so the body is
  straight-line (explicit comment svm_kernels.inl:1670-1683; b_x=0/b_x=1 split
  into two loops). Good for both auto-vec and the intrinsic paths.
- The reduction loops (`parallel_reduce` in
  `exec_meas_active_diagonal`/`_interfere`, `OP_EXP_VAL`) accumulate on plain
  `double` and use `std::norm` — with `-ffast-math` these vectorize into
  FMA-reduction chains.

**Obstacles / friction:**
- `std::complex<double>` is passed to intrinsics via
  `reinterpret_cast<double*>` (re/im interleaved). Complex multiply needs
  shuffle+FMA (`cmul_m512d`); it works but is not free SIMD — real/imag lane
  juggling costs shuffles.
- **Strided permutation gates** (`ARRAY_CNOT/CZ/SWAP` with a high axis) have
  large power-of-two strides → gather-like access, poor cache/TLB locality at
  high rank; SIMD width helps throughput but memory latency dominates.
- The **computed-goto dispatch itself is scalar** and cannot be vectorized —
  for low-rank circuits (§4a) this fixed per-opcode cost, not SIMD width, is the
  limiter.
- OpenMP only helps at active_k ≥ 18; from rank ~11–17 the code is single-core
  bandwidth-limited with SIMD but no thread parallelism (a visible gap).

---

## Key takeaways for the perf DB

1. **Shot loop is serial single-threaded** (svm.cc:299); the only parallelism is
   intra-op OpenMP gated at active_k ≥ 18 (svm_internal.h:60). Most benchmarks
   never cross that threshold → one core.
2. **Cost is bimodal:** frame/Clifford/dormant/classical ops are O(1)
   frame-only; array/expand/active-measurement ops are O(2^active_k). Runtime ≈
   Σ 2^k over the array-touching opcodes.
3. **f64 complex, 16 B/amp, one allocation per call** (huge-paged ≥ 2 MB); no
   hot-path malloc except optional record-keeping.
4. Hotspots shift with rank: **dispatch overhead (rank≤~6) → SIMD phasing/fold
   kernels (rank~7–13) → single-core memory bandwidth (rank~14–17) → multi-core
   bandwidth (rank≥18)**.
5. RNG is a single serial xoshiro256++ stream threaded across all shots — cheap
   but forces the serial shot ordering.
6. Array kernels are already AVX-512/AVX2 hand-tuned with `__restrict` and
   hoisted branches; the practical single-thread ceiling below rank 18 and the
   scalar computed-goto dispatch are the remaining structural limits.
