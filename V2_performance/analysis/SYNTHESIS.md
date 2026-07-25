# V2 Performance Synthesis — CPU + GPU characterization (2026-07-25)

Node: mi350x-es (gfx950 / MI355X / CDNA4). Clean kernel-vs-kernel timing via
rocprofv3 --kernel-trace (NOT host wall — the earlier baseline's flaw is fixed).

## THE HEADLINE: V2's perf profile is rank-dependent and INVERTED from intuition

| circuit | rank | tier | V2 kernel(µs) | SVM kernel(µs) | **V2/SVM** | verdict |
|---|---|---|---|---|---|---|
| frame_h | 0 | reg(coop) | 521 | 18.4 | **28.3×** | V2 catastrophically slower |
| circuit_d3 | 4 | reg(coop) | 6667 | 439 | **15.2×** | V2 catastrophically slower |
| qv10 | 10 | coop | 5691 | 4334 | **1.31×** | V2 slightly slower |
| surface_d7_t15 | 10 | coop | 36306 | 22008 | **1.65×** | V2 slower |
| surface_d9_t10 | 7 | coop | 72636 | 44227 | **1.64×** | V2 slower |
| surface_d7_t19 | 12 | global | 19324 | 19958 | **0.97×** | V2 ~par (wins) |
| surface_d9_t19 | 13 | global | 40650 | 44844 | **0.91×** | **V2 FASTER** |
| surface_d11_t15 | 11 | global | 70187 | 74793 | **0.94×** | **V2 FASTER** |

**V2 loses badly at low rank, ties in mid-coop, and WINS at the global tier.**
The reason is structural (below), and it dictates the entire optimization plan.

## WHY: it's occupancy + barriers + serial tid0, NOT instruction count

For qv10 (the cleanest coop A/B), V2 vs SVM counters:
- V2 does **FEWER** instructions: VALU 964M vs 1128M, LDS-insts 60.9M vs 89.2M.
- Yet V2 is 1.31× slower and burns **32% more wave-cycles** (SQ_WAVE_CYCLES
  11.9B vs 9.0B) for the identical 80k waves.
- V2 has **~500× more L2 misses** (TCC_MISS 17.4M vs 31K). SVM = 1 shot/thread
  (embarrassingly parallel, great locality); V2 = 256 threads cooperate on one
  shot in LDS, and the cooperative gather/scatter + reduction thrash L2.
- MFMA=0 for both ✓ (no GEMM mislowering). VGPR=64 (low, not register-bound).

**Root causes (ranked by leverage), corroborated by the static GPU analysis:**

1. **LDS occupancy ceiling (coop tier).** LDS=25088 B (25 KB) → only 2 wg/CU
   (8 waves/CU) resident → cannot hide barrier/HBM latency. The big four:
   lds_v 8KB + lds_red_scratch 8KB + lds_meas 4KB + lds_red0/1 4KB.
   **The GLOBAL kernel proves the thesis: it uses LDS=8704 B (amplitudes in
   HBM), gets higher occupancy, and WINS vs SVM.** Shrinking coop LDS is the #1
   lever.

2. **Barrier density + tid0 serialization (all tiers, worst at low rank).**
   ~52 static s_barrier sites; every opcode ends in ≥1 barrier; measurements
   cost 6-9 each. ~34 `if(t==0)` serial regions where 255/256 threads idle.
   For frame_h (rank 0, ALL frame ops = pure tid0 work) this is why V2 is 28×
   slower: SVM does frame ops 1-thread-per-shot with zero cooperation cost; V2
   spins up a 256-thread workgroup per shot to have thread 0 twiddle Pauli bits
   while 255 threads sit at barriers.

3. **Register/shot-packing mismatch at low rank.** At rank 0-4 the statevector
   is 1-16 amplitudes but V2 still launches 256 threads/shot → ~250 threads do
   nothing. SVM packs 1 shot/thread. This is the low-rank catastrophe.

## THE STRATEGIC PIVOT (what the data says to do)

- **Register/low-rank tier (rank 0-6): abandon the 256-thread-per-shot model.**
  Either (a) pack many shots per workgroup (1 shot/thread or 1 shot/wave like
  SVM) for low rank, or (b) route low-rank circuits to a distinct kernel. This
  is where 15-28× is being lost and where the biggest absolute win is.
- **Coop tier (rank 7-10): reclaim LDS to double occupancy.** Shrink/dynamic-
  size lds_meas (4KB→~circuit size) and lds_red_scratch (8KB, only SWAP_MEAS
  uses it) → LDS ~13KB → 4 wg/CU. Cut barriers (fuse active_k++, one-pass
  measurement). Expect the 1.3-1.65× gap to close or invert.
- **Global tier (rank 11+): already winning — protect it, minor tuning.**
  The XCD-interleaved work-steal (deferred for correctness) could add more.

## CPU side (for completeness — CPU is the f64 GOLD, not a perf target, but characterized)

- Shot loop is **serial single-threaded** (svm.cc:299); intra-op OpenMP only
  engages at active_k≥18 (svm_internal.h:60) → nearly all benchmarks run on ONE
  core. Biggest CPU lever = shot-level parallelism (needs RNG substream split;
  the single xoshiro stream threaded across shots is currently a correctness
  constraint).
- Cost bimodal: frame/classical O(1), array/expand/meas O(2^k). Hotspots shift:
  dispatch overhead (rank≤6) → SIMD kernels (7-13) → single-core BW (14-17) →
  multi-core BW (≥18). Already AVX-512/AVX2 hand-tuned.

## DATA LOCATIONS
- Per-circuit GPU JSON: V2_performance/gpu/<circuit>.json (kernel time, regs,
  LDS, all counters, HSA overhead).
- Table: V2_performance/analysis/gpu_profile_summary.{md,json}
- Static: V2_performance/gpu/gpu_kernel_static_characterization.md,
  V2_performance/cpu/cpu_static_characterization.md
- Tooling + agentic loop guide: V2_performance/tools/
