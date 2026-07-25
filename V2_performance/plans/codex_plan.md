# V2 GPU Optimization Plan (Prioritized, Quantified, Execution-Ready)

## 0) Baseline Facts Driving Priority

- V2 has a tier-inverted profile: catastrophic at low rank (`frame_h` 28.3x slower, `circuit_d3_p0.001` 15.2x slower), modestly slower in coop rank 7-10 (1.31-1.65x), and faster in global rank 11+ (0.91-0.97x). (V2_performance/analysis/SYNTHESIS.md:10, V2_performance/analysis/SYNTHESIS.md:11, V2_performance/analysis/SYNTHESIS.md:12, V2_performance/analysis/SYNTHESIS.md:13, V2_performance/analysis/SYNTHESIS.md:16)
- Coop V2 is not instruction-bound versus SVM on qv10: V2 has fewer VALU/LDS instructions but higher wave-cycles and massive L2 misses. (V2_performance/analysis/SYNTHESIS.md:25, V2_performance/analysis/SYNTHESIS.md:26, V2_performance/analysis/SYNTHESIS.md:28)
- Coop LDS footprint is 25,088 B and occupancy-caps at 2 wg/CU; barrier density is ~52 static sites; tid0 serialization is ~34 regions. (V2_performance/gpu/gpu_kernel_static_characterization.md:8, V2_performance/gpu/gpu_kernel_static_characterization.md:23, V2_performance/gpu/gpu_kernel_static_characterization.md:20, V2_performance/gpu/gpu_kernel_static_characterization.md:78)
- Global V2 wins with 8,704 B LDS, reinforcing occupancy as the dominant structural lever. (V2_performance/analysis/SYNTHESIS.md:38, V2_performance/analysis/SYNTHESIS.md:39, V2_performance/gpu/surface_d9_t19.json:5, V2_performance/gpu/surface_d9_t19.json:71)

## 1) Ranking Method

Priority score uses:

`score = expected_speedup_on_affected_tier * fraction_of_suite_cases_affected * implementation_confidence`

Where:
- `fraction_of_suite_cases_affected` uses current 8-circuit matrix in the perf DB (low rank 2/8, coop 3/8, global 3/8). (V2_performance/analysis/gpu_profile_summary.md:5, V2_performance/analysis/gpu_profile_summary.md:12)
- `implementation_confidence` is inverse-risk proxy (1.0 = lowest risk).

I also include expected full-suite time gain (from current kernel-time table) as a secondary sanity check. (V2_performance/analysis/gpu_profile_summary.md:5, V2_performance/analysis/gpu_profile_summary.md:12)

## 2) Prioritized Roadmap

| Rank | Optimization | Expected speedup (tier-local) | Suite cases affected | Confidence | Score | Expected full-suite gain* |
|---|---|---:|---:|---:|---:|---:|
| P0 | Add low-rank packed/register tier + routing (rank<=4 first, extend to <=6) | 8x-16x (mid 12x) | 25% (2/8) | 0.60 | **1.80** | ~2.6% now; much larger on low-rank-heavy workloads |
| P1 | Coop LDS reclamation to cross <16 KB and unlock 4 wg/CU | 1.30x-1.60x (mid 1.45x) | 37.5% (3/8) | 0.80 | **0.44** | ~14% |
| P2 | Coop barrier + tid0-serialization reduction | 1.15x-1.35x (mid 1.25x) | 37.5% (3/8) | 0.70 | **0.33** | ~9% |
| P3 | U4 live-range/AGPR pressure reduction | 1.05x-1.20x (mid 1.12x) | 37.5% (3/8, qv-like compute-heavy) | 0.75 | **0.32** | ~4% |
| P4 | Arithmetic relaxations (selective FMA/f32) behind strict canary gates | 1.10x-1.40x (mid 1.18x) | 37.5% | 0.35 | **0.15** | ~6% if safe; high rollback risk |
| P5 | Global-tier micro-tuning only (guard current wins) | 1.00x-1.08x | 37.5% | 0.85 | **0.03** | small; avoid regressions |

`*` Full-suite gain estimates are computed from current V2 per-circuit kernel-time shares in `gpu_profile_summary`.

Why P0 before P1 despite lower immediate suite-time share: low-rank regression factor (15-28x) is the largest qualitative failure mode and blocks parity claims for register-tier workloads. (V2_performance/analysis/SYNTHESIS.md:10, V2_performance/analysis/SYNTHESIS.md:11, V2_performance/analysis/SYNTHESIS.md:56)

## 3) P0 Design: Low-Rank Catastrophe Fix (Concrete)

### Decision

Implement a **separate V2 register-tier packed kernel** and **route by rank** in `v2_sample`.

- Start with `rank<=4` (matches existing SVM thread-tier boundary). (src/clifft/gpu/gpu_types.h:8)
- Extend to `rank<=6` only if scratch/register behavior is acceptable after profiling.

### Concrete kernel/routing design

1. Add `clifft_v2_thread` kernel (plain C -> amdgcn, no HIP) with **1 shot/thread**.
2. Keep `block=256`, `grid=ceil(shots/256)*256`; each lane owns full shot state, no inter-thread barriers in shot execution.
3. Keep existing `clifft_v2_coop` for rank 5-10 and `clifft_v2_global` for 11-19.
4. Update host routing in `v2_kernel.cc` currently binary coop/global split (`kCoopMaxRank=10`, `use_global`). (src/clifft/gpu/mlir/v2/v2_kernel.cc:50, src/clifft/gpu/mlir/v2/v2_kernel.cc:52, src/clifft/gpu/mlir/v2/v2_kernel.cc:55)

### Why this is the right fix

- Current coop model is structurally mismatched at low rank: still 256 threads/shot while active amplitudes are tiny and frame/classical work is tid0-only. (V2_performance/analysis/SYNTHESIS.md:50, V2_performance/analysis/SYNTHESIS.md:51, V2_performance/analysis/SYNTHESIS.md:45)
- Existing SVM already demonstrates the right topology: thread-tier exists and is used for low rank (`kThreadMaxPeakRank=4`). (src/clifft/gpu/gpu_types.h:8, src/clifft/gpu/sampler/hip_sampler.hip:2048, src/clifft/gpu/sampler/hip_sampler.hip:1778)

## 4) P1/P2 Coop Tier: Exact LDS + Barrier Changes in `coop_interpreter.c`

### P1-A: LDS reclamation (highest coop lever)

Current LDS heavy allocations:
- `lds_red_scratch[1024]` (8 KB), `lds_meas[4096]` (4 KB), `lds_red0/1[256]` (4 KB total). (src/clifft/gpu/mlir/v2/coop_interpreter.c:45, src/clifft/gpu/mlir/v2/coop_interpreter.c:46, src/clifft/gpu/mlir/v2/coop_interpreter.c:55, src/clifft/gpu/mlir/v2/coop_interpreter.c:56)

Concrete changes:

1. **Shrink reduction partial buffers**: `lds_red0[256], lds_red1[256] -> [4]`.
- Justification: `coop_reduce2` only indexes warp IDs 0..3 and `t<4`. (src/clifft/gpu/mlir/v2/coop_interpreter.c:189, src/clifft/gpu/mlir/v2/coop_interpreter.c:194, src/clifft/gpu/mlir/v2/coop_interpreter.c:196)

2. **Shrink scratch to true max need**: `lds_red_scratch[1024] -> [512]` for coop rank<=10.
- Justification: `OP_SWAP_MEAS_INTERFERE` scratch index is `< half`, and max `half=2^(10-1)=512`. (src/clifft/gpu/mlir/v2/coop_interpreter.c:706, src/clifft/gpu/mlir/v2/coop_interpreter.c:709)

3. **Pack measurement bits**: replace byte array `lds_meas[4096]` with bitset (`u64[64]` for 4096 bits).
- All measurement writes/reads are boolean semantics; convert all `lds_meas[idx]` load/store sites to bit accessors.
- Representative touch points: writes at dormant/active/swap/readout paths, reads in apply_pauli/observable/postselect. (src/clifft/gpu/mlir/v2/coop_interpreter.c:355, src/clifft/gpu/mlir/v2/coop_interpreter.c:410, src/clifft/gpu/mlir/v2/coop_interpreter.c:444, src/clifft/gpu/mlir/v2/coop_interpreter.c:698, src/clifft/gpu/mlir/v2/coop_interpreter.c:720, src/clifft/gpu/mlir/v2/coop_interpreter.c:764, src/clifft/gpu/mlir/v2/coop_interpreter.c:781)

Expected LDS result: ~25 KB -> ~13 KB class, enabling 4 wg/CU (target occupancy doubling), consistent with static occupancy analysis threshold (<16 KB). (V2_performance/gpu/gpu_kernel_static_characterization.md:151, V2_performance/gpu/gpu_kernel_static_characterization.md:158)

### P2-A: Barrier reduction (safe-first)

1. **EXPAND family remove redundant second barrier**:
- Targets: `OP_EXPAND`, `OP_EXPAND_T`, `OP_EXPAND_ROT` second barriers around `lds_active_k += 1`. (src/clifft/gpu/mlir/v2/coop_interpreter.c:375, src/clifft/gpu/mlir/v2/coop_interpreter.c:377, src/clifft/gpu/mlir/v2/coop_interpreter.c:389, src/clifft/gpu/mlir/v2/coop_interpreter.c:391, src/clifft/gpu/mlir/v2/coop_interpreter.c:593, src/clifft/gpu/mlir/v2/coop_interpreter.c:595)

2. **Measurement handlers drop redundant pre-update barriers**:
- `MEAS_ACTIVE_DIAGONAL`: remove barrier at 417; keep synchronization at 413 and 424. (src/clifft/gpu/mlir/v2/coop_interpreter.c:413, src/clifft/gpu/mlir/v2/coop_interpreter.c:417, src/clifft/gpu/mlir/v2/coop_interpreter.c:424)
- `MEAS_ACTIVE_INTERFERE`: remove barrier at 452; keep 446 and 459. (src/clifft/gpu/mlir/v2/coop_interpreter.c:446, src/clifft/gpu/mlir/v2/coop_interpreter.c:452, src/clifft/gpu/mlir/v2/coop_interpreter.c:459)
- `SWAP_MEAS_INTERFERE` degenerate/non-degenerate: remove barrier at 674 and 710; keep stage boundaries at 669/680 and 700/716. (src/clifft/gpu/mlir/v2/coop_interpreter.c:669, src/clifft/gpu/mlir/v2/coop_interpreter.c:674, src/clifft/gpu/mlir/v2/coop_interpreter.c:680, src/clifft/gpu/mlir/v2/coop_interpreter.c:700, src/clifft/gpu/mlir/v2/coop_interpreter.c:710, src/clifft/gpu/mlir/v2/coop_interpreter.c:716)

3. **`coop_reduce2` from 3 barriers -> 2** by removing trailing broadcast barrier when caller immediately enforces the next sync point.
- Current barriers in primitive: 195, 204, 206. (src/clifft/gpu/mlir/v2/coop_interpreter.c:195, src/clifft/gpu/mlir/v2/coop_interpreter.c:204, src/clifft/gpu/mlir/v2/coop_interpreter.c:206)

### P2-B: tid0 serialization reduction (frame-heavy paths)

Batch consecutive frame-only opcodes in interpreter loop so a run of `OP_FRAME_*` pays one synchronization boundary instead of one per opcode.

- Interpreter loop anchor: runtime switch loop. (src/clifft/gpu/mlir/v2/coop_interpreter.c:301, src/clifft/gpu/mlir/v2/coop_interpreter.c:303)
- Frame-op cases currently each pay an immediate barrier. (src/clifft/gpu/mlir/v2/coop_interpreter.c:304, src/clifft/gpu/mlir/v2/coop_interpreter.c:311, src/clifft/gpu/mlir/v2/coop_interpreter.c:343)

## 5) Correctness Invariant Impact (Explicit)

## Must-preserve invariants

- `cscale` f64 multiply then narrow behavior for branch-consistent semantics. (src/clifft/gpu/mlir/v2/coop_interpreter.c:122, src/clifft/gpu/mlir/v2/coop_interpreter.c:125)
- `coop_reduce2` summation/shuffle order (ds_bpermute-based) on branch decisions. (src/clifft/gpu/mlir/v2/coop_interpreter.c:182, src/clifft/gpu/mlir/v2/coop_interpreter.c:184, src/clifft/gpu/mlir/v2/coop_interpreter.c:190)
- compile with `-ffp-contract=off` on default path. (cmake/ClifftAmdgcn.cmake:60, cmake/ClifftAmdgcn.cmake:87)
- deterministic RNG formula `(rng() >> 11) * 0x1.0p-53`. (src/clifft/gpu/mlir/v2/coop_interpreter.c:86, src/clifft/gpu/mlir/v2/coop_interpreter.c:87)

### Optimizations and invariant risk

- **P0 low-rank packed kernel**: preserves invariant if it reuses same RNG formula and scalar arithmetic semantics; no cooperative reduction-order dependency in thread mode. Risk: RNG draw-order bugs if opcode flow diverges.
- **P1 LDS reclamation**: preserves invariant; storage/layout change only.
- **P2 barrier + tid0 serialization reductions**: preserves invariant if arithmetic order in branch-probability paths is unchanged.
- **P3 U4 live-range split**: medium risk; algebraic reassociation can change rounding if expression tree changes.
- **P4 selective FMA/f32 relaxations**: **high risk** to byte-exactness; keep behind opt-in experiment flag and never default until exhaustive exact-match validation passes.

## 6) KernelForge Recommendation: Manual First, Then Agentic Loop

Recommendation: **manual P0-P2 first**, wire KernelForge after structural bottlenecks are removed.

Why:
- Current bottlenecks are explicit and line-local (LDS over-allocation, barrier density, routing mismatch), so manual edits have high signal and low search requirement. (V2_performance/analysis/SYNTHESIS.md:35, V2_performance/analysis/SYNTHESIS.md:42, V2_performance/analysis/SYNTHESIS.md:56)
- KernelForge integration is feasible but requires driver shim/build wiring and SLURM orchestration before useful iteration. (V2_performance/tools/agentic_integration_guide.md:113, V2_performance/tools/agentic_integration_guide.md:121, V2_performance/tools/agentic_integration_guide.md:151, V2_performance/tools/agentic_integration_guide.md:155)

Execution stance:
- Do manual P0-P2 until ratios are near parity (`low-rank <=~2x`, `coop <=~1.1x`).
- Then use KernelForge for P3/P4 micro-variants and auto keep/revert with strict exact-match gate. (V2_performance/tools/agentic_integration_guide.md:38, V2_performance/tools/agentic_integration_guide.md:42, V2_performance/tools/agentic_integration_guide.md:72)

## 7) Measurement Plan (Per Optimization)

Use existing sweep harness and counter sets as canonical measurement protocol. (V2_performance/tools/profile_sweep.sh:26, V2_performance/tools/profile_sweep.sh:30, V2_performance/tools/profile_sweep.sh:45)

Common gates for every step:
1. Correctness gate (exact integer match): `passed_shots` + `observable_ones` vs SVM, seed-fixed, plus determinism rerun same seed.
2. Performance gate: kernel-trace medians and V2/SVM ratio.
3. Regression guard: no >3% slowdown on already-winning global circuits.

### P0 validation (low-rank kernel+routing)

Circuits:
- `frame_h`, `circuit_d3_p0.001` (primary), `qv10` (routing guard). (V2_performance/analysis/gpu_profile_summary.md:5, V2_performance/analysis/gpu_profile_summary.md:7)

Counters/metadata:
- `LDS_Block_Size`, `SQ_WAVE_CYCLES`, `SQ_INSTS_LDS`, `SQ_INSTS_VALU`, `TCC_MISS_sum`.

Success thresholds:
- `frame_h` ratio: 28.3x -> <=2.0x first milestone.
- `circuit_d3_p0.001` ratio: 15.2x -> <=2.5x first milestone.
- No regression >5% on rank>=7 circuits.

### P1 validation (LDS reclamation)

Circuits:
- `qv10`, `surface_d7_t15`, `surface_d9_t10` (coop core), plus one global guard (`surface_d9_t19`). (V2_performance/analysis/gpu_profile_summary.md:7, V2_performance/analysis/gpu_profile_summary.md:9, V2_performance/analysis/gpu_profile_summary.md:11, V2_performance/analysis/gpu_profile_summary.md:12)

Counters/metadata:
- `LDS_Block_Size` target <=14 KB, `SQ_WAVE_CYCLES/SQ_WAVES`, `SQ_WAIT_INST_LDS`, `TCC_MISS_sum`.

Success thresholds:
- Coop tier median speedup >=1.20x.
- `LDS_Block_Size` materially reduced from 25,088 B baseline.

### P2 validation (barriers + tid0)

Circuits:
- `frame_h` (frame-heavy), `qv10`, `surface_d7_t15`, `surface_d9_t10`.

Counters:
- `SQ_WAVE_CYCLES/SQ_WAVES` primary, `SQ_BUSY_CYCLES`, `SQ_WAIT_INST_LDS`, static barrier recount from disassembly/source audit.

Success thresholds:
- >=15% drop in `SQ_WAVE_CYCLES/SQ_WAVES` on coop set.
- No correctness drift on measurement-heavy cases.

### P3 validation (U4 pressure)

Circuits:
- `qv10` primary, `surface_d7_t15` secondary.

Checks:
- Kernel-time improvement with stable `VGPR_Count` (<=64 target), no scratch blow-up.
- Optional register metadata extraction from `.hsaco` each iteration.

### P4 validation (arithmetic relax experiments)

Circuits:
- `qv10`, `surface_d7_t15`, `surface_d9_t10`, `surface_d9_t19`.

Protocol:
- Multi-seed exact-match matrix (e.g., seeds 1, 2, 17, 101).
- If any mismatch vs SVM tallies: auto-reject variant.
- Only keep if exact-match pass and >=8% speedup on at least two coop benchmarks.

## 8) Execution Order (Concrete)

1. Implement P0 rank routing + thread-tier kernel skeleton; validate low-rank collapse immediately.
2. Implement P1 LDS compaction trio (`red[4]`, `scratch[512]`, `meas bitset`); reprofile coop.
3. Implement P2 safe barrier trims + frame-run batching; reprofile coop/frame-heavy.
4. Apply P3 U4 live-range split if coop still >1.1x.
5. Start KernelForge loop only after steps 1-4 are stable; scope it to P3/P4 variant search with strict exact-match gate.

This ordering minimizes risk while attacking both the catastrophic low-rank outlier and the dominant coop-time deficit.
