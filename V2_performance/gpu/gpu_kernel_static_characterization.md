# V2 GPU Kernel — Static Characterization

**Kernel:** `clifft_v2_coop` / `clifft_v2_global` (shared body `execute_shot`)
**Source:** `src/clifft/gpu/mlir/v2/coop_interpreter.c` (810 lines, plain C → amdgcn)
**Host driver:** `src/clifft/gpu/mlir/v2/v2_kernel.cc`
**ABI:** `src/clifft/gpu/mlir/v2/device_abi.h`
**Build:** `cmake/ClifftAmdgcn.cmake` (clang → llc → ld.lld, `-O2 -ffp-contract=off`, `+wavefrontsize64`, +ocml)
**Profiling baseline (rocprofv3, qv10 @ 20k shots, coop):** VGPR=64, SGPR=112, LDS=25088 B (~24.5 KB), Scratch=416 B, SQ_INSTS_MFMA=0, SQ_INSTS_VALU=9.64e8, SQ_WAVES=80000.

Model type: **runtime bytecode interpreter** — a `for (pc...) switch(opcode)` loop (line 301), deliberately *not* unrolled. One workgroup = 256 threads = **4 waves of 64 lanes** (wavefrontsize64) cooperates on **one shot**; the statevector lives in LDS (coop, rank≤10, 1024 amps) or HBM (global, rank 11-19). Both tiers call the same `execute_shot()`.

---

## 1. Thread / Cooperation Model

The 256 threads follow a **SIMT cooperative-strided** pattern: amplitude loops all use `for (i = t; i < iters; i += 256u)` so each thread owns a strided slice of the statevector. Classical bookkeeping (Pauli frame `lds_px/lds_pz`, RNG, measurement records, discard flag, `active_k`) is **single-owner: `t == 0` mutates, all others idle at a barrier**, then read the broadcast through LDS.

### `if (t == 0)` serial sections (thread 0 only, 255 idle)
Init (287), FRAME_CNOT (305), FRAME_CZ (314), FRAME_H (323), FRAME_S/S_DAG (333), FRAME_SWAP (337), MEAS_DORMANT_STATIC (347), MEAS_DORMANT_RANDOM (362), EXPAND `active_k++` (376), EXPAND_T `active_k++` (390), MEAS_ACTIVE_DIAGONAL sample (406) + post-compaction frame (418), MEAS_ACTIVE_INTERFERE sample (439) + frame (453), ARRAY_CNOT frame (472), ARRAY_CZ frame (488), ARRAY_SWAP frame (505), ARRAY_MULTI_CNOT frame (521), ARRAY_MULTI_CZ frame (541), ARRAY_H frame (560), ARRAY_S frame (573), EXPAND_ROT `active_k++` (594), ARRAY_U2 out-state (613), ARRAY_U4 out-state (644), SWAP_MEAS_INTERFERE ×3 (664,675,683,695,711), APPLY_PAULI (720), NOISE (729), NOISE_BLOCK (737), OBSERVABLE (760), READOUT_NOISE (770), POSTSELECT (777), final aggregation (800), global work-steal (882).
**Count: ~34 distinct `if (t == 0)` serial regions.**

### `barrier()` count (`__builtin_amdgcn_s_barrier`)
**~52 static `barrier()` sites.** By category:
- Init: 2 (286, 298)
- Frame ops (CNOT/CZ/H/S/SWAP): 1 barrier each = 5
- Dormant meas (STATIC/RANDOM): 1 each = 2
- EXPAND / EXPAND_T / EXPAND_ROT: **2 barriers each** (post-copy + post-`active_k++`) = 6
- MEAS_ACTIVE_DIAGONAL: **~6** (3 inside `coop_reduce2` at 195/204/206, +413, +417, +424)
- MEAS_ACTIVE_INTERFERE: **~6** (3 in coop_reduce2 +446 +452 +459)
- SWAP_MEAS_INTERFERE: **~8-9** (two reduce2 calls = 6, plus 669/674/680 or 700/708/710/716)
- ARRAY_CNOT/CZ/SWAP/MULTI_CNOT/MULTI_CZ/H/U2/U4: **2 barriers each** (post-amp + post-frame) = 16
- ARRAY_S/S_DAG, ARRAY_ROT, ARRAY_T: 1 barrier via `coop_apply_phase` (164) + frame barrier
- APPLY_PAULI/NOISE/NOISE_BLOCK/OBSERVABLE/READOUT_NOISE/DETECTOR: 1 each = 6
- POSTSELECT: 1 (784) + early-return guard
- Final: 1 (799)

**Barrier cost is dominant** because every opcode ends with ≥1 `s_barrier`, and *measurement* opcodes each pay 6-9 barriers (the `coop_reduce2` alone costs 3). A typical qv10 circuit issues hundreds of ops → thousands of dynamic barriers, each forcing all 4 waves to converge.

### Fully cooperative opcodes (all 256 threads compute)
EXPAND (374), EXPAND_T (388), EXPAND_ROT (592), ARRAY_CNOT (466), ARRAY_CZ (483), ARRAY_SWAP (499), ARRAY_MULTI_CNOT (513), ARRAY_MULTI_CZ (534), ARRAY_H (553), ARRAY_S/ROT/T via `coop_apply_phase` (160), ARRAY_U2 (605), ARRAY_U4 (629), plus the **reduction phase** of MEAS_ACTIVE_DIAGONAL/INTERFERE/SWAP_MEAS (the `cnorm` accumulation loops 400/432/659/688 and the fold-back loops 415/447/702/709). Init zeroing (285) is cooperative.

### tid0-only opcodes (255 threads idle at barrier)
**All FRAME_* ops** (305-343) — pure Pauli-frame bit twiddling, inherently scalar.
**Dormant measurements** (MEAS_DORMANT_STATIC/RANDOM, 347/362) — outcome from a single frame bit / one RNG draw.
**Measurement *sampling*** (the `sample_branch` + frame update *after* the cooperative reduction: 406, 439, 664, 695).
**All classical post-processing**: APPLY_PAULI, NOISE, NOISE_BLOCK, OBSERVABLE, READOUT_NOISE, POSTSELECT, DETECTOR, final aggregation, and the global work-steal atomic.

**Serialization hazard:** on frame-heavy / measurement-heavy circuits (QEC, surface codes), a large fraction of wall-clock is spent with 255/256 threads (99.6%) stalled at a barrier while `t==0` walks a frame or samples a branch.

---

## 2. LDS Budget

Every `extern __attribute__((address_space(3)))` global (lines 44-58) and its byte size:

| Global | Type × count | Bytes |
|---|---|---|
| `lds_v[1024]` | CV2Complex(8) × 1024 | **8192** |
| `lds_red_scratch[1024]` | CV2Complex(8) × 1024 | **8192** |
| `lds_meas[4096]` | u8 × 4096 | **4096** |
| `lds_red0[256]` | double × 256 | **2048** |
| `lds_red1[256]` | double × 256 | **2048** |
| `lds_px[5]` | u64 × 5 | 40 |
| `lds_pz[5]` | u64 × 5 | 40 |
| `lds_rng[4]` | u64 × 4 | 32 |
| `lds_obs[8]` | u8 × 8 | 8 |
| `lds_shot` | u64 | 8 |
| `lds_active_k` | u32 | 4 |
| `lds_next_noise` | u32 | 4 |
| `lds_xcd` | u32 | 4 |
| `lds_discarded` | u8 | 1 |
| `lds_branch` | u8 | 1 |
| **Sum (packed)** | | **24718 B** |

Rounded to LDS alloc granularity this matches the profiled **25088 B (~24.5 KB)**. The **big four are `lds_v` (8 KB) + `lds_red_scratch` (8 KB) + `lds_meas` (4 KB) + `lds_red0/1` (4 KB) = 24 KB (97 %)** of the budget.

### Occupancy implication (CDNA4, 64 KB LDS/CU)
`floor(64 KB / 24.5 KB) = 2` workgroups can co-reside per CU **from LDS alone**. With 4 waves/wg → **8 waves/CU** resident. This is the binding occupancy constraint (see §6).

### `lds_meas[4096]` is oversized
`V2_MAX_MEAS = 4096` (line 43) is a worst-case cap. Typical coop circuits (qv10) have tens–low-hundreds of measurement slots. `total_meas_slots` is a *known runtime kernarg* (line 293 only zeroes `i < total_meas_slots`), yet the LDS reservation is static at 4096 B. **Shrinking or dynamically sizing `lds_meas` is the single cheapest occupancy win** (see §7a). Likewise `lds_red_scratch` (8 KB) is only used by SWAP_MEAS_INTERFERE's strided fold (706-709); most coop circuits never touch it, yet it permanently costs 8 KB of the LDS budget.

---

## 3. Per-Opcode Work + Barrier Count (hot opcodes)

**EXPAND (371-379):** strided copy `v[i+half]=v[i]` over `half=2^active_k` (line 374), then `active_k++`. Contiguous LDS read/write (no scatter). f32 only. **2 barriers.**

**EXPAND_T / EXPAND_T_DAG (380-393):** same copy but multiplied by a fixed `(1/√2, ±1/√2)` phase via `cmul` (line 388) → 1 complex mul (4 f32 mul + 2 add, no FMA). Contiguous. **2 barriers.**

**ARRAY_CNOT (462-478):** gather/scatter via `scatter_bits_2(i, c, tg)` (two `insert_zero_bit`, line 467) → non-contiguous LDS. Swaps `v[base]`/`v[base|t_bit]`. `iters = 2^(active_k-2)`. No arithmetic (just a swap). tid0 frame update. **2 barriers.**

**ARRAY_CZ (479-494):** `scatter_bits_2` gather, negate one amplitude. **2 barriers.**

**ARRAY_H (550-566):** `scatter_bits_1` gather (line 554); butterfly `(a±b)/√2` via `cadd/csub` + **`cscale` (f64 scalar mul!)**. `iters=2^(active_k-1)`. **2 barriers.**

**ARRAY_U2 (598-620):** `scatter_bits_1` gather, applies a 2×2 complex matrix selected by the Pauli-frame in-state (`fused_u2[a].matrices[in_state]`). Inner: **4 `cmul` + 2 `cadd`** per iteration (lines 608-609). `iters=2^(active_k-1)`. **2 barriers.**

**ARRAY_U4 (621-651):** `scatter_bits_2` gather of a 4-tuple `v0..v3`, applies a 4×4 complex matrix (`fused_u4[a].entries[in_state].matrix`). Inner: **16 `cmul` + 12 `cadd`** per iteration (lines 633-640) — the heaviest arithmetic body. `iters=2^(active_k-2)`. **2 barriers.** The 16 live matrix elements + 4 live amplitudes drive register pressure (§5).

**coop_apply_phase (157-165):** used by ARRAY_S/S_DAG/ROT/T and EXPAND paths. Strided `v[insert_zero_bit(i,axis)|axis_bit] *= phase`, `iters=2^(active_k-1)`. 1 `cmul`. **1 barrier** (line 164).

**MEAS_ACTIVE_DIAGONAL (394-426):** two-pass. Pass 1: cooperative `cnorm` accumulation of `l0,l1` (f64) over `half=2^(active_k-1)` (400-403). Then **`coop_reduce2` (3 barriers)**. tid0 `sample_branch`. Conditional compaction copy `v[i]=v[i+half]` (415). Frame update. **~6 barriers total** (3 in reduce2 + 413 + 417 + 424).

**MEAS_ACTIVE_INTERFERE (427-461):** Pass 1: cooperative `cnorm(cadd)`/`cnorm(csub)` (f64) → `coop_reduce2` (3 barriers). tid0 sample. Pass 2: fold `(v[i]±v[i+half])/√2` via `cscale` (447-451). **~6 barriers.**

**coop_reduce2 (187-207):** the reduction primitive. Per call: intra-wave butterfly `off=32→1` (6 `shfl_xor_f64` steps, each = **2 `ds_bpermute`** for the hi/lo halves of the f64, lines 190-193) = 12 `ds_bpermute`; write partials to `lds_red0/1[warp]`; **barrier (195)**; 4-warp butterfly `off=2→1` (2 steps × 2 bpermute = 4 bpermute, only `t<4`); tid0 store; **barrier (204)**; broadcast; **barrier (206)**. **3 barriers + 16 `ds_bpermute` per reduce2 call**, and it is called once per active measurement (twice in SWAP_MEAS). This is the hottest LDS-traffic primitive.

---

## 4. Numerical / Precision Cost

The kernel is compiled `-ffp-contract=off` (cmake lines 60/87), which **forbids FMA fusion**. Every `cmul` (110-115) is 4 separate `v_mul_f32` + 2 `v_add_f32` instead of 2 mul + 2 `v_fma_f32`. For the multiply-add chains this is roughly **~1.5-2× the VALU instruction count**:
- `cmul` ×4 in ARRAY_U2 → 16 mul + 8 add (vs 8 fma-fused);
- ARRAY_U4's 16 `cmul` + 12 `cadd` → **64 mul + 32 add + 24 add**; with FMA this collapses to ~32 fma + tail. The observed **SQ_INSTS_VALU=9.64e8** is inflated by exactly this.

**f64 usage on CDNA4** (where f64 throughput is a fraction of f32):
- `cnorm` (132-135): each amplitude promoted to f64, squared, summed — the reduction pre-pass runs entirely in f64.
- `coop_reduce2` (187): all butterfly adds are f64; `ds_bpermute` moves 8-byte doubles as two 32-bit shuffles (line 174-175) → 2× the shuffle traffic of an f32 reduction.
- `cscale` (125-130): **f64 scalar multiply then narrow to f32** — an f64 mul per component for what is arithmetically an f32 scale, used in every ARRAY_H and every INTERFERE fold.
- `sample_branch`, `draw_next_noise`, `apply_noise_site`, RNG: f64 (correct, low volume, tid0-only).

**Where f64 is genuinely required:** the **measurement branch decision** must be byte-exact with the SVM/CPU gold path (same seed → same `passed_shots`). The f64 accumulation order in `coop_reduce2` and the f64→f32 narrowing in `cscale` are documented (lines 122-124, 182-186) as *deterministic-branch* requirements: relaxing them desyncs branch sampling on reduction-heavy rank-10 QEC circuits.

**Where f64 could be relaxed (with care):** `cnorm`/`cscale` in *non-measurement* paths (e.g. ARRAY_H's `cscale`, which never feeds a branch decision) could use f32 — the fold result is a state amplitude, not a probability compared against an RNG draw. The reduction feeding `sample_branch` must stay f64, but the *state renormalization* after the branch is chosen need not. This is a targeted, correctness-gated relaxation (§7d).

---

## 5. The AGPR Mystery (`v_accvgpr_write/read` with MFMA=0)

Disassembly shows `v_accvgpr_write` / `v_accvgpr_read` traffic even though **SQ_INSTS_MFMA=0** (no matrix engine use). On CDNA, when a kernel is *not* using MFMA, the **AGPR file can be repurposed by the register allocator as spill/overflow storage** — moving a value to an AGPR (`v_accvgpr_write`) is cheaper than spilling to scratch memory. Seeing AGPR traffic with MFMA=0 is the classic signature of **register pressure being relieved into AGPRs rather than scratch**.

**Most likely source: the ARRAY_U4 handler (621-651).** In its inner loop it holds simultaneously live:
- 4 complex amplitudes `v0,v1,v2,v3` (8 f32 = 8 VGPR),
- a pointer walk over `mat[4][4]` = **16 complex matrix elements** (up to 32 f32),
- 4 complex outputs accumulated via nested `cadd(cadd(cmul...),cadd(cmul...))` (633-640), which with `-ffp-contract=off` produces long, non-collapsible temporary chains.

That is a very large live-range peak inside one basic block. With the VGPR budget held to **64** (the profiled ceiling, chosen to keep occupancy at 8 waves/SIMD), the allocator has to park part of the U4 live set somewhere; it uses **AGPRs as extra registers** (416 B scratch is small, consistent with AGPRs absorbing most of the overflow instead of memory). The `CV2FusedU4Entry`/`fused_u4` struct's 16-entry matrix table (device_abi.h 57-64) is read through a pointer, so the elements materialize into registers on demand.

**Secondary contributor:** `coop_reduce2`'s f64 butterflies keep 4 doubles (`l0,l1` + their shuffled partners) = 8 VGPR live across 16 `ds_bpermute` + 3 barriers; f64 pairs also pressure the allocator.

**Hypothesis (to confirm with `--save-temps` / disassembly):** AGPR traffic concentrates in the U4 basic block. If confirmed, splitting the U4 matrix-apply into two half-passes (rows 0-1, then 2-3) or forcing the matrix load through LDS would shrink the live-range and drop the AGPR shuffles.

---

## 6. Occupancy Analysis

CDNA4 per-CU limits (per SIMD ×4): 512 VGPR/SIMD (256 addressable/wave shown as ceiling), 64 KB LDS/CU, 8 waves/SIMD max (10 on some parts), 16 waves/CU-ish typical.

- **VGPR = 64** → 256/64 ≈ *up to ~8 waves/SIMD* allowed by registers. **Not the limiter.**
- **LDS = 24.5 KB** → `floor(64 KB / 24.5 KB) = 2` workgroups/CU. **This is the limiter.**
- 256 threads/wg ÷ 64 = **4 waves/wg**; 2 wg/CU → **8 waves/CU resident** (2 waves/SIMD).

**The kernel is LDS-occupancy-bound, not register-bound.** VGPR headroom (64 of ~256) is wasted because LDS caps residency first.

**Is 8 waves/CU enough to hide latency?** Only 2 waves/SIMD is *marginal* — it is enough to hide short ALU/LDS latencies but **not enough to hide the long, frequent `s_barrier` stalls** or HBM latency in the global tier. Because nearly every opcode ends in a barrier, the 2 resident waves per SIMD spend much of their time co-stalled rather than overlapping. Latency hiding here is poor: the barrier-per-op structure means there is little independent work for the scheduler to interleave.

**Theoretical ceiling & what limits it:** if LDS dropped below **16 KB**, 4 wg/CU (16 waves/CU) would fit — the register file (VGPR=64) already supports that. So **LDS is the sole gate to doubling occupancy**, and `lds_v`+`lds_red_scratch`+`lds_meas` are where the 24 KB lives.

---

## 7. Top Optimization Opportunities (ranked)

**(a) Dynamically size / shrink `lds_meas` and `lds_red_scratch` → raise occupancy [HIGH].**
`lds_meas[4096]` (4 KB) is a worst-case cap; typical coop circuits use a fraction (`total_meas_slots` is known at dispatch, line 293). `lds_red_scratch[1024]` (8 KB) is used **only** by SWAP_MEAS_INTERFERE's strided fold (706-709) — most circuits never touch it. Trimming these two from 12 KB to ~1-2 KB drops LDS from 24.5 KB to ~13-14 KB → **2 wg/CU → 4 wg/CU (2× occupancy)**. Expected **~1.5-2× throughput** on latency-bound coop workloads. Cheapest, highest-leverage change; requires LDS layout / dynamic-LDS plumbing in the host dispatch (kernarg-driven LDS size) but no algorithm change.

**(b) Reduce tid0-serial frame/measurement sections [HIGH].**
~34 `if (t==0)` regions run with 255 threads idle at a barrier. FRAME_* ops and dormant measurements are the worst (pure scalar work stalling 4 waves). Options: batch consecutive frame ops so one barrier covers many; let lane 0 of *each wave* do independent frame words in parallel; or push frame-only op runs onto a separate cheap path. On frame-heavy QEC circuits this is a large fraction of wall time. Expected **1.2-1.6×** on frame/measurement-dense circuits.

**(c) Fuse the two-pass measurement into fewer barriers [MEDIUM-HIGH].**
MEAS_ACTIVE_DIAGONAL/INTERFERE each do reduce (6 barriers incl. reduce2) → sample → fold (2 more). The reduction and the fold both stream the same `v[i]/v[i+half]` pairs. The fold can't precede the branch decision, but the **reduce and the branch-independent part of the fold** can share a pass, and `coop_reduce2`'s 3 barriers can be cut to ~2 by broadcasting through a single LDS slot without the trailing re-read barrier (line 206). Fewer barriers per measurement × hundreds of measurements. Expected **1.1-1.3×** on measurement-heavy circuits.

**(d) f64 → f32 in branch-independent paths [MEDIUM].**
Keep f64 in the reduction that feeds `sample_branch` (determinism, lines 182-186) but relax `cscale` (125) and the fold arithmetic in ARRAY_H (556-557) and post-branch INTERFERE fold (450) to f32 where the result is a state amplitude, not a probability. f64 is markedly slower on CDNA4. Must be gated by correctness (same-seed passed_shots). Expected **1.1-1.25×**; risk: branch desync if applied too broadly — verify against SVM per circuit.

**(e) Restore FMA where it doesn't change the branch [MEDIUM].**
`-ffp-contract=off` doubles the multiply-add VALU count in `cmul`/U2/U4 and is the main driver of SQ_INSTS_VALU=9.64e8. Selectively enabling `-ffp-contract=fast` (or `#pragma fp_contract on`) on the *pure-transform* opcodes (ARRAY_U2/U4/H amplitude math) while keeping it off for the reduction path could nearly halve the VALU in the U4 body. Must confirm it doesn't perturb the measurement branch (it operates on state, not probabilities). Expected **1.2-1.4×** on U-gate-heavy circuits (qv10 is U-heavy).

**(f) Shrink ARRAY_U4 register/AGPR pressure [MEDIUM].**
The 16-element matrix + 4 amplitudes + non-collapsible add chains overflow to AGPRs (§5). Splitting the 4×4 apply into two 2-row passes, or staging the matrix in LDS, cuts the live-range and removes `v_accvgpr_write/read` traffic. Frees VGPR headroom (already non-binding) and reduces the U4 instruction count. Expected **1.05-1.2×** on U4-heavy circuits.

**(g) Cut redundant barriers around `active_k++` [LOW-MEDIUM].**
EXPAND/EXPAND_T/EXPAND_ROT use **2 barriers** (post-copy, then post-`active_k++`, lines 375-378/389-391/593-595). The `active_k++` is a single tid0 store; it can be folded so only one barrier separates the copy from the next op's read. Removes ~1 barrier per expansion op. Expected **1.02-1.1×**.

**(h) `coop_reduce2` `ds_bpermute` efficiency [LOW-MEDIUM].**
Each f64 shuffle is 2 `ds_bpermute` (hi/lo, lines 174-175); 16 per reduce2 call. Packing both f64 partials (`l0,l1`) into a single wider shuffle, or using `v_permlane`/DPP for the intra-wave steps, could halve the shuffle count. Must preserve exact summation order (line 182-186). Expected **1.05-1.15×** on reduction-heavy circuits.

---

### Cross-cutting note
The kernel is **latency-bound, LDS-occupancy-gated, and barrier-dense**, not compute- or bandwidth-bound (MFMA=0, VGPR headroom unused). The highest-leverage moves are (a) reclaiming LDS to double occupancy and (b)/(c)/(g) cutting the barrier/serial-section count — these attack the actual bottleneck (co-stalled waves at barriers). The f64/FMA relaxations (d)/(e) are secondary VALU wins gated on branch-determinism verification against the SVM gold path.
