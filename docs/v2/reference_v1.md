# clifft GPU Code-Generation Reference (V1)

Source-of-truth analysis of the three GPU execution backends, written to inform a
V2 redesign. All citations are `file:line` against the tree at
`/shared/jmonsalv/quantum/clifft_rl/clifft` as of 2026-07-24. No source was
modified to produce this document.

---

## 1. Architecture overview

clifft is a Pauli-frame-tracking quantum-circuit simulator. A circuit is compiled
by the back-end (`lower()`, `backend.h:356`) into a `CompiledModule`
(`backend.h:324`): a flat `std::vector<Instruction>` **bytecode**, a `ConstantPool`
(`backend.h:255`) of heavy side data referenced by index, and metadata including
`peak_rank` (max active statevector dimension `k`, ≤19) and `num_qubits`.

### The operand / bytecode model

Each `Instruction` is exactly 32 bytes (`static_assert`, `backend.h:163`): a 1-byte
`Opcode`, flag/axis fields, and a 24-byte payload `union` with seven variants
(`math`, `classical`, `pauli`, `multi_gate`, `u2`, `u4`, `exp_val`;
`backend.h:111-160`). There are ~40 opcodes in `enum class Opcode`
(`backend.h:25-86`), grouped as: **frame** ops (zero-cost, update only the `p_x`/`p_z`
Pauli frame), **array** ops (mutate the amplitude vector `v[]` and the frame),
**expand** ops (grow `k → k+1`), **measurement** ops (sample/collapse), **noise**
ops, and classical ops (`APPLY_PAULI`, `POSTSELECT`, `OBSERVABLE`, `EXP_VAL`,
`DETECTOR`). The five `*_FORCED` measurement variants are synthesized at runtime by
the CPU path only and are never emitted by the compiler (`backend.h:62-74`).

The statevector is stored in a **local active basis** of `2^k` complex amplitudes,
where `k = active_k` grows on `EXPAND` and shrinks on active measurement. `peak_rank`
bounds `2^k` and selects the GPU tier. The Pauli frame is a pair of bit-vectors
`px`/`pz` of `kPauliWords = 5` × `uint64` = 320 qubits (`gpu_types.h:30`), sized
independently of `peak_rank`.

### The three tiers (by `peak_rank`)

Boundaries are compile-time constants in `gpu_types.h:8-14`:

| Tier | `peak_rank` | Amplitude storage | Parallelism |
|------|-------------|-------------------|-------------|
| Register | 0–4 (`kThreadMaxPeakRank`) | per-thread `alloca` / private (`≤16` amps) | 1 shot / thread |
| Coop | 5–10 (`kSharedMaxPeakRank`) | LDS `addrspace(3)` (`≤1024` amps) | 256 threads / 1 shot |
| Global | 11–19 (`kGlobalMaxPeakRank`) | HBM, per-workgroup slot | work-stealing across workgroups |

Note `StatevectorSqueezePass` minimizes `peak_rank` at compile time, so many
nominally "high-rank" circuits collapse to register tier.

### The three backends & their pipelines

1. **SVM (interpreter, reference).** A flat interpreter that walks the bytecode at
   runtime. CPU version `svm/svm_kernels.inl` (threaded computed-goto dispatch at
   `svm_kernels.inl:2183`, f64, AVX-512/AVX2/scalar + OpenMP). GPU version
   `gpu/sampler/hip_sampler.hip`: three `__global__` entry points
   (`sample_kernel` `:1778`, `sample_kernel_coop` `:1861`,
   `sample_kernel_global_coop` `:1937`) around two device interpreter loops
   (`execute_shot` `:728`, `execute_shot_coop` `:1528`). Tier is chosen host-side in
   `gpu_sample_survivors` (`:2034`) from `flat.peak_rank`. No per-circuit compile
   step — one prebuilt kernel interprets any bytecode.

2. **Hybrid (compiled HIP).** C++ string templates emit a bespoke HIP source file
   per circuit, then compile it with **`clang++ -x hip --offload-device-only`**
   (not `hipcc`, despite comments) and `clang-offload-bundler --unbundle`
   (`codegen/kernel_cache.cc:112-168`), caching the `.hsaco` under
   `$HOME/.clifft/kernel_cache/`. Per-op dispatch is a `switch` in
   `emit_instructions.cc` (register `:27`, coop/global `:247`); tier generators are
   `emit_register_kernel.cc:16`, `emit_coop_kernel.cc:25`, `emit_global_kernel.cc:29`.
   Loaded and dispatched via HSA (`hsa_load_kernel`), no `hipModuleLoad`.

3. **MLIR (compiled, textual LLVM dialect).** Emits a textual MLIR module per
   circuit (`mlir_emit.cc`, ~3425 lines), pipes it through
   `mlir-opt --canonicalize --cse --convert-func-to-llvm` → `mlir-translate
   --mlir-to-llvmir` → `opt`/`llc`/`ld.lld` → `.hsaco` → HSA dispatch
   (`mlir_codegen.cc:28-142`, `mlir_kernel_cache.cc:94-214`). Three entry points
   emit the three tiers: `emit_mlir_text` (`mlir_emit.cc:2154`),
   `emit_mlir_text_coop` (`:2624`), `emit_mlir_text_global` (`:2988`). Caching keyed
   on `fnv1a(llvmir + gpu_arch)` (`mlir_kernel_cache.cc:200`).

All three backends **must produce identical sampling results** for the same seed.

---

## 2. The operand set

Legend for kind: **F** = frame-only, **A** = amplitude-mutating, **M** =
measurement, **N** = noise, **C** = classical/other. Citations give the
implementation site in each backend. SVM-GPU register handlers are in
`hip_sampler.hip`; Hybrid device functions live in `codegen/ops/*.inc`; MLIR
emitters in `mlir/ops/mlir_*.inc` (`case` label line).

| Opcode | Kind | SVM CPU (`svm_kernels.inl`) | SVM GPU (`hip_sampler.hip`) | Hybrid (`codegen/ops`) | MLIR (`mlir/ops`) |
|--------|------|------|------|------|------|
| OP_FRAME_CNOT | F | `exec_frame_cnot` :67 | `frame_cnot` :277 | frame_ops.inc:5 | mlir_frame_ops.inc:31 |
| OP_FRAME_CZ | F | :77 | :281 | frame_ops.inc:9 | mlir_frame_ops.inc:46 |
| OP_FRAME_H | F | :90 | :285 | frame_ops.inc:13 | mlir_frame_ops.inc:6 |
| OP_FRAME_S | F | :103 | :291 | frame_ops.inc:17 | mlir_frame_ops.inc:19 |
| OP_FRAME_S_DAG | F | :749 | :291 | frame_ops.inc:17 | mlir_frame_ops.inc:20 |
| OP_FRAME_SWAP | F | :113 | :299 | frame_ops.inc:20 | mlir_frame_ops.inc:61 |
| OP_ARRAY_CNOT | A | `exec_array_cnot` :124 | `array_cnot` :308 | array_ops.inc:15 | mlir_array_ops.inc:169 |
| OP_ARRAY_CZ | A | :255 | `array_cz` :321 | array_ops.inc:29 | mlir_array_ops.inc:184 |
| OP_ARRAY_SWAP | A | :369 | `array_swap` :332 | array_ops.inc:41 | mlir_array_ops.inc:245 |
| OP_ARRAY_MULTI_CNOT | A | :464 | `array_multi_cnot` :345 | multi_qubit_ops.inc:4 | mlir_array_ops.inc:366 |
| OP_ARRAY_MULTI_CZ | A | :559 | `array_multi_cz` :364 | multi_qubit_ops.inc:22 | mlir_array_ops.inc:467 |
| OP_ARRAY_H | A | :639 | `array_h` :382 | array_ops.inc:55 | mlir_array_ops.inc:11 |
| OP_ARRAY_S | A | :854 | `array_s` :405 | array_ops.inc:69 | mlir_array_ops.inc:24 |
| OP_ARRAY_S_DAG | A | :862 | `array_s` :405 | array_ops.inc:69 | mlir_array_ops.inc:35 |
| OP_ARRAY_T | A | :951 | `array_t` :410 | array_ops.inc:76 | mlir_array_ops.inc:46 |
| OP_ARRAY_T_DAG | A | :972 | `array_t` :410 | array_ops.inc:76 | mlir_array_ops.inc:109 |
| OP_ARRAY_ROT | A | :1044 | `array_rot` :608 | array_ops.inc:86 | mlir_array_ops.inc:299 |
| OP_ARRAY_U2 | A | :1091 | `array_u2` :626 | unitary_ops.inc:23 | mlir_array_ops.inc:546 |
| OP_ARRAY_U4 | A | :1310 | `array_u4` :645 | unitary_ops.inc:45 | mlir_array_ops.inc:550 |
| OP_EXPAND | A | :876 | `expand_plain` :422 | expand_ops.inc:4 | mlir_expand_ops.inc:5 |
| OP_EXPAND_T | A | :995 | `expand_t` :430 | expand_ops.inc:12 | mlir_expand_ops.inc:9 |
| OP_EXPAND_T_DAG | A | :1016 | `expand_t` :430 | expand_ops.inc:12 | mlir_expand_ops.inc:13 |
| OP_EXPAND_ROT | A | :1065 | `expand_rot` :615 | expand_ops.inc:24 | mlir_expand_ops.inc:17 |
| OP_MEAS_DORMANT_STATIC | M | :1540 | inline (`execute_shot`) | emit_instructions.cc:105 | mlir_measurement_ops.inc:6 |
| OP_MEAS_DORMANT_RANDOM | M | :1551 | `meas_dormant_random` :444 | measurement_ops.inc:13 | mlir_measurement_ops.inc:77 |
| OP_MEAS_ACTIVE_DIAGONAL | M | :1573 | `meas_active_diagonal` :452 / `coop_` :1213 | measurement_ops.inc:23 | mlir_measurement_ops.inc:81 |
| OP_MEAS_ACTIVE_INTERFERE | M | :1634 | `meas_active_interfere` :476 / `coop_` :1248 | measurement_ops.inc:40 | mlir_measurement_ops.inc:85 |
| OP_SWAP_MEAS_INTERFERE | M | :1728 | `swap_meas_interfere` :503 / `coop_` :1289 | measurement_ops.inc:65 | mlir_measurement_ops.inc:89 |
| OP_APPLY_PAULI | N/C | :1869 | `apply_pauli` :550 | frame_ops.inc:32 | mlir_noise_ops.inc:399 |
| OP_NOISE | N | :1886 | `exec_noise` :559 | emit_instructions.cc:137 | mlir_noise_ops.inc:7 |
| OP_NOISE_BLOCK | N | :1926 | `exec_noise_block` :578 | emit_instructions.cc:155 | mlir_noise_ops.inc:158 |
| OP_READOUT_NOISE | N | :1937 | (register inline) | emit_instructions.cc:182 | mlir_noise_ops.inc:331 |
| OP_DETECTOR | C | :1964 | no-op/barrier (`:827`/`:1733`) | emit_instructions.cc:187 | mlir_measurement_ops.inc:75 (no-op) |
| OP_POSTSELECT | C | :1984 | `exec_postselect` :586 | emit_instructions.cc:190 | mlir_noise_ops.inc:366 |
| OP_OBSERVABLE | C | :2007 | `exec_observable` :598 | emit_instructions.cc:202 | mlir_measurement_ops.inc:35 |
| OP_EXP_VAL | C | :2031 | `exec_exp_val` :682 | exp_val_ops.inc:8 | mlir_exp_val_ops.inc:7 (discard — unimpl.) |
| OP_*_FORCED (×5) | M | `svm_forced_kernels.*` | not implemented | not implemented | discard (mlir_measurement_ops.inc:93) |

The `*_FORCED` opcodes and `OP_DETECTOR`'s record-writing exist only in the CPU
path; the GPU backends treat `DETECTOR` as a barrier/no-op and mark `*_FORCED` shots
discarded (`mlir_measurement_ops.inc:93-101`).

---

## 3. How MLIR codegen works today

### Structure of the three emitters

`emit_mlir_text` / `_coop` / `_global` (`mlir_emit.cc:2154` / `:2624` / `:2988`)
share one skeleton: emit a `module`, intrinsic decls (`emit_gpu_intrinsic_decls`
`:487`), the shared `@clifft_log` (`:1422`) and (if noisy) `@clifft_draw_next_noise`
(`:1562`) functions, LDS globals, then one `amdgpu_kernelcc` function. Inside, they
declare constants, allocate state, then run a **compile-time loop over the bytecode**
(`for pc … switch(op)`, register `:2377-2414`) that `#include`s the six
`ops/mlir_*.inc` files as the body of the `switch` (`:2387-2392`). Each op emits its
IR **inline at that program counter** — there is no runtime loop over the bytecode;
the entire circuit is flattened into straight-line IR. Compile-time `active_k`
(`g_emit_ak`, `:340`) is tracked so each gate's inner loop trip count `2^(k-1)` is a
constant. The `.inc` files call back into free functions in `mlir_emit.cc` through
per-tier lambda bindings (`:2313-2366`) so the same op file works for all tiers.

### State storage per tier

- **Register** (`:2197-2228`): `px`/`pz`/`active_k`/`discarded`/`v[2^peak]`/`meas`/
  `obs` all in `alloca` `addrspace(5)` (private), addrspace-cast to generic. Amplitude
  loads use `emit_load_v` (`:210`); alignment-8 is deliberately **excluded** here to
  avoid a VGPR-spill catastrophe (comment `:216-218`, "P1.9").
- **Coop** (`:2638-2718`): amplitudes in LDS `@lds_v` (`addrspace(3)`,
  `emit_lds_global` `:721`); `lds_amplitudes=true` switches load/store to `ptr<3>`
  with alignment-8 (`emit_load_v` `:212,218`). Frame/meas/reduction scalars also LDS.
  Init and every gate loop stride 256 threads (`loop_init="%tidx"`, `step="%c256"`;
  e.g. `emit_array_h_static` `:364-365`).
- **Global** (`:3001-3068`): frame/classical state in LDS; amplitudes in HBM via
  kernel-arg `%global_v`, sliced per workgroup at `BIDX * (1<<kGlobalMaxPeakRank)`
  (`:3053-3057`). Scratch is allocated at **half** stride (`:3064`) — a full-stride
  bug previously faulted for `bidx≥1` (comment `:3058-3063`).

### RNG, frame, measurement

RNG is xoshiro256** seeded per shot (`emit_rng_seed`, called `:2304`). In register
tier each thread runs its own RNG; in **coop/global only thread 0 draws**, then
broadcasts through LDS. Frame bit ops are `emit_bit_get/set/xor` (`:109-184`), five
64-bit words. Measurements compute branch probabilities by a warp-shuffle +
LDS reduction (`emit_coop_reduce2` `:566`, `emit_wavefront_reduce_i64` `:679`, using
`llvm.amdgcn.ds.bpermute`), sample the branch on thread 0, and fold cooperatively.
Cooperative divergent regions are wrapped by `emit_tid0_guard_begin/end`
(`:693-719`), with a `_nobarrier` variant because **a barrier reached by only some
threads hangs AMDGPU** (comment `:711-714`).

### Shared-function extractions

Two hot bodies were hoisted out of per-op inlining into module-level `llvm.func`s:
`@clifft_log` (`:1422`, IEEE decomposition + atanh series) and
`@clifft_draw_next_noise` (`:1562`, gap-sampling binary search over the hazard
table). Because **MLIR functions are isolated regions** (cannot reference
kernel-scope SSA values), each re-declares the constants it needs and takes the state
pointers as arguments (`:1568-1572`, call site `:1586`). `OP_NOISE_BLOCK` was also
converted from per-site unrolling to a **runtime MLIR loop** over sites
(`mlir_noise_ops.inc:158`, header/body/exit at `:183-321`; comment `:164`).

---

## 4. Cross-backend consistency model

There is **no shared implementation** — every opcode is hand-written three times.
Consistency is maintained only by convention plus adversarial cross-backend diffing
of sampled outputs (same seed → same passed-shot count / observable parities).

The "gold" hierarchy is:

1. **SVM CPU** (`svm_kernels.inl`, f64 `std::complex<double>`) is the truest
   reference — full double precision, deterministic.
2. **Hybrid** (f32 amplitude storage, f64 accumulation) is the **GPU gold**: it is
   the reference the GPU backends target, since GPU f32 storage legitimately diverges
   from CPU f64 on borderline stabilizer branches.
3. **MLIR must bit-match Hybrid.** The MLIR emitters deliberately replicate Hybrid's
   f32-store/f64-accumulate split: `emit_cscale_f64` extends to f64, scales, truncates
   (`:272-298`, comment "Match gold cscale"); `emit_cnorm` squares in f64
   (`:825-828`). It also replicates the exact RNG-draw ordering — e.g.
   `draw_next_noise` early-returns without consuming an RNG value once noise is
   exhausted, or every downstream measurement desyncs (comment
   `mlir_emit.cc:1455-1459`).

Because state layout must also match across backends, struct/word conventions are
shared through constants (`kPauliWords`, `kMaxMeas`, `kMaxObs`, tier ranks in
`gpu_types.h`) and the `block_counts` byte offsets are hard-coded identically in all
GPU reductions (`passed` at 0, `logical_errors` at 8, observables at `16 + i*8`;
`mlir_emit.cc:2560-2608`).

---

## 5. Catalogue of problems & root causes

### 5.1 Duplication tax
Every opcode is implemented ~3× (SVM interp, Hybrid template, MLIR IR emitter),
often 4× counting CPU vs GPU SVM and register vs coop. A fix in one backend does not
propagate. Concrete: `OP_ARRAY_MULTI_CNOT` had `px`/`pz` swapped **only in the MLIR
frame update** while SVM and Hybrid were correct — the fix and its post-mortem are
inline at `mlir_array_ops.inc:426-463` ("The previous emission had px/pz swapped
here, corrupting the frame for controlled…"). This is a whole class of
"one-backend-diverges" bugs that only cross-backend diffing catches.

### 5.2 IR bloat (full unrolling)
The MLIR backend flattens the entire bytecode into straight-line IR with **no runtime
loop over instructions** (`mlir_emit.cc:2377`). A 16432-op circuit (`surface_d11`)
produces ~20 MB of LLVM-IR that `llc`/`lld` cannot link in reasonable time. The
compile path is size-adaptive as a band-aid: `>4 MB` drops to `llc -O2`, `>16 MB` to
`-O1` and skips the separate `opt` pass (`mlir_kernel_cache.cc:104-141`).

Two bloat sources were already mitigated:
- `OP_NOISE_BLOCK` used to unroll one full site-block per site (~1200 sites for
  `circuit_d5`, ~100k lines / 32 MB IR that OOM'd `llc`) → now a runtime loop over the
  site index that reads per-site metadata from `%noise_sites_ptr[si*16]`
  (`mlir_noise_ops.inc:158-330`, comment `:164-169`).
- `log(x)` and `draw_next_noise` were inlined at every noise draw (e.g.
  `surface_d9_t5`: 1444 draws × ~110/~317 lines) → hoisted to `@clifft_log` /
  `@clifft_draw_next_noise` (`mlir_emit.cc:1417-1428`, `:1557-1576`).

**Still unsolved:** each `OP_NOISE` instruction emits a bounded per-site body (an
inner *runtime* channel loop `nch_hdr/body/done` plus a `kPauliWords`-unrolled frame
XOR; `mlir_noise_ops.inc:7-149`), so the bloat scales with the **number of NOISE
instructions**, not per-draw unrolling — and with ~1200 sites this is the single
largest IR contributor (~55%) in large noisy circuits. `OP_ARRAY_U2`/`U4` remain
matrix-unrolled: each unrolls per incoming-frame-state branch (U2: 4 branches ≈
150–200 IR ops; U4: up to 16 branches ≈ 800–1500+ IR ops **per instruction**), each
branch a runtime amplitude loop wrapping a fully unrolled complex matmul
(`emit_array_u2` `mlir_emit.cc:1813`, `emit_array_u4` `:1920`). This is open task #134,
and `mlir_kernel_cache.cc:89-90` notes "surface codes emit 40MB+ IR from U2/U4
inlining."

### 5.3 Register spills & occupancy
The register-tier kernel reports **private = 115 KB/thread** — spill scratch produced
by the giant unrolled straight-line body — which destroys occupancy. The emitter
already fights this defensively by omitting the alignment-8 attribute on register-tier
amplitude loads/stores (`mlir_emit.cc:216-218`, "EXCLUDED from register tier to avoid
VGPR spill catastrophe"). Root cause is the same as 5.2: no loops means the register
allocator sees one enormous basic-block chain.

### 5.4 Performance gap
MLIR is currently **5–60× slower** than SVM/Hybrid. Measured O2 `sample_seconds`
(10000 shots, MI350X):

| Circuit | SVM | Hybrid | MLIR | MLIR slowdown |
|---------|-----|--------|------|---------------|
| frame_h | 0.066 | 0.053 | 0.271 | 5× |
| qv10 | 0.074 | 0.058 | 3.54 | 61× |
| circuit_d5 | 0.078 | 0.065 | 1.22 | 19× |
| surface_d9_t10 | 0.090 | 0.112 | 5.50 | 49× |

SVM/Hybrid sit at ~0.05–0.11 s across the board; MLIR ranges 0.27–5.5 s. The gap
tracks IR size/spills (5.2/5.3), and the coop/global tiers additionally have high
barrier/reduction overhead and thread-0-serialized RNG (§3).

### 5.5 Correctness fragility
Achieving cross-backend parity took enormous effort. Documented hazards:
- **Barriers:** divergent barriers hang AMDGPU → the `_nobarrier` guard variant and
  careful placement (`mlir_emit.cc:711-719`).
- **RNG threading:** coop/global must draw only on thread 0 and broadcast; the noise
  exhaustion early-return must not consume an extra RNG value or all downstream draws
  desync (`:1455-1459`).
- **Precision:** f32 store vs f64 accumulate must match Hybrid exactly or borderline
  stabilizer branches flip (`emit_cscale_f64` `:272`, `emit_cnorm` `:825`).
- **Struct/stride layout:** the global scratch half-stride fault (`:3058-3063`); the
  `block_counts` byte offsets must match across backends.
- **Frame width:** the frame was hardcoded to 2×`uint64` (128 qubits) and widened to
  `kPauliWords = 5` (320 qubits); `emit_zero_frame` (`:733`) now zeroes all 5 words —
  zeroing only words 0/1 left high words uninitialized for >128-qubit circuits.

---

## 6. Lessons learned

**What made bugs hard.** The triple (often quadruple) implementation means a single
gate has three independently-maintained truth tables; divergence is invisible until a
sampled-output diff. Because the GPU legitimately differs from CPU f64, there is no
single bit-exact oracle — MLIR must match *Hybrid's* rounding, not CPU's, so "wrong"
and "just f32" look alike. AMDGPU's undefined behavior on divergent barriers means
correctness bugs and hangs share the same surface.

**Patterns that worked.**
- **Shared `llvm.func` extraction** for hot, large bodies (`@clifft_log`,
  `@clifft_draw_next_noise`) collapses hundreds of thousands of IR lines to one
  definition + short call sites — the single biggest IR-size win so far. Constraint:
  isolated regions force re-declaring constants and passing state pointers as args.
- **Runtime MLIR loops** in place of per-site unrolling (`OP_NOISE_BLOCK`) — the
  template for shrinking `OP_NOISE` and U2/U4 next.
- **tid0-only + broadcast** for RNG/frame/classical updates in cooperative tiers,
  with barrier-safe guard helpers.
- **Adversarial cross-backend diffing** (same seed → same passed-shots / observable
  parities) is the only reliable detector of one-backend-diverges bugs; it is what
  surfaced the `MULTI_CNOT` px/pz swap and the noise RNG-desync.

**Implication for V2.** The duplication tax (5.1) and unrolling-driven bloat/spills
(5.2/5.3) are the same root cause viewed two ways: opcode semantics live in
hand-written per-backend emitters, and the MLIR emitter chooses full unrolling over a
runtime interpreter-style loop. A V2 that (a) defines each opcode's semantics **once**
in a form all backends consume, and (b) emits a **runtime loop over bytecode** (or at
least loops over repeated structure) directly attacks both the correctness fragility
and the 5–60× performance gap.
