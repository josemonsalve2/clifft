# GPU Sampler Qubit Limit

## Previous limit: 128 qubits

The GPU sampler was hardcoded to support at most 128 qubits. This prevented
surface code circuits at distance d=9 (188 qubits) and higher from running.

### Root cause

The Pauli frame representation used a fixed-width pair of `uint64_t` arrays
(`x[2]`, `z[2]`) across three layers:

| Layer | File | Structures affected |
|-------|------|---------------------|
| Type definitions | `gpu_types.h` | `GpuMask`, `GpuChannel`, `GpuExpValMask` |
| Host flattening | `device_program.cc` | `flatten_arena_mask()`, noise channel copy, exp-val mask copy, `validate_program()` |
| Device kernels | `hip_sampler.hip` | `ShotState`, `__shared__ px/pz`, all XOR apply-pauli paths, exp-val parity computation |

With `uint64_t[2]`, the frame can encode 128 qubit indices (bits 0-127). Any
circuit with `num_qubits > 128` was rejected by `validate_program()` at line
284 of `device_program.cc`.

### Symptom

Running `build-gpu/run_gpu --circuit tests/fixtures/large/surface_d9_r9.stim`
(188 qubits) produced no `peak_rank` or `shots_per_second` output -- the
program threw a `std::runtime_error` before reaching the kernel launch.

## Fix: parameterized width via `kPauliWords`

Introduced two new constants in `gpu_types.h`:

```cpp
constexpr uint32_t kPauliWords = 4;           // 4 * 64 = 256 qubit frame width
constexpr uint32_t kMaxQubits = kPauliWords * 64;
```

All `uint64_t x[2]` / `z[2]` arrays now use `x[kPauliWords]` / `z[kPauliWords]`,
and all per-word XOR / popcount / copy loops iterate over `kPauliWords` instead
of being unrolled for exactly 2 words.

### Files changed

1. **`gpu_types.h`** -- `GpuMask`, `GpuChannel`, `GpuExpValMask` arrays
   widened to `[kPauliWords]`.

2. **`device_program.cc`** -- `validate_program()` now checks against
   `kMaxQubits` (256). `flatten_arena_mask()`, noise channel flatten, and
   exp-val mask flatten all use `for (w = 0; w < kPauliWords; ...)` loops.

3. **`hip_sampler.hip`** -- `ShotState.px/pz`, `__shared__ px/pz` in the
   shared-coop and global-coop kernels, `apply_pauli_to_frame()`, all coop
   `OP_APPLY_PAULI` / `OP_NOISE` / `OP_NOISE_BLOCK` XOR paths, and both
   `exec_exp_val()` / `coop_exec_exp_val()` parity + dormant-X computations.

### Coverage

| Surface code | Qubits | Words needed | Status |
|-------------|--------|-------------|--------|
| d=3 | 26 | 1 | was working, still works |
| d=5 | 72 | 2 | was working, still works |
| d=7 | 130 | 3 | **now works** (was blocked at 128) |
| d=9 | 188 | 3 | **now works** |
| d=11 | 252 | 4 | **now works** |
| d=13 | 376 | 6 | needs `kPauliWords = 6` |

### VGPR cost analysis

Each additional Pauli word adds 4 VGPRs to `ShotState` (2 for px, 2 for pz).
Going from 2 to 4 words adds 8 VGPRs per thread.

- **Per-thread kernel** (`sample_kernel`): ShotState is in registers. The 8
  extra VGPRs may reduce occupancy slightly on GFX942 (MI300X has 512 VGPRs
  per SIMD). At kThreadMaxPeakRank=4, the v[] array is 16 amplitudes = 32
  VGPRs for re/im, so the total per-thread state is moderate. Impact: minimal.

- **Shared-coop kernel** (`sample_kernel_coop`): px/pz live in LDS. Adding 4
  words of LDS (32 bytes) is negligible vs. the 8KB v[] array already in LDS.

- **Global-coop kernel** (`sample_kernel_global_coop`): px/pz in LDS. Same
  negligible impact.

### Future: supporting d=13 and beyond

To support surface_d13 (376 qubits), change `kPauliWords` from 4 to 6. This
is a single-line change. The loop-based code adapts automatically.

For very large circuits (>512 qubits), consider a dynamically-sized bitset
approach, but this would require significant kernel refactoring since device
code cannot use `std::vector`.
