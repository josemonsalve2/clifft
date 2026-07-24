// device_abi.h — the frozen host<->device wire ABI for the MLIR-V2 backend.
//
// This header is included by BOTH the host driver (C++) and the plain-C device
// kernels (compiled to amdgcn). It must therefore be valid C AND C++ with NO
// HIP types, no C++-only constructs. All structs mirror the byte layout the
// device interpreter reads out of the flattened program (device_program.cc /
// gpu_types.h). device_abi_checks.cc static_asserts these against the host
// gpu_types.h structs so drift is caught at compile time (CODEX_REVIEW.md §8.6).
//
// ABI VERSION: bump when any struct layout, enum value, or kernarg order
// changes. Cached .hsaco keyed partly on this.
#ifndef CLIFFT_GPU_MLIR_V2_DEVICE_ABI_H
#define CLIFFT_GPU_MLIR_V2_DEVICE_ABI_H

#include <stdint.h>

#define CLIFFT_V2_ABI_VERSION 1u

// Frame width in 64-bit words (mirrors host kPauliWords). Config-driven per R4;
// pinned to the host value at build time. If the host changes kPauliWords the
// static_assert in device_abi_checks.cc fails.
#ifndef CLIFFT_V2_PAULI_WORDS
#define CLIFFT_V2_PAULI_WORDS 5
#endif

#define CLIFFT_V2_MAX_OBS 8u
#define CLIFFT_V2_MAX_EXP 8u

// ---- Complex amplitude (f32 storage; f64 used only in reductions) ----------
// MUST be 8-byte aligned to match host GpuComplex (__attribute__((aligned(8)))),
// otherwise struct padding of GpuFusedU2/U4 diverges. Use the attribute (works
// in both C and C++ under clang/gcc) rather than _Alignas for portability here.
typedef struct __attribute__((aligned(8))) { float re; float im; } CV2Complex;

// ---- Instruction (mirrors GpuInstr, 24 bytes) ------------------------------
typedef struct {
    uint8_t  opcode;
    uint8_t  flags;
    uint16_t axis_1;
    uint16_t axis_2;
    uint32_t a;
    uint32_t b;
    uint64_t mask;
    double   weight_re;
    double   weight_im;
} CV2Instr;

// ---- Fused single-axis 2x2 unitary (mirrors GpuFusedU2Entry exactly) --------
// Indexed by in_state (Pauli frame px|pz<<1, 0..3). matrices[s] is a flattened
// 2x2 = 4 complex; gamma per state; out_state per state.
typedef struct {
    CV2Complex matrices[4][4];          // [in_state][2x2 flattened]
    CV2Complex gamma_multipliers[4];
    uint8_t    out_states[4];
} CV2FusedU2Entry;

// ---- Fused 2-axis 4x4 unitary (mirrors GpuFusedU4Entry exactly) -------------
typedef struct {
    struct {
        CV2Complex matrix[4][4];        // 4x4 complex
        CV2Complex gamma_multiplier;
        uint8_t    out_state;
    } entries[16];                       // indexed by 4-bit in_state
} CV2FusedU4Entry;

// ---- Kernel arguments for the coop interpreter -----------------------------
// Packed in this exact order into the AQL kernarg region. The device kernel
// signature must match. Pointers are 64-bit device addresses.
typedef struct {
    uint64_t instrs;              // const CV2Instr*
    uint32_t num_instrs;
    uint32_t peak_rank;
    uint32_t total_meas_slots;
    uint32_t num_observables;
    uint64_t seed;
    uint64_t shot_offset;
    uint64_t shots;
    uint64_t block_counts;        // CV2BlockCounts* (per-workgroup results)
    uint64_t fused_u2;            // const GpuFusedU2Entry*
    uint64_t fused_u4;            // const GpuFusedU4Entry*
    uint64_t observable_offsets;  // const uint32_t*
    uint64_t observable_targets;  // const uint32_t*
} CV2KernArgs;

// ---- Per-workgroup result accumulator (mirrors BlockCounts) ----------------
typedef struct {
    uint64_t passed;
    uint64_t logical_errors;
    uint64_t observable_ones[CLIFFT_V2_MAX_OBS];
    double   exp_val_sums[CLIFFT_V2_MAX_EXP];
    uint64_t exp_val_count;
} CV2BlockCounts;

#endif  // CLIFFT_GPU_MLIR_V2_DEVICE_ABI_H
