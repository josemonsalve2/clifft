// device_abi_checks.cc — compile-time verification that the V2 device ABI
// (device_abi.h, shared with plain-C device code) byte-matches the host
// structs in gpu_types.h. If any host struct changes layout, this fails to
// compile — catching drift before it becomes a silent GPU miscompute
// (CODEX_REVIEW.md §8.6). This file has no runtime code.
#include "clifft/gpu/mlir/v2/device_abi.h"
#include "clifft/gpu/gpu_types.h"
#include "clifft/backend/backend.h"

#include <cstddef>
#include <type_traits>

// The V2 device interpreter (coop_interpreter.c) hardcodes opcode integer
// values in a plain-C enum. Verify the anchor values match the authoritative
// clifft::Opcode enum so a reorder in backend.h can never silently misroute the
// device switch (this caught an off-by-one: EXPAND/MEAS were numbered -1).
static_assert(static_cast<int>(clifft::Opcode::OP_FRAME_CNOT) == 0, "opcode FRAME_CNOT");
static_assert(static_cast<int>(clifft::Opcode::OP_FRAME_SWAP) == 5, "opcode FRAME_SWAP");
static_assert(static_cast<int>(clifft::Opcode::OP_ARRAY_CNOT) == 6, "opcode ARRAY_CNOT");
static_assert(static_cast<int>(clifft::Opcode::OP_ARRAY_U4) == 18, "opcode ARRAY_U4");
static_assert(static_cast<int>(clifft::Opcode::OP_EXPAND) == 19, "opcode EXPAND");
static_assert(static_cast<int>(clifft::Opcode::OP_MEAS_DORMANT_STATIC) == 23, "opcode MEAS_DORMANT_STATIC");
static_assert(static_cast<int>(clifft::Opcode::OP_SWAP_MEAS_INTERFERE) == 27, "opcode SWAP_MEAS_INTERFERE");
static_assert(static_cast<int>(clifft::Opcode::OP_APPLY_PAULI) == 33, "opcode APPLY_PAULI");
static_assert(static_cast<int>(clifft::Opcode::OP_DETECTOR) == 37, "opcode DETECTOR");
static_assert(static_cast<int>(clifft::Opcode::OP_OBSERVABLE) == 39, "opcode OBSERVABLE");
static_assert(static_cast<int>(clifft::Opcode::OP_EXP_VAL) == 40, "opcode EXP_VAL");

namespace {

using clifft::gpu::GpuInstr;
using clifft::gpu::GpuComplex;
using clifft::gpu::GpuFusedU2Entry;
using clifft::gpu::GpuFusedU4Entry;
using clifft::gpu::BlockCounts;
using clifft::gpu::kMaxObs;
using clifft::gpu::kMaxExpVals;
using clifft::gpu::kPauliWords;

// --- Frame width / limits pinned to host ---
static_assert(CLIFFT_V2_PAULI_WORDS == kPauliWords,
              "V2 ABI kPauliWords mismatch — rebuild device_abi.h with -DCLIFFT_V2_PAULI_WORDS matching host");
static_assert(CLIFFT_V2_MAX_OBS == kMaxObs, "V2 MAX_OBS mismatch");
static_assert(CLIFFT_V2_MAX_EXP == kMaxExpVals, "V2 MAX_EXP mismatch");

// --- Complex ---
static_assert(sizeof(CV2Complex) == sizeof(GpuComplex), "CV2Complex size");
static_assert(alignof(CV2Complex) == alignof(GpuComplex) || sizeof(CV2Complex) == 8,
              "CV2Complex align");
static_assert(offsetof(CV2Complex, re) == offsetof(GpuComplex, re), "complex.re");
static_assert(offsetof(CV2Complex, im) == offsetof(GpuComplex, im), "complex.im");

// --- Instruction (the hot per-op record) ---
static_assert(sizeof(CV2Instr) == sizeof(GpuInstr), "CV2Instr size");
static_assert(offsetof(CV2Instr, opcode)    == offsetof(GpuInstr, opcode),    "instr.opcode");
static_assert(offsetof(CV2Instr, flags)     == offsetof(GpuInstr, flags),     "instr.flags");
static_assert(offsetof(CV2Instr, axis_1)    == offsetof(GpuInstr, axis_1),    "instr.axis_1");
static_assert(offsetof(CV2Instr, axis_2)    == offsetof(GpuInstr, axis_2),    "instr.axis_2");
static_assert(offsetof(CV2Instr, a)         == offsetof(GpuInstr, a),         "instr.a");
static_assert(offsetof(CV2Instr, b)         == offsetof(GpuInstr, b),         "instr.b");
static_assert(offsetof(CV2Instr, mask)      == offsetof(GpuInstr, mask),      "instr.mask");
static_assert(offsetof(CV2Instr, weight_re) == offsetof(GpuInstr, weight_re), "instr.weight_re");
static_assert(offsetof(CV2Instr, weight_im) == offsetof(GpuInstr, weight_im), "instr.weight_im");

// --- Fused U2 ---
static_assert(sizeof(CV2FusedU2Entry) == sizeof(GpuFusedU2Entry), "CV2FusedU2Entry size");
static_assert(offsetof(CV2FusedU2Entry, matrices)          == offsetof(GpuFusedU2Entry, matrices),          "u2.matrices");
static_assert(offsetof(CV2FusedU2Entry, gamma_multipliers) == offsetof(GpuFusedU2Entry, gamma_multipliers), "u2.gamma");
static_assert(offsetof(CV2FusedU2Entry, out_states)        == offsetof(GpuFusedU2Entry, out_states),        "u2.out_states");

// --- Fused U4 ---
static_assert(sizeof(CV2FusedU4Entry) == sizeof(GpuFusedU4Entry), "CV2FusedU4Entry size");
// entries[] element stride must match
static_assert(sizeof(((CV2FusedU4Entry*)0)->entries[0]) ==
              sizeof(((GpuFusedU4Entry*)0)->entries[0]), "u4 entry stride");

// --- BlockCounts (result aggregation) ---
static_assert(sizeof(CV2BlockCounts) == sizeof(BlockCounts), "CV2BlockCounts size");
static_assert(offsetof(CV2BlockCounts, passed)          == offsetof(BlockCounts, passed),          "bc.passed");
static_assert(offsetof(CV2BlockCounts, logical_errors)  == offsetof(BlockCounts, logical_errors),  "bc.logical_errors");
static_assert(offsetof(CV2BlockCounts, observable_ones) == offsetof(BlockCounts, observable_ones), "bc.observable_ones");
static_assert(offsetof(CV2BlockCounts, exp_val_sums)    == offsetof(BlockCounts, exp_val_sums),    "bc.exp_val_sums");
static_assert(offsetof(CV2BlockCounts, exp_val_count)   == offsetof(BlockCounts, exp_val_count),   "bc.exp_val_count");

}  // namespace
