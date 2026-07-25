// v2_specializer.h — per-circuit C-source specializer for the MLIR-V2 backend.
//
// Emits a specialized amdgcn kernel for one circuit: a straight-line sequence of
// the shared v2_op_*() calls (from v2_ops.h) with COMPILE-TIME-CONSTANT operands
// and per-instruction active_k. The opcode switch/pc-loop is gone (resolved at
// emit time); loop bounds like 1<<(active_k-2) fold to literals under -O2. The
// 2^k amplitude sweeps stay LOOPS (R5: specialize the body + sequence, do NOT
// unroll the sweeps — that was V1's IR-bloat disease).
//
// Byte-exact with the interpreter (and GPU-SVM) because it calls the identical
// v2_op_*() bodies. This is the C-source stand-in for the eventual MLIR dialect
// (each emitted call == one dialect op); R5 says measure clang-per-circuit
// before committing to the dialect.
#pragma once

#include "clifft/gpu/device_program.h"

#include <string>

namespace clifft::gpu::v2 {

// The tier a specialized kernel targets (mirrors the interpreter's tiers).
enum class SpecTier { Register, Coop, Global };

// Pick the tier for a peak_rank (same thresholds as v2_sample).
SpecTier spec_tier_for_rank(uint32_t peak_rank);

// Emit specialized C source for `flat`. `kernel_name` is the emitted
// amdgpu_kernel symbol. Returns the complete .c text (ready to compile via the
// ClifftAmdgcn pipeline). Throws if the circuit uses an opcode the specializer
// cannot emit yet (caller should fall back to the interpreter).
std::string emit_specialized_kernel(const FlattenedProgram& flat,
                                    const std::string& kernel_name,
                                    SpecTier tier);

}  // namespace clifft::gpu::v2
