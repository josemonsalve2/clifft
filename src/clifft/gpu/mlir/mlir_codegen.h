#pragma once

#ifdef CLIFFT_ENABLE_MLIR

#include "clifft/gpu/device_program.h"

#include <string>

namespace clifft {
namespace gpu {

/// Generate textual MLIR (LLVM dialect), optimize with mlir-opt, then translate
/// to LLVM-IR text via mlir-translate.  The resulting LLVM-IR string is suitable
/// for compilation by llc --march=amdgcn.
///
/// Uses subprocess tools from the LLVM module install:
///   mlir-opt --canonicalize --cse --convert-func-to-llvm
///   mlir-translate --mlir-to-llvmir
///
/// Covers register-tier ops (peak_rank ≤ 4): H, S, T, CNOT, frame ops,
/// dormant measurements, observables.  Unsupported ops set a discard flag.
///
/// Returns empty string on failure.
std::string generate_mlir_kernel_llvmir(const FlattenedProgram& flat,
                                         const std::string& gpu_arch);
std::string generate_mlir_kernel_llvmir_coop(const FlattenedProgram& flat,
                                              const std::string& gpu_arch);
std::string generate_mlir_kernel_llvmir_global(const FlattenedProgram& flat,
                                                const std::string& gpu_arch);

}  // namespace gpu
}  // namespace clifft

#endif  // CLIFFT_ENABLE_MLIR
