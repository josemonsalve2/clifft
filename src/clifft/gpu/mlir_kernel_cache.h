#pragma once

#ifdef CLIFFT_ENABLE_MLIR

#include "clifft/gpu/device_program.h"
#include "clifft/gpu/hsa_kernel_dispatch.h"

namespace clifft {
namespace gpu {

/// Compile (or load from cache) the MLIR→LLVM-IR megakernel for the given
/// program.  Uses llc + lld (or clang++ fallback) to produce a .hsaco.
/// Loads via HSA. Caches at ~/.clifft/kernel_cache/mlir/{hash}_{arch}.hsaco
/// Returns an invalid HsaLoadedKernel on failure.
HsaLoadedKernel compile_or_load_mlir_kernel(const FlattenedProgram& flat);

}  // namespace gpu
}  // namespace clifft

#endif  // CLIFFT_ENABLE_MLIR
