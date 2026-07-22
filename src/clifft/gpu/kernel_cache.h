#pragma once

#include "clifft/gpu/device_program.h"
#include "clifft/gpu/hsa_kernel_dispatch.h"

#include <string>

namespace clifft {
namespace gpu {

/// Compile the kernel source (via clang++ subprocess) and load it via HSA.
/// Caches .hsaco files on disk at ~/.clifft/kernel_cache/{hash}.hsaco
/// Returns an invalid HsaLoadedKernel on failure (caller falls back to SVM).
HsaLoadedKernel compile_or_load_kernel(const FlattenedProgram& flat);

/// Compile and load a cooperative (LDS-tier) kernel.
/// peak_rank 5–10.  Kernel function name: "compiled_sample_kernel_coop".
HsaLoadedKernel compile_or_load_kernel_coop(const FlattenedProgram& flat);

/// Compile and load a global-coop kernel.
/// peak_rank 11–19.  Kernel function name: "compiled_sample_kernel_global".
HsaLoadedKernel compile_or_load_kernel_global(const FlattenedProgram& flat);

/// Free resources associated with a loaded kernel.
void free_compiled_kernel(HsaLoadedKernel& lk);

}  // namespace gpu
}  // namespace clifft
