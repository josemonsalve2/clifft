// v2_compile_cache.h — runtime amdgcn compile + content-addressed cache for the
// V2 specializer. Compiles emitted C source through the SAME no-HIP+ocml
// pipeline as cmake/ClifftAmdgcn.cmake (clang -> llvm-link ocml -> opt -> llc ->
// ld.lld -> .hsaco), caching by a hash of (source + toolchain + arch) so a given
// circuit is compiled ONCE. Compile time is never on the sampling hot path and
// is never counted in benchmarks (warm the cache first).
#pragma once

#include <string>

namespace clifft::gpu::v2 {

// Compile `csrc` (a complete .c produced by the specializer) to a .hsaco and
// return its path. `key` is a stable identifier (e.g. circuit hash) used for the
// cache filename; identical (key, source, toolchain) reuse the cached .hsaco.
// Throws std::runtime_error on compile failure (caller falls back to interpreter).
std::string compile_specialized(const std::string& csrc, const std::string& key);

// True if the runtime toolchain (clang/llc/lld/opt/llvm-link + ocml bitcode) is
// available; if not, the specializer cannot be used and callers fall back.
bool specializer_toolchain_available();

}  // namespace clifft::gpu::v2
