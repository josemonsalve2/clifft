// empty_kernel.c — the smallest possible amdgcn kernel.
//
// The point of an empty kernel is that its GPU execution time is a floor
// (a few hundred ns of wave launch + s_endpgm), so end-to-end launch-to-
// completion latency measured on the host is DOMINATED by the runtime's
// dispatch path. That is exactly the quantity we want to attribute to
// HIP vs raw HSA.
//
// It takes one pointer arg and writes one byte so the compiler cannot delete
// the body and so the kernarg segment is non-trivially sized (matching the
// real kernels, which pass ~20 pointer/scalar args).
//
// Built with the SAME flags as the V2 specialized kernels
// (see v2_compile_cache.cc): --target=amdgcn-amd-amdhsa -ffreestanding
// -nostdlib -nogpulib -std=c23 -O2.

typedef unsigned int u32;
typedef unsigned long u64;

__attribute__((visibility("default")))
__attribute__((amdgpu_kernel))
void bench_empty(unsigned char* out, u64 a, u64 b, u64 c, u64 d) {
    u32 t = __builtin_amdgcn_workitem_id_x();
    if (t == 0u) out[0] = (unsigned char)(a ^ b ^ c ^ d);
}
