// probe_kernel.c — MLIR-V2 P0(a) toolchain proof. Plain C, NO HIP.
// Compiled to amdgcn via clang/llc/ld.lld (see cmake/ClifftAmdgcn.cmake).
// Each thread writes out[i] = i*2, proving the no-HIP build + HSA
// load/dispatch/readback path works end-to-end before any operand code.
//
// Kernel args are passed by the HSA AQL packet's kernarg region. For the
// amdgpu_kernel calling convention the arguments are laid out in order with
// natural alignment; the host packs {u32* out; u32 n;} to match.

typedef unsigned int u32;

// Must be GLOBAL + default visibility so HSA can resolve the .kd symbol by
// name (hsa_executable_get_symbol_by_name only sees non-local symbols).
__attribute__((amdgpu_kernel, visibility("default")))
void clifft_v2_probe(u32* out, u32 n) {
    u32 tid = __builtin_amdgcn_workitem_id_x();
    u32 bid = __builtin_amdgcn_workgroup_id_x();
    u32 i = bid * 256u + tid;
    if (i < n) {
        out[i] = i * 2u;
    }
}
