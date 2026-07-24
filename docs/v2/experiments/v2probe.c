// Plain C, no HIP. A trivial amdgpu kernel: each thread writes its id.
typedef unsigned int u32;
__attribute__((amdgpu_kernel)) void probe_kernel(u32* out, u32 n) {
    u32 tid = __builtin_amdgcn_workitem_id_x();
    u32 bid = __builtin_amdgcn_workgroup_id_x();
    u32 i = bid * 256u + tid;
    if (i < n) out[i] = i * 2u;
}
