typedef unsigned int u32;
// extern -> external linkage, NO initializer (matches MLIR llvm.mlir.global external)
extern __attribute__((address_space(3))) float lds_v[1024];
__attribute__((amdgpu_kernel)) void coop_probe(float* g, u32 n) {
    u32 tid = __builtin_amdgcn_workitem_id_x();
    for (u32 i = tid; i < 1024u; i += 256u) lds_v[i] = (float)i;
    __builtin_amdgcn_s_barrier();
    float x = lds_v[tid & 1023u];
    u32 lane = tid & 63u;
    for (u32 off = 32u; off > 0u; off >>= 1) {
        int peer = __builtin_amdgcn_ds_bpermute((int)((lane ^ off) << 2), __builtin_bit_cast(int, x));
        x += __builtin_bit_cast(float, peer);
    }
    __builtin_amdgcn_s_barrier();
    if (tid == 0u && n > 0u) g[0] = x;
}
