typedef unsigned int u32;
// LDS: no initializer. Clang emits addrspace(3) uninitialized when marked so.
__attribute__((address_space(3))) float lds_v[1024];  // no static, no init
__attribute__((amdgpu_kernel)) void coop_probe(float* g, u32 n) {
    u32 tid = __builtin_amdgcn_workitem_id_x();
    for (u32 i = tid; i < 1024u; i += 256u) lds_v[i] = (float)i;
    __builtin_amdgcn_s_barrier();
    u32 lane = tid & 63u;
    float x = lds_v[tid & 1023u];
    for (u32 off = 32u; off > 0u; off >>= 1) {
        int xi = __builtin_bit_cast(int, x);
        int peer = __builtin_amdgcn_ds_bpermute((int)((lane ^ off) << 2), xi);
        x += __builtin_bit_cast(float, peer);
    }
    __builtin_amdgcn_s_barrier();
    if (tid == 0u && n > 0u) g[0] = x;
}
