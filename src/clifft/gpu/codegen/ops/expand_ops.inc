// expand_ops.inc — Expand (rank-increase) device functions (register + coop tiers)
// Included by emit_preamble.cc. Do not compile standalone.

static const char* kExpandPlain = R"HIP(
DEV static void expand_plain(ShotState& st) {
    uint64_t half = 1ULL << st.active_k;
    for (uint64_t i = 0; i < half; ++i) st.v[i + half] = st.v[i];
    st.active_k++;
}
)HIP";

static const char* kExpandT = R"HIP(
DEV static void expand_t(ShotState& st, uint32_t axis, bool dagger) {
    uint64_t half = 1ULL << st.active_k;
    bool px = bit_get(st.px, axis);
    double imag = dagger ? -kInvSqrt2 : kInvSqrt2;
    if (px) imag = -imag;
    GpuComplex phase = {(float)kInvSqrt2, (float)imag};
    for (uint64_t i = 0; i < half; ++i) st.v[i + half] = cmul(st.v[i], phase);
    st.active_k++;
}
)HIP";

static const char* kExpandRot = R"HIP(
DEV static void expand_rot(ShotState& st, uint32_t axis, double z_re, double z_im) {
    uint64_t half = 1ULL << st.active_k;
    bool px = bit_get(st.px, axis);
    double im = px ? -z_im : z_im;
    GpuComplex phase = {(float)z_re, (float)im};
    for (uint64_t i = 0; i < half; ++i) st.v[i + half] = cmul(st.v[i], phase);
    st.active_k++;
}
)HIP";

// -----------------------------------------------------------------------
// Cooperative (LDS-tier) expand device functions
// -----------------------------------------------------------------------

static const char* kCoopExpandPlain = R"HIP(
DEV static void coop_expand_plain(GpuComplex* v, uint32_t* active_k_ptr) {
    uint32_t k = *active_k_ptr;
    uint64_t half = 1ULL << k;
    for (uint64_t i = TIDX; i < half; i += BSIZE)
        store_complex64_lds(v, i + half, v[i]);
    BARRIER();
    if (TIDX == 0) (*active_k_ptr)++;
    BARRIER();
}
)HIP";

static const char* kCoopExpandT = R"HIP(
DEV static void coop_expand_t(GpuComplex* v, const uint64_t* px,
                               uint32_t* active_k_ptr, uint32_t axis, bool dagger) {
    uint32_t k = *active_k_ptr;
    uint64_t half = 1ULL << k;
    bool pxv = bit_get(px, axis);
    double imag = dagger ? -kInvSqrt2 : kInvSqrt2;
    if (pxv) imag = -imag;
    GpuComplex phase = {(float)kInvSqrt2, (float)imag};
    for (uint64_t i = TIDX; i < half; i += BSIZE)
        store_complex64_lds(v, i + half, cmul(v[i], phase));
    BARRIER();
    if (TIDX == 0) (*active_k_ptr)++;
    BARRIER();
}
)HIP";

static const char* kCoopExpandRot = R"HIP(
DEV static void coop_expand_rot(GpuComplex* v, const uint64_t* px,
                                 uint32_t* active_k_ptr, uint32_t axis,
                                 double z_re, double z_im) {
    uint32_t k = *active_k_ptr;
    uint64_t half = 1ULL << k;
    bool pxv = bit_get(px, axis);
    double im = pxv ? -z_im : z_im;
    GpuComplex phase = {(float)z_re, (float)im};
    for (uint64_t i = TIDX; i < half; i += BSIZE)
        store_complex64_lds(v, i + half, cmul(v[i], phase));
    BARRIER();
    if (TIDX == 0) (*active_k_ptr)++;
    BARRIER();
}
)HIP";
