// array_ops.inc — Array sweep device functions (register + coop tiers)
// Included by emit_preamble.cc. Do not compile standalone.

static const char* kApplyPhase = R"HIP(
DEV static void apply_phase(ShotState& st, uint32_t axis, GpuComplex phase) {
    uint64_t axis_bit = 1ULL << axis;
    uint64_t iters = 1ULL << (st.active_k - 1);
    for (uint64_t i = 0; i < iters; ++i) {
        uint64_t idx = scatter_bits_1(i, axis) | axis_bit;
        st.v[idx] = cmul(st.v[idx], phase);
    }
}
)HIP";

static const char* kArrayCnot = R"HIP(
DEV static void array_cnot(ShotState& st, uint32_t c, uint32_t t) {
    uint64_t c_bit = 1ULL << c, t_bit = 1ULL << t;
    uint64_t iters = 1ULL << (st.active_k - 2);
    for (uint64_t i = 0; i < iters; ++i) {
        uint64_t base = scatter_bits_2(i, c, t) | c_bit;
        GpuComplex tmp = st.v[base];
        st.v[base] = st.v[base | t_bit];
        st.v[base | t_bit] = tmp;
    }
    frame_cnot(st, c, t);
}
)HIP";

static const char* kArrayCz = R"HIP(
DEV static void array_cz(ShotState& st, uint32_t a, uint32_t b) {
    uint64_t both_bits = (1ULL << a) | (1ULL << b);
    uint64_t iters = 1ULL << (st.active_k - 2);
    for (uint64_t i = 0; i < iters; ++i) {
        uint64_t idx = scatter_bits_2(i, a, b) | both_bits;
        st.v[idx].re = -st.v[idx].re; st.v[idx].im = -st.v[idx].im;
    }
    frame_cz(st, a, b);
}
)HIP";

static const char* kArraySwap = R"HIP(
DEV static void array_swap(ShotState& st, uint32_t a, uint32_t b) {
    uint64_t a_bit = 1ULL << a, b_bit = 1ULL << b;
    uint64_t iters = 1ULL << (st.active_k - 2);
    for (uint64_t i = 0; i < iters; ++i) {
        uint64_t base = scatter_bits_2(i, a, b);
        GpuComplex tmp = st.v[base | a_bit];
        st.v[base | a_bit] = st.v[base | b_bit];
        st.v[base | b_bit] = tmp;
    }
    frame_swap(st, a, b);
}
)HIP";

static const char* kArrayH = R"HIP(
DEV static void array_h(ShotState& st, uint32_t axis) {
    uint64_t axis_bit = 1ULL << axis;
    uint64_t iters = 1ULL << (st.active_k - 1);
    for (uint64_t i = 0; i < iters; ++i) {
        uint64_t idx0 = scatter_bits_1(i, axis), idx1 = idx0 | axis_bit;
        GpuComplex a = st.v[idx0], b = st.v[idx1];
        st.v[idx0] = cscale(cadd(a, b), kInvSqrt2);
        st.v[idx1] = cscale(csub(a, b), kInvSqrt2);
    }
    frame_h(st, axis);
}
)HIP";

static const char* kArrayS = R"HIP(
DEV static void array_s(ShotState& st, uint32_t axis, bool dagger) {
    apply_phase(st, axis, dagger ? (GpuComplex){0.0f, -1.0f} : (GpuComplex){0.0f, 1.0f});
    frame_s(st, axis);
}
)HIP";

static const char* kArrayT = R"HIP(
DEV static void array_t(ShotState& st, uint32_t axis, bool dagger) {
    bool px = bit_get(st.px, axis);
    if (axis >= st.active_k) return;
    double imag = dagger ? -kInvSqrt2 : kInvSqrt2;
    if (px) imag = -imag;
    apply_phase(st, axis, (GpuComplex){(float)kInvSqrt2, (float)imag});
}
)HIP";

static const char* kArrayRot = R"HIP(
DEV static void array_rot(ShotState& st, uint32_t axis, double z_re, double z_im) {
    if (axis < st.active_k) {
        bool px = bit_get(st.px, axis);
        double im = px ? -z_im : z_im;
        apply_phase(st, axis, (GpuComplex){(float)z_re, (float)im});
    }
}
)HIP";

// -----------------------------------------------------------------------
// Cooperative (LDS-tier) array sweep device functions
// -----------------------------------------------------------------------

static const char* kCoopArrayH = R"HIP(
DEV static void coop_array_h(GpuComplex* v, uint32_t active_k, uint32_t axis) {
    uint64_t axis_bit = 1ULL << axis;
    uint64_t iters = 1ULL << (active_k - 1);
    for (uint64_t i = TIDX; i < iters; i += BSIZE) {
        uint64_t idx0 = scatter_bits_1(i, axis);
        uint64_t idx1 = idx0 | axis_bit;
        GpuComplex a = v[idx0], b = v[idx1];
        store_complex64_lds(v, idx0, cscale(cadd(a, b), kInvSqrt2));
        store_complex64_lds(v, idx1, cscale(csub(a, b), kInvSqrt2));
    }
    BARRIER();
}
)HIP";

static const char* kCoopArrayCnot = R"HIP(
DEV static void coop_array_cnot(GpuComplex* v, uint64_t* px, uint64_t* pz,
                                uint32_t active_k, uint32_t c, uint32_t t) {
    uint64_t c_bit = 1ULL << c, t_bit = 1ULL << t;
    uint64_t iters = 1ULL << (active_k - 2);
    for (uint64_t i = TIDX; i < iters; i += BSIZE) {
        uint64_t base = scatter_bits_2(i, c, t) | c_bit;
        GpuComplex tmp = v[base];
        store_complex64_lds(v, base, v[base | t_bit]);
        store_complex64_lds(v, base | t_bit, tmp);
    }
    BARRIER();
    if (TIDX == 0) frame_cnot_raw(px, pz, c, t);
    BARRIER();
}
)HIP";

static const char* kCoopArrayCz = R"HIP(
DEV static void coop_array_cz(GpuComplex* v, uint64_t* px, uint64_t* pz,
                               uint32_t active_k, uint32_t a, uint32_t b) {
    uint64_t both_bits = (1ULL << a) | (1ULL << b);
    uint64_t iters = 1ULL << (active_k - 2);
    for (uint64_t i = TIDX; i < iters; i += BSIZE) {
        uint64_t idx = scatter_bits_2(i, a, b) | both_bits;
        GpuComplex c = v[idx];
        store_complex64_lds(v, idx, {-c.re, -c.im});
    }
    BARRIER();
    if (TIDX == 0) frame_cz_raw(px, pz, a, b);
    BARRIER();
}
)HIP";

static const char* kCoopArrayS = R"HIP(
DEV static void coop_array_s(GpuComplex* v, uint64_t* px, uint64_t* pz,
                              uint32_t active_k, uint32_t axis, bool dagger) {
    GpuComplex phase = dagger ? GpuComplex{0.0f, -1.0f} : GpuComplex{0.0f, 1.0f};
    uint64_t axis_bit = 1ULL << axis;
    uint64_t iters = 1ULL << (active_k - 1);
    for (uint64_t i = TIDX; i < iters; i += BSIZE) {
        uint64_t idx = scatter_bits_1(i, axis) | axis_bit;
        store_complex64_lds(v, idx, cmul(v[idx], phase));
    }
    BARRIER();
    if (TIDX == 0) frame_s_raw(px, pz, axis);
    BARRIER();
}
)HIP";

static const char* kCoopArrayT = R"HIP(
DEV static void coop_array_t(GpuComplex* v, const uint64_t* px, uint32_t active_k,
                              uint32_t axis, bool dagger) {
    if (axis >= active_k) return;
    bool pxv = bit_get(px, axis);
    double imag = dagger ? -kInvSqrt2 : kInvSqrt2;
    if (pxv) imag = -imag;
    GpuComplex phase = {(float)kInvSqrt2, (float)imag};
    uint64_t axis_bit = 1ULL << axis;
    uint64_t iters = 1ULL << (active_k - 1);
    for (uint64_t i = TIDX; i < iters; i += BSIZE) {
        uint64_t idx = scatter_bits_1(i, axis) | axis_bit;
        store_complex64_lds(v, idx, cmul(v[idx], phase));
    }
    BARRIER();
}
)HIP";

static const char* kCoopArraySwap = R"HIP(
DEV static void coop_array_swap(GpuComplex* v, uint64_t* px, uint64_t* pz,
                                uint32_t active_k, uint32_t a, uint32_t b) {
    uint64_t a_bit = 1ULL << a;
    uint64_t b_bit = 1ULL << b;
    uint64_t iters = 1ULL << (active_k - 2);
    for (uint64_t i = TIDX; i < iters; i += BSIZE) {
        uint64_t base = scatter_bits_2(i, a, b);
        GpuComplex tmp_a = v[base | a_bit];
        GpuComplex tmp_b = v[base | b_bit];
        store_complex64_lds(v, base | a_bit, tmp_b);
        store_complex64_lds(v, base | b_bit, tmp_a);
    }
    BARRIER();
    if (TIDX == 0) {
        bit_swap(px, a, px, b);
        bit_swap(pz, a, pz, b);
    }
    BARRIER();
}
)HIP";

static const char* kCoopArrayRot = R"HIP(
DEV static void coop_array_rot(GpuComplex* v, const uint64_t* px, uint32_t active_k,
                                uint32_t axis, double z_re, double z_im) {
    if (axis >= active_k) { BARRIER(); return; }
    bool pxv = bit_get(px, axis);
    double im = pxv ? -z_im : z_im;
    GpuComplex phase = {(float)z_re, (float)im};
    uint64_t axis_bit = 1ULL << axis;
    uint64_t iters = 1ULL << (active_k - 1);
    for (uint64_t i = TIDX; i < iters; i += BSIZE) {
        uint64_t idx = scatter_bits_1(i, axis) | axis_bit;
        store_complex64_lds(v, idx, cmul(v[idx], phase));
    }
    BARRIER();
}
)HIP";
