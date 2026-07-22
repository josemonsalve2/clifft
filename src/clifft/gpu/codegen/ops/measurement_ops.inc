// measurement_ops.inc — Measurement device functions (register + coop tiers)
// Included by emit_preamble.cc. Do not compile standalone.

static const char* kPreambleSampleBranch = R"HIP(
DEV static inline __attribute__((always_inline)) uint8_t sample_branch(Rng& rng, double prob0, double prob1, double total) {
    double eps = kDustEpsilon * total;
    if (prob1 <= eps) return 0;
    if (prob0 <= eps) return 1;
    return (rng.uniform() * total < prob0) ? 0 : 1;
}
)HIP";

static const char* kMeasDormantRandom = R"HIP(
DEV static void meas_dormant_random(ShotState& st, Rng& rng, uint32_t axis,
                                    uint32_t classical_idx, bool sign) {
    uint8_t m_abs = rng.uniform() < 0.5 ? 0 : 1;
    bit_set(st.px, axis, m_abs != 0);
    bit_set(st.pz, axis, false);
    st.meas[classical_idx] = m_abs ^ (uint8_t)sign;
}
)HIP";

static const char* kMeasActiveDiag = R"HIP(
DEV static void meas_active_diagonal(ShotState& st, Rng& rng, uint32_t axis,
                                     uint32_t classical_idx, bool sign) {
    uint64_t half = 1ULL << (st.active_k - 1);
    bool px = bit_get(st.px, axis);
    double p0 = 0.0, p1 = 0.0;
    for (uint64_t i = 0; i < half; ++i) { p0 += cnorm(st.v[i]); p1 += cnorm(st.v[i + half]); }
    uint8_t b = sample_branch(rng, p0, p1, p0 + p1);
    uint8_t m_abs = b ^ (uint8_t)px;
    st.meas[classical_idx] = m_abs ^ (uint8_t)sign;
    if (b != 0) { for (uint64_t i = 0; i < half; ++i) st.v[i] = st.v[i + half]; }
    st.active_k--;
    bit_set(st.px, axis, m_abs != 0);
    bit_set(st.pz, axis, false);
}
)HIP";

static const char* kMeasActiveInterfere = R"HIP(
DEV static void meas_active_interfere(ShotState& st, Rng& rng, uint32_t axis,
                                      uint32_t classical_idx, bool sign) {
    uint64_t half = 1ULL << (st.active_k - 1);
    bool pz = bit_get(st.pz, axis);
    double p_plus = 0.0, p_minus = 0.0;
    for (uint64_t i = 0; i < half; ++i) {
        GpuComplex sum = cadd(st.v[i], st.v[i + half]);
        GpuComplex diff = csub(st.v[i], st.v[i + half]);
        p_plus += cnorm(sum); p_minus += cnorm(diff);
    }
    uint8_t b_x = sample_branch(rng, p_plus, p_minus, p_plus + p_minus);
    uint8_t m_abs = b_x ^ (uint8_t)pz;
    st.meas[classical_idx] = m_abs ^ (uint8_t)sign;
    for (uint64_t i = 0; i < half; ++i) {
        GpuComplex folded = (b_x == 0) ? cadd(st.v[i], st.v[i + half])
                                       : csub(st.v[i], st.v[i + half]);
        st.v[i] = cscale(folded, kInvSqrt2);
    }
    st.active_k--;
    bit_set(st.px, axis, m_abs != 0);
    bit_set(st.pz, axis, false);
}
)HIP";

static const char* kSwapMeasInterfere = R"HIP(
DEV static void swap_meas_interfere(ShotState& st, Rng& rng, uint32_t from, uint32_t to,
                                    uint32_t classical_idx, bool sign) {
    if (from == to) { meas_active_interfere(st, rng, to, classical_idx, sign); return; }
    frame_swap(st, from, to);
    bool pz = bit_get(st.pz, to);
    uint64_t half = 1ULL << to;
    uint64_t f_bit = 1ULL << from;
    double p_plus = 0.0, p_minus = 0.0;
    for (uint64_t idx = 0; idx < half; ++idx) {
        uint64_t b_f = (idx >> from) & 1ULL;
        uint64_t base = (idx & ~f_bit) | (b_f << to);
        GpuComplex sum = cadd(st.v[base], st.v[base | f_bit]);
        GpuComplex diff = csub(st.v[base], st.v[base | f_bit]);
        p_plus += cnorm(sum); p_minus += cnorm(diff);
    }
    uint8_t b_x = sample_branch(rng, p_plus, p_minus, p_plus + p_minus);
    for (uint64_t idx = 0; idx < half; ++idx) {
        uint64_t b_f = (idx >> from) & 1ULL;
        uint64_t base = (idx & ~f_bit) | (b_f << to);
        st.v[idx] = cscale((b_x == 0) ? cadd(st.v[base], st.v[base | f_bit])
                                      : csub(st.v[base], st.v[base | f_bit]),
                           kInvSqrt2);
    }
    st.active_k--;
    uint8_t m_abs = b_x ^ (uint8_t)pz;
    st.meas[classical_idx] = m_abs ^ (uint8_t)sign;
    bit_set(st.px, to, m_abs != 0);
    bit_set(st.pz, to, false);
}
)HIP";

static const char* kCoopMeasActiveDiag = R"HIP(
DEV static void coop_meas_active_diagonal(GpuComplex* v, uint64_t* px, uint64_t* pz,
                                          uint32_t* active_k_ptr, uint8_t* meas,
                                          double* red0, double* red1,
                                          Rng& rng, uint32_t axis, uint32_t classical_idx,
                                          bool sign) {
    uint32_t active_k = *active_k_ptr;
    bool pxv = bit_get(px, axis);
    uint64_t half = 1ULL << (active_k - 1);
    double p0_local = 0.0, p1_local = 0.0;
    for (uint64_t i = TIDX; i < half; i += BSIZE) {
        p0_local += cnorm(v[i]);
        p1_local += cnorm(v[i + half]);
    }
    double p0_out = 0.0, p1_out = 0.0;
    coop_reduce2(p0_local, p1_local, red0, red1, p0_out, p1_out);

    LDS uint8_t branch_val;
    if (TIDX == 0) {
        branch_val = sample_branch(rng, p0_out, p1_out, p0_out + p1_out);
    }
    BARRIER();
    uint8_t b = branch_val;

    if (b != 0) {
        for (uint64_t i = TIDX; i < half; i += BSIZE)
            store_complex64_lds(v, i, v[i + half]);
    }
    BARRIER();
    if (TIDX == 0) {
        uint8_t m_abs = b ^ (uint8_t)pxv;
        meas[classical_idx] = m_abs ^ (uint8_t)sign;
        (*active_k_ptr)--;
        bit_set(px, axis, m_abs != 0);
        bit_set(pz, axis, false);
    }
    BARRIER();
}
)HIP";

static const char* kCoopMeasActiveInterfere = R"HIP(
DEV static void coop_meas_active_interfere(GpuComplex* v, uint64_t* px, uint64_t* pz,
                                           uint32_t* active_k_ptr, uint8_t* meas,
                                           double* red0, double* red1,
                                           Rng& rng, uint32_t axis,
                                           uint32_t classical_idx, bool sign) {
    uint32_t active_k = *active_k_ptr;
    bool pzv = bit_get(pz, axis);
    uint64_t half = 1ULL << (active_k - 1);
    double lp = 0.0, lm = 0.0;
    for (uint64_t i = TIDX; i < half; i += BSIZE) {
        GpuComplex vi = v[i];
        GpuComplex vh = v[i + half];
        GpuComplex sum  = cadd(vi, vh);
        GpuComplex diff = csub(vi, vh);
        lp += cnorm(sum);
        lm += cnorm(diff);
    }
    double p_plus = 0.0, p_minus = 0.0;
    coop_reduce2(lp, lm, red0, red1, p_plus, p_minus);

    LDS uint8_t branch_val;
    if (TIDX == 0) {
        branch_val = sample_branch(rng, p_plus, p_minus, p_plus + p_minus);
    }
    BARRIER();
    uint8_t b = branch_val;

    for (uint64_t i = TIDX; i < half; i += BSIZE) {
        GpuComplex vi = v[i];
        GpuComplex vh = v[i + half];
        GpuComplex folded = (b == 0) ? cadd(vi, vh) : csub(vi, vh);
        store_complex64_lds(v, i, cscale(folded, kInvSqrt2));
    }
    BARRIER();
    if (TIDX == 0) {
        uint8_t m_abs = b ^ (uint8_t)pzv;
        meas[classical_idx] = m_abs ^ (uint8_t)sign;
        (*active_k_ptr)--;
        bit_set(px, axis, m_abs != 0);
        bit_set(pz, axis, false);
    }
    BARRIER();
}
)HIP";

static const char* kCoopSwapMeasInterfere = R"HIP(
DEV static void coop_swap_meas_interfere(GpuComplex* v, GpuComplex* scratch,
                                         uint64_t* px, uint64_t* pz,
                                         uint32_t* active_k_ptr, uint8_t* meas,
                                         double* red0, double* red1,
                                         Rng& rng, uint32_t from, uint32_t to,
                                         uint32_t classical_idx, bool sign) {
    if (from == to) {
        coop_meas_active_interfere(v, px, pz, active_k_ptr, meas, red0, red1,
                                   rng, to, classical_idx, sign);
        return;
    }
    if (TIDX == 0) {
        bit_swap(px, from, px, to);
        bit_swap(pz, from, pz, to);
    }
    BARRIER();
    bool pzv = bit_get(pz, to);
    uint64_t half  = 1ULL << to;
    uint64_t f_bit = 1ULL << from;
    double lp = 0.0, lm = 0.0;
    for (uint64_t idx = TIDX; idx < half; idx += BSIZE) {
        uint64_t b_f  = (idx >> from) & 1ULL;
        uint64_t base = (idx & ~f_bit) | (b_f << to);
        GpuComplex vb = v[base];
        GpuComplex vf = v[base | f_bit];
        GpuComplex sum  = cadd(vb, vf);
        GpuComplex diff = csub(vb, vf);
        lp += cnorm(sum);
        lm += cnorm(diff);
    }
    double p_plus = 0.0, p_minus = 0.0;
    coop_reduce2(lp, lm, red0, red1, p_plus, p_minus);

    LDS uint8_t branch_val;
    if (TIDX == 0) {
        branch_val = sample_branch(rng, p_plus, p_minus, p_plus + p_minus);
    }
    BARRIER();
    uint8_t b = branch_val;

    for (uint64_t idx = TIDX; idx < half; idx += BSIZE) {
        uint64_t b_f  = (idx >> from) & 1ULL;
        uint64_t base = (idx & ~f_bit) | (b_f << to);
        GpuComplex vb = v[base];
        GpuComplex vf = v[base | f_bit];
        store_complex64_lds(scratch, idx,
            cscale((b == 0) ? cadd(vb, vf) : csub(vb, vf), kInvSqrt2));
    }
    BARRIER();
    for (uint64_t idx = TIDX; idx < half; idx += BSIZE)
        store_complex64_lds(v, idx, scratch[idx]);
    BARRIER();
    if (TIDX == 0) {
        (*active_k_ptr)--;
        uint8_t m_abs = b ^ (uint8_t)pzv;
        meas[classical_idx] = m_abs ^ (uint8_t)sign;
        bit_set(px, to, m_abs != 0);
        bit_set(pz, to, false);
    }
    BARRIER();
}
)HIP";
