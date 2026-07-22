// multi_qubit_ops.inc — Multi-controlled array sweep device functions (register + coop tiers)
// Included by emit_preamble.cc. Do not compile standalone.

static const char* kArrayMultiCnot = R"HIP(
DEV static void array_multi_cnot(ShotState& st, uint32_t target, uint64_t ctrl_mask) {
    uint64_t t_bit = 1ULL << target;
    uint64_t half = 1ULL << (st.active_k - 1);
    for (uint64_t idx = 0; idx < half; ++idx) {
        uint64_t actual = scatter_bits_1(idx, target);
        if ((POPCOUNT64(actual & ctrl_mask) & 1) != 0) {
            GpuComplex tmp = st.v[actual];
            st.v[actual] = st.v[actual | t_bit];
            st.v[actual | t_bit] = tmp;
        }
    }
    for (uint32_t c = 0; c < st.active_k; ++c) {
        if ((ctrl_mask >> c) & 1ULL) frame_cnot(st, c, target);
    }
}
)HIP";

static const char* kArrayMultiCz = R"HIP(
DEV static void array_multi_cz(ShotState& st, uint32_t control, uint64_t target_mask) {
    uint64_t c_bit = 1ULL << control;
    uint64_t half = 1ULL << (st.active_k - 1);
    for (uint64_t idx = 0; idx < half; ++idx) {
        uint64_t actual = scatter_bits_1(idx, control) | c_bit;
        if ((POPCOUNT64(actual & target_mask) & 1) != 0) {
            st.v[actual].re = -st.v[actual].re; st.v[actual].im = -st.v[actual].im;
        }
    }
    for (uint32_t t = 0; t < st.active_k; ++t) {
        if ((target_mask >> t) & 1ULL) frame_cz(st, control, t);
    }
}
)HIP";

// -----------------------------------------------------------------------
// Cooperative (LDS-tier) multi-qubit device functions
// -----------------------------------------------------------------------

static const char* kCoopArrayMultiCnot = R"HIP(
DEV static void coop_array_multi_cnot(GpuComplex* v, uint64_t* px, uint64_t* pz,
                                      uint32_t active_k, uint32_t target,
                                      uint64_t ctrl_mask) {
    uint64_t t_bit = 1ULL << target;
    uint64_t half = 1ULL << (active_k - 1);
    for (uint64_t idx = TIDX; idx < half; idx += BSIZE) {
        uint64_t actual = scatter_bits_1(idx, target);
        bool parity = (POPCOUNT64(actual & ctrl_mask) & 1) != 0;
        if (parity) {
            GpuComplex tmp  = v[actual];
            GpuComplex tmp2 = v[actual | t_bit];
            store_complex64_lds(v, actual,         tmp2);
            store_complex64_lds(v, actual | t_bit, tmp);
        }
    }
    BARRIER();
    if (TIDX == 0) {
        for (uint32_t c = 0; c < active_k; ++c) {
            if ((ctrl_mask >> c) & 1ULL) {
                bool px_c = bit_get(px, c);
                bool pz_t = bit_get(pz, target);
                bit_xor(px, target, px_c);
                bit_xor(pz, c,      pz_t);
            }
        }
    }
    BARRIER();
}
)HIP";

static const char* kCoopArrayMultiCz = R"HIP(
DEV static void coop_array_multi_cz(GpuComplex* v, uint64_t* px, uint64_t* pz,
                                    uint32_t active_k, uint32_t control,
                                    uint64_t target_mask) {
    uint64_t c_bit = 1ULL << control;
    uint64_t half = 1ULL << (active_k - 1);
    for (uint64_t idx = TIDX; idx < half; idx += BSIZE) {
        uint64_t actual = scatter_bits_1(idx, control) | c_bit;
        bool negate = (POPCOUNT64(actual & target_mask) & 1) != 0;
        if (negate) {
            GpuComplex val = v[actual];
            store_complex64_lds(v, actual, {-val.re, -val.im});
        }
    }
    BARRIER();
    if (TIDX == 0) {
        for (uint32_t t = 0; t < active_k; ++t) {
            if ((target_mask >> t) & 1ULL) {
                bool px_c = bit_get(px, control);
                bool px_t = bit_get(px, t);
                bit_xor(pz, t,       px_c);
                bit_xor(pz, control, px_t);
            }
        }
    }
    BARRIER();
}
)HIP";
