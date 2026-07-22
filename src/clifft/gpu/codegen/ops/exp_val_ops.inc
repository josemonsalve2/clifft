// exp_val_ops.inc — Expectation value device functions (register + coop tiers)
// Included by emit_preamble.cc. Do not compile standalone.

static const char* kExpValMaskType = R"HIP(
struct GpuExpValMask { uint64_t x[2]; uint64_t z[2]; uint8_t sign; };
)HIP";

static const char* kExecExpVal = R"HIP(
DEV static void exec_exp_val_compiled(ShotState& st, const GpuExpValMask* masks,
                                      uint32_t cp_idx, uint32_t ev_idx) {
    const GpuExpValMask& mask = masks[cp_idx];
    bool frame_sign = mask.sign != 0;
    int parity = POPCOUNT64(mask.x[0] & st.pz[0]) + POPCOUNT64(mask.x[1] & st.pz[1])
               + POPCOUNT64(mask.z[0] & st.px[0]) + POPCOUNT64(mask.z[1] & st.px[1]);
    if (parity & 1) frame_sign = !frame_sign;
    uint32_t k = st.active_k;
    uint64_t active_mask_w0 = (k >= 64) ? ~uint64_t{0} : ((k == 0) ? 0 : ((uint64_t{1} << k) - 1));
    uint64_t dormant_x = mask.x[0] & ~active_mask_w0;
    if (k < 64) dormant_x |= mask.x[1];
    if (dormant_x != 0) { st.exp_vals[ev_idx] = 0.0; return; }
    uint64_t x_active = mask.x[0] & active_mask_w0;
    uint64_t z_active = mask.z[0] & active_mask_w0;
    uint32_t yz_count = POPCOUNT64(x_active & z_active);
    double phase_re = 0.0, phase_im = 0.0;
    switch (yz_count & 3) {
        case 0: phase_re = 1.0; break;
        case 1: phase_im = 1.0; break;
        case 2: phase_re = -1.0; break;
        case 3: phase_im = -1.0; break;
    }
    if (frame_sign) { phase_re = -phase_re; phase_im = -phase_im; }
    uint64_t dim = uint64_t{1} << k;
    double num_re = 0.0, num_im = 0.0, denom = 0.0;
    for (uint64_t j = 0; j < dim; ++j) {
        denom += cnorm(st.v[j]);
        uint64_t j_xor = j ^ x_active;
        double z_sign = (POPCOUNT64(j & z_active) & 1) ? -1.0 : 1.0;
        double a_re = (double)st.v[j_xor].re;
        double a_im = -(double)st.v[j_xor].im;
        num_re += (a_re * z_sign * st.v[j].re - a_im * z_sign * st.v[j].im);
        num_im += (a_re * z_sign * st.v[j].im + a_im * z_sign * st.v[j].re);
    }
    double result_re = num_re * phase_re - num_im * phase_im;
    st.exp_vals[ev_idx] = (denom > 0.0) ? result_re / denom : 0.0;
}
)HIP";

static const char* kCoopExecExpVal = R"HIP(
DEV static void coop_exec_exp_val(GpuComplex* v, const uint64_t* px, const uint64_t* pz,
                                   uint32_t active_k, double* exp_vals,
                                   const GpuExpValMask* masks,
                                   uint32_t cp_idx, uint32_t ev_idx) {
    if (TIDX == 0) {
        const GpuExpValMask& mask = masks[cp_idx];
        bool frame_sign = mask.sign != 0;
        int parity = POPCOUNT64(mask.x[0] & pz[0]) + POPCOUNT64(mask.x[1] & pz[1])
                   + POPCOUNT64(mask.z[0] & px[0]) + POPCOUNT64(mask.z[1] & px[1]);
        if (parity & 1) frame_sign = !frame_sign;

        uint32_t k = active_k;
        uint64_t active_mask_w0 = (k >= 64) ? ~uint64_t{0}
                                  : ((k == 0) ? uint64_t{0} : ((uint64_t{1} << k) - 1));
        uint64_t dormant_x = mask.x[0] & ~active_mask_w0;
        if (k < 64) dormant_x |= mask.x[1];
        if (dormant_x != 0) {
            exp_vals[ev_idx] = 0.0;
        } else {
            uint64_t x_active = mask.x[0] & active_mask_w0;
            uint64_t z_active = mask.z[0] & active_mask_w0;
            uint32_t yz_count = POPCOUNT64(x_active & z_active);
            double phase_re = 0.0, phase_im = 0.0;
            switch (yz_count & 3) {
                case 0: phase_re =  1.0; break;
                case 1: phase_im =  1.0; break;
                case 2: phase_re = -1.0; break;
                case 3: phase_im = -1.0; break;
            }
            if (frame_sign) { phase_re = -phase_re; phase_im = -phase_im; }
            uint64_t dim = uint64_t{1} << k;
            double num_re = 0.0, num_im = 0.0, denom = 0.0;
            for (uint64_t j = 0; j < dim; ++j) {
                GpuComplex vj = v[j];
                denom += cnorm(vj);
                uint64_t j_xor    = j ^ x_active;
                GpuComplex vj_xor = v[j_xor];
                double z_sign = (POPCOUNT64(j & z_active) & 1) ? -1.0 : 1.0;
                double a_re = (double)vj_xor.re;
                double a_im = -(double)vj_xor.im;
                num_re += (a_re * z_sign * vj.re - a_im * z_sign * vj.im);
                num_im += (a_re * z_sign * vj.im + a_im * z_sign * vj.re);
            }
            double result_re = num_re * phase_re - num_im * phase_im;
            exp_vals[ev_idx] = (denom > 0.0) ? result_re / denom : 0.0;
        }
    }
    BARRIER();
}
)HIP";
