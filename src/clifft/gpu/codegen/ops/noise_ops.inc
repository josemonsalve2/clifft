// noise_ops.inc — Noise device functions and type declarations (register + coop tiers)
// Included by emit_preamble.cc. Do not compile standalone.

static const char* kNoiseTypes = R"HIP(
struct GpuChannel { uint64_t x[2]; uint64_t z[2]; double prob; };
struct GpuNoiseSite  { uint32_t offset; uint32_t count; double prob_sum; };
)HIP";

static const char* kReadoutNoiseType = R"HIP(
struct GpuReadoutNoise { uint32_t meas_idx; double prob; };
)HIP";

static const char* kPauliMaskType = R"HIP(
struct GpuMask { uint64_t x[2]; uint64_t z[2]; uint8_t sign; };
)HIP";

static const char* kNoiseHelper = R"HIP(
DEV static void draw_next_noise_compiled(ShotState& st, Rng& rng,
                                         const double* noise_hazards,
                                         uint32_t num_noise_sites) {
    if (num_noise_sites == 0 || st.next_noise_idx >= num_noise_sites) {
        st.next_noise_idx = 0xffffffffu;
        return;
    }
    double current_hazard = (st.next_noise_idx == 0) ? 0.0
                            : noise_hazards[st.next_noise_idx - 1];
    double u = rng.uniform();
    double gap = -log(1.0 - u);
    double target = current_hazard + gap;
    uint32_t lo = 0, hi = num_noise_sites;
    while (lo < hi) {
        uint32_t mid = lo + ((hi - lo) >> 1);
        if (noise_hazards[mid] <= target) lo = mid + 1; else hi = mid;
    }
    st.next_noise_idx = (lo >= num_noise_sites) ? 0xffffffffu : lo;
}
)HIP";

static const char* kCoopNoiseHelper = R"HIP(
DEV static void coop_draw_next_noise(uint32_t* next_noise_idx, Rng& rng,
                                     const double* noise_hazards,
                                     uint32_t num_noise_sites) {
    if (num_noise_sites == 0 || *next_noise_idx >= num_noise_sites) {
        *next_noise_idx = 0xffffffffu;
        return;
    }
    double current_hazard = (*next_noise_idx == 0) ? 0.0
                            : noise_hazards[*next_noise_idx - 1];
    double u = rng.uniform();
    double gap = -log(1.0 - u);
    double target = current_hazard + gap;
    uint32_t lo = 0, hi = num_noise_sites;
    while (lo < hi) {
        uint32_t mid = lo + ((hi - lo) >> 1);
        if (noise_hazards[mid] <= target) lo = mid + 1; else hi = mid;
    }
    *next_noise_idx = (lo >= num_noise_sites) ? 0xffffffffu : lo;
}
)HIP";
