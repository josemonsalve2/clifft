// frame_ops.inc — Frame-only device functions (Pauli frame tracking)
// Included by emit_preamble.cc. Do not compile standalone.

static const char* kFrameOps = R"HIP(
DEV static inline __attribute__((always_inline)) void frame_cnot(ShotState& st, uint32_t c, uint32_t t) {
    bool px_c = bit_get(st.px, c), pz_t = bit_get(st.pz, t);
    bit_xor(st.px, t, px_c); bit_xor(st.pz, c, pz_t);
}
DEV static inline __attribute__((always_inline)) void frame_cz(ShotState& st, uint32_t c, uint32_t t) {
    bool px_c = bit_get(st.px, c), px_t = bit_get(st.px, t);
    bit_xor(st.pz, t, px_c); bit_xor(st.pz, c, px_t);
}
DEV static inline __attribute__((always_inline)) void frame_h(ShotState& st, uint32_t axis) {
    bool px = bit_get(st.px, axis), pz = bit_get(st.pz, axis);
    bit_set(st.px, axis, pz); bit_set(st.pz, axis, px);
}
DEV static inline __attribute__((always_inline)) void frame_s(ShotState& st, uint32_t axis) {
    bit_xor(st.pz, axis, bit_get(st.px, axis));
}
DEV static inline __attribute__((always_inline)) void frame_swap(ShotState& st, uint32_t a, uint32_t b) {
    bit_swap(st.px, a, st.px, b); bit_swap(st.pz, a, st.pz, b);
}
)HIP";

static const char* kApplyPauliToFrame = R"HIP(
DEV static void apply_pauli_to_frame(ShotState& st, const uint64_t* x, const uint64_t* z) {
    st.px[0] ^= x[0]; st.px[1] ^= x[1];
    st.pz[0] ^= z[0]; st.pz[1] ^= z[1];
}
)HIP";

static const char* kApplyPauli = R"HIP(
DEV static void apply_pauli(ShotState& st, const GpuMask* masks, uint32_t mask_idx,
                            uint32_t condition_idx) {
    if (st.meas[condition_idx] == 0) return;
    const GpuMask& mask = masks[mask_idx];
    apply_pauli_to_frame(st, mask.x, mask.z);
}
)HIP";

static const char* kCoopFrameRaw = R"HIP(
DEV static inline __attribute__((always_inline)) void frame_cnot_raw(uint64_t* px, uint64_t* pz, uint32_t c, uint32_t t) {
    bool px_c = bit_get(px, c), pz_t = bit_get(pz, t);
    bit_xor(px, t, px_c); bit_xor(pz, c, pz_t);
}
DEV static inline __attribute__((always_inline)) void frame_cz_raw(uint64_t* px, uint64_t* pz, uint32_t a, uint32_t b) {
    bool px_a = bit_get(px, a), px_b = bit_get(px, b);
    bit_xor(pz, b, px_a); bit_xor(pz, a, px_b);
}
DEV static inline __attribute__((always_inline)) void frame_h_raw(uint64_t* px, uint64_t* pz, uint32_t axis) {
    bool pxv = bit_get(px, axis), pzv = bit_get(pz, axis);
    bit_set(px, axis, pzv); bit_set(pz, axis, pxv);
}
DEV static inline __attribute__((always_inline)) void frame_s_raw(uint64_t* px, uint64_t* pz, uint32_t axis) {
    bit_xor(pz, axis, bit_get(px, axis));
}
DEV static inline __attribute__((always_inline)) void frame_swap_raw(uint64_t* px, uint64_t* pz, uint32_t a, uint32_t b) {
    bit_swap(px, a, px, b); bit_swap(pz, a, pz, b);
}
)HIP";
