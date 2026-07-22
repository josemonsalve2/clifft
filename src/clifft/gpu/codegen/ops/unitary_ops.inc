// unitary_ops.inc — Unitary gate device functions: U2/U4 types + array ops (register + coop tiers)
// Included by emit_preamble.cc. Do not compile standalone.

static const char* kFusedU2Type = R"HIP(
struct GpuFusedU2Entry {
    GpuComplex matrices[4][4];
    GpuComplex gamma_multipliers[4];
    uint8_t out_states[4];
};
)HIP";

static const char* kFusedU4Type = R"HIP(
struct GpuFusedU4Entry {
    struct Entry {
        GpuComplex matrix[4][4];
        GpuComplex gamma_multiplier;
        uint8_t out_state;
    };
    Entry entries[16];
};
)HIP";

static const char* kArrayU2 = R"HIP(
DEV static void array_u2_compiled(ShotState& st, uint32_t axis,
                                  const GpuFusedU2Entry* table, uint32_t cp_idx) {
    const GpuFusedU2Entry& entry = table[cp_idx];
    uint8_t in_state = (bit_get(st.pz, axis) ? 2 : 0) | (bit_get(st.px, axis) ? 1 : 0);
    uint8_t out = entry.out_states[in_state];
    bit_set(st.px, axis, (out & 1) != 0);
    bit_set(st.pz, axis, (out & 2) != 0);
    if (axis >= st.active_k) return;
    const GpuComplex* mat = entry.matrices[in_state];
    uint64_t axis_bit = 1ULL << axis;
    uint64_t iters = 1ULL << (st.active_k - 1);
    for (uint64_t i = 0; i < iters; ++i) {
        uint64_t idx0 = scatter_bits_1(i, axis);
        uint64_t idx1 = idx0 | axis_bit;
        GpuComplex a = st.v[idx0], b = st.v[idx1];
        st.v[idx0] = cadd(cmul(a, mat[0]), cmul(b, mat[1]));
        st.v[idx1] = cadd(cmul(a, mat[2]), cmul(b, mat[3]));
    }
}
)HIP";

static const char* kArrayU4 = R"HIP(
DEV static void array_u4_compiled(ShotState& st, uint32_t axis_lo, uint32_t axis_hi,
                                  const GpuFusedU4Entry* table, uint32_t cp_idx) {
    const GpuFusedU4Entry& node = table[cp_idx];
    uint8_t px_lo = bit_get(st.px, axis_lo) ? 1 : 0;
    uint8_t pz_lo = bit_get(st.pz, axis_lo) ? 1 : 0;
    uint8_t px_hi = bit_get(st.px, axis_hi) ? 1 : 0;
    uint8_t pz_hi = bit_get(st.pz, axis_hi) ? 1 : 0;
    uint8_t in_state = (pz_hi << 3) | (px_hi << 2) | (pz_lo << 1) | px_lo;
    const GpuFusedU4Entry::Entry& entry = node.entries[in_state];
    uint8_t out = entry.out_state;
    bit_set(st.px, axis_lo, (out & 1) != 0);
    bit_set(st.pz, axis_lo, (out & 2) != 0);
    bit_set(st.px, axis_hi, (out & 4) != 0);
    bit_set(st.pz, axis_hi, (out & 8) != 0);
    if (axis_hi >= st.active_k) return;
    uint64_t lo_bit = 1ULL << axis_lo;
    uint64_t hi_bit = 1ULL << axis_hi;
    uint64_t iters = 1ULL << (st.active_k - 2);
    for (uint64_t i = 0; i < iters; ++i) {
        uint64_t base = scatter_bits_2(i, axis_lo, axis_hi);
        uint64_t i0 = base;
        uint64_t i1 = base | lo_bit;
        uint64_t i2 = base | hi_bit;
        uint64_t i3 = base | lo_bit | hi_bit;
        GpuComplex v0 = st.v[i0], v1 = st.v[i1], v2 = st.v[i2], v3 = st.v[i3];
        st.v[i0] = cadd(cadd(cmul(v0, entry.matrix[0][0]), cmul(v1, entry.matrix[0][1])),
                        cadd(cmul(v2, entry.matrix[0][2]), cmul(v3, entry.matrix[0][3])));
        st.v[i1] = cadd(cadd(cmul(v0, entry.matrix[1][0]), cmul(v1, entry.matrix[1][1])),
                        cadd(cmul(v2, entry.matrix[1][2]), cmul(v3, entry.matrix[1][3])));
        st.v[i2] = cadd(cadd(cmul(v0, entry.matrix[2][0]), cmul(v1, entry.matrix[2][1])),
                        cadd(cmul(v2, entry.matrix[2][2]), cmul(v3, entry.matrix[2][3])));
        st.v[i3] = cadd(cadd(cmul(v0, entry.matrix[3][0]), cmul(v1, entry.matrix[3][1])),
                        cadd(cmul(v2, entry.matrix[3][2]), cmul(v3, entry.matrix[3][3])));
    }
}
)HIP";

static const char* kCoopArrayU2 = R"HIP(
DEV static void coop_array_u2(GpuComplex* v, uint64_t* px, uint64_t* pz,
                               uint32_t active_k, uint32_t axis,
                               const GpuFusedU2Entry* table, uint32_t cp_idx) {
    const GpuFusedU2Entry& entry = table[cp_idx];
    uint8_t in_state = (bit_get(pz, axis) ? 2 : 0) | (bit_get(px, axis) ? 1 : 0);
    uint8_t out = entry.out_states[in_state];
    const GpuComplex* mat = entry.matrices[in_state];
    if (TIDX == 0) {
        bit_set(px, axis, (out & 1) != 0);
        bit_set(pz, axis, (out & 2) != 0);
    }
    BARRIER();
    if (axis >= active_k) return;
    uint64_t axis_bit = 1ULL << axis;
    uint64_t iters = 1ULL << (active_k - 1);
    for (uint64_t i = TIDX; i < iters; i += BSIZE) {
        uint64_t idx0 = scatter_bits_1(i, axis);
        uint64_t idx1 = idx0 | axis_bit;
        GpuComplex a = v[idx0], b = v[idx1];
        store_complex64_lds(v, idx0, cadd(cmul(a, mat[0]), cmul(b, mat[1])));
        store_complex64_lds(v, idx1, cadd(cmul(a, mat[2]), cmul(b, mat[3])));
    }
    BARRIER();
}
)HIP";

static const char* kCoopArrayU4 = R"HIP(
DEV static void coop_array_u4(GpuComplex* v, uint64_t* px, uint64_t* pz,
                               uint32_t active_k, uint32_t axis_lo, uint32_t axis_hi,
                               const GpuFusedU4Entry* table, uint32_t cp_idx) {
    const GpuFusedU4Entry& node = table[cp_idx];
    uint8_t px_lo = bit_get(px, axis_lo) ? 1 : 0, pz_lo = bit_get(pz, axis_lo) ? 1 : 0;
    uint8_t px_hi = bit_get(px, axis_hi) ? 1 : 0, pz_hi = bit_get(pz, axis_hi) ? 1 : 0;
    uint8_t in_state = (pz_hi << 3) | (px_hi << 2) | (pz_lo << 1) | px_lo;
    const GpuFusedU4Entry::Entry& entry = node.entries[in_state];
    uint8_t out = entry.out_state;
    if (TIDX == 0) {
        bit_set(px, axis_lo, (out & 1) != 0); bit_set(pz, axis_lo, (out & 2) != 0);
        bit_set(px, axis_hi, (out & 4) != 0); bit_set(pz, axis_hi, (out & 8) != 0);
    }
    BARRIER();
    if (axis_hi >= active_k) return;
    uint64_t lo_bit = 1ULL << axis_lo, hi_bit = 1ULL << axis_hi;
    uint64_t iters = 1ULL << (active_k - 2);
    for (uint64_t i = TIDX; i < iters; i += BSIZE) {
        uint64_t base = scatter_bits_2(i, axis_lo, axis_hi);
        GpuComplex v0 = v[base], v1 = v[base|lo_bit], v2 = v[base|hi_bit], v3 = v[base|lo_bit|hi_bit];
        store_complex64_lds(v, base, cadd(cadd(cmul(v0,entry.matrix[0][0]),cmul(v1,entry.matrix[0][1])),cadd(cmul(v2,entry.matrix[0][2]),cmul(v3,entry.matrix[0][3]))));
        store_complex64_lds(v, base|lo_bit, cadd(cadd(cmul(v0,entry.matrix[1][0]),cmul(v1,entry.matrix[1][1])),cadd(cmul(v2,entry.matrix[1][2]),cmul(v3,entry.matrix[1][3]))));
        store_complex64_lds(v, base|hi_bit, cadd(cadd(cmul(v0,entry.matrix[2][0]),cmul(v1,entry.matrix[2][1])),cadd(cmul(v2,entry.matrix[2][2]),cmul(v3,entry.matrix[2][3]))));
        store_complex64_lds(v, base|lo_bit|hi_bit, cadd(cadd(cmul(v0,entry.matrix[3][0]),cmul(v1,entry.matrix[3][1])),cadd(cmul(v2,entry.matrix[3][2]),cmul(v3,entry.matrix[3][3]))));
    }
    BARRIER();
}
)HIP";
