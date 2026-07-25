// v2_ops_body.inc — the per-opcode operand functions. Included by v2_ops.h.
//
// Each v2_op_* is the EXACT body of the corresponding interpreter case, with
// operands (axes, flags, table indices) and `active_k` passed by value. The
// interpreter passes runtime values; the specializer passes constants that fold.
// Byte-exact with SVM is inherited from these bodies being the single source.
//
// Convention: `t` = v2_tid(), `active_k` = st->active_k BEFORE the op. Ops that
// change rank update st->active_k themselves (so the caller re-reads it).

// ---- frame ops (tid0-only, O(1)) --------------------------------------------
static inline void v2_op_frame_cnot(V2State* st, u32 ctrl, u32 tgt) {
    u32 t = v2_tid();
    if (IS_OWNER) {
        int px_c = fget(st->px, ctrl), pz_tt = fget(st->pz, tgt);
        fxor(st->px, tgt, px_c); fxor(st->pz, ctrl, pz_tt);
    }
    v2_barrier();
}
static inline void v2_op_frame_cz(V2State* st, u32 a, u32 b) {
    u32 t = v2_tid();
    if (IS_OWNER) {
        int px_a = fget(st->px, a), px_b = fget(st->px, b);
        fxor(st->pz, b, px_a); fxor(st->pz, a, px_b);
    }
    v2_barrier();
}
static inline void v2_op_frame_h(V2State* st, u32 axis) {
    u32 t = v2_tid();
    if (IS_OWNER) {
        int px = fget(st->px, axis), pz = fget(st->pz, axis);
        fset(st->px, axis, pz); fset(st->pz, axis, px);
    }
    v2_barrier();
}
static inline void v2_op_frame_s(V2State* st, u32 axis) {
    u32 t = v2_tid();
    if (IS_OWNER) { int px = fget(st->px, axis); fxor(st->pz, axis, px); }
    v2_barrier();
}
static inline void v2_op_frame_swap(V2State* st, u32 a, u32 b) {
    u32 t = v2_tid();
    if (IS_OWNER) {
        int px_a = fget(st->px, a), px_b = fget(st->px, b);
        fset(st->px, a, px_b); fset(st->px, b, px_a);
        int pz_a = fget(st->pz, a), pz_b = fget(st->pz, b);
        fset(st->pz, a, pz_b); fset(st->pz, b, pz_a);
    }
    v2_barrier();
}

// ---- dormant measurements (tid0-only) ---------------------------------------
static inline void v2_op_meas_dormant_static(V2State* st, u32 axis, u32 slot, u8 flags) {
    u32 t = v2_tid();
    if (IS_OWNER) {
        u8 mval;
        if (flags & FLAG_IDENTITY) mval = (flags & FLAG_SIGN) ? 1u : 0u;
        else {
            u8 outcome = (u8)fget(st->px, axis);
            mval = outcome ^ (u8)((flags & FLAG_SIGN) != 0);
        }
        if (slot < V2_MAX_MEAS) mset(st->meas, slot, mval);
    }
    v2_barrier();
}
static inline void v2_op_meas_dormant_random(V2State* st, u32 axis, u32 slot, u8 flags) {
    u32 t = v2_tid();
    if (IS_OWNER) {
        u8 m_abs = (rng_uniform(st->rng) < 0.5) ? 0u : 1u;
        fset(st->px, axis, m_abs != 0); fset(st->pz, axis, 0);
        if (slot < V2_MAX_MEAS) mset(st->meas, slot, m_abs ^ (u8)((flags & FLAG_SIGN) != 0));
    }
    v2_barrier();
}

// ---- expand family (grows k) ------------------------------------------------
static inline void v2_op_expand(V2State* st, CV2Complex* v, u32 active_k) {
    u32 t = v2_tid();
    u32 half = 1u << active_k;
    for (u32 i = t; i < half; i += V2_STRIDE) v[i + half] = v[i];
    v2_barrier();
    if (IS_OWNER) st->active_k = active_k + 1;
    v2_barrier();
}
static inline void v2_op_expand_t(V2State* st, CV2Complex* v, u32 active_k, u32 axis, int dagger) {
    u32 t = v2_tid();
    u32 half = 1u << active_k;
    int px = fget(st->px, axis);
    double imag = dagger ? -V2_INV_SQRT2 : V2_INV_SQRT2;
    if (px) imag = -imag;
    CV2Complex phase; phase.re = (float)V2_INV_SQRT2; phase.im = (float)imag;
    for (u32 i = t; i < half; i += V2_STRIDE) v[i + half] = cmul(v[i], phase);
    v2_barrier();
    if (IS_OWNER) st->active_k = active_k + 1;
    v2_barrier();
}
static inline void v2_op_expand_rot(V2State* st, CV2Complex* v, u32 active_k,
                                    u32 axis, double wre, double wim) {
    u32 t = v2_tid();
    u32 half = 1u << active_k;
    int px = fget(st->px, axis);
    double im = px ? -wim : wim;
    CV2Complex phase; phase.re = (float)wre; phase.im = (float)im;
    for (u32 i = t; i < half; i += V2_STRIDE) v[i + half] = cmul(v[i], phase);
    v2_barrier();
    if (IS_OWNER) st->active_k = active_k + 1;
    v2_barrier();
}

// ---- active measurements (shrinks k; cooperative reduction) -----------------
static inline void v2_op_meas_active_diagonal(V2State* st, CV2Complex* v, u32 active_k,
                                              u32 axis, u32 slot, u8 flags) {
    u32 t = v2_tid();
    u32 half = 1u << (active_k - 1u);
    int px = fget(st->px, axis);
    double l0 = 0.0, l1 = 0.0;
    for (u32 i = t; i < half; i += V2_STRIDE) { l0 += cnorm(v[i]); l1 += cnorm(v[i + half]); }
    double p0, p1; V2_REDUCE2(t, l0, l1, &p0, &p1);
    if (IS_OWNER) {
        u8 b = sample_branch(st->rng, p0, p1, p0 + p1);
        st->branch = b;
        u8 m_abs = b ^ (u8)px;
        if (slot < V2_MAX_MEAS) mset(st->meas, slot, m_abs ^ (u8)((flags & FLAG_SIGN) != 0));
    }
    v2_barrier();
    if (st->branch != 0) { for (u32 i = t; i < half; i += V2_STRIDE) v[i] = v[i + half]; }
    v2_barrier();
    if (IS_OWNER) {
        st->active_k = active_k - 1;
        u8 m_abs = mget(st->meas, slot) ^ (u8)((flags & FLAG_SIGN) != 0);
        fset(st->px, axis, m_abs != 0); fset(st->pz, axis, 0);
    }
    v2_barrier();
}
static inline void v2_op_meas_active_interfere(V2State* st, CV2Complex* v, u32 active_k,
                                               u32 axis, u32 slot, u8 flags) {
    u32 t = v2_tid();
    u32 half = 1u << (active_k - 1u);
    int pz = fget(st->pz, axis);
    double lp = 0.0, lm = 0.0;
    for (u32 i = t; i < half; i += V2_STRIDE) {
        CV2Complex vi = v[i], vh = v[i + half];
        lp += cnorm(cadd(vi, vh)); lm += cnorm(csub(vi, vh));
    }
    double p_plus, p_minus; V2_REDUCE2(t, lp, lm, &p_plus, &p_minus);
    if (IS_OWNER) {
        u8 b = sample_branch(st->rng, p_plus, p_minus, p_plus + p_minus);
        st->branch = b;
        u8 m_abs = b ^ (u8)pz;
        if (slot < V2_MAX_MEAS) mset(st->meas, slot, m_abs ^ (u8)((flags & FLAG_SIGN) != 0));
    }
    v2_barrier();
    for (u32 i = t; i < half; i += V2_STRIDE) {
        CV2Complex vi = v[i], vh = v[i + half];
        CV2Complex folded = (st->branch == 0) ? cadd(vi, vh) : csub(vi, vh);
        v[i] = cscale(folded, V2_INV_SQRT2);
    }
    v2_barrier();
    if (IS_OWNER) {
        st->active_k = active_k - 1;
        u8 m_abs = mget(st->meas, slot) ^ (u8)((flags & FLAG_SIGN) != 0);
        fset(st->px, axis, m_abs != 0); fset(st->pz, axis, 0);
    }
    v2_barrier();
}

// ---- array Clifford ops -----------------------------------------------------
static inline void v2_op_array_cnot(V2State* st, CV2Complex* v, u32 active_k, u32 c, u32 tg) {
    u32 t = v2_tid();
    u64 c_bit = 1ull << c, t_bit = 1ull << tg;
    u64 iters = 1ull << (active_k - 2u);
    for (u64 i = t; i < iters; i += V2_STRIDE) {
        u64 base = scatter_bits_2(i, c, tg) | c_bit;
        CV2Complex a = v[base], b = v[base | t_bit];
        v[base] = b; v[base | t_bit] = a;
    }
    v2_barrier();
    if (IS_OWNER) { int px_c = fget(st->px, c), pz_t = fget(st->pz, tg); fxor(st->px, tg, px_c); fxor(st->pz, c, pz_t); }
    v2_barrier();
}
static inline void v2_op_array_cz(V2State* st, CV2Complex* v, u32 active_k, u32 a, u32 b) {
    u32 t = v2_tid();
    u64 both = (1ull << a) | (1ull << b);
    u64 iters = 1ull << (active_k - 2u);
    for (u64 i = t; i < iters; i += V2_STRIDE) {
        u64 idx = scatter_bits_2(i, a, b) | both;
        CV2Complex c = v[idx]; c.re = -c.re; c.im = -c.im; v[idx] = c;
    }
    v2_barrier();
    if (IS_OWNER) { int px_a = fget(st->px, a), px_b = fget(st->px, b); fxor(st->pz, b, px_a); fxor(st->pz, a, px_b); }
    v2_barrier();
}
static inline void v2_op_array_swap(V2State* st, CV2Complex* v, u32 active_k, u32 a, u32 b) {
    u32 t = v2_tid();
    u64 a_bit = 1ull << a, b_bit = 1ull << b;
    u64 iters = 1ull << (active_k - 2u);
    for (u64 i = t; i < iters; i += V2_STRIDE) {
        u64 base = scatter_bits_2(i, a, b);
        CV2Complex ta = v[base | a_bit], tb = v[base | b_bit];
        v[base | a_bit] = tb; v[base | b_bit] = ta;
    }
    v2_barrier();
    if (IS_OWNER) { fswap(st->px, a, b); fswap(st->pz, a, b); }
    v2_barrier();
}
static inline void v2_op_array_multi_cnot(V2State* st, CV2Complex* v, u32 active_k, u32 tg, u64 ctrl_mask) {
    u32 t = v2_tid();
    u64 t_bit = 1ull << tg;
    u64 half = 1ull << (active_k - 1u);
    for (u64 idx = t; idx < half; idx += V2_STRIDE) {
        u64 actual = scatter_bits_1(idx, tg);
        if (__builtin_popcountll(actual & ctrl_mask) & 1) {
            CV2Complex a = v[actual], b = v[actual | t_bit];
            v[actual] = b; v[actual | t_bit] = a;
        }
    }
    v2_barrier();
    if (IS_OWNER) {
        for (u32 c = 0; c < active_k; ++c) if ((ctrl_mask >> c) & 1ull) {
            int px_c = fget(st->px, c), pz_t = fget(st->pz, tg);
            fxor(st->px, tg, px_c); fxor(st->pz, c, pz_t);
        }
    }
    v2_barrier();
}
static inline void v2_op_array_multi_cz(V2State* st, CV2Complex* v, u32 active_k, u32 ctrl, u64 target_mask) {
    u32 t = v2_tid();
    u64 c_bit = 1ull << ctrl;
    u64 half = 1ull << (active_k - 1u);
    for (u64 idx = t; idx < half; idx += V2_STRIDE) {
        u64 actual = scatter_bits_1(idx, ctrl) | c_bit;
        if (__builtin_popcountll(actual & target_mask) & 1) {
            CV2Complex vv = v[actual]; vv.re = -vv.re; vv.im = -vv.im; v[actual] = vv;
        }
    }
    v2_barrier();
    if (IS_OWNER) {
        for (u32 tg = 0; tg < active_k; ++tg) if ((target_mask >> tg) & 1ull) {
            int px_c = fget(st->px, ctrl), px_t = fget(st->px, tg);
            fxor(st->pz, tg, px_c); fxor(st->pz, ctrl, px_t);
        }
    }
    v2_barrier();
}
static inline void v2_op_array_h(V2State* st, CV2Complex* v, u32 active_k, u32 axis) {
    u32 t = v2_tid();
    u64 axis_bit = 1ull << axis;
    u64 iters = 1ull << (active_k - 1u);
    for (u64 i = t; i < iters; i += V2_STRIDE) {
        u64 i0 = scatter_bits_1(i, axis), i1 = i0 | axis_bit;
        CV2Complex a = v[i0], b = v[i1];
        v[i0] = cscale(cadd(a, b), V2_INV_SQRT2);
        v[i1] = cscale(csub(a, b), V2_INV_SQRT2);
    }
    v2_barrier();
    if (IS_OWNER) { int px = fget(st->px, axis), pz = fget(st->pz, axis); fset(st->px, axis, pz); fset(st->pz, axis, px); }
    v2_barrier();
}
static inline void v2_op_array_s(V2State* st, CV2Complex* v, u32 active_k, u32 axis, int dagger) {
    u32 t = v2_tid();
    CV2Complex ph; ph.re = 0.0f; ph.im = dagger ? -1.0f : 1.0f;
    coop_apply_phase(t, v, active_k, axis, ph);
    if (IS_OWNER) { int px = fget(st->px, axis); fxor(st->pz, axis, px); }
    v2_barrier();
}
static inline void v2_op_array_t(V2State* st, CV2Complex* v, u32 active_k, u32 axis, int dagger) {
    u32 t = v2_tid();
    if (axis >= active_k) { v2_barrier(); return; }
    int px = fget(st->px, axis);
    double imag = dagger ? -V2_INV_SQRT2 : V2_INV_SQRT2;
    if (px) imag = -imag;
    CV2Complex phase; phase.re = (float)V2_INV_SQRT2; phase.im = (float)imag;
    coop_apply_phase(t, v, active_k, axis, phase);
}
static inline void v2_op_array_rot(V2State* st, CV2Complex* v, u32 active_k, u32 axis, double wre, double wim) {
    u32 t = v2_tid();
    if (axis < active_k) {
        int px = fget(st->px, axis);
        double im = px ? -wim : wim;
        CV2Complex phase; phase.re = (float)wre; phase.im = (float)im;
        coop_apply_phase(t, v, active_k, axis, phase);
    } else v2_barrier();
}
static inline void v2_op_array_u2(V2State* st, CV2Complex* v, u32 active_k, u32 axis,
                                  const CV2FusedU2Entry* fused_u2, u32 cp) {
    u32 t = v2_tid();
    int in_state = (fget(st->pz, axis) ? 2 : 0) | (fget(st->px, axis) ? 1 : 0);
    const CV2Complex* mat = fused_u2[cp].matrices[in_state];
    if (axis < active_k) {
        u64 axis_bit = 1ull << axis;
        u64 iters = 1ull << (active_k - 1u);
        for (u64 i = t; i < iters; i += V2_STRIDE) {
            u64 i0 = scatter_bits_1(i, axis), i1 = i0 | axis_bit;
            CV2Complex a = v[i0], b = v[i1];
            v[i0] = cadd(cmul(a, mat[0]), cmul(b, mat[1]));
            v[i1] = cadd(cmul(a, mat[2]), cmul(b, mat[3]));
        }
    }
    v2_barrier();
    if (IS_OWNER) {
        u8 out = fused_u2[cp].out_states[in_state];
        fset(st->px, axis, (out & 1) != 0); fset(st->pz, axis, (out & 2) != 0);
    }
    v2_barrier();
}
static inline void v2_op_array_u4(V2State* st, CV2Complex* v, u32 active_k, u32 lo, u32 hi,
                                  const CV2FusedU4Entry* fused_u4, u32 cp) {
    u32 t = v2_tid();
    int in_state = (fget(st->pz, hi) << 3) | (fget(st->px, hi) << 2)
                 | (fget(st->pz, lo) << 1) | fget(st->px, lo);
    const CV2Complex (*mat)[4] = fused_u4[cp].entries[in_state].matrix;
    if (hi < active_k) {
        u64 lo_bit = 1ull << lo, hi_bit = 1ull << hi;
        u64 iters = 1ull << (active_k - 2u);
        for (u64 i = t; i < iters; i += V2_STRIDE) {
            u64 base = scatter_bits_2(i, lo, hi);
            CV2Complex v0 = v[base], v1 = v[base | lo_bit];
            CV2Complex v2 = v[base | hi_bit], v3 = v[base | lo_bit | hi_bit];
            v[base] = cadd(cadd(cmul(v0, mat[0][0]), cmul(v1, mat[0][1])),
                               cadd(cmul(v2, mat[0][2]), cmul(v3, mat[0][3])));
            v[base | lo_bit] = cadd(cadd(cmul(v0, mat[1][0]), cmul(v1, mat[1][1])),
                               cadd(cmul(v2, mat[1][2]), cmul(v3, mat[1][3])));
            v[base | hi_bit] = cadd(cadd(cmul(v0, mat[2][0]), cmul(v1, mat[2][1])),
                               cadd(cmul(v2, mat[2][2]), cmul(v3, mat[2][3])));
            v[base | lo_bit | hi_bit] = cadd(cadd(cmul(v0, mat[3][0]), cmul(v1, mat[3][1])),
                               cadd(cmul(v2, mat[3][2]), cmul(v3, mat[3][3])));
        }
    }
    v2_barrier();
    if (IS_OWNER) {
        u8 out = fused_u4[cp].entries[in_state].out_state;
        fset(st->px, lo, (out & 1) != 0); fset(st->pz, lo, (out & 2) != 0);
        fset(st->px, hi, (out & 4) != 0); fset(st->pz, hi, (out & 8) != 0);
    }
    v2_barrier();
}

// ---- swap-measure-interfere (fused swap + X-fold; shrinks k) -----------------
static inline void v2_op_swap_meas_interfere(V2State* st, CV2Complex* v, CV2Complex* scratch,
                                             u32 active_k, u32 from, u32 to, u32 slot, u8 flags) {
    u32 t = v2_tid();
    if (from == to) {
        u32 half = 1u << (active_k - 1u);
        int pz = fget(st->pz, to);
        double lp = 0.0, lm = 0.0;
        for (u32 i = t; i < half; i += V2_STRIDE) {
            CV2Complex vi = v[i], vh = v[i + half];
            lp += cnorm(cadd(vi, vh)); lm += cnorm(csub(vi, vh));
        }
        double pp, pm; V2_REDUCE2(t, lp, lm, &pp, &pm);
        if (IS_OWNER) {
            u8 bb = sample_branch(st->rng, pp, pm, pp + pm); st->branch = bb;
            u8 m_abs = bb ^ (u8)pz;
            if (slot < V2_MAX_MEAS) mset(st->meas, slot, m_abs ^ (u8)((flags & FLAG_SIGN) != 0));
        }
        v2_barrier();
        for (u32 i = t; i < half; i += V2_STRIDE) {
            CV2Complex vi = v[i], vh = v[i + half];
            v[i] = cscale((st->branch == 0) ? cadd(vi, vh) : csub(vi, vh), V2_INV_SQRT2);
        }
        v2_barrier();
        if (IS_OWNER) {
            st->active_k = active_k - 1;
            u8 m_abs = mget(st->meas, slot) ^ (u8)((flags & FLAG_SIGN) != 0);
            fset(st->px, to, m_abs != 0); fset(st->pz, to, 0);
        }
        v2_barrier();
        return;
    }
    if (IS_OWNER) { fswap(st->px, from, to); fswap(st->pz, from, to); }
    v2_barrier();
    int pz = fget(st->pz, to);
    u64 half = 1ull << to, f_bit = 1ull << from;
    double lp = 0.0, lm = 0.0;
    for (u64 idx = t; idx < half; idx += V2_STRIDE) {
        u64 b_f = (idx >> from) & 1ull;
        u64 base = (idx & ~f_bit) | (b_f << to);
        CV2Complex vb = v[base], vf = v[base | f_bit];
        lp += cnorm(cadd(vb, vf)); lm += cnorm(csub(vb, vf));
    }
    double pp, pm; V2_REDUCE2(t, lp, lm, &pp, &pm);
    if (IS_OWNER) {
        u8 bb = sample_branch(st->rng, pp, pm, pp + pm); st->branch = bb;
        u8 m_abs = bb ^ (u8)pz;
        if (slot < V2_MAX_MEAS) mset(st->meas, slot, m_abs ^ (u8)((flags & FLAG_SIGN) != 0));
    }
    v2_barrier();
    for (u64 idx = t; idx < half; idx += V2_STRIDE) {
        u64 b_f = (idx >> from) & 1ull;
        u64 base = (idx & ~f_bit) | (b_f << to);
        CV2Complex vb = v[base], vf = v[base | f_bit];
        scratch[idx] = cscale((st->branch == 0) ? cadd(vb, vf) : csub(vb, vf), V2_INV_SQRT2);
    }
    v2_barrier();
    for (u64 idx = t; idx < half; idx += V2_STRIDE) v[idx] = scratch[idx];
    v2_barrier();
    if (IS_OWNER) {
        st->active_k = active_k - 1;
        u8 m_abs = mget(st->meas, slot) ^ (u8)((flags & FLAG_SIGN) != 0);
        fset(st->px, to, m_abs != 0); fset(st->pz, to, 0);
    }
    v2_barrier();
}

// ---- classical / noise / probe ----------------------------------------------
static inline void v2_op_apply_pauli(V2State* st, const CV2Mask* pauli_masks, u32 cp, u32 cond_slot) {
    u32 t = v2_tid();
    if (IS_OWNER && mget(st->meas, cond_slot) != 0) {
        const CV2Mask* m = &pauli_masks[cp];
        for (u32 w = 0; w < CLIFFT_V2_PAULI_WORDS; ++w) { st->px[w] ^= m->x[w]; st->pz[w] ^= m->z[w]; }
    }
    v2_barrier();
}
static inline void v2_op_noise(V2State* st, const CV2NoiseSite* sites, const CV2Channel* channels,
                               const double* hazards, u32 num_sites, u32 site) {
    u32 t = v2_tid();
    if (IS_OWNER && site == st->next_noise) {
        apply_noise_site(st, sites, channels, site);
        st->next_noise = site + 1u;
        draw_next_noise(st, hazards, num_sites);
    }
    v2_barrier();
}
static inline void v2_op_noise_block(V2State* st, const CV2NoiseSite* sites, const CV2Channel* channels,
                                     const double* hazards, u32 num_sites, u32 start, u32 count) {
    u32 t = v2_tid();
    if (IS_OWNER) {
        u32 end = start + count;
        while (st->next_noise >= start && st->next_noise < end) {
            u32 site_idx = st->next_noise;
            apply_noise_site(st, sites, channels, site_idx);
            st->next_noise = site_idx + 1u;
            draw_next_noise(st, hazards, num_sites);
        }
    }
    v2_barrier();
}
static inline void v2_op_readout_noise(V2State* st, const CV2ReadoutNoise* readout_noise, u32 idx) {
    u32 t = v2_tid();
    if (IS_OWNER) {
        CV2ReadoutNoise r = readout_noise[idx];
        if (rng_uniform(st->rng) < r.prob) mxor1(st->meas, r.meas_idx);
    }
    v2_barrier();
}
static inline void v2_op_observable(V2State* st, const u32* observable_offsets,
                                    const u32* observable_targets, u32 idx, u32 obs_slot) {
    u32 t = v2_tid();
    if (IS_OWNER) {
        u32 s0 = observable_offsets[idx], e0 = observable_offsets[idx + 1];
        u8 parity = 0;
        for (u32 k = s0; k < e0; ++k) parity ^= mget(st->meas, observable_targets[k]);
        if (obs_slot < CLIFFT_V2_MAX_OBS) st->obs[obs_slot] ^= parity;
    }
    v2_barrier();
}
// Returns 1 if the shot is discarded (caller must return from the kernel).
static inline int v2_op_postselect(V2State* st, const u32* detector_offsets,
                                   const u32* detector_targets, u32 idx, u8 flags) {
    u32 t = v2_tid();
    if (IS_OWNER) {
        u32 s0 = detector_offsets[idx], e0 = detector_offsets[idx + 1];
        u8 parity = (flags & FLAG_EXPECTED_ONE) ? 1u : 0u;
        for (u32 k = s0; k < e0; ++k) parity ^= mget(st->meas, detector_targets[k]);
        if (parity != 0) st->discarded = 1;
    }
    v2_barrier();
    return st->discarded;
}
static inline void v2_op_detector(void) { v2_barrier(); }

// ---- per-shot init + final aggregation (shared) -----------------------------
static inline void v2_shot_init(V2State* st, CV2Complex* v, u32 amp_capacity, u64 shot_id,
                                u32 num_observables, u64 seed,
                                const double* noise_hazards, u32 num_noise_sites) {
    u32 t = v2_tid();
    for (u32 i = t; i < amp_capacity; i += V2_STRIDE) { v[i].re = 0.0f; v[i].im = 0.0f; }
    v2_barrier();
    if (IS_OWNER) {
        for (u32 w = 0; w < CLIFFT_V2_PAULI_WORDS; ++w) { st->px[w] = 0; st->pz[w] = 0; }
        st->active_k = 0; st->discarded = 0;
        v[0].re = 1.0f; v[0].im = 0.0f;
        for (u32 i = 0; i < num_observables; ++i) st->obs[i] = 0;
        for (u32 w = 0; w < V2_MEAS_WORDS; ++w) st->meas[w] = 0;
        st->next_noise = 0;
        rng_seed(st->rng, seed, shot_id);
        draw_next_noise(st, noise_hazards, num_noise_sites);
    }
    v2_barrier();
}
static inline void v2_shot_aggregate(V2State* st, CV2BlockCounts* block_counts, u32 num_observables) {
    u32 t = v2_tid();
    v2_barrier();
    if (IS_OWNER) {
        if (!st->discarded) {
            __atomic_fetch_add(&block_counts->passed, 1UL, __ATOMIC_RELAXED);
            for (u32 i = 0; i < num_observables && i < CLIFFT_V2_MAX_OBS; ++i) {
                if (st->obs[i] != 0)
                    __atomic_fetch_add(&block_counts->observable_ones[i], 1UL, __ATOMIC_RELAXED);
            }
        }
    }
}
