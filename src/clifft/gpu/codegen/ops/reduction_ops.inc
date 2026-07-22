// reduction_ops.inc — Cooperative state, shared-memory reductions, and warp-shuffle BlockCounts reduction
// Included by emit_preamble.cc. Do not compile standalone.

static const char* kPreambleCoopState = R"HIP(
// LDS amplitude array size = 1 << PEAK_RANK (baked at compile time)
// Padded by 1 per 32-element bank group to avoid bank conflicts.
// __shared__ GpuComplex v[COOP_AMPLITUDES]; // declared in kernel, size baked

struct CoopShotStateCompiled {
GpuComplex* v;        // LDS amplitude array
uint64_t* px;         // Pauli-X frame (2 words)
uint64_t* pz;         // Pauli-Z frame (2 words)
uint32_t* active_k;
uint32_t* next_noise_idx;
uint8_t* meas;
uint8_t* obs;
uint8_t* discarded;
double* red0;         // reduction scratch [BSIZE]
double* red1;
};

// Bit-cast float to uint32 without UB (LLVM optimizes to a register move)
DEV static inline __attribute__((always_inline)) uint32_t float_as_uint(float x) {
    uint32_t tmp;
    __builtin_memcpy(&tmp, &x, sizeof(tmp));
    return tmp;
}
DEV static inline __attribute__((always_inline)) void store_complex64_lds(GpuComplex* v, uint32_t idx, GpuComplex c) {
    uint64_t packed = (uint64_t)float_as_uint(c.re) | ((uint64_t)float_as_uint(c.im) << 32);
    *reinterpret_cast<uint64_t*>(v + idx) = packed;
}
)HIP";

// Cooperative warp-shuffle + shared-memory reduction for double (used in coop measurements)
// Two-value reduce matching SVM coop_reduce2 exactly for bit-identical results
static const char* kCoopReduceDouble = R"HIP(
DEV static double coop_reduce_sum(double val, double* smem) {
    uint32_t tid = TIDX;
    uint32_t lane = tid & 63u;
    uint32_t warp_id = tid >> 6;
    for (int off = 32; off > 0; off >>= 1) val += shfl_xor_f64(val, off);
    if (lane == 0) smem[warp_id] = val;
    BARRIER();
    if (tid < 4) {
        val = smem[tid];
        for (int off = 2; off > 0; off >>= 1) val += shfl_xor_f64(val, off);
        if (tid == 0) smem[0] = val;
    }
    BARRIER();
    return smem[0];
}
DEV static void coop_reduce2(double local0, double local1,
                              double* red0, double* red1,
                              double& out0, double& out1) {
    uint32_t tid = TIDX;
    uint32_t lane = tid & 63u;
    uint32_t warp_id = tid >> 6;
    for (int off = 32; off > 0; off >>= 1) {
        local0 += shfl_xor_f64(local0, off);
        local1 += shfl_xor_f64(local1, off);
    }
    if (lane == 0) {
        red0[warp_id] = local0;
        red1[warp_id] = local1;
    }
    BARRIER();
    if (tid < 4) {
        local0 = red0[tid];
        local1 = red1[tid];
        for (int off = 2; off > 0; off >>= 1) {
            local0 += shfl_xor_f64(local0, off);
            local1 += shfl_xor_f64(local1, off);
        }
    }
    if (tid == 0) {
        red0[0] = local0;
        red1[0] = local1;
    }
    BARRIER();
    out0 = red0[0];
    out1 = red1[0];
}
)HIP";

// -----------------------------------------------------------------------
// Warp-shuffle reduction for BlockCounts (GEAK OPT-1)
//
// Replaces the original shared-memory tree reduction with a two-phase
// approach:
//   Phase 1: Intra-wavefront reduction via __shfl_xor (AMD wavefront=64)
//   Phase 2: Inter-wavefront reduction via shared memory (4 values for
//            blockDim=256 = 4 wavefronts)
// -----------------------------------------------------------------------
static const char* kWarpShuffleReduction = R"HIP(
    // --- Warp-shuffle reduction (GEAK OPT-1) ---
    // Phase 1: intra-wavefront reduction via __shfl_xor (wavefront = 64)
    for (int offset = 32; offset > 0; offset >>= 1) {
        local_passed    += shfl_xor_i32((int)local_passed, offset);
        local_logical   += shfl_xor_i32((int)local_logical, offset);
    }
    {
        static LDS uint64_t wf_passed[4];
        static LDS uint64_t wf_logical[4];
        uint32_t warp_id = tid >> 6;
        uint32_t lane = tid & 63;
        if (lane == 0) {
            wf_passed[warp_id] = local_passed;
            wf_logical[warp_id] = local_logical;
        }
        BARRIER();
        // Phase 2: first 4 threads reduce the wavefront partials
        if (tid < 4) {
            local_passed = wf_passed[tid];
            local_logical = wf_logical[tid];
            for (int offset = 2; offset > 0; offset >>= 1) {
                local_passed  += shfl_xor_i32((int)local_passed, offset);
                local_logical += shfl_xor_i32((int)local_logical, offset);
            }
        }
    }
    // Observable reduction: per-observable warp shuffle
    for (uint32_t oi = 0; oi < num_observables; ++oi) {
        for (int offset = 32; offset > 0; offset >>= 1) {
            local_obs[oi] += shfl_xor_i32((int)local_obs[oi], offset);
        }
    }
    {
        static LDS uint64_t wf_obs[kMaxObs][4];
        uint32_t warp_id = tid >> 6;
        uint32_t lane = tid & 63;
        for (uint32_t oi = 0; oi < num_observables; ++oi) {
            if (lane == 0) wf_obs[oi][warp_id] = local_obs[oi];
        }
        BARRIER();
        if (tid < 4) {
            for (uint32_t oi = 0; oi < num_observables; ++oi) {
                local_obs[oi] = wf_obs[oi][tid];
                for (int offset = 2; offset > 0; offset >>= 1) {
                    local_obs[oi] += shfl_xor_i32((int)local_obs[oi], offset);
                }
            }
        }
    }
    // Exp-val reduction
    for (uint32_t ei = 0; ei < num_exp_vals; ++ei) {
        for (int offset = 32; offset > 0; offset >>= 1) {
            local_exp[ei] += shfl_xor_f64(local_exp[ei], offset);
        }
    }
    {
        static LDS double wf_exp[kMaxExpVals][4];
        uint32_t warp_id = tid >> 6;
        uint32_t lane = tid & 63;
        for (uint32_t ei = 0; ei < num_exp_vals; ++ei) {
            if (lane == 0) wf_exp[ei][warp_id] = local_exp[ei];
        }
        BARRIER();
        if (tid < 4) {
            for (uint32_t ei = 0; ei < num_exp_vals; ++ei) {
                local_exp[ei] = wf_exp[ei][tid];
                for (int offset = 2; offset > 0; offset >>= 1) {
                    local_exp[ei] += shfl_xor_f64(local_exp[ei], offset);
                }
            }
        }
    }
    if (tid == 0) {
        BlockCounts out;
        out.passed = local_passed;
        out.logical_errors = local_logical;
        for (uint32_t oi = 0; oi < kMaxObs; ++oi) {
            out.observable_ones[oi] = (oi < num_observables) ? local_obs[oi] : 0;
        }
        for (uint32_t ei = 0; ei < kMaxExpVals; ++ei) {
            out.exp_val_sums[ei] = (ei < num_exp_vals) ? local_exp[ei] : 0.0;
        }
        out.exp_val_count = local_passed;
        block_counts[BIDX] = out;
    }
)HIP";
