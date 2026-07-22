// mfma_sweep_bench.hip
//
// Microbenchmarks comparing scalar VALU vs MFMA approaches for quantum gate
// amplitude array sweeps on MI300X (gfx942).
//
// Build:
//   hipcc -O3 --offload-arch=gfx942 -ffp-contract=off \
//         -o mfma_sweep_bench mfma_sweep_bench.hip
//
// Run:
//   ./mfma_sweep_bench
//
// Each benchmark is run in isolation with warm-up and timing via hipEvents.
// All kernels operate on a shared-memory amplitude array of 2^RANK complex
// amplitudes and apply a gate sweep, mimicking the clifft coop kernel pattern.

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// Ranks to benchmark (active qubit dimension = 2^RANK amplitudes)
static constexpr int RANKS[] = {6, 8, 10};
static constexpr int NUM_RANKS = 3;

// Number of sequential gate applications per kernel launch (simulates a
// circuit layer with multiple gates on different axes)
static constexpr int GATES_PER_LAUNCH = 20;

// Benchmark repetitions for timing
static constexpr int WARMUP_ITERS = 5;
static constexpr int BENCH_ITERS = 50;

// Workgroup size (matches clifft coop kernel)
static constexpr int BLOCK_SIZE = 256;

// Maximum amplitude array size
static constexpr int MAX_AMPLITUDES = 1 << 10; // 1024

// ---------------------------------------------------------------------------
// Complex arithmetic (matches clifft exactly)
// ---------------------------------------------------------------------------

struct Complex {
    float re;
    float im;
};

__device__ inline Complex cadd(Complex a, Complex b) {
    return {a.re + b.re, a.im + b.im};
}

__device__ inline Complex cmul(Complex a, Complex b) {
    return {a.re * b.re - a.im * b.im,
            a.re * b.im + a.im * b.re};
}

__device__ inline uint64_t insert_zero_bit(uint64_t val, uint32_t pos) {
    uint64_t mask = (1ULL << pos) - 1ULL;
    return (val & mask) | ((val & ~mask) << 1);
}

__device__ inline uint64_t scatter_bits_1(uint64_t val, uint32_t bit_pos) {
    return insert_zero_bit(val, bit_pos);
}

__device__ inline uint64_t scatter_bits_2(uint64_t val, uint32_t bit1, uint32_t bit2) {
    uint32_t lo = min(bit1, bit2);
    uint32_t hi = max(bit1, bit2);
    val = insert_zero_bit(val, lo);
    return insert_zero_bit(val, hi);
}

// ---------------------------------------------------------------------------
// Benchmark 1: Scalar U2 sweep (current clifft implementation)
// ---------------------------------------------------------------------------

__device__ __noinline__
void scalar_u2_sweep(Complex* v, uint32_t active_k,
                     uint32_t axis, const Complex mat[4]) {
    uint64_t axis_bit = 1ULL << axis;
    uint64_t iters = 1ULL << (active_k - 1);
    for (uint64_t i = threadIdx.x; i < iters; i += blockDim.x) {
        uint64_t idx0 = scatter_bits_1(i, axis);
        uint64_t idx1 = idx0 | axis_bit;
        Complex a = v[idx0], b = v[idx1];
        v[idx0] = cadd(cmul(a, mat[0]), cmul(b, mat[1]));
        v[idx1] = cadd(cmul(a, mat[2]), cmul(b, mat[3]));
    }
}

__global__ void bench_scalar_u2(int active_k, int num_gates) {
    __shared__ Complex v[MAX_AMPLITUDES];

    // Initialize: |0> state
    uint32_t dim = 1u << active_k;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        v[i] = (i == 0) ? Complex{1.0f, 0.0f} : Complex{0.0f, 0.0f};
    }
    __syncthreads();

    // Hadamard-like gate matrix: (1/sqrt2) * [[1, 1], [1, -1]]
    const float s = 0.70710678118654752f;
    Complex mat[4] = {{s, 0.0f}, {s, 0.0f}, {s, 0.0f}, {-s, 0.0f}};

    // Apply gates on rotating axes
    for (int g = 0; g < num_gates; ++g) {
        uint32_t axis = g % active_k;
        scalar_u2_sweep(v, active_k, axis, mat);
        __syncthreads();
    }

    // Prevent dead-code elimination: write one value to global
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Use volatile to prevent optimization
        volatile float sink = v[0].re + v[dim - 1].im;
        (void)sink;
    }
}

// ---------------------------------------------------------------------------
// Benchmark 2: Scalar U4 sweep (current clifft 2-qubit gate)
// ---------------------------------------------------------------------------

__device__ __noinline__
void scalar_u4_sweep(Complex* v, uint32_t active_k,
                     uint32_t axis_lo, uint32_t axis_hi,
                     const Complex mat[4][4]) {
    uint64_t lo_bit = 1ULL << axis_lo;
    uint64_t hi_bit = 1ULL << axis_hi;
    uint64_t iters = 1ULL << (active_k - 2);
    for (uint64_t i = threadIdx.x; i < iters; i += blockDim.x) {
        uint64_t base = scatter_bits_2(i, axis_lo, axis_hi);
        uint64_t i0 = base;
        uint64_t i1 = base | lo_bit;
        uint64_t i2 = base | hi_bit;
        uint64_t i3 = base | lo_bit | hi_bit;
        Complex v0 = v[i0], v1 = v[i1], v2 = v[i2], v3 = v[i3];
        v[i0] = cadd(cadd(cmul(v0, mat[0][0]), cmul(v1, mat[0][1])),
                     cadd(cmul(v2, mat[0][2]), cmul(v3, mat[0][3])));
        v[i1] = cadd(cadd(cmul(v0, mat[1][0]), cmul(v1, mat[1][1])),
                     cadd(cmul(v2, mat[1][2]), cmul(v3, mat[1][3])));
        v[i2] = cadd(cadd(cmul(v0, mat[2][0]), cmul(v1, mat[2][1])),
                     cadd(cmul(v2, mat[2][2]), cmul(v3, mat[2][3])));
        v[i3] = cadd(cadd(cmul(v0, mat[3][0]), cmul(v1, mat[3][1])),
                     cadd(cmul(v2, mat[3][2]), cmul(v3, mat[3][3])));
    }
}

__global__ void bench_scalar_u4(int active_k, int num_gates) {
    __shared__ Complex v[MAX_AMPLITUDES];

    uint32_t dim = 1u << active_k;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        v[i] = (i == 0) ? Complex{1.0f, 0.0f} : Complex{0.0f, 0.0f};
    }
    __syncthreads();

    // CNOT-like gate matrix
    Complex mat[4][4] = {
        {{1,0}, {0,0}, {0,0}, {0,0}},
        {{0,0}, {1,0}, {0,0}, {0,0}},
        {{0,0}, {0,0}, {0,0}, {1,0}},
        {{0,0}, {0,0}, {1,0}, {0,0}}
    };

    for (int g = 0; g < num_gates; ++g) {
        uint32_t axis_lo = g % active_k;
        uint32_t axis_hi = (g + 1) % active_k;
        if (axis_lo == axis_hi) axis_hi = (axis_hi + 1) % active_k;
        scalar_u4_sweep(v, active_k, axis_lo, axis_hi, mat);
        __syncthreads();
    }

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        volatile float sink = v[0].re + v[dim - 1].im;
        (void)sink;
    }
}

// ---------------------------------------------------------------------------
// Benchmark 3: MFMA-based U2 sweep using v_mfma_f32_4x4x1f32
//
// Strategy: Embed complex 2x2 gate as real 4x4 matrix, batch 16 independent
// 4x4 matrix-vector products per MFMA call.  Each 4x4 product maps to one
// amplitude pair: [re(a), im(a), re(b), im(b)]^T.
//
// The 4x4 real embedding of complex 2x2 [[U00, U01], [U10, U11]] is:
//   [re(U00) -im(U00) re(U01) -im(U01)]
//   [im(U00)  re(U00) im(U01)  re(U01)]
//   [re(U10) -im(U10) re(U11) -im(U11)]
//   [im(U10)  re(U10) im(U11)  re(U11)]
//
// v_mfma_f32_4x4x1f32 computes D = A * B^T + C with K=1.
// For a full 4x4 matvec we need 4 calls (K=1 rank-1 updates):
//   D = col0(M)*row0(v) + col1(M)*row1(v) + col2(M)*row2(v) + col3(M)*row3(v)
// ---------------------------------------------------------------------------

__device__ __noinline__
void mfma_u2_sweep(Complex* v, uint32_t active_k, uint32_t axis,
                   const Complex umat[4]) {
    // Build real 4x4 embedding of the complex 2x2 gate
    float M[4][4];
    M[0][0] =  umat[0].re; M[0][1] = -umat[0].im; M[0][2] =  umat[1].re; M[0][3] = -umat[1].im;
    M[1][0] =  umat[0].im; M[1][1] =  umat[0].re; M[1][2] =  umat[1].im; M[1][3] =  umat[1].re;
    M[2][0] =  umat[2].re; M[2][1] = -umat[2].im; M[2][2] =  umat[3].re; M[2][3] = -umat[3].im;
    M[3][0] =  umat[2].im; M[3][1] =  umat[2].re; M[3][2] =  umat[3].im; M[3][3] =  umat[3].re;

    uint64_t axis_bit = 1ULL << axis;
    uint64_t num_pairs = 1ULL << (active_k - 1);

    // Each wavefront (64 threads) processes 16 pairs per MFMA call.
    // The 4x4x1 instruction processes 16 independent 4x4 blocks.
    // Lane assignment: lane l is in block l/4, position l%4.
    //
    // For each batch of 16 pairs, we need 4 MFMA calls (K=4 total).
    // We process pairs in batches assigned to wavefronts.

    uint32_t warp_id = threadIdx.x >> 6;  // wavefront id within block
    uint32_t lane = threadIdx.x & 63;     // lane within wavefront
    uint32_t num_wavefronts = blockDim.x >> 6;

    uint32_t block_in_lane = lane >> 2;   // which of 16 blocks this lane is in
    uint32_t pos_in_block = lane & 3;     // position within the 4x4 block (0..3)

    for (uint64_t batch_base = warp_id * 16; batch_base < num_pairs;
         batch_base += num_wavefronts * 16) {

        uint64_t pair_idx = batch_base + block_in_lane;

        // Gather: load amplitude pair into a 4-element vector [re(a), im(a), re(b), im(b)]
        float vec[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (pair_idx < num_pairs) {
            uint64_t idx0 = scatter_bits_1(pair_idx, axis);
            uint64_t idx1 = idx0 | axis_bit;
            vec[0] = v[idx0].re;
            vec[1] = v[idx0].im;
            vec[2] = v[idx1].re;
            vec[3] = v[idx1].im;
        }

        // Accumulator: 4 floats per lane (one column of the 4x4 result)
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        // 4 rank-1 updates: D += col_k(M) * vec[k]
        // For v_mfma_f32_4x4x1f32:
        //   A register: the column of M (one float, broadcast by pos_in_block)
        //   B register: the corresponding vec element (one float)
        //   C/D registers: 4 floats (the accumulator for this block's 4x4 output)
        for (int k = 0; k < 4; ++k) {
            float a_val = M[pos_in_block][k];  // row pos_in_block, col k
            float b_val = vec[k];

            // v_mfma_f32_4x4x1f32: D = A * B + C
            // Each of 16 blocks does a 4x4 rank-1 update
            typedef __attribute__((__vector_size__(4 * sizeof(float)))) float float4_t;
            float4_t c_vec;
            c_vec[0] = acc[0]; c_vec[1] = acc[1]; c_vec[2] = acc[2]; c_vec[3] = acc[3];

            float4_t d_vec = __builtin_amdgcn_mfma_f32_4x4x1f32(a_val, b_val, c_vec, 0, 0, 0);

            acc[0] = d_vec[0]; acc[1] = d_vec[1]; acc[2] = d_vec[2]; acc[3] = d_vec[3];
        }

        // Scatter: write results back
        if (pair_idx < num_pairs) {
            uint64_t idx0 = scatter_bits_1(pair_idx, axis);
            uint64_t idx1 = idx0 | axis_bit;
            // The result for this block is in acc[0..3],
            // but the MFMA output layout distributes across lanes within the block.
            // For 4x4x1f32 with 16 blocks: lane l contributes to block l/4.
            // Within the block, acc[pos_in_block] has the result for row pos_in_block.
            // We need to read from the correct lane.
            //
            // Actually, in the 4x4 block, after the rank-1 accumulation,
            // acc[j] for lane (block*4 + pos) holds D[pos][j].
            // We want the column vector result: D * e_col = column of D.
            // The matvec result is in the pos_in_block-th row across all k.
            //
            // For a matvec M*x, the result[row] = sum_k M[row][k]*x[k].
            // After our 4 rank-1 updates, acc[j] at lane (block*4+row) = sum_k M[row][k] * x[k]
            // when j == row (diagonal). But MFMA computes outer products, not inner products.
            //
            // Actually MFMA 4x4x1 computes D[i][j] = A[i]*B[j] + C[i][j] where A is
            // one column and B is one row. After 4 such rank-1 updates:
            // D[i][j] = sum_k A_k[i] * B_k[j]
            // This is a full matrix multiply result M[i][j].
            // But we want a matrix-VECTOR product, not a matrix-matrix product.
            //
            // To get matvec M*x: set B_k = x[k] (same value for all j in that rank-1),
            // then D[i][j] = sum_k M_col_k[i] * x[k] for all j.
            // But that gives the SAME value in every column j -- wasteful but correct.
            // Just read D[i][0] for any column.

            // Result for row pos_in_block is in acc[0] (any column works since B was scalar)
            // No -- MFMA 4x4 distributes 4 elements across 4 lanes of one block.
            // Lane (block*4 + pos) holds D[pos][0..3]. After scalar-B broadcast,
            // D[pos][j] = result[pos] for all j.
            // So acc[0] == acc[1] == acc[2] == acc[3] == result[pos_in_block].

            float result_val = acc[0]; // result for row = pos_in_block

            // pos_in_block 0 -> re(a'), 1 -> im(a'), 2 -> re(b'), 3 -> im(b')
            // We need to scatter back. Only one lane per element writes.
            if (pos_in_block == 0) v[idx0].re = result_val;
            if (pos_in_block == 1) v[idx0].im = result_val;
            if (pos_in_block == 2) v[idx1].re = result_val;
            if (pos_in_block == 3) v[idx1].im = result_val;
        }
    }
}

__global__ void bench_mfma_u2(int active_k, int num_gates) {
    __shared__ Complex v[MAX_AMPLITUDES];

    uint32_t dim = 1u << active_k;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        v[i] = (i == 0) ? Complex{1.0f, 0.0f} : Complex{0.0f, 0.0f};
    }
    __syncthreads();

    const float s = 0.70710678118654752f;
    Complex mat[4] = {{s, 0.0f}, {s, 0.0f}, {s, 0.0f}, {-s, 0.0f}};

    for (int g = 0; g < num_gates; ++g) {
        uint32_t axis = g % active_k;
        mfma_u2_sweep(v, active_k, axis, mat);
        __syncthreads();
    }

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        volatile float sink = v[0].re + v[dim - 1].im;
        (void)sink;
    }
}

// ---------------------------------------------------------------------------
// Benchmark 4: Vectorized scalar U2 with float2 LDS loads
//
// Optimization: load GpuComplex as float2 (single 64-bit LDS load) and
// use FMA intrinsics for the complex multiply.
// ---------------------------------------------------------------------------

__device__ __noinline__
void vectorized_u2_sweep(Complex* v, uint32_t active_k,
                         uint32_t axis, const Complex mat[4]) {
    uint64_t axis_bit = 1ULL << axis;
    uint64_t iters = 1ULL << (active_k - 1);

    // Preload gate matrix elements into registers
    float m0r = mat[0].re, m0i = mat[0].im;
    float m1r = mat[1].re, m1i = mat[1].im;
    float m2r = mat[2].re, m2i = mat[2].im;
    float m3r = mat[3].re, m3i = mat[3].im;

    for (uint64_t i = threadIdx.x; i < iters; i += blockDim.x) {
        uint64_t idx0 = scatter_bits_1(i, axis);
        uint64_t idx1 = idx0 | axis_bit;

        // Load as float2 (64-bit LDS load)
        float2 a = *reinterpret_cast<float2*>(&v[idx0]);
        float2 b = *reinterpret_cast<float2*>(&v[idx1]);

        // cmul(a, mat[0]) + cmul(b, mat[1])
        float r0_re = __builtin_fmaf(a.x, m0r, -a.y * m0i) +
                      __builtin_fmaf(b.x, m1r, -b.y * m1i);
        float r0_im = __builtin_fmaf(a.x, m0i,  a.y * m0r) +
                      __builtin_fmaf(b.x, m1i,  b.y * m1r);

        // cmul(a, mat[2]) + cmul(b, mat[3])
        float r1_re = __builtin_fmaf(a.x, m2r, -a.y * m2i) +
                      __builtin_fmaf(b.x, m3r, -b.y * m3i);
        float r1_im = __builtin_fmaf(a.x, m2i,  a.y * m2r) +
                      __builtin_fmaf(b.x, m3i,  b.y * m3r);

        // Store as float2 (64-bit LDS store)
        *reinterpret_cast<float2*>(&v[idx0]) = make_float2(r0_re, r0_im);
        *reinterpret_cast<float2*>(&v[idx1]) = make_float2(r1_re, r1_im);
    }
}

__global__ void bench_vectorized_u2(int active_k, int num_gates) {
    __shared__ Complex v[MAX_AMPLITUDES];

    uint32_t dim = 1u << active_k;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        v[i] = (i == 0) ? Complex{1.0f, 0.0f} : Complex{0.0f, 0.0f};
    }
    __syncthreads();

    const float s = 0.70710678118654752f;
    Complex mat[4] = {{s, 0.0f}, {s, 0.0f}, {s, 0.0f}, {-s, 0.0f}};

    for (int g = 0; g < num_gates; ++g) {
        uint32_t axis = g % active_k;
        vectorized_u2_sweep(v, active_k, axis, mat);
        __syncthreads();
    }

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        volatile float sink = v[0].re + v[dim - 1].im;
        (void)sink;
    }
}

// ---------------------------------------------------------------------------
// Benchmark 5: Hadamard (specialized, no general cmul)
//
// H gate: (1/sqrt2) * [[1,1],[1,-1]]
// a' = (a + b) / sqrt2, b' = (a - b) / sqrt2
// This avoids 4 cmul entirely -- just add/sub + scale.
// Measures the "speed of light" for 1-qubit sweeps.
// ---------------------------------------------------------------------------

__device__ __noinline__
void hadamard_sweep(Complex* v, uint32_t active_k, uint32_t axis) {
    uint64_t axis_bit = 1ULL << axis;
    uint64_t iters = 1ULL << (active_k - 1);
    const float s = 0.70710678118654752f;

    for (uint64_t i = threadIdx.x; i < iters; i += blockDim.x) {
        uint64_t idx0 = scatter_bits_1(i, axis);
        uint64_t idx1 = idx0 | axis_bit;
        float2 a = *reinterpret_cast<float2*>(&v[idx0]);
        float2 b = *reinterpret_cast<float2*>(&v[idx1]);

        float2 sum  = make_float2((a.x + b.x) * s, (a.y + b.y) * s);
        float2 diff = make_float2((a.x - b.x) * s, (a.y - b.y) * s);

        *reinterpret_cast<float2*>(&v[idx0]) = sum;
        *reinterpret_cast<float2*>(&v[idx1]) = diff;
    }
}

__global__ void bench_hadamard(int active_k, int num_gates) {
    __shared__ Complex v[MAX_AMPLITUDES];

    uint32_t dim = 1u << active_k;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        v[i] = (i == 0) ? Complex{1.0f, 0.0f} : Complex{0.0f, 0.0f};
    }
    __syncthreads();

    for (int g = 0; g < num_gates; ++g) {
        uint32_t axis = g % active_k;
        hadamard_sweep(v, active_k, axis);
        __syncthreads();
    }

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        volatile float sink = v[0].re + v[dim - 1].im;
        (void)sink;
    }
}

// ---------------------------------------------------------------------------
// Benchmark 6: Diagonal phase gate (Rz / S / T style)
//
// Phase gate: [[1, 0], [0, e^{i*theta}]]
// Only the |1> amplitudes are multiplied by a complex phase.
// This is purely element-wise -- no pair interaction.
// ---------------------------------------------------------------------------

__device__ __noinline__
void phase_sweep(Complex* v, uint32_t active_k, uint32_t axis,
                 Complex phase) {
    uint64_t axis_bit = 1ULL << axis;
    uint64_t iters = 1ULL << (active_k - 1);

    for (uint64_t i = threadIdx.x; i < iters; i += blockDim.x) {
        uint64_t idx = scatter_bits_1(i, axis) | axis_bit;
        float2 amp = *reinterpret_cast<float2*>(&v[idx]);
        float re = amp.x * phase.re - amp.y * phase.im;
        float im = amp.x * phase.im + amp.y * phase.re;
        *reinterpret_cast<float2*>(&v[idx]) = make_float2(re, im);
    }
}

__global__ void bench_phase(int active_k, int num_gates) {
    __shared__ Complex v[MAX_AMPLITUDES];

    uint32_t dim = 1u << active_k;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        v[i] = (i == 0) ? Complex{1.0f, 0.0f} : Complex{0.0f, 0.0f};
    }
    __syncthreads();

    // T gate phase: e^{i*pi/4}
    Complex phase = {0.70710678118654752f, 0.70710678118654752f};

    for (int g = 0; g < num_gates; ++g) {
        uint32_t axis = g % active_k;
        phase_sweep(v, active_k, axis, phase);
        __syncthreads();
    }

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        volatile float sink = v[0].re + v[dim - 1].im;
        (void)sink;
    }
}

// ---------------------------------------------------------------------------
// Benchmark 7: Mixed circuit pattern (H + CNOT + T sequence)
//
// Simulates a typical QEC circuit layer:
//   H on qubit 0 -> CNOT(0,1) -> T on qubit 2 -> H on qubit 3 -> ...
// Uses the actual clifft gate mix.
// ---------------------------------------------------------------------------

__global__ void bench_mixed_circuit(int active_k, int num_gates) {
    __shared__ Complex v[MAX_AMPLITUDES];

    uint32_t dim = 1u << active_k;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        v[i] = (i == 0) ? Complex{1.0f, 0.0f} : Complex{0.0f, 0.0f};
    }
    __syncthreads();

    const float s = 0.70710678118654752f;
    Complex h_mat[4] = {{s, 0.0f}, {s, 0.0f}, {s, 0.0f}, {-s, 0.0f}};
    Complex t_phase = {s, s};
    // CNOT as U4: identity with rows 2,3 swapped
    Complex cnot[4][4] = {
        {{1,0}, {0,0}, {0,0}, {0,0}},
        {{0,0}, {1,0}, {0,0}, {0,0}},
        {{0,0}, {0,0}, {0,0}, {1,0}},
        {{0,0}, {0,0}, {1,0}, {0,0}}
    };

    for (int g = 0; g < num_gates; ++g) {
        int gate_type = g % 3;  // cycle: H, CNOT, T
        if (gate_type == 0) {
            // Hadamard on axis g/3 % active_k
            hadamard_sweep(v, active_k, (g / 3) % active_k);
        } else if (gate_type == 1 && active_k >= 2) {
            // CNOT on two axes
            uint32_t a_lo = (g / 3) % active_k;
            uint32_t a_hi = (a_lo + 1) % active_k;
            scalar_u4_sweep(v, active_k, a_lo, a_hi, cnot);
        } else {
            // T gate (phase) on axis
            phase_sweep(v, active_k, (g / 3 + 2) % active_k, t_phase);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        volatile float sink = v[0].re + v[dim - 1].im;
        (void)sink;
    }
}

// ---------------------------------------------------------------------------
// Benchmark harness
// ---------------------------------------------------------------------------

inline void check(hipError_t err, const char* msg) {
    if (err != hipSuccess) {
        fprintf(stderr, "HIP error at %s: %s\n", msg, hipGetErrorString(err));
        exit(1);
    }
}

struct BenchResult {
    const char* name;
    int rank;
    double us_per_launch;    // microseconds per kernel launch
    double ns_per_gate;      // nanoseconds per gate application
    double gflops;           // estimated GFLOP/s
};

BenchResult run_bench(const char* name, void(*kernel)(int, int),
                      int rank, int num_gates,
                      int warmup, int iters) {
    hipEvent_t start, stop;
    check(hipEventCreate(&start), "create start");
    check(hipEventCreate(&stop), "create stop");

    // Warm up
    for (int i = 0; i < warmup; ++i) {
        kernel<<<1, BLOCK_SIZE>>>(rank, num_gates);
    }
    check(hipDeviceSynchronize(), "warmup sync");

    // Timed runs
    check(hipEventRecord(start), "record start");
    for (int i = 0; i < iters; ++i) {
        kernel<<<1, BLOCK_SIZE>>>(rank, num_gates);
    }
    check(hipEventRecord(stop), "record stop");
    check(hipEventSynchronize(stop), "event sync");

    float ms = 0.0f;
    check(hipEventElapsedTime(&ms, start, stop), "elapsed time");

    hipEventDestroy(start);
    hipEventDestroy(stop);

    double total_us = (double)ms * 1000.0;
    double us_per_launch = total_us / iters;
    double ns_per_gate = (us_per_launch * 1000.0) / num_gates;

    // Estimate FLOP:
    // 1-qubit gate: 2^(k-1) pairs, each pair: 4 cmul (8 FMA) + 2 cadd (4 add) = 20 FLOP
    // 2-qubit gate: 2^(k-2) quads, each quad: 16 cmul + 12 cadd = 88 FLOP
    // Phase gate: 2^(k-1) elements, each: 1 cmul = 4 FMA + 2 add = 6 FLOP
    // We use 20 FLOP per pair as default estimate for U2
    uint64_t pairs = 1ULL << (rank - 1);
    double flop_per_gate = pairs * 20.0;
    double total_flop = flop_per_gate * num_gates * iters;
    double gflops = total_flop / (total_us * 1000.0);

    return {name, rank, us_per_launch, ns_per_gate, gflops};
}

int main() {
    // Print device info
    hipDeviceProp_t prop;
    check(hipGetDeviceProperties(&prop, 0), "get props");
    printf("Device: %s (%s)\n", prop.name, prop.gcnArchName);
    printf("CUs: %d, Clock: %d MHz, LDS/CU: %d KB\n",
           prop.multiProcessorCount, prop.clockRate / 1000,
           (int)(prop.sharedMemPerBlock / 1024));
    printf("Block size: %d threads, Gates per launch: %d\n", BLOCK_SIZE, GATES_PER_LAUNCH);
    printf("Warmup: %d iters, Bench: %d iters\n\n", WARMUP_ITERS, BENCH_ITERS);

    printf("%-25s %5s %12s %12s %10s\n",
           "Benchmark", "Rank", "us/launch", "ns/gate", "GFLOP/s");
    printf("%-25s %5s %12s %12s %10s\n",
           "-------------------------", "-----", "------------",
           "------------", "----------");

    std::vector<BenchResult> results;

    for (int ri = 0; ri < NUM_RANKS; ++ri) {
        int rank = RANKS[ri];

        // Scalar U2 (current clifft)
        auto r1 = run_bench("scalar_u2", bench_scalar_u2,
                            rank, GATES_PER_LAUNCH, WARMUP_ITERS, BENCH_ITERS);
        results.push_back(r1);

        // Vectorized U2 (float2 loads + FMA)
        auto r2 = run_bench("vectorized_u2", bench_vectorized_u2,
                            rank, GATES_PER_LAUNCH, WARMUP_ITERS, BENCH_ITERS);
        results.push_back(r2);

        // MFMA U2
        auto r3 = run_bench("mfma_u2", bench_mfma_u2,
                            rank, GATES_PER_LAUNCH, WARMUP_ITERS, BENCH_ITERS);
        results.push_back(r3);

        // Hadamard (speed of light)
        auto r4 = run_bench("hadamard_sol", bench_hadamard,
                            rank, GATES_PER_LAUNCH, WARMUP_ITERS, BENCH_ITERS);
        results.push_back(r4);

        // Phase gate (diagonal, minimal work)
        auto r5 = run_bench("phase_gate", bench_phase,
                            rank, GATES_PER_LAUNCH, WARMUP_ITERS, BENCH_ITERS);
        results.push_back(r5);

        // Scalar U4
        if (rank >= 2) {
            auto r6 = run_bench("scalar_u4", bench_scalar_u4,
                                rank, GATES_PER_LAUNCH, WARMUP_ITERS, BENCH_ITERS);
            results.push_back(r6);
        }

        // Mixed circuit
        if (rank >= 2) {
            auto r7 = run_bench("mixed_circuit", bench_mixed_circuit,
                                rank, GATES_PER_LAUNCH, WARMUP_ITERS, BENCH_ITERS);
            results.push_back(r7);
        }

        printf("\n");
    }

    // Print results table
    for (const auto& r : results) {
        printf("%-25s %5d %12.2f %12.1f %10.2f\n",
               r.name, r.rank, r.us_per_launch, r.ns_per_gate, r.gflops);
    }

    // Print comparison summary
    printf("\n=== Speedup Summary (vs scalar_u2 baseline) ===\n\n");
    printf("%-25s %5s %10s\n", "Benchmark", "Rank", "Speedup");
    printf("%-25s %5s %10s\n", "-------------------------", "-----", "----------");

    for (int ri = 0; ri < NUM_RANKS; ++ri) {
        int rank = RANKS[ri];
        double baseline_ns = -1.0;
        for (const auto& r : results) {
            if (r.rank == rank && strcmp(r.name, "scalar_u2") == 0) {
                baseline_ns = r.ns_per_gate;
                break;
            }
        }
        if (baseline_ns <= 0) continue;

        for (const auto& r : results) {
            if (r.rank == rank) {
                double speedup = baseline_ns / r.ns_per_gate;
                printf("%-25s %5d %9.2fx\n", r.name, rank, speedup);
            }
        }
        printf("\n");
    }

    return 0;
}
