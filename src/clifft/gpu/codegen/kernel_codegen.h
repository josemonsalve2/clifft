#pragma once

#include "clifft/backend/backend.h"
#include "clifft/gpu/device_program.h"
#include "clifft/gpu/gpu_types.h"

#include <cstdint>
#include <string>
#include <vector>

namespace clifft {
namespace gpu {

/// Tracks which device helper functions a circuit actually needs.
/// Populated by analyzing the bytecode; used to conditionally emit only
/// the required preamble sections and device functions.
struct UsedFunctions {
    // Frame operations
    bool frame_cnot = false;
    bool frame_cz = false;
    bool frame_h = false;
    bool frame_s = false;
    bool frame_swap = false;

    // Array sweep operations
    bool array_cnot = false;
    bool array_cz = false;
    bool array_swap = false;
    bool array_multi_cnot = false;
    bool array_multi_cz = false;
    bool array_h = false;
    bool array_s = false;
    bool array_t = false;
    bool array_rot = false;
    bool array_u2 = false;
    bool array_u4 = false;

    // Expand operations
    bool expand_plain = false;
    bool expand_t = false;
    bool expand_rot = false;

    // Measurement operations
    bool meas_dormant_static = false;
    bool meas_dormant_random = false;
    bool meas_active_diagonal = false;
    bool meas_active_interfere = false;
    bool swap_meas_interfere = false;

    // Pauli / noise / classical
    bool apply_pauli = false;
    bool noise = false;
    bool noise_block = false;
    bool readout_noise = false;
    bool postselect = false;
    bool observable = false;
    bool exp_val = false;

    // Derived flags (computed by compute_derived())
    bool needs_rng = false;           // true if any measurement or noise
    bool needs_array_sweep = false;   // true if any array_* or expand_*
    bool needs_complex_ops = false;   // true if any array sweep needs cmul/cadd/csub
    bool needs_apply_phase = false;   // true if array_s/t/rot
    bool needs_frame_ops = false;     // true if any frame op or array op that calls frame
    bool needs_scatter = false;       // true if any array 2-qubit gate or expand
    bool needs_sample_branch = false; // true if any active measurement

    /// Compute derived flags from the primary flags.
    void compute_derived();
};

/// Analyze a flattened program and determine which device functions are used.
UsedFunctions analyze_used_functions(const FlattenedProgram& flat);

/// Describes a pair of consecutive instructions that are independent (their
/// amplitude index sets are disjoint) and can be double-buffered.
struct PipelineOp {
    size_t first_pc;   ///< Index of the first instruction in flat.instrs
    size_t second_pc;  ///< Index of the immediately following instruction
};

/// Analyze a flattened program for pipeline/double-buffer opportunities.
/// Returns pairs of consecutive instructions that operate on disjoint amplitude
/// index sets (no data dependency between them).  Used by LDS- and global-tier
/// compiled kernels to decide whether to emit double-buffered code.
std::vector<PipelineOp> analyze_pipeline_opportunities(const FlattenedProgram& flat);

/// Generate a complete, self-contained HIP kernel source string for the given
/// program — register tier (peak_rank <= kThreadMaxPeakRank=4).
///
/// The kernel function is named "compiled_sample_kernel" and has the signature:
///   __global__ void compiled_sample_kernel(
///       uint64_t shot_offset, uint64_t shots, uint64_t seed,
///       BlockCounts* block_counts,
///       uint32_t num_observables, uint32_t num_exp_vals);
std::string generate_compiled_kernel(const FlattenedProgram& flat);

/// Generate a cooperative (LDS-tier) HIP kernel source string for the given
/// program — shared-memory tier (peak_rank 5–10).
/// One block = one shot; amplitude array lives in __shared__ GpuComplex v[N].
/// The kernel function is named "compiled_sample_kernel_coop" and has the same
/// parameter signature as compiled_sample_kernel.
std::string generate_compiled_kernel_coop(const FlattenedProgram& flat);

/// Generate a global-coop HIP kernel source string for the given program —
/// global-memory tier (peak_rank 11–26).
/// Amplitude array lives in HBM global_v; uses per-XCD work stealing.
/// The kernel function is named "compiled_sample_kernel_global" with signature:
///   __global__ void compiled_sample_kernel_global(
///       uint64_t shot_offset, uint64_t shots, uint64_t seed,
///       GpuComplex* global_v, GpuComplex* global_scratch,
///       uint64_t* work_counter, BlockCounts* block_counts,
///       uint32_t num_observables, uint32_t num_exp_vals);
std::string generate_compiled_kernel_global(const FlattenedProgram& flat);

}  // namespace gpu
}  // namespace clifft
