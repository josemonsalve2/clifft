#include "clifft/gpu/codegen/codegen_types.h"

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

// String constants referenced directly by this translation unit
#include "ops/frame_ops.inc"
#include "ops/array_ops.inc"
#include "ops/multi_qubit_ops.inc"
#include "ops/expand_ops.inc"
#include "ops/measurement_ops.inc"
#include "ops/noise_ops.inc"
#include "ops/unitary_ops.inc"
#include "ops/exp_val_ops.inc"
#include "ops/reduction_ops.inc"

namespace clifft {
namespace gpu {

// -----------------------------------------------------------------------
// Public: generate_compiled_kernel_coop (LDS/cooperative tier)
// -----------------------------------------------------------------------
std::string generate_compiled_kernel_coop(const FlattenedProgram& flat) {
    UsedFunctions uf = analyze_used_functions(flat);
    bool has_noise = uf.noise || uf.noise_block;

    auto pipe_ops = analyze_pipeline_opportunities(flat);

    uint32_t peak = flat.peak_rank;
    uint32_t num_amplitudes = 1u << peak;

    std::ostringstream out;

    // Preamble: types, scatter, complex, rng, coop state + LDS helpers, frame raw, coop funcs
    codegen::emit_preamble(out, uf);
    out << kPreambleCoopState;
    out << kCoopFrameRaw;
    out << kCoopReduceDouble;
    if (uf.needs_complex_ops) {
        out << kCoopArrayH;
        out << kCoopArrayCnot;
        out << kCoopArrayCz;
        if (uf.array_s) out << kCoopArrayS;
        if (uf.array_t) out << kCoopArrayT;
        if (uf.array_u2) out << kCoopArrayU2;
        if (uf.array_u4) out << kCoopArrayU4;
    }
    if (uf.expand_plain) out << kCoopExpandPlain;
    if (uf.expand_t) out << kCoopExpandT;
    if (uf.expand_rot) out << kCoopExpandRot;
    if (uf.meas_active_diagonal) out << kCoopMeasActiveDiag;
    if (uf.meas_active_interfere || uf.swap_meas_interfere) out << kCoopMeasActiveInterfere;
    if (uf.swap_meas_interfere) out << kCoopSwapMeasInterfere;
    if (uf.array_swap) out << kCoopArraySwap;
    if (uf.array_multi_cnot) out << kCoopArrayMultiCnot;
    if (uf.array_multi_cz) out << kCoopArrayMultiCz;
    if (uf.array_rot) out << kCoopArrayRot;
    if (uf.exp_val) out << kCoopExecExpVal;

    // Constant pool
    codegen::emit_constant_pool(out, flat, uf);
    // Dummy ShotState for apply_pauli (uses ShotState& in existing helper)
    if (uf.apply_pauli) {
        out << kApplyPauliToFrame;
        out << kApplyPauli;
    }
    if (has_noise) {
        out << kCoopNoiseHelper;
    }

    // Kernel signature
    out << "\nextern \"C\" __global__ __attribute__((amdgpu_flat_work_group_size(1, 256))) void compiled_sample_kernel_coop(\n"
        << "    uint64_t shot_offset, uint64_t shots, uint64_t seed,\n"
        << "    BlockCounts* block_counts,\n"
        << "    uint32_t num_observables, uint32_t num_exp_vals\n"
        << ") {\n";

    // One block = one shot
    out << "    uint64_t batch_shot_id = BIDX;\n"
        << "    if (batch_shot_id >= shots) return;\n"
        << "\n";

    // LDS amplitude array -- padded by 1 per 32-element group to avoid bank conflicts
    // For MI300X: LDS has 64 banks of 4 bytes each. GpuComplex = 8 bytes = 2 banks.
    // Adding 1 to the stride per 32-element row avoids the conflict pattern.
    out << "    LDS GpuComplex v[" << num_amplitudes << "];\n";
    out << "    LDS uint64_t px[2];\n"
        << "    LDS uint64_t pz[2];\n"
        << "    LDS uint32_t active_k_shared;\n"
        << "    LDS uint32_t next_noise_idx_shared;\n"
        << "    LDS uint8_t discarded_shared;\n"
        << "    LDS uint8_t meas[kMaxMeas];\n"
        << "    LDS uint8_t obs[kMaxObs];\n"
        << "    LDS double red0[256];\n"
        << "    LDS double red1[256];\n";
    if (uf.swap_meas_interfere) {
        // scratch_v is used as a temporary during coop_swap_meas_interfere.
        // We need half the peak amplitude count (1 << (peak_rank-1)) entries.
        uint32_t scratch_size = (peak > 0) ? (num_amplitudes / 2) : 1u;
        out << "    LDS GpuComplex scratch_v[" << scratch_size << "];\n";
    } else {
        out << "    GpuComplex* scratch_v = nullptr; // unused\n";
    }
    if (uf.exp_val) {
        out << "    LDS double exp_vals[kMaxExpVals];\n";
    }
    out << "\n";

    // Aliases for pointers
    out << "    uint32_t* active_k_ptr = &active_k_shared;\n"
        << "    uint32_t* next_noise_idx = &next_noise_idx_shared;\n"
        << "    uint8_t* discarded_ptr = &discarded_shared;\n"
        << "\n";

    // Cooperative initialization -- all threads participate to zero v[]
    out << "    for (uint32_t i = TIDX; i < " << num_amplitudes << "u; i += BSIZE) {\n"
        << "        v[i] = {0.0f, 0.0f};\n"
        << "    }\n"
        << "    if (TIDX == 0) {\n"
        << "        px[0] = 0; px[1] = 0;\n"
        << "        pz[0] = 0; pz[1] = 0;\n"
        << "        active_k_shared = 0;\n"
        << "        next_noise_idx_shared = 0;\n"
        << "        discarded_shared = 0;\n"
        << "        v[0] = {1.0f, 0.0f};\n"
        << "        for (uint32_t i = 0; i < num_observables; ++i) obs[i] = 0;\n";
    if (uf.exp_val) {
        out << "        for (uint32_t i = 0; i < num_exp_vals; ++i) exp_vals[i] = 0.0;\n";
    }
    out << "    }\n"
        << "    BARRIER();\n"
        << "\n";

    if (uf.needs_rng) {
        out << "    Rng rng;\n"
            << "    if (TIDX == 0) {\n"
            << "        rng.seed(seed, shot_offset + batch_shot_id);\n";
        if (has_noise) {
            out << "        coop_draw_next_noise(next_noise_idx, rng, kNoiseHazards, "
                << flat.noise_sites.size() << "u);\n";
        }
        out << "    }\n"
            << "    BARRIER();\n\n";
    }

    // Instruction sequence
    out << "    // --- Begin cooperative instruction sequence (" << flat.instrs.size() << " ops) ---\n";
    out << "    // Pipeline opportunities found: " << pipe_ops.size() << "\n";
    codegen::emit_coop_instructions(out, flat, pipe_ops);
    out << "    // --- End cooperative instruction sequence ---\n";

    // Accumulate into block_counts (one count record total -- coop uses atomic adds)
    out << "\n"
        << "done:\n"
        << "    BARRIER();\n"
        << "    if (TIDX == 0 && discarded_shared == 0) {\n"
        << "        bool any_logical = false;\n"
        << "        for (uint32_t i = 0; i < num_observables; ++i) {\n"
        << "            uint8_t val = obs[i];\n";
    if (uf.observable) {
        out << "            if (kExpectedObservables[i] != 0) val ^= 1u;\n";
    }
    out << "            if (val != 0) {\n"
        << "                __atomic_fetch_add(reinterpret_cast<unsigned long long*>(\n"
        << "                    &block_counts->observable_ones[i]), 1ULL, __ATOMIC_RELAXED);\n"
        << "                any_logical = true;\n"
        << "            }\n"
        << "        }\n"
        << "        __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&block_counts->passed), 1ULL, __ATOMIC_RELAXED);\n"
        << "        if (any_logical)\n"
        << "            __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&block_counts->logical_errors), 1ULL, __ATOMIC_RELAXED);\n";
    if (uf.exp_val) {
        out << "        for (uint32_t i = 0; i < num_exp_vals; ++i) {\n"
            << "            __atomic_fetch_add(reinterpret_cast<unsigned long long*>(\n"
            << "                &block_counts->exp_val_count), 1ULL, __ATOMIC_RELAXED);\n"
            << "            // Use double-precision atomic add via CAS loop\n"
            << "            unsigned long long* addr = reinterpret_cast<unsigned long long*>(\n"
            << "                &block_counts->exp_val_sums[i]);\n"
            << "            unsigned long long old_val = __atomic_load_n(addr, __ATOMIC_RELAXED);\n"
            << "            unsigned long long new_val;\n"
            << "            do {\n"
            << "                double old_d;\n"
            << "                __builtin_memcpy(&old_d, &old_val, sizeof(double));\n"
            << "                double new_d = old_d + exp_vals[i];\n"
            << "                __builtin_memcpy(&new_val, &new_d, sizeof(double));\n"
            << "            } while (!__atomic_compare_exchange_n(addr, &old_val, new_val,\n"
            << "                         false, __ATOMIC_RELAXED, __ATOMIC_RELAXED));\n"
            << "        }\n";
    }
    out << "    }\n"
        << "}\n";

    return out.str();
}

}  // namespace gpu
}  // namespace clifft
