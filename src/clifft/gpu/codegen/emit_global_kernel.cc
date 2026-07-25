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
// Public: generate_compiled_kernel_global
//
// Global-tier kernel: one shot shared across BSIZE threads, amplitude
// array in HBM.  Per-XCD work-stealing reuses the same counter layout as
// the SVM global-coop kernel.
// -----------------------------------------------------------------------
std::string generate_compiled_kernel_global(const FlattenedProgram& flat) {
    UsedFunctions uf = analyze_used_functions(flat);
    bool has_noise = uf.noise || uf.noise_block;
    auto pipe_ops = analyze_pipeline_opportunities(flat);

    std::ostringstream out;

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

    codegen::emit_constant_pool(out, flat, uf);
    if (uf.apply_pauli) {
        out << kApplyPauliToFrame;
        out << kApplyPauli;
    }
    if (has_noise) out << kCoopNoiseHelper;

    // Kernel signature -- global_v and global_scratch are HBM buffers
    out << "\nextern \"C\" __global__ __attribute__((amdgpu_flat_work_group_size(1, 256))) void compiled_sample_kernel_global(\n"
        << "    uint64_t shot_offset, uint64_t shots, uint64_t seed,\n"
        << "    GpuComplex* __restrict__ global_v,\n"
        << "    GpuComplex* __restrict__ global_scratch,\n"
        << "    uint64_t* work_counter,\n"
        << "    BlockCounts* block_counts,\n"
        << "    uint32_t num_observables, uint32_t num_exp_vals\n"
        << ") {\n";

    // Each block has a dedicated HBM slot for its amplitude array
    // Per-slot stride is derived from THIS circuit's peak_rank (matching the
    // host allocation in hip_sampler.hip), NOT from kGlobalMaxPeakRank — a
    // max-derived stride would scale every allocation with the cap.
    const uint64_t slot_amps = 1ull << flat.peak_rank;
    out << "    uint32_t slot = BIDX;\n"
        << "    GpuComplex* v = global_v + (size_t)slot * " << slot_amps << "ull;\n";
    if (uf.swap_meas_interfere) {
        // scratch_v: the host allocates scratch at HALF the per-slot amplitude
        // count, so the scratch stride is half the v stride.
        out << "    GpuComplex* scratch_v = global_scratch + (size_t)slot * "
            << (slot_amps / 2) << "ull;\n";
    } else {
        out << "    GpuComplex* scratch_v = nullptr; // unused\n";
    }
    out << "\n";

    out << "    LDS uint64_t px[2];\n"
        << "    LDS uint64_t pz[2];\n"
        << "    LDS uint32_t active_k_shared;\n"
        << "    LDS uint32_t next_noise_idx_shared;\n"
        << "    LDS uint8_t discarded_shared;\n"
        << "    LDS uint8_t meas[kMaxMeas];\n"
        << "    LDS uint8_t obs[kMaxObs];\n"
        << "    LDS double red0[256];\n"
        << "    LDS double red1[256];\n"
        << "    LDS uint64_t batch_shot_id_shared;\n";
    if (uf.exp_val) {
        out << "    LDS double exp_vals[kMaxExpVals];\n";
    }
    out << "\n"
        << "    uint32_t* active_k_ptr = &active_k_shared;\n"
        << "    uint32_t* next_noise_idx = &next_noise_idx_shared;\n"
        << "    uint8_t* discarded_ptr = &discarded_shared;\n"
        << "\n";

    // Per-XCD work stealing: same pattern as SVM global-coop kernel
    uint32_t global_num_amplitudes = 1u << flat.peak_rank;
    out << "    // Per-XCD work stealing (8 XCDs for MI300X)\n"
        << "    uint32_t xcd_id;\n"
        << "    asm volatile(\"s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)\" : \"=s\"(xcd_id));\n"
        << "    xcd_id = xcd_id % " << kNumXCDs << "u;\n"
        << "    uint64_t* my_counter = work_counter + xcd_id;\n"
        << "\n"
        << "    while (true) {\n"
        << "        if (TIDX == 0) {\n"
        << "            uint64_t local_slot = __atomic_fetch_add(reinterpret_cast<unsigned long long*>(my_counter),\n"
        << "                                           (unsigned long long)1, __ATOMIC_RELAXED);\n"
        << "            batch_shot_id_shared = local_slot * " << kNumXCDs << "u + xcd_id;\n"
        << "        }\n"
        << "        BARRIER();\n"
        << "        uint64_t my_shot_offset = batch_shot_id_shared;\n"
        << "        if (my_shot_offset >= shots) break;\n"
        << "\n"
        << "        // Initialize state for this shot -- cooperative v[] zeroing\n"
        << "        for (uint32_t i = TIDX; i < " << global_num_amplitudes << "u; i += BSIZE) {\n"
        << "            v[i] = {0.0f, 0.0f};\n"
        << "        }\n"
        << "        if (TIDX == 0) {\n"
        << "            px[0] = 0; px[1] = 0; pz[0] = 0; pz[1] = 0;\n"
        << "            active_k_shared = 0;\n"
        << "            next_noise_idx_shared = 0;\n"
        << "            discarded_shared = 0;\n"
        << "            v[0] = {1.0f, 0.0f};\n"
        << "            for (uint32_t i = 0; i < num_observables; ++i) obs[i] = 0;\n";
    if (uf.exp_val) {
        out << "            for (uint32_t i = 0; i < num_exp_vals; ++i) exp_vals[i] = 0.0;\n";
    }
    out << "        }\n"
        << "        BARRIER();\n\n";

    if (uf.needs_rng) {
        out << "        Rng rng;\n"
            << "        if (TIDX == 0) {\n"
            << "            rng.seed(seed, shot_offset + my_shot_offset);\n";
        if (has_noise) {
            out << "            coop_draw_next_noise(next_noise_idx, rng, kNoiseHazards, "
                << flat.noise_sites.size() << "u);\n";
        }
        out << "        }\n"
            << "        BARRIER();\n\n";
    }

    // Instruction sequence
    out << "        // --- Begin global-tier instruction sequence (" << flat.instrs.size() << " ops) ---\n";
    out << "        // Pipeline opportunities: " << pipe_ops.size() << "\n";
    // Emit instructions with 8-space indent for the inner while loop
    // We reuse emit_coop_instructions but need extra indentation
    {
        std::ostringstream inner;
        codegen::emit_coop_instructions(inner, flat, pipe_ops);
        std::string line;
        std::istringstream ss(inner.str());
        while (std::getline(ss, line)) {
            out << "    " << line << "\n";
        }
    }
    out << "        // --- End global-tier instruction sequence ---\n\n";

    out << "    done:\n"
        << "        BARRIER();\n"
        << "        if (TIDX == 0 && discarded_shared == 0) {\n"
        << "            bool any_logical = false;\n"
        << "            for (uint32_t i = 0; i < num_observables; ++i) {\n"
        << "                uint8_t val = obs[i];\n";
    if (uf.observable) {
        out << "                if (kExpectedObservables[i] != 0) val ^= 1u;\n";
    }
    out << "                if (val != 0) {\n"
        << "                    __atomic_fetch_add(reinterpret_cast<unsigned long long*>(\n"
        << "                        &block_counts->observable_ones[i]), 1ULL, __ATOMIC_RELAXED);\n"
        << "                    any_logical = true;\n"
        << "                }\n"
        << "            }\n"
        << "            __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&block_counts->passed), 1ULL, __ATOMIC_RELAXED);\n"
        << "            if (any_logical)\n"
        << "                __atomic_fetch_add(reinterpret_cast<unsigned long long*>(&block_counts->logical_errors), 1ULL, __ATOMIC_RELAXED);\n";
    if (uf.exp_val) {
        out << "            for (uint32_t i = 0; i < num_exp_vals; ++i) {\n"
            << "                __atomic_fetch_add(reinterpret_cast<unsigned long long*>(\n"
            << "                    &block_counts->exp_val_count), 1ULL, __ATOMIC_RELAXED);\n"
            << "                unsigned long long* addr = reinterpret_cast<unsigned long long*>(\n"
            << "                    &block_counts->exp_val_sums[i]);\n"
            << "                unsigned long long old_val = __atomic_load_n(addr, __ATOMIC_RELAXED);\n"
            << "                unsigned long long new_val;\n"
            << "                do {\n"
            << "                    double old_d;\n"
            << "                    __builtin_memcpy(&old_d, &old_val, sizeof(double));\n"
            << "                    double new_d = old_d + exp_vals[i];\n"
            << "                    __builtin_memcpy(&new_val, &new_d, sizeof(double));\n"
            << "                } while (!__atomic_compare_exchange_n(addr, &old_val, new_val,\n"
            << "                             false, __ATOMIC_RELAXED, __ATOMIC_RELAXED));\n"
            << "            }\n";
    }
    out << "        }\n"
        << "    } // while (true) work-stealing loop\n"
        << "}\n";

    return out.str();
}

}  // namespace gpu
}  // namespace clifft
