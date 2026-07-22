#include "clifft/gpu/codegen/codegen_types.h"

#include <cstdint>
#include <sstream>
#include <string>

// String constants referenced directly by this translation unit
#include "ops/reduction_ops.inc"

namespace clifft {
namespace gpu {

// -----------------------------------------------------------------------
// Public: generate_compiled_kernel (register tier)
// -----------------------------------------------------------------------
std::string generate_compiled_kernel(const FlattenedProgram& flat) {
    UsedFunctions uf = analyze_used_functions(flat);
    bool has_noise = uf.noise || uf.noise_block;

    std::ostringstream out;

    // Emit conditional preamble
    codegen::emit_preamble(out, uf);

    // Emit constant pool (only arrays used by the circuit)
    codegen::emit_constant_pool(out, flat, uf);

    // Emit device functions (only those called by the circuit)
    codegen::emit_needed_functions(out, uf);

    // Emit the kernel function
    out << "\nextern \"C\" __global__ __attribute__((amdgpu_flat_work_group_size(1, 256))) void compiled_sample_kernel(\n"
        << "    uint64_t shot_offset, uint64_t shots, uint64_t seed,\n"
        << "    BlockCounts* block_counts,\n"
        << "    uint32_t num_observables, uint32_t num_exp_vals\n"
        << ") {\n"
        << "    uint32_t tid = TIDX;\n"
        << "    uint64_t batch_shot_id = (uint64_t)BIDX * BSIZE + tid;\n"
        << "\n"
        << "    uint64_t local_passed = 0;\n"
        << "    uint64_t local_logical = 0;\n"
        << "    uint64_t local_obs[kMaxObs];\n"
        << "    double local_exp[kMaxExpVals];\n"
        << "    for (uint32_t i = 0; i < kMaxObs; ++i) local_obs[i] = 0;\n"
        << "    for (uint32_t i = 0; i < kMaxExpVals; ++i) local_exp[i] = 0.0;\n"
        << "\n"
        << "    if (batch_shot_id < shots) {\n"
        << "        uint64_t shot_id = shot_offset + batch_shot_id;\n";

    // Only emit RNG init if needed
    if (uf.needs_rng) {
        out << "        Rng rng;\n"
            << "        rng.seed(seed, shot_id);\n";
    }

    out << "\n"
        << "        ShotState st;\n"
        << "        st.px[0] = 0; st.px[1] = 0;\n"
        << "        st.pz[0] = 0; st.pz[1] = 0;\n"
        << "        st.active_k = 0;\n"
        << "        st.next_noise_idx = 0;\n"
        << "        st.discarded = false;\n"
        << "        st.v[0] = {1.0f, 0.0f};\n"
        << "        for (uint32_t i = 0; i < num_observables; ++i) st.obs[i] = 0;\n"
        << "        for (uint32_t i = 0; i < num_exp_vals; ++i) st.exp_vals[i] = 0.0;\n"
        << "\n";

    // Initial noise draw (only if circuit has noise)
    if (has_noise) {
        out << "        // Initial noise draw\n"
            << "        draw_next_noise_compiled(st, rng, kNoiseHazards, "
            << flat.noise_sites.size() << "u);\n\n";
    }

    // Straight-line instruction sequence
    out << "        // --- Begin instruction sequence (" << flat.instrs.size() << " ops) ---\n";
    codegen::emit_instructions(out, flat);
    out << "        // --- End instruction sequence ---\n";

    // Accumulate results
    out << "\n"
        << "    done:\n"
        << "        if (!st.discarded) {\n"
        << "            local_passed = 1;\n"
        << "            bool any_logical = false;\n"
        << "            for (uint32_t i = 0; i < num_observables; ++i) {\n"
        << "                uint8_t val = st.obs[i];\n";

    if (uf.observable) {
        out << "                if (kExpectedObservables[i] != 0) val ^= 1u;\n";
    }

    out << "                if (val != 0) {\n"
        << "                    local_obs[i] = 1;\n"
        << "                    any_logical = true;\n"
        << "                }\n"
        << "            }\n"
        << "            local_logical = any_logical ? 1 : 0;\n"
        << "            for (uint32_t i = 0; i < num_exp_vals; ++i) {\n"
        << "                local_exp[i] = st.exp_vals[i];\n"
        << "            }\n"
        << "        }\n"
        << "    } // end if (batch_shot_id < shots)\n"
        << "\n";

    // Warp-shuffle reduction (GEAK OPT-1)
    out << kWarpShuffleReduction;

    out << "}\n";

    return out.str();
}

}  // namespace gpu
}  // namespace clifft
