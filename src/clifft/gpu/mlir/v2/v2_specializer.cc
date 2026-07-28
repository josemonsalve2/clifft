// v2_specializer.cc — emit per-circuit specialized amdgcn C source. See header.
#include "clifft/gpu/mlir/v2/v2_specializer.h"
#include "clifft/gpu/gpu_types.h"
#include "clifft/backend/backend.h"   // Opcode enum values

#include <cstdlib>
#include <sstream>
#include <stdexcept>

namespace clifft::gpu::v2 {

SpecTier spec_tier_for_rank(uint32_t peak_rank) {
    if (peak_rank <= 4) return SpecTier::Register;
    if (peak_rank <= 10) return SpecTier::Coop;
    return SpecTier::Global;
}

namespace {

using clifft::Opcode;

// Emit one instruction as a v2_op_*() call with constant operands. `k` is the
// active_k BEFORE this instruction (statically tracked). Advances *k for
// rank-changing ops. Returns false if the opcode is unsupported (caller aborts
// -> interpreter fallback).
bool emit_instr(std::ostream& o, const GpuInstr& in, uint32_t* k) {
    const uint32_t a1 = in.axis_1, a2 = in.axis_2, a = in.a, b = in.b;
    const uint32_t flags = in.flags;
    auto op = static_cast<Opcode>(in.opcode);
    o << "    ";
    switch (op) {
    case Opcode::OP_FRAME_CNOT: o << "v2_op_frame_cnot(st, " << a1 << "u, " << a2 << "u);"; break;
    case Opcode::OP_FRAME_CZ:   o << "v2_op_frame_cz(st, " << a1 << "u, " << a2 << "u);"; break;
    case Opcode::OP_FRAME_H:    o << "v2_op_frame_h(st, " << a1 << "u);"; break;
    case Opcode::OP_FRAME_S:
    case Opcode::OP_FRAME_S_DAG: o << "v2_op_frame_s(st, " << a1 << "u);"; break;
    case Opcode::OP_FRAME_SWAP: o << "v2_op_frame_swap(st, " << a1 << "u, " << a2 << "u);"; break;
    case Opcode::OP_MEAS_DORMANT_STATIC:
        o << "v2_op_meas_dormant_static(st, " << a1 << "u, " << a << "u, " << flags << "u);"; break;
    case Opcode::OP_MEAS_DORMANT_RANDOM:
        o << "v2_op_meas_dormant_random(st, " << a1 << "u, " << a << "u, " << flags << "u);"; break;
    case Opcode::OP_EXPAND:
        o << "v2_op_expand(st, v, " << *k << "u);"; ++*k; break;
    case Opcode::OP_EXPAND_T:
        o << "v2_op_expand_t(st, v, " << *k << "u, " << a1 << "u, 0);"; ++*k; break;
    case Opcode::OP_EXPAND_T_DAG:
        o << "v2_op_expand_t(st, v, " << *k << "u, " << a1 << "u, 1);"; ++*k; break;
    case Opcode::OP_EXPAND_ROT:
        o << "v2_op_expand_rot(st, v, " << *k << "u, " << a1 << "u, "
          << in.weight_re << ", " << in.weight_im << ");"; ++*k; break;
    case Opcode::OP_MEAS_ACTIVE_DIAGONAL:
        o << "v2_op_meas_active_diagonal(st, v, " << *k << "u, " << a1 << "u, " << a << "u, " << flags << "u);";
        if (*k) --*k; break;
    case Opcode::OP_MEAS_ACTIVE_INTERFERE:
        o << "v2_op_meas_active_interfere(st, v, " << *k << "u, " << a1 << "u, " << a << "u, " << flags << "u);";
        if (*k) --*k; break;
    case Opcode::OP_ARRAY_CNOT: o << "v2_op_array_cnot(st, v, " << *k << "u, " << a1 << "u, " << a2 << "u);"; break;
    case Opcode::OP_ARRAY_CZ:   o << "v2_op_array_cz(st, v, " << *k << "u, " << a1 << "u, " << a2 << "u);"; break;
    case Opcode::OP_ARRAY_SWAP: o << "v2_op_array_swap(st, v, " << *k << "u, " << a1 << "u, " << a2 << "u);"; break;
    case Opcode::OP_ARRAY_MULTI_CNOT:
        o << "v2_op_array_multi_cnot(st, v, " << *k << "u, " << a1 << "u, " << in.mask << "ull);"; break;
    case Opcode::OP_ARRAY_MULTI_CZ:
        o << "v2_op_array_multi_cz(st, v, " << *k << "u, " << a1 << "u, " << in.mask << "ull);"; break;
    case Opcode::OP_ARRAY_H:    o << "v2_op_array_h(st, v, " << *k << "u, " << a1 << "u);"; break;
    case Opcode::OP_ARRAY_S:    o << "v2_op_array_s(st, v, " << *k << "u, " << a1 << "u, 0);"; break;
    case Opcode::OP_ARRAY_S_DAG:o << "v2_op_array_s(st, v, " << *k << "u, " << a1 << "u, 1);"; break;
    case Opcode::OP_ARRAY_T:    o << "v2_op_array_t(st, v, " << *k << "u, " << a1 << "u, 0);"; break;
    case Opcode::OP_ARRAY_T_DAG:o << "v2_op_array_t(st, v, " << *k << "u, " << a1 << "u, 1);"; break;
    case Opcode::OP_ARRAY_ROT:
        o << "v2_op_array_rot(st, v, " << *k << "u, " << a1 << "u, "
          << in.weight_re << ", " << in.weight_im << ");"; break;
    case Opcode::OP_ARRAY_U2:
        o << "v2_op_array_u2(st, v, " << *k << "u, " << a1 << "u, fused_u2, " << a << "u);"; break;
    case Opcode::OP_ARRAY_U4:
        o << "v2_op_array_u4(st, v, " << *k << "u, " << a1 << "u, " << a2 << "u, fused_u4, " << a << "u);"; break;
    case Opcode::OP_SWAP_MEAS_INTERFERE:
        o << "v2_op_swap_meas_interfere(st, v, scratch, " << *k << "u, " << a1 << "u, " << a2 << "u, "
          << a << "u, " << flags << "u);";
        if (*k) --*k; break;
    case Opcode::OP_APPLY_PAULI: o << "v2_op_apply_pauli(st, pauli_masks, " << a << "u, " << b << "u);"; break;
    case Opcode::OP_NOISE:
        o << "v2_op_noise(st, noise_sites, noise_channels, noise_hazards, num_noise_sites, " << a << "u);"; break;
    case Opcode::OP_NOISE_BLOCK:
        o << "v2_op_noise_block(st, noise_sites, noise_channels, noise_hazards, num_noise_sites, "
          << a << "u, " << b << "u);"; break;
    case Opcode::OP_READOUT_NOISE: o << "v2_op_readout_noise(st, readout_noise, " << a << "u);"; break;
    case Opcode::OP_OBSERVABLE:
        o << "v2_op_observable(st, observable_offsets, observable_targets, " << a << "u, " << b << "u);"; break;
    case Opcode::OP_POSTSELECT:
        o << "if (v2_op_postselect(st, detector_offsets, detector_targets, " << a << "u, " << flags << "u)) return;"; break;
    case Opcode::OP_DETECTOR: o << "v2_op_detector();"; break;
    default:
        return false;  // unsupported (forced variants, exp_val) -> fallback
    }
    o << "\n";
    return true;
}

const char* tier_macro(SpecTier t) {
    return t == SpecTier::Register ? "V2_REGISTER" : "V2_COOP_OR_GLOBAL";
}

}  // namespace

std::string emit_specialized_kernel(const FlattenedProgram& flat,
                                    const std::string& kernel_name,
                                    SpecTier tier) {
    std::ostringstream o;
    o.precision(17);  // exact double literals for weights
    // The register build must define V2_REGISTER BEFORE including v2_ops.h.
    if (tier == SpecTier::Register) o << "#define V2_REGISTER 1\n";
    // Emit the FP-carrying noise ops as NON-inlined calls: each call site is an
    // optimization barrier the -O2 FP scheduler cannot reorder across, so the N
    // straight-lined noise ops behave like the interpreter's single loop-body
    // (which the pc-loop iteration already fences). This is the specializer
    // analog of resolving noise to a loop, and it restores byte-exactness on
    // READOUT_NOISE/NOISE_BLOCK-heavy circuits (circuit_d5) without touching the
    // interpreter/register builds (they keep always_inline).
    // V2_SPEC_NOISE_INLINE=1 flips this back to always_inline (the interpreter's
    // own setting) so the noinline-fence hypothesis can be A/B tested rather
    // than assumed: if a circuit's gate verdict is unchanged with the fence
    // removed, the fence is not what that circuit's divergence hinges on.
    if (getenv("V2_SPEC_NOISE_INLINE"))
        o << "#define V2_NOISE_ATTR __attribute__((always_inline))\n";
    else
        o << "#define V2_NOISE_ATTR __attribute__((noinline))\n";
    // V2_U4_PREFETCH=1 software-pipelines the U4 butterfly (see v2_ops.h). It
    // is byte-exact by construction -- only the fetch point moves, not the
    // arithmetic -- so it is a pure latency/occupancy trade to be A/B measured.
    if (getenv("V2_U4_PREFETCH"))
        o << "#define V2_U4_PREFETCH 1\n";
    o << "#include \"clifft/gpu/mlir/v2/v2_ops.h\"\n\n";

    // --- specialized straight-line body (shared by the tier wrapper) ---------
    // Emitted ONCE; the per-instruction operands + active_k are constants.
    o << "static void spec_body(V2State* st, CV2Complex* v, CV2Complex* scratch,\n"
      << "                      u32 amp_capacity, u64 shot_id, u32 num_observables, u64 seed,\n"
      << "                      CV2BlockCounts* block_counts,\n"
      << "                      const CV2FusedU2Entry* fused_u2, const CV2FusedU4Entry* fused_u4,\n"
      << "                      const u32* observable_offsets, const u32* observable_targets,\n"
      << "                      const CV2NoiseSite* noise_sites, const CV2Channel* noise_channels,\n"
      << "                      const double* noise_hazards, u32 num_noise_sites,\n"
      << "                      const CV2Mask* pauli_masks, const CV2ReadoutNoise* readout_noise,\n"
      << "                      const u32* detector_offsets, const u32* detector_targets,\n"
      << "                      u32 expected_obs_mask) {\n"
      << "    v2_shot_init(st, v, amp_capacity, shot_id, num_observables, seed, noise_hazards, num_noise_sites);\n";

    uint32_t k = 0;
    for (uint32_t pc = 0; pc < flat.instrs.size(); ++pc) {
        if (!emit_instr(o, flat.instrs[pc], &k))
            throw std::runtime_error("v2 specializer: unsupported opcode " +
                                     std::to_string((int)flat.instrs[pc].opcode) + " at pc " +
                                     std::to_string(pc));
    }
    o << "    v2_shot_aggregate(st, block_counts, num_observables, expected_obs_mask);\n"
      << "}\n\n";

    // --- tier wrapper (same shape as the interpreter kernels) ----------------
    const char* args =
        "const CV2Instr* instrs, u32 num_instrs, u32 peak_rank,\n"
        "    u32 total_meas_slots, u32 num_observables,\n"
        "    u64 seed, u64 shot_offset, u64 shots,\n"
        "    CV2BlockCounts* block_counts,\n"
        "    const CV2FusedU2Entry* fused_u2, const CV2FusedU4Entry* fused_u4,\n"
        "    const u32* observable_offsets, const u32* observable_targets,\n"
        "    const CV2NoiseSite* noise_sites, const CV2Channel* noise_channels,\n"
        "    const double* noise_hazards, u32 num_noise_sites, u32 expected_obs_mask,\n"
        "    const CV2Mask* pauli_masks, const CV2ReadoutNoise* readout_noise,\n"
        "    const u32* detector_offsets, const u32* detector_targets";

    const char* fwd =
        "num_observables, seed, block_counts, fused_u2, fused_u4,\n"
        "                  observable_offsets, observable_targets, noise_sites, noise_channels,\n"
        "                  noise_hazards, num_noise_sites, pauli_masks, readout_noise,\n"
        "                  detector_offsets, detector_targets, expected_obs_mask";

    if (tier == SpecTier::Register) {
        o << "#define V2_REG_MAX_AMP 16\n"
          << "__attribute__((amdgpu_kernel, visibility(\"default\")))\n"
          << "void " << kernel_name << "(" << args << ") {\n"
          << "    (void)peak_rank; (void)num_instrs; (void)instrs; (void)total_meas_slots;\n"
          << "    u64 shot_id = shot_offset + ((u64)v2_bid() * 256ul + (u64)__builtin_amdgcn_workitem_id_x());\n"
          << "    if (shot_id >= shots) return;\n"
          << "    V2State st; CV2Complex vloc[V2_REG_MAX_AMP]; CV2Complex sloc[V2_REG_MAX_AMP/2];\n"
          << "    spec_body(&st, vloc, sloc, V2_REG_MAX_AMP, shot_id, " << fwd << ");\n"
          << "}\n";
    } else if (tier == SpecTier::Coop) {
        // Coop tier: 1 workgroup/shot, amplitudes + classical state in LDS.
        // Same LDS globals the interpreter's coop kernel uses.
        o << "extern __attribute__((address_space(3))) CV2Complex lds_v[V2_MAX_AMP];\n"
          << "extern __attribute__((address_space(3))) CV2Complex lds_red_scratch[V2_SCRATCH_AMP];\n"
          << "extern __attribute__((address_space(3))) V2State lds_state;\n"
          << "__attribute__((amdgpu_kernel, visibility(\"default\")))\n"
          << "void " << kernel_name << "(" << args << ") {\n"
          << "    (void)peak_rank; (void)num_instrs; (void)instrs; (void)total_meas_slots;\n"
          << "    u64 shot_id = shot_offset + (u64)v2_bid();\n"
          << "    if (shot_id >= shots) return;\n"
          << "    spec_body((V2State*)&lds_state, (CV2Complex*)lds_v, (CV2Complex*)lds_red_scratch,\n"
          << "              V2_MAX_AMP, shot_id, " << fwd << ");\n"
          << "}\n";
    } else {  // SpecTier::Global
        // Global tier (rank 11-26): amplitudes in HBM (one slice per resident
        // workgroup), classical state in LDS, shots drained via a single atomic
        // work-steal counter. Same spec_body (scatter index math constant-folded
        // by the emitted constant operands — the primary lever: it removes the
        // 2x-VALU per-amplitude scatter recompute the runtime interpreter pays).
        // amp_capacity = 1<<peak_rank is a compile-time constant here.
        const uint32_t amp_cap = 1u << flat.peak_rank;
        o << "extern __attribute__((address_space(3))) V2State lds_state;\n"
          << "extern __attribute__((address_space(3))) unsigned long lds_shot;\n"
          << "__attribute__((amdgpu_kernel, visibility(\"default\")))\n"
          << "void " << kernel_name << "(" << args << ",\n"
          << "    CV2Complex* global_v, CV2Complex* global_scratch, u64* work_counter) {\n"
          << "    (void)peak_rank; (void)num_instrs; (void)instrs; (void)total_meas_slots;\n"
          << "    u32 t = v2_tid();\n"
          << "    u32 slot = v2_bid();\n"
          << "    const u64 amp_capacity = " << amp_cap << "ull;\n"
          << "    CV2Complex* v = global_v + (u64)slot * amp_capacity;\n"
          << "    CV2Complex* scratch = global_scratch + (u64)slot * (amp_capacity >> 1);\n"
          << "    for (;;) {\n"
          << "        if (t == 0) lds_shot = __atomic_fetch_add(&work_counter[0], 1UL, __ATOMIC_RELAXED);\n"
          << "        v2_barrier();\n"
          << "        u64 batch_shot = lds_shot;\n"
          << "        if (batch_shot >= shots) return;\n"
          << "        spec_body((V2State*)&lds_state, v, scratch, (u32)amp_capacity,\n"
          << "                  shot_offset + batch_shot, " << fwd << ");\n"
          << "        v2_barrier();\n"
          << "    }\n"
          << "}\n";
    }
    (void)tier_macro;
    return o.str();
}

}  // namespace clifft::gpu::v2
