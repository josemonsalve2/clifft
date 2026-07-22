#include "clifft/gpu/codegen/codegen_types.h"

#include <cstdint>
#include <iomanip>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace clifft {
namespace gpu {
namespace codegen {

using Op = clifft::Opcode;

// -----------------------------------------------------------------------
// Emit straight-line instructions (register-tier)
// -----------------------------------------------------------------------
void emit_instructions(std::ostringstream& out, const FlattenedProgram& flat) {
    for (size_t pc = 0; pc < flat.instrs.size(); ++pc) {
        const GpuInstr& instr = flat.instrs[pc];
        auto op = static_cast<Op>(instr.opcode);
        bool sign = (instr.flags & kFlagSign) != 0;
        bool identity = (instr.flags & kFlagIdentity) != 0;
        bool expected_one = (instr.flags & kFlagExpectedOne) != 0;

        switch (op) {
            case Op::OP_FRAME_CNOT:
                out << "    frame_cnot(st, " << instr.axis_1 << "u, " << instr.axis_2 << "u);\n";
                break;
            case Op::OP_FRAME_CZ:
                out << "    frame_cz(st, " << instr.axis_1 << "u, " << instr.axis_2 << "u);\n";
                break;
            case Op::OP_FRAME_H:
                out << "    frame_h(st, " << instr.axis_1 << "u);\n";
                break;
            case Op::OP_FRAME_S:
            case Op::OP_FRAME_S_DAG:
                out << "    frame_s(st, " << instr.axis_1 << "u);\n";
                break;
            case Op::OP_FRAME_SWAP:
                out << "    frame_swap(st, " << instr.axis_1 << "u, " << instr.axis_2 << "u);\n";
                break;
            case Op::OP_ARRAY_CNOT:
                out << "    array_cnot(st, " << instr.axis_1 << "u, " << instr.axis_2 << "u);\n";
                break;
            case Op::OP_ARRAY_CZ:
                out << "    array_cz(st, " << instr.axis_1 << "u, " << instr.axis_2 << "u);\n";
                break;
            case Op::OP_ARRAY_SWAP:
                out << "    array_swap(st, " << instr.axis_1 << "u, " << instr.axis_2 << "u);\n";
                break;
            case Op::OP_ARRAY_MULTI_CNOT:
                out << "    array_multi_cnot(st, " << instr.axis_1
                    << "u, 0x" << std::hex << instr.mask << std::dec << "ULL);\n";
                break;
            case Op::OP_ARRAY_MULTI_CZ:
                out << "    array_multi_cz(st, " << instr.axis_1
                    << "u, 0x" << std::hex << instr.mask << std::dec << "ULL);\n";
                break;
            case Op::OP_ARRAY_H:
                out << "    array_h(st, " << instr.axis_1 << "u);\n";
                break;
            case Op::OP_ARRAY_S:
                out << "    array_s(st, " << instr.axis_1 << "u, false);\n";
                break;
            case Op::OP_ARRAY_S_DAG:
                out << "    array_s(st, " << instr.axis_1 << "u, true);\n";
                break;
            case Op::OP_ARRAY_T:
                out << "    array_t(st, " << instr.axis_1 << "u, false);\n";
                break;
            case Op::OP_ARRAY_T_DAG:
                out << "    array_t(st, " << instr.axis_1 << "u, true);\n";
                break;
            case Op::OP_ARRAY_ROT:
                out << "    array_rot(st, " << instr.axis_1 << "u, "
                    << hex_double(instr.weight_re) << ", " << hex_double(instr.weight_im) << ");\n";
                break;
            case Op::OP_ARRAY_U2:
                if (!flat.fused_u2.empty()) {
                    out << "    array_u2_compiled(st, " << instr.axis_1 << "u, kFusedU2, "
                        << instr.a << "u);\n";
                }
                break;
            case Op::OP_ARRAY_U4:
                if (!flat.fused_u4.empty()) {
                    out << "    array_u4_compiled(st, " << instr.axis_1 << "u, "
                        << instr.axis_2 << "u, kFusedU4, " << instr.a << "u);\n";
                }
                break;
            case Op::OP_EXPAND:
                out << "    expand_plain(st);\n";
                break;
            case Op::OP_EXPAND_T:
                out << "    expand_t(st, " << instr.axis_1 << "u, false);\n";
                break;
            case Op::OP_EXPAND_T_DAG:
                out << "    expand_t(st, " << instr.axis_1 << "u, true);\n";
                break;
            case Op::OP_EXPAND_ROT:
                out << "    expand_rot(st, " << instr.axis_1 << "u, "
                    << hex_double(instr.weight_re) << ", " << hex_double(instr.weight_im) << ");\n";
                break;
            case Op::OP_MEAS_DORMANT_STATIC:
                if (identity) {
                    out << "    st.meas[" << instr.a << "u] = " << (sign ? 1u : 0u) << "u;\n";
                } else {
                    out << "    st.meas[" << instr.a << "u] = "
                        << "(bit_get(st.px, " << instr.axis_1 << "u) ? 1u : 0u)"
                        << (sign ? " ^ 1u" : "") << ";\n";
                }
                break;
            case Op::OP_MEAS_DORMANT_RANDOM:
                out << "    meas_dormant_random(st, rng, " << instr.axis_1 << "u, "
                    << instr.a << "u, " << (sign ? "true" : "false") << ");\n";
                break;
            case Op::OP_MEAS_ACTIVE_DIAGONAL:
                out << "    meas_active_diagonal(st, rng, " << instr.axis_1 << "u, "
                    << instr.a << "u, " << (sign ? "true" : "false") << ");\n";
                break;
            case Op::OP_MEAS_ACTIVE_INTERFERE:
                out << "    meas_active_interfere(st, rng, " << instr.axis_1 << "u, "
                    << instr.a << "u, " << (sign ? "true" : "false") << ");\n";
                break;
            case Op::OP_SWAP_MEAS_INTERFERE:
                out << "    swap_meas_interfere(st, rng, " << instr.axis_1 << "u, "
                    << instr.axis_2 << "u, " << instr.a << "u, "
                    << (sign ? "true" : "false") << ");\n";
                break;
            case Op::OP_APPLY_PAULI:
                if (!flat.pauli_masks.empty()) {
                    out << "    apply_pauli(st, kPauliMasks, " << instr.a << "u, "
                        << instr.b << "u);\n";
                }
                break;
            case Op::OP_NOISE:
                out << "    if (st.next_noise_idx == " << instr.a << "u) {\n"
                    << "        const GpuNoiseSite* site = &kNoiseSites[" << instr.a << "u];\n"
                    << "        double roll = rng.uniform() * site->prob_sum;\n"
                    << "        double cumulative = 0.0;\n"
                    << "        for (uint32_t k = 0; k < site->count; ++k) {\n"
                    << "            const GpuChannel* ch = &kNoiseChannels[site->offset + k];\n"
                    << "            cumulative += ch->prob;\n"
                    << "            if (roll < cumulative) {\n"
                    << "                apply_pauli_to_frame(st, ch->x, ch->z);\n"
                    << "                break;\n"
                    << "            }\n"
                    << "        }\n"
                    << "        st.next_noise_idx = " << instr.a << "u + 1;\n"
                    << "        draw_next_noise_compiled(st, rng, kNoiseHazards, "
                    << flat.noise_sites.size() << "u);\n"
                    << "    }\n";
                break;
            case Op::OP_NOISE_BLOCK:
                {
                    uint32_t start = instr.a;
                    uint32_t end = start + instr.b;
                    out << "    {\n"
                        << "        uint32_t _noise_end = " << end << "u;\n"
                        << "        while (st.next_noise_idx >= " << start << "u && "
                        << "st.next_noise_idx < _noise_end) {\n"
                        << "            uint32_t _si = st.next_noise_idx;\n"
                        << "            const GpuNoiseSite* site = &kNoiseSites[_si];\n"
                        << "            double roll = rng.uniform() * site->prob_sum;\n"
                        << "            double cumulative = 0.0;\n"
                        << "            for (uint32_t k = 0; k < site->count; ++k) {\n"
                        << "                const GpuChannel* ch = &kNoiseChannels[site->offset + k];\n"
                        << "                cumulative += ch->prob;\n"
                        << "                if (roll < cumulative) {\n"
                        << "                    apply_pauli_to_frame(st, ch->x, ch->z);\n"
                        << "                    break;\n"
                        << "                }\n"
                        << "            }\n"
                        << "            st.next_noise_idx = _si + 1;\n"
                        << "            draw_next_noise_compiled(st, rng, kNoiseHazards, "
                        << flat.noise_sites.size() << "u);\n"
                        << "        }\n"
                        << "    }\n";
                }
                break;
            case Op::OP_READOUT_NOISE:
                out << "    if (rng.uniform() < kReadoutNoise[" << instr.a << "u].prob) {\n"
                    << "        st.meas[kReadoutNoise[" << instr.a << "u].meas_idx] ^= 1u;\n"
                    << "    }\n";
                break;
            case Op::OP_DETECTOR:
                // No-op at runtime
                break;
            case Op::OP_POSTSELECT:
                out << "    {\n";
                out << "        uint32_t _ps_start = kDetectorOffsets[" << instr.a << "u];\n";
                out << "        uint32_t _ps_end = kDetectorOffsets[" << instr.a << "u + 1u];\n";
                out << "        uint8_t _ps_parity = " << (expected_one ? 1u : 0u) << "u;\n";
                out << "        for (uint32_t i = _ps_start; i < _ps_end; ++i) {\n";
                out << "            _ps_parity ^= st.meas[kDetectorTargets[i]];\n";
                out << "        }\n";
                out << "        if (_ps_parity != 0) { st.discarded = true; }\n";
                out << "    }\n";
                out << "    if (st.discarded) goto done;\n";
                break;
            case Op::OP_OBSERVABLE:
                out << "    {\n";
                out << "        uint32_t _ob_start = kObservableOffsets[" << instr.a << "u];\n";
                out << "        uint32_t _ob_end = kObservableOffsets[" << instr.a << "u + 1u];\n";
                out << "        uint8_t _ob_parity = 0u;\n";
                out << "        for (uint32_t i = _ob_start; i < _ob_end; ++i) {\n";
                out << "            _ob_parity ^= st.meas[kObservableTargets[i]];\n";
                out << "        }\n";
                out << "        st.obs[" << instr.b << "u] ^= _ob_parity;\n";
                out << "    }\n";
                break;
            case Op::OP_EXP_VAL:
                if (!flat.exp_val_masks.empty()) {
                    out << "    exec_exp_val_compiled(st, kExpValMasks, " << instr.a << "u, "
                        << instr.b << "u);\n";
                }
                break;
            default:
                out << "    st.discarded = true; goto done; // unknown opcode "
                    << (unsigned)instr.opcode << "\n";
                break;
        }
    }
}

// -----------------------------------------------------------------------
// emit_coop_instructions: like emit_instructions but for cooperative kernels
// Uses coop_* functions and thread-0-only guards for frame/classical ops
// -----------------------------------------------------------------------
void emit_coop_instructions(std::ostringstream& out, const FlattenedProgram& flat,
                                    const std::vector<PipelineOp>& pipe_ops) {
    // Build pipeline-pair lookup for O(1) check
    std::set<size_t> pipeline_second_pcs;
    for (const auto& p : pipe_ops) {
        (void)p;  // used indirectly below
        pipeline_second_pcs.insert(p.second_pc);
    }

    for (size_t pc = 0; pc < flat.instrs.size(); ++pc) {
        const GpuInstr& instr = flat.instrs[pc];
        auto op = static_cast<Op>(instr.opcode);
        bool sign = (instr.flags & kFlagSign) != 0;
        bool identity = (instr.flags & kFlagIdentity) != 0;
        bool expected_one = (instr.flags & kFlagExpectedOne) != 0;

        switch (op) {
            // Frame-only ops: executed by thread 0 only, then sync
            case Op::OP_FRAME_CNOT:
                out << "    if (TIDX == 0) frame_cnot_raw(px, pz, "
                    << instr.axis_1 << "u, " << instr.axis_2 << "u);\n"
                    << "    BARRIER();\n";
                break;
            case Op::OP_FRAME_CZ:
                out << "    if (TIDX == 0) frame_cz_raw(px, pz, "
                    << instr.axis_1 << "u, " << instr.axis_2 << "u);\n"
                    << "    BARRIER();\n";
                break;
            case Op::OP_FRAME_H:
                out << "    if (TIDX == 0) frame_h_raw(px, pz, " << instr.axis_1 << "u);\n"
                    << "    BARRIER();\n";
                break;
            case Op::OP_FRAME_S:
                out << "    if (TIDX == 0) frame_s_raw(px, pz, " << instr.axis_1 << "u);\n"
                    << "    BARRIER();\n";
                break;
            case Op::OP_FRAME_S_DAG:
                out << "    if (TIDX == 0) frame_s_raw(px, pz, " << instr.axis_1 << "u);\n"
                    << "    BARRIER();\n";
                break;
            case Op::OP_FRAME_SWAP:
                out << "    if (TIDX == 0) frame_swap_raw(px, pz, "
                    << instr.axis_1 << "u, " << instr.axis_2 << "u);\n"
                    << "    BARRIER();\n";
                break;

            // Array sweep ops: cooperative, include internal sync
            case Op::OP_ARRAY_CNOT:
                out << "    coop_array_cnot(v, px, pz, *active_k_ptr, "
                    << instr.axis_1 << "u, " << instr.axis_2 << "u);\n";
                break;
            case Op::OP_ARRAY_CZ:
                out << "    coop_array_cz(v, px, pz, *active_k_ptr, "
                    << instr.axis_1 << "u, " << instr.axis_2 << "u);\n";
                break;
            case Op::OP_ARRAY_H:
                out << "    coop_array_h(v, *active_k_ptr, " << instr.axis_1 << "u);\n";
                break;
            case Op::OP_ARRAY_S:
                out << "    coop_array_s(v, px, pz, *active_k_ptr, "
                    << instr.axis_1 << "u, false);\n";
                break;
            case Op::OP_ARRAY_S_DAG:
                out << "    coop_array_s(v, px, pz, *active_k_ptr, "
                    << instr.axis_1 << "u, true);\n";
                break;
            case Op::OP_ARRAY_T:
                out << "    coop_array_t(v, px, *active_k_ptr, "
                    << instr.axis_1 << "u, false);\n";
                break;
            case Op::OP_ARRAY_T_DAG:
                out << "    coop_array_t(v, px, *active_k_ptr, "
                    << instr.axis_1 << "u, true);\n";
                break;
            case Op::OP_ARRAY_U2:
                if (!flat.fused_u2.empty()) {
                    out << "    coop_array_u2(v, px, pz, *active_k_ptr, "
                        << instr.axis_1 << "u, kFusedU2, " << instr.a << "u);\n";
                }
                break;
            case Op::OP_ARRAY_U4:
                if (!flat.fused_u4.empty()) {
                    out << "    coop_array_u4(v, px, pz, *active_k_ptr, "
                        << instr.axis_1 << "u, " << instr.axis_2 << "u, kFusedU4, "
                        << instr.a << "u);\n";
                }
                break;

            // Expand ops: cooperative
            case Op::OP_EXPAND:
                out << "    coop_expand_plain(v, active_k_ptr);\n";
                break;
            case Op::OP_EXPAND_T:
                out << "    coop_expand_t(v, px, active_k_ptr, "
                    << instr.axis_1 << "u, false);\n";
                break;
            case Op::OP_EXPAND_T_DAG:
                out << "    coop_expand_t(v, px, active_k_ptr, "
                    << instr.axis_1 << "u, true);\n";
                break;

            // Measurements: thread 0 handles dormant/classical
            case Op::OP_MEAS_DORMANT_STATIC:
                if (identity) {
                    out << "    if (TIDX == 0) meas[" << instr.a << "u] = "
                        << (sign ? 1u : 0u) << "u;\n"
                        << "    BARRIER();\n";
                } else {
                    out << "    if (TIDX == 0) meas[" << instr.a << "u] = "
                        << "(bit_get(px, " << instr.axis_1 << "u) ? 1u : 0u)"
                        << (sign ? " ^ 1u" : "") << ";\n"
                        << "    BARRIER();\n";
                }
                break;
            case Op::OP_MEAS_DORMANT_RANDOM:
                out << "    if (TIDX == 0) {\n"
                    << "        uint8_t _m = rng.uniform() < 0.5 ? 0 : 1;\n"
                    << "        bit_set(px, " << instr.axis_1 << "u, _m != 0);\n"
                    << "        bit_set(pz, " << instr.axis_1 << "u, false);\n"
                    << "        meas[" << instr.a << "u] = _m ^ " << (sign ? 1u : 0u) << "u;\n"
                    << "    }\n"
                    << "    BARRIER();\n";
                break;
            case Op::OP_MEAS_ACTIVE_DIAGONAL:
                out << "    coop_meas_active_diagonal(v, px, pz, active_k_ptr, meas, "
                    << "red0, red1, rng, " << instr.axis_1 << "u, "
                    << instr.a << "u, " << (sign ? "true" : "false") << ");\n";
                break;

            // Apply Pauli: thread 0 only -- directly XOR px/pz arrays
            case Op::OP_APPLY_PAULI:
                if (!flat.pauli_masks.empty()) {
                    out << "    if (TIDX == 0) {\n"
                        << "        if (meas[" << instr.b << "u] != 0) {\n"
                        << "            const GpuMask& _pm = kPauliMasks[" << instr.a << "u];\n"
                        << "            px[0] ^= _pm.x[0]; px[1] ^= _pm.x[1];\n"
                        << "            pz[0] ^= _pm.z[0]; pz[1] ^= _pm.z[1];\n"
                        << "        }\n"
                        << "    }\n"
                        << "    BARRIER();\n";
                }
                break;

            // Noise: thread 0 only -- must call coop_draw_next_noise to keep RNG in sync
            case Op::OP_NOISE:
                out << "    if (TIDX == 0) {\n"
                    << "        if (*next_noise_idx == " << instr.a << "u) {\n"
                    << "            const GpuNoiseSite* site = &kNoiseSites[" << instr.a << "u];\n"
                    << "            double roll = rng.uniform() * site->prob_sum;\n"
                    << "            double cumulative = 0.0;\n"
                    << "            for (uint32_t k = 0; k < site->count; ++k) {\n"
                    << "                const GpuChannel* ch = &kNoiseChannels[site->offset + k];\n"
                    << "                cumulative += ch->prob;\n"
                    << "                if (roll < cumulative) {\n"
                    << "                    px[0] ^= ch->x[0]; px[1] ^= ch->x[1];\n"
                    << "                    pz[0] ^= ch->z[0]; pz[1] ^= ch->z[1];\n"
                    << "                    break;\n"
                    << "                }\n"
                    << "            }\n"
                    << "            *next_noise_idx = " << instr.a << "u + 1;\n"
                    << "            coop_draw_next_noise(next_noise_idx, rng, kNoiseHazards, "
                    << flat.noise_sites.size() << "u);\n"
                    << "        }\n"
                    << "    }\n"
                    << "    BARRIER();\n";
                break;

            // Postselect
            case Op::OP_POSTSELECT:
                out << "    if (TIDX == 0) {\n"
                    << "        uint32_t _ps_start = kDetectorOffsets[" << instr.a << "u];\n"
                    << "        uint32_t _ps_end = kDetectorOffsets[" << instr.a << "u + 1u];\n"
                    << "        uint8_t _ps_parity = " << (expected_one ? 1u : 0u) << "u;\n"
                    << "        for (uint32_t i = _ps_start; i < _ps_end; ++i)\n"
                    << "            _ps_parity ^= meas[kDetectorTargets[i]];\n"
                    << "        if (_ps_parity != 0) *discarded_ptr = 1;\n"
                    << "    }\n"
                    << "    BARRIER();\n"
                    << "    if (*discarded_ptr) goto done;\n";
                break;

            // Observable
            case Op::OP_OBSERVABLE:
                out << "    if (TIDX == 0) {\n"
                    << "        uint32_t _ob_start = kObservableOffsets[" << instr.a << "u];\n"
                    << "        uint32_t _ob_end = kObservableOffsets[" << instr.a << "u + 1u];\n"
                    << "        uint8_t _ob_parity = 0u;\n"
                    << "        for (uint32_t i = _ob_start; i < _ob_end; ++i)\n"
                    << "            _ob_parity ^= meas[kObservableTargets[i]];\n"
                    << "        obs[" << instr.b << "u] ^= _ob_parity;\n"
                    << "    }\n"
                    << "    BARRIER();\n";
                break;

            // Array swap
            case Op::OP_ARRAY_SWAP:
                out << "    coop_array_swap(v, px, pz, *active_k_ptr, "
                    << instr.axis_1 << "u, " << instr.axis_2 << "u);\n";
                break;

            // Multi-controlled CNOT
            case Op::OP_ARRAY_MULTI_CNOT:
                out << "    coop_array_multi_cnot(v, px, pz, *active_k_ptr, "
                    << instr.axis_1 << "u, 0x" << std::hex << instr.mask << std::dec << "ULL);\n";
                break;

            // Multi-controlled CZ
            case Op::OP_ARRAY_MULTI_CZ:
                out << "    coop_array_multi_cz(v, px, pz, *active_k_ptr, "
                    << instr.axis_1 << "u, 0x" << std::hex << instr.mask << std::dec << "ULL);\n";
                break;

            // Z-rotation on an axis
            case Op::OP_ARRAY_ROT:
                out << "    coop_array_rot(v, px, *active_k_ptr, "
                    << instr.axis_1 << "u, " << hex_double(instr.weight_re)
                    << ", " << hex_double(instr.weight_im) << ");\n";
                break;

            // Expand with Z-rotation phase
            case Op::OP_EXPAND_ROT:
                out << "    coop_expand_rot(v, px, active_k_ptr, "
                    << instr.axis_1 << "u, " << hex_double(instr.weight_re)
                    << ", " << hex_double(instr.weight_im) << ");\n";
                break;

            // Active interfere measurement
            case Op::OP_MEAS_ACTIVE_INTERFERE:
                out << "    coop_meas_active_interfere(v, px, pz, active_k_ptr, meas, "
                    << "red0, red1, rng, " << instr.axis_1 << "u, "
                    << instr.a << "u, " << (sign ? "true" : "false") << ");\n";
                break;

            // Swap+measure with interference
            case Op::OP_SWAP_MEAS_INTERFERE:
                out << "    coop_swap_meas_interfere(v, scratch_v, px, pz, active_k_ptr, meas, "
                    << "red0, red1, rng, " << instr.axis_1 << "u, " << instr.axis_2 << "u, "
                    << instr.a << "u, " << (sign ? "true" : "false") << ");\n";
                break;

            // Expectation value computation
            case Op::OP_EXP_VAL:
                if (!flat.exp_val_masks.empty()) {
                    out << "    coop_exec_exp_val(v, px, pz, *active_k_ptr, exp_vals, "
                        << "kExpValMasks, " << instr.a << "u, " << instr.b << "u);\n";
                }
                break;

            case Op::OP_NOISE_BLOCK:
                out << "    if (TIDX == 0) {\n"
                    << "        uint32_t _nb_end = " << instr.a << "u + " << instr.b << "u;\n"
                    << "        while (*next_noise_idx >= " << instr.a << "u && *next_noise_idx < _nb_end) {\n"
                    << "            uint32_t _nb_site_idx = *next_noise_idx;\n"
                    << "            const GpuNoiseSite* site = &kNoiseSites[_nb_site_idx];\n"
                    << "            double roll = rng.uniform() * site->prob_sum;\n"
                    << "            double cumulative = 0.0;\n"
                    << "            for (uint32_t k = 0; k < site->count; ++k) {\n"
                    << "                const GpuChannel* ch = &kNoiseChannels[site->offset + k];\n"
                    << "                cumulative += ch->prob;\n"
                    << "                if (roll < cumulative) {\n"
                    << "                    px[0] ^= ch->x[0]; px[1] ^= ch->x[1];\n"
                    << "                    pz[0] ^= ch->z[0]; pz[1] ^= ch->z[1];\n"
                    << "                    break;\n"
                    << "                }\n"
                    << "            }\n"
                    << "            *next_noise_idx = _nb_site_idx + 1;\n"
                    << "            coop_draw_next_noise(next_noise_idx, rng, kNoiseHazards, "
                    << flat.noise_sites.size() << "u);\n"
                    << "        }\n"
                    << "    }\n"
                    << "    BARRIER();\n";
                break;

            case Op::OP_READOUT_NOISE:
                out << "    if (TIDX == 0) {\n"
                    << "        const GpuReadoutNoise* _rn = &kReadoutNoise[" << instr.a << "u];\n"
                    << "        if (rng.uniform() < _rn->prob) {\n"
                    << "            meas[_rn->meas_idx] ^= 1;\n"
                    << "        }\n"
                    << "    }\n"
                    << "    BARRIER();\n";
                break;

            case Op::OP_DETECTOR:
                out << "    BARRIER();\n";
                break;

            default:
                out << "    if (TIDX == 0) *discarded_ptr = 1;\n"
                    << "    goto done;\n";
                break;
        }
    }
}

}  // namespace codegen
}  // namespace gpu
}  // namespace clifft
