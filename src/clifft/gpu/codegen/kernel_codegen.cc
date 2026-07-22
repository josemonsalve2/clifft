#include "clifft/gpu/codegen/kernel_codegen.h"
#include "clifft/gpu/gpu_types.h"

#include <cstdint>
#include <vector>

namespace clifft {
namespace gpu {

using Op = clifft::Opcode;

// -----------------------------------------------------------------------
// UsedFunctions::compute_derived()
// -----------------------------------------------------------------------
void UsedFunctions::compute_derived() {
    needs_rng = meas_dormant_random || meas_active_diagonal ||
                meas_active_interfere || swap_meas_interfere ||
                noise || noise_block || readout_noise;

    needs_array_sweep = array_cnot || array_cz || array_swap ||
                        array_multi_cnot || array_multi_cz ||
                        array_h || array_s || array_t ||
                        array_rot || array_u2 || array_u4 ||
                        expand_plain || expand_t || expand_rot;

    // Complex arithmetic (cmul, cadd, csub, cscale, cnorm) is needed by
    // any array gate that transforms amplitudes (not just frame updates),
    // and by active measurements that compute probabilities.
    needs_complex_ops = array_h || array_s || array_t || array_rot ||
                        array_u2 || array_u4 ||
                        expand_t || expand_rot ||
                        meas_active_diagonal || meas_active_interfere ||
                        swap_meas_interfere || exp_val;

    needs_apply_phase = array_s || array_t || array_rot;

    // Frame ops are needed directly, and also called implicitly by array
    // 2-qubit ops (array_cnot calls frame_cnot, etc.)
    needs_frame_ops = frame_cnot || frame_cz || frame_h || frame_s || frame_swap ||
                      array_cnot || array_cz || array_swap ||
                      array_multi_cnot || array_multi_cz ||
                      array_h || array_s || array_t ||
                      swap_meas_interfere;

    // Scatter bits needed for 2-qubit array ops, 1-qubit array sweeps, expands
    needs_scatter = array_cnot || array_cz || array_swap ||
                    array_multi_cnot || array_multi_cz ||
                    array_h || array_s || array_t || array_rot ||
                    array_u2 || array_u4 ||
                    expand_plain || expand_t || expand_rot;

    needs_sample_branch = meas_active_diagonal || meas_active_interfere ||
                          swap_meas_interfere;
}

// -----------------------------------------------------------------------
// analyze_used_functions
// -----------------------------------------------------------------------
UsedFunctions analyze_used_functions(const FlattenedProgram& flat) {
    UsedFunctions uf{};
    for (const auto& instr : flat.instrs) {
        auto op = static_cast<Op>(instr.opcode);
        switch (op) {
            case Op::OP_FRAME_CNOT:      uf.frame_cnot = true; break;
            case Op::OP_FRAME_CZ:        uf.frame_cz = true; break;
            case Op::OP_FRAME_H:         uf.frame_h = true; break;
            case Op::OP_FRAME_S:
            case Op::OP_FRAME_S_DAG:     uf.frame_s = true; break;
            case Op::OP_FRAME_SWAP:      uf.frame_swap = true; break;
            case Op::OP_ARRAY_CNOT:      uf.array_cnot = true; break;
            case Op::OP_ARRAY_CZ:        uf.array_cz = true; break;
            case Op::OP_ARRAY_SWAP:      uf.array_swap = true; break;
            case Op::OP_ARRAY_MULTI_CNOT:uf.array_multi_cnot = true; break;
            case Op::OP_ARRAY_MULTI_CZ:  uf.array_multi_cz = true; break;
            case Op::OP_ARRAY_H:         uf.array_h = true; break;
            case Op::OP_ARRAY_S:
            case Op::OP_ARRAY_S_DAG:     uf.array_s = true; break;
            case Op::OP_ARRAY_T:
            case Op::OP_ARRAY_T_DAG:     uf.array_t = true; break;
            case Op::OP_ARRAY_ROT:       uf.array_rot = true; break;
            case Op::OP_ARRAY_U2:        uf.array_u2 = true; break;
            case Op::OP_ARRAY_U4:        uf.array_u4 = true; break;
            case Op::OP_EXPAND:          uf.expand_plain = true; break;
            case Op::OP_EXPAND_T:
            case Op::OP_EXPAND_T_DAG:    uf.expand_t = true; break;
            case Op::OP_EXPAND_ROT:      uf.expand_rot = true; break;
            case Op::OP_MEAS_DORMANT_STATIC:  uf.meas_dormant_static = true; break;
            case Op::OP_MEAS_DORMANT_RANDOM:  uf.meas_dormant_random = true; break;
            case Op::OP_MEAS_ACTIVE_DIAGONAL: uf.meas_active_diagonal = true; break;
            case Op::OP_MEAS_ACTIVE_INTERFERE:uf.meas_active_interfere = true; break;
            case Op::OP_SWAP_MEAS_INTERFERE:  uf.swap_meas_interfere = true; break;
            case Op::OP_APPLY_PAULI:     uf.apply_pauli = true; break;
            case Op::OP_NOISE:           uf.noise = true; break;
            case Op::OP_NOISE_BLOCK:     uf.noise_block = true; break;
            case Op::OP_READOUT_NOISE:   uf.readout_noise = true; break;
            case Op::OP_POSTSELECT:      uf.postselect = true; break;
            case Op::OP_OBSERVABLE:       uf.observable = true; break;
            case Op::OP_EXP_VAL:         uf.exp_val = true; break;
            default: break;
        }
    }
    uf.compute_derived();
    return uf;
}

// -----------------------------------------------------------------------
// analyze_pipeline_opportunities
//
// Two consecutive array-sweep instructions are independent when the set of
// amplitude indices they touch is disjoint.  For a single-qubit gate on axis
// A, the indices touched are those with bit A at a specific value -- i.e. they
// span ALL amplitudes with any pattern for other bits.  Two single-qubit gates
// on different axes are independent.  Two-qubit gates on axes (A, B) vs (C, D)
// are independent iff {A, B} cap {C, D} = empty.
//
// We encode the "qubit footprint" of an instruction as a bitmask over axis
// indices.  Zero footprint means the instruction doesn't touch amplitudes at
// all (frame-only or classical).
// -----------------------------------------------------------------------
static uint64_t instr_amplitude_footprint(const GpuInstr& instr) {
    auto op = static_cast<Op>(instr.opcode);
    switch (op) {
        // Two-qubit array ops: footprint is both axes
        case Op::OP_ARRAY_CNOT:
        case Op::OP_ARRAY_CZ:
        case Op::OP_ARRAY_SWAP:
            return (1ULL << instr.axis_1) | (1ULL << instr.axis_2);

        // Multi-qubit: footprint is axis_1 plus ctrl_mask
        case Op::OP_ARRAY_MULTI_CNOT:
        case Op::OP_ARRAY_MULTI_CZ:
            return instr.mask | (1ULL << instr.axis_1);

        // One-qubit array ops: footprint is axis_1 only
        case Op::OP_ARRAY_H:
        case Op::OP_ARRAY_S:
        case Op::OP_ARRAY_S_DAG:
        case Op::OP_ARRAY_T:
        case Op::OP_ARRAY_T_DAG:
        case Op::OP_ARRAY_ROT:
        case Op::OP_ARRAY_U2:
            return 1ULL << instr.axis_1;

        // Two-qubit unitary: both axes
        case Op::OP_ARRAY_U4:
            return (1ULL << instr.axis_1) | (1ULL << instr.axis_2);

        // Expand ops: touch all current amplitudes -> footprint = all bits
        case Op::OP_EXPAND:
        case Op::OP_EXPAND_T:
        case Op::OP_EXPAND_T_DAG:
        case Op::OP_EXPAND_ROT:
            return ~uint64_t{0};

        // Measurement: collapses one axis -- not safe to pipeline with anything
        case Op::OP_MEAS_ACTIVE_DIAGONAL:
        case Op::OP_MEAS_ACTIVE_INTERFERE:
        case Op::OP_SWAP_MEAS_INTERFERE:
            return ~uint64_t{0};

        // Frame-only, classical, dormant meas, noise, postselect: no amplitude access
        default:
            return 0;
    }
}

std::vector<PipelineOp> analyze_pipeline_opportunities(const FlattenedProgram& flat) {
    std::vector<PipelineOp> result;
    size_t n = flat.instrs.size();
    for (size_t pc = 0; pc + 1 < n; ++pc) {
        uint64_t fp0 = instr_amplitude_footprint(flat.instrs[pc]);
        uint64_t fp1 = instr_amplitude_footprint(flat.instrs[pc + 1]);
        // Both must have a non-trivial footprint (amplitude-touching) and be disjoint
        if (fp0 != 0 && fp1 != 0 && (fp0 & fp1) == 0) {
            result.push_back({pc, pc + 1});
        }
    }
    return result;
}

}  // namespace gpu
}  // namespace clifft
