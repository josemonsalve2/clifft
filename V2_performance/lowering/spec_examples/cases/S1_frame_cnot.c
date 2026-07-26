// TIER: register
// S1 -- frame operand folding. OP_FRAME_CNOT touches only the Pauli frame:
// two bit reads and two bit XORs at word/bit offsets derived from the axes.
// With constant axes those offsets are compile-time; with runtime axes each
// one costs a shift/mask chain plus a dynamic index into st->px / st->pz.
#include "../harness.h"
CASE_KERNEL_HEAD {
    (void)v; (void)scratch; (void)fused_u2; (void)fused_u4; (void)pauli_masks;
    (void)noise_sites; (void)noise_channels; (void)noise_hazards; (void)num_noise_sites;
    (void)readout_noise; (void)obs_off; (void)obs_tgt;
#if SPEC_FORM
    (void)instrs; (void)pc;
    v2_op_frame_cnot(st, 3u, 10u);
    v2_op_frame_cnot(st, 3u, 12u);
    v2_op_frame_cz(st, 3u, 6u);
#else
    CV2Instr i0 = IN(0), i1 = IN(1), i2 = IN(2);
    v2_op_frame_cnot(st, i0.axis_1, i0.axis_2);
    v2_op_frame_cnot(st, i1.axis_1, i1.axis_2);
    v2_op_frame_cz(st, i2.axis_1, i2.axis_2);
#endif
}
