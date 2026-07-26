// TIER: coop
// S6 -- fused-matrix table lookup. v2_op_array_u2 indexes fused_u2[cp] then
// picks matrices[in_state] where in_state comes from the LIVE Pauli frame, so
// the row is NOT foldable -- only the table entry `cp` and the axis are. This
// case bounds how much specialization can buy for a data-dependent op.
#include "../harness.h"
CASE_KERNEL_HEAD {
    (void)scratch; (void)fused_u4; (void)pauli_masks;
    (void)noise_sites; (void)noise_channels; (void)noise_hazards; (void)num_noise_sites;
    (void)readout_noise; (void)obs_off; (void)obs_tgt;
#if SPEC_FORM
    (void)instrs; (void)pc;
    v2_op_array_u2(st, v, 8u, 4u, fused_u2, 17u);
#else
    CV2Instr i0 = IN(0);
    v2_op_array_u2(st, v, st->active_k, i0.axis_1, fused_u2, i0.a);
#endif
}
