// TIER: register
// S8 -- Pauli-mask index folding. OP_APPLY_PAULI XORs a stored mask into the
// frame; the mask INDEX and the measurement slot fold, the mask CONTENTS do not
// (they live in a device buffer). Bounds how much a table-driven op can gain.
#include "../harness.h"
CASE_KERNEL_HEAD {
    (void)v; (void)scratch; (void)fused_u2; (void)fused_u4;
    (void)noise_sites; (void)noise_channels; (void)noise_hazards; (void)num_noise_sites;
    (void)readout_noise; (void)obs_off; (void)obs_tgt;
#if SPEC_FORM
    (void)instrs; (void)pc;
    v2_op_apply_pauli(st, pauli_masks, 0u, 21u);
    v2_op_apply_pauli(st, pauli_masks, 1u, 22u);
    v2_op_apply_pauli(st, pauli_masks, 2u, 23u);
#else
    CV2Instr i0 = IN(0), i1 = IN(1), i2 = IN(2);
    v2_op_apply_pauli(st, pauli_masks, i0.a, i0.b);
    v2_op_apply_pauli(st, pauli_masks, i1.a, i1.b);
    v2_op_apply_pauli(st, pauli_masks, i2.a, i2.b);
#endif
}
