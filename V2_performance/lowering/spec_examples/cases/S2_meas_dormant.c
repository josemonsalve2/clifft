// TIER: register
// S2 -- flag folding on dormant measurements. `flags` selects between the
// FLAG_IDENTITY constant-outcome path and the frame-read path, and supplies the
// sign XOR. Constant flags delete one of the two paths outright; runtime flags
// keep both plus the test.
#include "../harness.h"
CASE_KERNEL_HEAD {
    (void)v; (void)scratch; (void)fused_u2; (void)fused_u4; (void)pauli_masks;
    (void)noise_sites; (void)noise_channels; (void)noise_hazards; (void)num_noise_sites;
    (void)readout_noise; (void)obs_off; (void)obs_tgt;
#if SPEC_FORM
    (void)instrs; (void)pc;
    v2_op_meas_dormant_static(st, 6u, 33u, 0u);
    v2_op_meas_dormant_static(st, 8u, 32u, 1u);
    v2_op_meas_dormant_random(st, 13u, 30u, 0u);
#else
    CV2Instr i0 = IN(0), i1 = IN(1), i2 = IN(2);
    v2_op_meas_dormant_static(st, i0.axis_1, i0.a, i0.flags);
    v2_op_meas_dormant_static(st, i1.axis_1, i1.a, i1.flags);
    v2_op_meas_dormant_random(st, i2.axis_1, i2.a, i2.flags);
#endif
}
