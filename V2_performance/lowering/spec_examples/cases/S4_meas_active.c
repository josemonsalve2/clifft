// TIER: coop
// S4 -- rank-folded cooperative reduction. v2_op_meas_active_interfere sweeps
// 2^(k-1) amplitudes accumulating two f64 partials, then coop_reduce2's
// warp butterfly. Constant k fixes the trip count and the LDS offsets;
// runtime k leaves the shift, the compare, and the address math dynamic.
#include "../harness.h"
CASE_KERNEL_HEAD {
    (void)scratch; (void)fused_u2; (void)fused_u4; (void)pauli_masks;
    (void)noise_sites; (void)noise_channels; (void)noise_hazards; (void)num_noise_sites;
    (void)readout_noise; (void)obs_off; (void)obs_tgt;
#if SPEC_FORM
    (void)instrs; (void)pc;
    v2_op_meas_active_interfere(st, v, 8u, 5u, 12u, 0u);
#else
    CV2Instr i0 = IN(0);
    v2_op_meas_active_interfere(st, v, st->active_k, i0.axis_1, i0.a, i0.flags);
#endif
}
