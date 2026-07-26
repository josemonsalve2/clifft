// TIER: coop
// S3 -- STATIC RANK TRACKING. This is the load-bearing specialization: the
// specializer knows active_k at every program point, so `half = 1u << active_k`
// becomes a literal and the strided sweep gets a compile-time trip count. The
// interpreter must re-read st->active_k after every rank-changing op.
// Note what is NOT happening: the sweep is still a LOOP. V1 unrolled it.
#include "../harness.h"
CASE_KERNEL_HEAD {
    (void)scratch; (void)fused_u2; (void)fused_u4; (void)pauli_masks;
    (void)noise_sites; (void)noise_channels; (void)noise_hazards; (void)num_noise_sites;
    (void)readout_noise; (void)obs_off; (void)obs_tgt;
#if SPEC_FORM
    (void)instrs; (void)pc;
    v2_op_expand_t(st, v, 0u, 0u, 1);
    v2_op_expand_t(st, v, 1u, 3u, 0);
    v2_op_expand(st, v, 2u);
#else
    CV2Instr i0 = IN(0), i1 = IN(1), i2 = IN(2);
    v2_op_expand_t(st, v, st->active_k, i0.axis_1, 1);
    v2_op_expand_t(st, v, st->active_k, i1.axis_1, 0);
    (void)i2; v2_op_expand(st, v, st->active_k);
#endif
}
