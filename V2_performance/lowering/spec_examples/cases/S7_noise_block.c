// TIER: register
// S7 -- the noise RUNTIME LOOP. OP_NOISE_BLOCK covers [start, start+count)
// noise sites but consumes only those the PRNG selects, so the loop is
// genuinely data-dependent and stays a loop in BOTH forms. Specialization folds
// only the bounds. This is the op whose straight-lining regressed performance
// (see the noise-fence knob, V2_SPEC_NOISE_INLINE).
#include "../harness.h"
CASE_KERNEL_HEAD {
    (void)v; (void)scratch; (void)fused_u2; (void)fused_u4; (void)pauli_masks;
    (void)readout_noise; (void)obs_off; (void)obs_tgt;
#if SPEC_FORM
    (void)instrs; (void)pc;
    v2_op_noise_block(st, noise_sites, noise_channels, noise_hazards, num_noise_sites, 131u, 57u);
    v2_op_noise(st, noise_sites, noise_channels, noise_hazards, num_noise_sites, 188u);
#else
    CV2Instr i0 = IN(0), i1 = IN(1);
    v2_op_noise_block(st, noise_sites, noise_channels, noise_hazards, num_noise_sites, i0.a, i0.b);
    v2_op_noise(st, noise_sites, noise_channels, noise_hazards, num_noise_sites, i1.a);
#endif
}
