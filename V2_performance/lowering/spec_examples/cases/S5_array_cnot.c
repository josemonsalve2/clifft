// TIER: coop
// S5 -- scatter-index folding. The array two-qubit ops build each amplitude
// index with scatter_bits_2(i, a, b): insert two zero bits at positions a and b.
// Constant axes collapse that to a fixed mask/shift sequence and turn
// 1ull<<(k-2) into a literal trip count.
#include "../harness.h"
CASE_KERNEL_HEAD {
    (void)scratch; (void)fused_u2; (void)fused_u4; (void)pauli_masks;
    (void)noise_sites; (void)noise_channels; (void)noise_hazards; (void)num_noise_sites;
    (void)readout_noise; (void)obs_off; (void)obs_tgt;
#if SPEC_FORM
    (void)instrs; (void)pc;
    v2_op_array_cnot(st, v, 8u, 2u, 5u);
    v2_op_array_cz(st, v, 8u, 1u, 6u);
#else
    CV2Instr i0 = IN(0), i1 = IN(1);
    v2_op_array_cnot(st, v, st->active_k, i0.axis_1, i0.axis_2);
    v2_op_array_cz(st, v, st->active_k, i1.axis_1, i1.axis_2);
#endif
}
