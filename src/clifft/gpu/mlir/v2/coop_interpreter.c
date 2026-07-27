// coop_interpreter.c — MLIR-V2 RUNTIME interpreter (plain C -> amdgcn, NO HIP).
//
// A for(pc)switch over the circuit bytecode that dispatches to the shared
// v2_op_*() operand library (v2_ops.h). One workgroup = one shot (coop/global),
// or one thread = one shot (register, -DV2_REGISTER). The SAME v2_op_*() bodies
// are emitted straight-line, with constant operands, by the per-circuit
// specializer (v2_specializer.cc) — so interpreter and specialized kernels are
// byte-exact with each other and with GPU-SVM by construction.
//
// Compiled to amdgcn via cmake/ClifftAmdgcn.cmake (clang/llc/ld.lld, no -x hip).

#include "clifft/gpu/mlir/v2/v2_ops.h"

// ----- amplitude + reduction LDS (coop/global only; register uses registers) -
#ifndef V2_REGISTER
extern __attribute__((address_space(3))) CV2Complex lds_v[V2_MAX_AMP];
extern __attribute__((address_space(3))) CV2Complex lds_red_scratch[V2_SCRATCH_AMP];
extern __attribute__((address_space(3))) V2State lds_state;   // one shot's classical state
extern __attribute__((address_space(3))) u64 lds_shot;        // global tier: claimed shot
#endif

// =============================================================================
// execute_shot — the RUNTIME dispatcher. Thin switch over the bytecode calling
// the shared v2_op_*() bodies. `active_k` is re-read from st each iteration so
// rank-changing ops (expand/measure) are visible to the next op.
// =============================================================================
static void execute_shot(V2State* st, CV2Complex* v, CV2Complex* scratch,
                         u32 amp_capacity, u64 shot_id,
                         const CV2Instr* instrs, u32 num_instrs,
                         u32 total_meas_slots, u32 num_observables,
                         u64 seed,
                         CV2BlockCounts* block_counts,
                         const CV2FusedU2Entry* fused_u2,
                         const CV2FusedU4Entry* fused_u4,
                         const u32* observable_offsets,
                         const u32* observable_targets,
                         const CV2NoiseSite* noise_sites,
                         const CV2Channel* noise_channels,
                         const double* noise_hazards,
                         u32 num_noise_sites,
                         const CV2Mask* pauli_masks,
                         const CV2ReadoutNoise* readout_noise,
                         const u32* detector_offsets,
                         const u32* detector_targets,
                         u32 expected_obs_mask) {
    (void)total_meas_slots;
    v2_shot_init(st, v, amp_capacity, shot_id, num_observables, seed,
                 noise_hazards, num_noise_sites);

    for (u32 pc = 0; pc < num_instrs; ++pc) {
        CV2Instr ins = instrs[pc];
        u32 k = st->active_k;
        switch (ins.opcode) {
        case OP_FRAME_CNOT:  v2_op_frame_cnot(st, ins.axis_1, ins.axis_2); break;
        case OP_FRAME_CZ:    v2_op_frame_cz(st, ins.axis_1, ins.axis_2); break;
        case OP_FRAME_H:     v2_op_frame_h(st, ins.axis_1); break;
        case OP_FRAME_S:
        case OP_FRAME_S_DAG: v2_op_frame_s(st, ins.axis_1); break;
        case OP_FRAME_SWAP:  v2_op_frame_swap(st, ins.axis_1, ins.axis_2); break;
        case OP_MEAS_DORMANT_STATIC: v2_op_meas_dormant_static(st, ins.axis_1, ins.a, ins.flags); break;
        case OP_MEAS_DORMANT_RANDOM: v2_op_meas_dormant_random(st, ins.axis_1, ins.a, ins.flags); break;
        case OP_EXPAND:        v2_op_expand(st, v, k); break;
        case OP_EXPAND_T:      v2_op_expand_t(st, v, k, ins.axis_1, 0); break;
        case OP_EXPAND_T_DAG:  v2_op_expand_t(st, v, k, ins.axis_1, 1); break;
        case OP_EXPAND_ROT:    v2_op_expand_rot(st, v, k, ins.axis_1, ins.weight_re, ins.weight_im); break;
        case OP_MEAS_ACTIVE_DIAGONAL:  v2_op_meas_active_diagonal(st, v, k, ins.axis_1, ins.a, ins.flags); break;
        case OP_MEAS_ACTIVE_INTERFERE: v2_op_meas_active_interfere(st, v, k, ins.axis_1, ins.a, ins.flags); break;
        case OP_ARRAY_CNOT:    v2_op_array_cnot(st, v, k, ins.axis_1, ins.axis_2); break;
        case OP_ARRAY_CZ:      v2_op_array_cz(st, v, k, ins.axis_1, ins.axis_2); break;
        case OP_ARRAY_SWAP:    v2_op_array_swap(st, v, k, ins.axis_1, ins.axis_2); break;
        case OP_ARRAY_MULTI_CNOT: v2_op_array_multi_cnot(st, v, k, ins.axis_1, ins.mask); break;
        case OP_ARRAY_MULTI_CZ:   v2_op_array_multi_cz(st, v, k, ins.axis_1, ins.mask); break;
        case OP_ARRAY_H:       v2_op_array_h(st, v, k, ins.axis_1); break;
        case OP_ARRAY_S:       v2_op_array_s(st, v, k, ins.axis_1, 0); break;
        case OP_ARRAY_S_DAG:   v2_op_array_s(st, v, k, ins.axis_1, 1); break;
        case OP_ARRAY_T:       v2_op_array_t(st, v, k, ins.axis_1, 0); break;
        case OP_ARRAY_T_DAG:   v2_op_array_t(st, v, k, ins.axis_1, 1); break;
        case OP_ARRAY_ROT:     v2_op_array_rot(st, v, k, ins.axis_1, ins.weight_re, ins.weight_im); break;
        case OP_ARRAY_U2:      v2_op_array_u2(st, v, k, ins.axis_1, fused_u2, ins.a); break;
        case OP_ARRAY_U4:      v2_op_array_u4(st, v, k, ins.axis_1, ins.axis_2, fused_u4, ins.a); break;
        case OP_SWAP_MEAS_INTERFERE: v2_op_swap_meas_interfere(st, v, scratch, k, ins.axis_1, ins.axis_2, ins.a, ins.flags); break;
        case OP_APPLY_PAULI:   v2_op_apply_pauli(st, pauli_masks, ins.a, ins.b); break;
        case OP_NOISE:         v2_op_noise(st, noise_sites, noise_channels, noise_hazards, num_noise_sites, ins.a); break;
        case OP_NOISE_BLOCK:   v2_op_noise_block(st, noise_sites, noise_channels, noise_hazards, num_noise_sites, ins.a, ins.b); break;
        case OP_READOUT_NOISE: v2_op_readout_noise(st, readout_noise, ins.a); break;
        case OP_OBSERVABLE:    v2_op_observable(st, observable_offsets, observable_targets, ins.a, ins.b); break;
        case OP_POSTSELECT:    if (v2_op_postselect(st, detector_offsets, detector_targets, ins.a, ins.flags)) return; break;
        case OP_DETECTOR:      v2_op_detector(); break;
        default: {
            u32 t = v2_tid();
            if (IS_OWNER) st->discarded = 1;   // loud: unimplemented opcode
            v2_barrier();
            break;
        }
        }
    }

    v2_shot_aggregate(st, block_counts, num_observables, expected_obs_mask);
}

// The coop and global tiers use LDS + workgroup barriers; DEFAULT build.
#ifndef V2_REGISTER

// =============================================================================
// COOP tier kernel (rank <= 10). One workgroup per shot; amplitudes in LDS.
// =============================================================================
__attribute__((amdgpu_kernel, visibility("default")))
void clifft_v2_coop(const CV2Instr* instrs, u32 num_instrs, u32 peak_rank,
                    u32 total_meas_slots, u32 num_observables,
                    u64 seed, u64 shot_offset, u64 shots,
                    CV2BlockCounts* block_counts,
                    const CV2FusedU2Entry* fused_u2,
                    const CV2FusedU4Entry* fused_u4,
                    const u32* observable_offsets,
                    const u32* observable_targets,
                    const CV2NoiseSite* noise_sites,
                    const CV2Channel* noise_channels,
                    const double* noise_hazards,
                    u32 num_noise_sites, u32 expected_obs_mask,
                    const CV2Mask* pauli_masks,
                    const CV2ReadoutNoise* readout_noise,
                    const u32* detector_offsets,
                    const u32* detector_targets) {
    (void)peak_rank;
    u64 shot_id = shot_offset + (u64)v2_bid();
    if (shot_id >= shots) return;
    execute_shot((V2State*)&lds_state, (CV2Complex*)lds_v,
                 (CV2Complex*)lds_red_scratch, V2_MAX_AMP,
                 shot_id, instrs, num_instrs, total_meas_slots, num_observables,
                 seed, block_counts, fused_u2, fused_u4,
                 observable_offsets, observable_targets,
                 noise_sites, noise_channels, noise_hazards, num_noise_sites,
                 pauli_masks, readout_noise, detector_offsets, detector_targets,
                 expected_obs_mask);
}

// =============================================================================
// GLOBAL tier kernel (rank 11-26). Amplitudes in HBM; single global atomic work
// queue drains all shots.
// =============================================================================
__attribute__((amdgpu_kernel, visibility("default")))
void clifft_v2_global(const CV2Instr* instrs, u32 num_instrs, u32 peak_rank,
                      u32 total_meas_slots, u32 num_observables,
                      u64 seed, u64 shot_offset, u64 shots,
                      CV2BlockCounts* block_counts,
                      const CV2FusedU2Entry* fused_u2,
                      const CV2FusedU4Entry* fused_u4,
                      const u32* observable_offsets,
                      const u32* observable_targets,
                      const CV2NoiseSite* noise_sites,
                      const CV2Channel* noise_channels,
                      const double* noise_hazards,
                      u32 num_noise_sites, u32 expected_obs_mask,
                      const CV2Mask* pauli_masks,
                      const CV2ReadoutNoise* readout_noise,
                      const u32* detector_offsets,
                      const u32* detector_targets,
                      CV2Complex* global_v,
                      CV2Complex* global_scratch,
                      u64* work_counter) {
    u32 t = v2_tid();
    u32 slot = v2_bid();
    u64 amp_capacity = 1ull << peak_rank;
    CV2Complex* v = global_v + (u64)slot * amp_capacity;
    CV2Complex* scratch = global_scratch + (u64)slot * (amp_capacity >> 1);
    for (;;) {
        if (t == 0) lds_shot = __atomic_fetch_add(&work_counter[0], 1UL, __ATOMIC_RELAXED);
        v2_barrier();
        u64 batch_shot = lds_shot;
        if (batch_shot >= shots) return;
        execute_shot((V2State*)&lds_state, v, scratch, (u32)amp_capacity,
                     shot_offset + batch_shot, instrs, num_instrs,
                     total_meas_slots, num_observables, seed, block_counts,
                     fused_u2, fused_u4, observable_offsets, observable_targets,
                     noise_sites, noise_channels, noise_hazards, num_noise_sites,
                     pauli_masks, readout_noise, detector_offsets, detector_targets,
                 expected_obs_mask);
        v2_barrier();
    }
}

#else  // V2_REGISTER

// =============================================================================
// REGISTER tier kernel (rank <= 4). ONE shot per thread; state in registers.
// =============================================================================
#define V2_REG_MAX_AMP 16
__attribute__((amdgpu_kernel, visibility("default")))
void clifft_v2_register(const CV2Instr* instrs, u32 num_instrs, u32 peak_rank,
                        u32 total_meas_slots, u32 num_observables,
                        u64 seed, u64 shot_offset, u64 shots,
                        CV2BlockCounts* block_counts,
                        const CV2FusedU2Entry* fused_u2,
                        const CV2FusedU4Entry* fused_u4,
                        const u32* observable_offsets,
                        const u32* observable_targets,
                        const CV2NoiseSite* noise_sites,
                        const CV2Channel* noise_channels,
                        const double* noise_hazards,
                        u32 num_noise_sites, u32 expected_obs_mask,
                        const CV2Mask* pauli_masks,
                        const CV2ReadoutNoise* readout_noise,
                        const u32* detector_offsets,
                        const u32* detector_targets) {
    (void)peak_rank;
    u64 shot_id = shot_offset + ((u64)v2_bid() * 256ul + (u64)__builtin_amdgcn_workitem_id_x());
    if (shot_id >= shots) return;
    V2State st;
    CV2Complex vloc[V2_REG_MAX_AMP];
    CV2Complex sloc[V2_REG_MAX_AMP / 2];
    execute_shot(&st, vloc, sloc, V2_REG_MAX_AMP, shot_id, instrs, num_instrs,
                 total_meas_slots, num_observables, seed, block_counts,
                 fused_u2, fused_u4, observable_offsets, observable_targets,
                 noise_sites, noise_channels, noise_hazards, num_noise_sites,
                 pauli_masks, readout_noise, detector_offsets, detector_targets,
                 expected_obs_mask);
}

#endif  // V2_REGISTER
