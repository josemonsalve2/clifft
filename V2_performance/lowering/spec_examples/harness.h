// harness.h — shared scaffolding for the per-specialization A/B cases.
//
// Every case file compiles TWICE from the same source:
//   -DSPEC_FORM=0  "interpreter form": operands arrive in a CV2Instr loaded
//                  from memory, exactly as coop_interpreter.c's for(pc)switch
//                  supplies them. Nothing about the operand is known at
//                  compile time.
//   -DSPEC_FORM=1  "specialized form": operands are the literal constants that
//                  v2_specializer.cc emits for that instruction.
//
// Both forms call the IDENTICAL v2_op_* body out of v2_ops.h, so the .s diff
// isolates precisely one variable: whether the operands were constants.
//
// The kernel is exported so llc keeps it; `bench_in` is volatile-loaded so the
// interpreter form cannot be constant-folded through the CV2Instr.

#include "clifft/gpu/mlir/v2/v2_ops.h"

// LDS blocks are `extern` here for the same reason the production coop sources
// declare them that way: addrspace(3) globals may not carry an initializer, so
// the definition is materialized by the AMDGPU backend at link time. We only
// emit assembly (-S), so leaving them undefined is fine.
#ifndef V2_REGISTER
extern __attribute__((address_space(3))) CV2Complex lds_v[V2_MAX_AMP];
extern __attribute__((address_space(3))) CV2Complex lds_red_scratch[V2_SCRATCH_AMP];
#endif

// The interpreter form reads its operands out of `instrs`, a plain
// `const CV2Instr*` kernel argument indexed by a runtime `pc`, which is exactly
// how coop_interpreter.c's for(pc)switch supplies them. NOT volatile: marking
// it volatile would force system-coherent (sc0 sc1) reads the real interpreter
// never issues and would overstate the interpreter's cost. The operands are
// unknowable at compile time simply because `instrs` points at device memory.
#define CASE_KERNEL_HEAD                                                    \
    __attribute__((visibility("default"))) __attribute__((amdgpu_kernel))   \
    void case_kernel(V2State* st, CV2Complex* v, CV2Complex* scratch,       \
                     const CV2Instr* instrs, u32 pc,                        \
                     const CV2FusedU2Entry* fused_u2,                       \
                     const CV2FusedU4Entry* fused_u4,                       \
                     const CV2Mask* pauli_masks,                            \
                     const CV2NoiseSite* noise_sites,                       \
                     const CV2Channel* noise_channels,                      \
                     const double* noise_hazards, u32 num_noise_sites,      \
                     const CV2ReadoutNoise* readout_noise,                  \
                     const u32* obs_off, const u32* obs_tgt)

// Fetch instruction `pc+n`, the interpreter's access pattern.
#define IN(n) (instrs[pc + (n)])
