# V2 de-risking experiments

## exp1: plain-C -> amdgcn -> .hsaco with NO HIP (2026-07-24) — PASSED
Proves V2 device authoring option #1 (plain-C operand library, HSA-dispatched,
no HIP) is buildable on the project toolchain
(/shared/jmonsalv/software/modules/llvm/upstream_05082025).

Files: v2probe.c (plain C amdgpu_kernel, no HIP headers), v2probe.ll (its IR).

Chain (all succeeded):
  clang --target=amdgcn-amd-amdhsa -mcpu=gfx950 -ffreestanding -nostdlib \
        -emit-llvm -S -O2 -o v2probe.ll v2probe.c
  llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -mattr=+wavefrontsize64 \
        -filetype=obj -O2 -o v2probe.o v2probe.ll
  ld.lld -shared -o v2probe.hsaco v2probe.o

Result: valid "AMD GPU" DYN ELF with `probe_kernel` FUNC + `probe_kernel.kd`
kernel descriptor — exactly what HSA hsa_executable_load needs. Used
__builtin_amdgcn_workitem_id_x / workgroup_id_x (no HIP runtime).

Implication: the operand library can be authored in plain C, compiled to
amdgcn, HSA-loaded via the existing pipeline back-half. R6 (no HIP) satisfied
with the lowest-infra option. NOT YET TESTED: actual GPU dispatch of this
kernel via the HSA runtime (next step), and LDS/barriers/atomics in plain C.

## exp2: coop-tier primitives in plain C (LDS + barrier + shuffle), NO HIP (2026-07-24) — PASSED
Files: v2coop3.c. Confirms the coop tier's core device primitives build without HIP:
- LDS: `extern __attribute__((address_space(3))) float lds_v[N];`  (MUST be
  extern/uninitialized — a plain `static`/global gets a zeroinitializer which
  llc REJECTS for addrspace(3): "unsupported initializer for address space".
  This mirrors the MLIR path's `llvm.mlir.global external @lds_*`.)
- workgroup barrier: `__builtin_amdgcn_s_barrier()`
- wave shuffle / reduction: `__builtin_amdgcn_ds_bpermute(addr, val)`
- thread/workgroup ids: `__builtin_amdgcn_workitem_id_x/workgroup_id_x`
- cooperative strided loop over LDS: fine.
Same clang/llc/ld.lld chain as exp1 -> valid .hsaco.

CONCLUSION (exp1+exp2): plain-C -> amdgcn -> HSA is a viable device-authoring
language for the V2 operand library (register AND coop primitives), NO HIP.
This closes the two biggest toolchain-reality risks in V2.md before writing the
backend. GOTCHA: LDS globals must be `extern` (no initializer).
NOT YET TESTED on-GPU (login-node compile only): actual HSA dispatch +
numerical correctness of a real operand; agent-scope atomics for global-tier
work-stealing; passing the bytecode/ConstantPool as kernel args.
