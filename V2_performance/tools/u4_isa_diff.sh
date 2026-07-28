#!/bin/bash
# Compare the U4 butterfly's generated ISA with and without V2_U4_PREFETCH.
#
# Runs on the LOGIN NODE -- no GPU, no ROCm needed. clang targets amdgcn
# directly, so the register/waitcnt trade can be inspected without waiting for
# a SLURM allocation. This is what produced the table in commit 1134ffc.
#
# Counts are taken PER LOOP BODY, not per function: whole-function totals
# include the prologue's loads and hide the change. The loop is what runs
# 2^(k-2) times.
#
# Usage: V2_performance/tools/u4_isa_diff.sh [rank] [lo] [hi]
set -u

BASE="$(cd "$(dirname "$0")/../.." && pwd)"
RANK=${1:-22}; LO=${2:-3}; HI=${3:-9}
CL=${LLVM_PREFIX:-/shared/jmonsalv/software/modules/llvm/upstream_05082025}/bin/clang
ARCH=${CLIFFT_V2_AMDGPU_ARCH:-gfx950}
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT

cat > "$OUT/tu.c" <<EOF
#include "clifft/gpu/mlir/v2/v2_ops.h"
__attribute__((amdgpu_kernel)) void k(V2State* st, CV2Complex* v,
                                      const CV2FusedU4Entry* f) {
    v2_op_array_u4(st, v, ${RANK}u, ${LO}u, ${HI}u, f, 0u);
}
EOF

echo "arch=$ARCH rank=$RANK lo=$LO hi=$HI"
printf "%-12s %6s %6s %6s %8s %7s\n" arm instr loads vmwait VGPR spills
for P in 0 1; do
  # -ffp-contract=off is a CORRECTNESS contract here, not a tuning flag: the
  # interpreter and specializer must produce bit-identical arithmetic.
  "$CL" -x c -O2 --target=amdgcn-amd-amdhsa -mcpu="$ARCH" -nogpulib \
        -ffreestanding -ffp-contract=off -DV2_U4_PREFETCH=$P \
        -I"$BASE/src" "$OUT/tu.c" -S -o "$OUT/a$P.s" || { echo "compile failed"; exit 1; }

  vgpr=$(grep -m1 'vgpr_count' "$OUT/a$P.s" | tr -dc 0-9)
  spill=$(grep -c 'scratch_store\|scratch_load' "$OUT/a$P.s")
  # The loop body is the basic block containing global_loads that ends in a
  # backward branch.
  read -r n l w < <(awk '
    /^\.LBB[0-9_]+:/ { inb=1; n=0; l=0; w=0 }
    inb { n++
          if (/global_load/)        l++
          if (/s_waitcnt vmcnt/)    w++ }
    /s_cbranch|s_branch/ && inb { if (l>0) { print n, l, w; exit } ; inb=0 }
  ' "$OUT/a$P.s")
  arm=$([ "$P" = 1 ] && echo pipelined || echo baseline)
  printf "%-12s %6s %6s %6s %8s %7s\n" "$arm" "${n:-?}" "${l:-?}" "${w:-?}" "${vgpr:-?}" "$spill"
done
echo
echo "Fewer loads/vmwait = more overlap. Higher VGPR = less occupancy."
echo "rank>=22 kernels sit at the 128 VGPR / 64 AGPR cap, so this is a"
echo "trade, not a win -- see u4_prefetch_ab.sh for the hardware A/B."
