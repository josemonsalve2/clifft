#!/bin/bash
# Byte-exactness of V2_U4_PREFETCH, checked ON THE CPU.
#
# The prefetch commit (1134ffc) claims the software-pipelined U4 butterfly is
# byte-exact with the baseline: only the fetch point moves, and the read-ahead
# cannot alias the write-behind because scatter_bits_2 is injective and the
# quads it generates partition the amplitude array.
#
# That is an argument. Given that V2_DUST_EPS and the bare-s_barrier bug BOTH
# surfaced as PRNG desync from a handful of differing bits, an argument is not
# enough -- but the claim is about integer index disjointness and float
# arithmetic, neither of which needs a GPU. So it is testable here, in seconds,
# instead of behind a SLURM queue.
#
# How the tiers are covered:
#   stride 1   (register tier) -- one thread sweeps every quad
#   stride 256 (coop/global)   -- 256 interleaved threads over one array, which
#                                 is the only configuration that exercises the
#                                 CROSS-THREAD half of the disjointness claim
#
# The macro overrides in arm.c are what make the strided case runnable off-GPU:
# -DV2_REGISTER suppresses the amdgcn intrinsics and LDS externs, then V2_STRIDE
# and v2_tid() are redefined to the coop values. The U4 sweep has no barrier
# inside its loop (the barrier is after it), so nothing else needs emulating.
#
# A NEGATIVE CONTROL runs last. A byte-exactness test that cannot fail proves
# nothing, so one amplitude's sign is flipped and the harness must report the
# mismatch -- if the control passes silently, the clean runs above it are void.
#
# Usage: V2_performance/tools/u4_exact/run.sh
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
BASE="$(cd "$HERE/../../.." && pwd)"
OUT=$(mktemp -d); trap 'rm -rf "$OUT"' EXIT

# The U4 op body alone, so the redefined V2_STRIDE/v2_tid take effect. Pulled
# from the real source at run time -- a copy would silently rot.
python3 - "$BASE" "$OUT" <<'PY'
import pathlib, sys
base, out = sys.argv[1], sys.argv[2]
src = pathlib.Path(base, 'src/clifft/gpu/mlir/v2/v2_ops_body.inc').read_text()
i = src.index('static inline __attribute__((always_inline)) void v2_op_array_u4')
j = src.index('// ---- swap-measure-interfere', i)
pathlib.Path(out, 'u4_loop_only.h').write_text(
    src[i:j].replace('v2_op_array_u4', 'u4_strided', 1))
PY

CFLAGS="-O2 -std=c11 -ffp-contract=off -I$BASE/src -I$OUT"
# -ffp-contract=off is a CORRECTNESS contract, not a tuning flag: an FMA
# contracted on one side and not the other would change results by itself and
# make this test measure the compiler rather than the change.

build() {  # $1=harness.c
  gcc $CFLAGS -DV2_U4_PREFETCH=1 -DWRAPNAME=pf_run   -c "$HERE/arm.c" -o "$OUT/pf.o"   || exit 1
  gcc $CFLAGS -DV2_U4_PREFETCH=0 -DWRAPNAME=base_run -c "$HERE/arm.c" -o "$OUT/base.o" || exit 1
  gcc $CFLAGS -c "$1" -o "$OUT/h.o" || exit 1
  gcc "$OUT/h.o" "$OUT/pf.o" "$OUT/base.o" -o "$OUT/t" || exit 1
}

echo "=== prefetch vs baseline, stride 256 / 256 threads ==="
build "$HERE/harness.c"
rc=0
for K in 9 11 14 16 18; do "$OUT/t" "$K" || rc=1; done

echo
echo "=== negative control (one amplitude sign flipped) ==="
sed 's|for (u32 t = 0; t < 256; t++) base_run|b[n/3].re = -b[n/3].re;\n        for (u32 t = 0; t < 256; t++) base_run|' \
    "$HERE/harness.c" > "$OUT/neg.c"
build "$OUT/neg.c"
if "$OUT/t" 14 >/dev/null 2>&1; then
  echo "  *** CONTROL PASSED -- the harness cannot detect a difference. ***"
  echo "  *** Every result above is meaningless. ***"
  rc=1
else
  echo "  control correctly FAILED -- the harness can see a one-amplitude change."
fi

echo
[ $rc -eq 0 ] && echo "RESULT: byte-exact" || echo "RESULT: FAILED"
exit $rc
