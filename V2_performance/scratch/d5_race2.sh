#!/usr/bin/env bash
# d5_race2.sh -- localize the spec kernel's NONDETERMINISM (proven in job 50358:
# spec disagreed with ITSELF on 4/6 seeds; interp was deterministic on 6/6).
#
# The bisect found ZERO diverging shots when each shot ran as its own dispatch
# (1 workgroup). The race therefore needs concurrent workgroups -> it is in
# shared state, i.e. LDS. Both kernels declare the same extern LDS globals, but
# the specializer additionally passes V2State* into NOINLINE functions, which
# forces the classical state through GENERIC (flat) pointers instead of direct
# ds_ accesses. Arms:
#
#   noinline_off : V2_SPEC_NOISE_INLINE=1 removes the noinline attribute, so the
#                  noise ops inline exactly as in the interpreter and the flat
#                  path disappears. If the kernel becomes SELF-CONSISTENT, the
#                  noinline+generic-pointer path is the race.
#   (baseline reproduces the failure for comparison in the same job/node.)
set -uo pipefail
cd "${CLIFFT:?}"
export V2_NO_CPU=1 V2_SPECIALIZE=1 V2_SPECIALIZE_VERBOSE=1 V2_GATE_SELFTEST=1 V2_GATE_SHOTS=5000
CIRC=tests/fixtures/large/circuit_d5_p0.001.stim
BASE="$PWD/V2_performance/scratch/race2_cache"; rm -rf "$BASE"

arm() {
  local name=$1; shift
  export V2_SPEC_CACHE_DIR="$BASE/$name"; mkdir -p "$V2_SPEC_CACHE_DIR"
  echo "########## ARM: $name ($*) ##########"
  env "$@" ./build-v2-nohip/run_v2 --circuit "$CIRC" --shots 500 --seed 42 2>&1 \
    | grep -E 'v2-selftest|v2-spec|rror' || true
  echo
}
arm baseline      V2_ARM=baseline
arm noinline_off  V2_SPEC_NOISE_INLINE=1
echo RACE2_DONE
