#!/usr/bin/env bash
# d5_race3.sh -- how does the spec kernel's nondeterminism scale with the number
# of CONCURRENT workgroups?
#
# Established so far:
#   * spec disagrees with ITSELF (job 50358/50361) -> a race, not codegen/rounding.
#   * noinline is NOT the cause (removing it made 5/6 seeds nondeterministic).
#   * bisect at 1 shot/dispatch = 1 workgroup found ZERO diverging shots.
#
# The last fact is the lever: if the race needs workgroups to run CONCURRENTLY,
# the divergence count must grow with occupancy. This sweeps the shot count,
# which is exactly the number of workgroups (coop tier = 1 workgroup/shot).
# A count that stays 0 at small N and grows with N confirms a cross-workgroup
# race -- i.e. state that is supposed to be per-workgroup LDS but is in fact
# being shared, or amplitude/reduction scratch surviving across shots.
set -uo pipefail
cd "${CLIFFT:?}"
export V2_NO_CPU=1 V2_SPECIALIZE=1 V2_SPECIALIZE_VERBOSE=1 V2_GATE_SELFTEST=1
CIRC=tests/fixtures/large/circuit_d5_p0.001.stim
BASE="$PWD/V2_performance/scratch/race3_cache"; rm -rf "$BASE"
for N in 1 2 8 64 256 1024 5000; do
  export V2_SPEC_CACHE_DIR="$BASE/$N"; mkdir -p "$V2_SPEC_CACHE_DIR"
  echo "########## workgroups (shots) = $N ##########"
  V2_GATE_SHOTS=$N ./build-v2-nohip/run_v2 --circuit "$CIRC" --shots 500 --seed 42 2>&1 \
    | grep -E 'spec-vs-spec' || true
  echo
done
echo RACE3_DONE
