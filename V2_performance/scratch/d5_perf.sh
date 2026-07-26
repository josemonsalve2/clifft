#!/bin/bash
# Perf recovery for the circuits that were forced onto the interpreter by the
# coop_r10_n1720 gate failure. Now that the gate passes (barrier fence fix),
# the specializer should be selected for them.
#
# Reports v2_kernel_seconds (kernel time, not host wall). 5 runs per arm, and
# both arms run back-to-back on the SAME node so the comparison is valid --
# never compare numbers across node types on this heterogeneous partition.
# V2_NO_CPU skips the f64 CPU reference, which otherwise dominates wall time
# and is irrelevant to kernel timing.
set -u
export CLIFFT=/shared/jmonsalv/quantum/clifft_rl/clifft
cd "$CLIFFT"
echo "NODE=$(hostname)"
BASE=V2_performance/scratch/perf_cache; rm -rf $BASE; mkdir -p $BASE
export V2_NO_CPU=1

for C in tests/fixtures/large/circuit_d5_p0.001.stim \
         tests/fixtures/large/circuit_d5_p0.0005.stim \
         tests/fixtures/large/circuit_d5_p0.002.stim \
         tests/fixtures/large/circuit_d5_p0.003.stim \
         tests/fixtures/large/circuit_d5_p0.005.stim \
         tests/fixtures/large/circuit_d3_p0.001.stim; do
  [ -f "$C" ] || continue
  N=$(basename "$C" .stim)
  # warm the spec cache + gate first so neither is timed
  V2_SPECIALIZE=1 V2_SPEC_CACHE_DIR=$BASE ./build-v2-nohip/run_v2 --circuit "$C" \
      --shots 10000 --seed 42 >/dev/null 2>&1
  for MODE in interp spec; do
    ENVV=""; [ "$MODE" = spec ] && ENVV="V2_SPECIALIZE=1"
    printf "%-28s %-6s " "$N" "$MODE"
    for i in 1 2 3 4 5; do
      T=$(env $ENVV V2_SPEC_CACHE_DIR=$BASE ./build-v2-nohip/run_v2 --circuit "$C" \
           --shots 10000 --seed 42 2>/dev/null | grep -oP '"v2_kernel_seconds": \K[0-9.e-]+')
      printf "%s " "$T"
    done
    echo
  done
done
