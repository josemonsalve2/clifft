#!/bin/bash
# Is the "match": false (cpu_observable_ones=169 vs v2=167) caused by the
# specializer, or does the plain interpreter show it too?
#
# The correctness GATE (interp vs spec, both GPU) is now exact on 6/6 seeds, so
# if the interpreter shows the same CPU delta then this is a pre-existing
# CPU-reference divergence on a separate axis, not something the fence fix
# introduced. Run both with V2_SPECIALIZE unset vs set, same seed.
set -u
export CLIFFT=/shared/jmonsalv/quantum/clifft_rl/clifft
cd "$CLIFFT"
BASE=V2_performance/scratch/cpuref_cache; rm -rf $BASE; mkdir -p $BASE
for SEED in 42 1 7; do
  echo "########## seed=$SEED ##########"
  echo "--- interpreter only (V2_SPECIALIZE unset) ---"
  ./build-v2-nohip/run_v2 --circuit tests/fixtures/large/circuit_d5_p0.001.stim \
     --shots 500 --seed $SEED 2>/dev/null | grep -E "cpu_observable|v2_observable|match"
  echo "--- specialized (V2_SPECIALIZE=1) ---"
  V2_SPECIALIZE=1 V2_SPEC_CACHE_DIR=$BASE ./build-v2-nohip/run_v2 \
     --circuit tests/fixtures/large/circuit_d5_p0.001.stim \
     --shots 500 --seed $SEED 2>/dev/null | grep -E "cpu_observable|v2_observable|match"
done
