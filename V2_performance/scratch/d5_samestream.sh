#!/bin/bash
# Same-stream test: force the CPU and GPU onto IDENTICAL PRNG state, then
# compare a single shot. This separates "different RNG stream" from "different
# logic" -- the only two things that can make the two disagree.
#
# Both use splitmix64 -> xoshiro256++ with the same constants. They differ only
# in seed derivation:
#   CPU (svm.h):     z = seed
#   GPU (v2_ops.h):  z = seed ^ (0x9e3779b97f4a7c15 * (shot_id + 1))
# so for shot 0 the GPU's effective z is seed ^ 0x9e3779b97f4a7c15. Passing THAT
# value as the CPU's --seed makes both start from the same 256-bit state.
#
# With shots=1 there is exactly one shot (id 0) on each side, same state, same
# circuit. If the implementations agree, cpu_observable_ones == v2_observable_ones
# EXACTLY. A mismatch here is a real logic bug and is fully reproducible.
set -u
export CLIFFT=/shared/jmonsalv/quantum/clifft_rl/clifft
cd "$CLIFFT"
echo "NODE=$(hostname)"
BASE=V2_performance/scratch/samestream_cache; rm -rf $BASE; mkdir -p $BASE
export V2_SPECIALIZE=1 V2_SPEC_CACHE_DIR=$BASE
G=11400714819323198527   # = 42 ^ 0x9e3779b97f4a7c15, the GPU's shot-0 z for seed 42

for C in tests/fixtures/large/circuit_d5_p0.001.stim tests/fixtures/large/circuit_d3_p0.001.stim; do
  echo "===== $(basename $C) ====="
  echo "--- GPU seed=42 (its shot 0), 1 shot ---"
  ./build-v2-nohip/run_v2 --circuit "$C" --shots 1 --seed 42 2>/dev/null \
    | grep -E 'cpu_passed|v2_passed|cpu_observable|v2_observable'
  echo "--- CPU seed=$G should reproduce the SAME shot ---"
  ./build-v2-nohip/run_v2 --circuit "$C" --shots 1 --seed $G 2>/dev/null \
    | grep -E 'cpu_passed|v2_passed|cpu_observable|v2_observable'
done
