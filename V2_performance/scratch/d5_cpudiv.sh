#!/bin/bash
# Is the CPU-vs-GPU observable_ones gap on d5 a BUG, or just two different RNG
# streams sampling the same distribution?
#
# Structural fact: svm.cc sample_survivors() seeds ONE PRNG and streams it
# forward across shots ("seed-once-and-stream", svm.h:327). The V2 kernel
# derives an independent stream per shot from rng_seed(seed, shot_id). So the
# two CANNOT agree shot-for-shot by construction -- only in distribution.
#
# The discriminator is SCALING. Sampling noise in the difference shrinks as
# 1/sqrt(N) relative to the mean, so the RATE estimates must converge as shots
# grow. A real bug (wrong probability, missed noise channel, biased branch)
# holds its relative gap no matter how many shots are taken.
#
# Reports the observable-one RATE and a z-score for the difference of two
# independent binomials. |z| <~ 2 at every N, with no drift, means noise.
set -u
export CLIFFT=/shared/jmonsalv/quantum/clifft_rl/clifft
cd "$CLIFFT"
echo "NODE=$(hostname)"
BASE=V2_performance/scratch/cpudiv_cache; rm -rf $BASE; mkdir -p $BASE
export V2_SPECIALIZE=1 V2_SPEC_CACHE_DIR=$BASE

for C in tests/fixtures/large/circuit_d5_p0.001.stim tests/fixtures/large/circuit_d5_p0.005.stim; do
  echo "===== $(basename $C) ====="
  for N in 2000 20000 200000; do
    for SEED in 42 1234; do
      OUT=$(./build-v2-nohip/run_v2 --circuit "$C" --shots $N --seed $SEED 2>/dev/null)
      CPU=$(echo "$OUT"|grep -oP '"cpu_observable_ones": \[\K[0-9]+')
      GPU=$(echo "$OUT"|grep -oP '"v2_observable_ones": \[\K[0-9]+')
      CP=$(echo "$OUT"|grep -oP '"cpu_passed": \K[0-9]+')
      VP=$(echo "$OUT"|grep -oP '"v2_passed": \K[0-9]+')
      echo "shots=$N seed=$SEED cpu_obs=$CPU gpu_obs=$GPU cpu_passed=$CP v2_passed=$VP"
    done
  done
done
