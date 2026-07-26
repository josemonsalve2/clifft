#!/bin/bash
# Test the fp32-amplitude hypothesis for the CPU-vs-GPU divergence.
#
# device_abi.h:31 says it outright: "Complex amplitude (f32 storage; f64 used
# only in reductions)". The CPU (svm.h:142) uses std::complex<double>. So the
# GPU carries ~7 decimal digits where the CPU carries ~16. Branch probabilities
# are reduced in f64 but from f32 inputs, so p0/total is only good to ~1e-6
# relative after summing 2^k terms. When rand*total lands within that of p0 the
# two sides take DIFFERENT branches on the SAME PRNG stream, and every draw
# afterwards decorrelates.
#
# Prediction: the flip probability scales with the number of amplitudes summed,
# ~sqrt(2^peak_rank)*2^-24. So agreement should DEGRADE with peak_rank.
# rank 0-4 (register tier, few amplitudes) should agree nearly always;
# rank 10 (d5 coop) should disagree at a small but visible rate.
#
# Method: same-stream comparison (CPU seeded with the GPU's shot-0 splitmix
# input, so both start from an identical 256-bit xoshiro state), 1 shot, many
# seeds, grouped by circuit rank.
set -u
export CLIFFT=/shared/jmonsalv/quantum/clifft_rl/clifft
cd "$CLIFFT"
echo "NODE=$(hostname)"
BASE=V2_performance/scratch/fp32_cache; rm -rf $BASE; mkdir -p $BASE
export V2_SPECIALIZE=1 V2_SPEC_CACHE_DIR=$BASE
GOLDEN=11400714819323198527

for C in tests/fixtures/large/circuit_d3_p0.001.stim \
         tests/fixtures/large/circuit_d5_p0.001.stim \
         tests/fixtures/large/circuit_d5_p0.005.stim; do
  [ -f "$C" ] || continue
  RANK=$(V2_DUMP_OPCODES=1 ./build-v2-nohip/run_v2 --circuit "$C" --shots 1 --seed 1 2>&1 \
         | grep -oP 'PEAK_RANK \K[0-9]+' | head -1)
  AG=0; DIS=0
  for S in $(seq 1 60); do
    CPUSEED=$(python3 -c "print($S ^ $GOLDEN)")
    G=$(./build-v2-nohip/run_v2 --circuit "$C" --shots 1 --seed $S 2>/dev/null \
        | grep -oP '"v2_observable_ones": \[\K[0-9]+')
    K=$(./build-v2-nohip/run_v2 --circuit "$C" --shots 1 --seed $CPUSEED 2>/dev/null \
        | grep -oP '"cpu_observable_ones": \[\K[0-9]+')
    [ "$G" = "$K" ] && AG=$((AG+1)) || DIS=$((DIS+1))
  done
  echo "$(basename $C) peak_rank=$RANK agree=$AG disagree=$DIS"
done
