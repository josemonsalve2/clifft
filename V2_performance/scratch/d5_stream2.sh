#!/bin/bash
# Widen the same-stream test: for many user seeds S, run the GPU at seed S
# (shot 0 only) and the CPU at seed (S ^ 0x9e3779b97f4a7c15), which is the GPU's
# effective shot-0 splitmix input. Both then start from the SAME 256-bit
# xoshiro state, so a disagreement is a logic difference, not RNG.
#
# Reports the GPU value from the first run and the CPU value from the second,
# which is the actual apples-to-apples pair. (Each run also prints the other
# side's number, but those use a different stream and are not comparable.)
set -u
export CLIFFT=/shared/jmonsalv/quantum/clifft_rl/clifft
cd "$CLIFFT"
echo "NODE=$(hostname)"
BASE=V2_performance/scratch/stream2_cache; rm -rf $BASE; mkdir -p $BASE
export V2_SPECIALIZE=1 V2_SPEC_CACHE_DIR=$BASE
GOLDEN=11400714819323198527   # 0x9e3779b97f4a7c15

for C in tests/fixtures/large/circuit_d5_p0.001.stim tests/fixtures/large/circuit_d3_p0.001.stim; do
  echo "===== $(basename $C) ====="
  AGREE=0; DISAGREE=0
  for S in $(seq 1 40); do
    CPUSEED=$(python3 -c "print($S ^ $GOLDEN)")
    G=$(./build-v2-nohip/run_v2 --circuit "$C" --shots 1 --seed $S 2>/dev/null \
        | grep -oP '"v2_observable_ones": \[\K[0-9]+')
    C2=$(./build-v2-nohip/run_v2 --circuit "$C" --shots 1 --seed $CPUSEED 2>/dev/null \
        | grep -oP '"cpu_observable_ones": \[\K[0-9]+')
    if [ "$G" = "$C2" ]; then AGREE=$((AGREE+1)); else DISAGREE=$((DISAGREE+1));
      echo "  seed=$S DISAGREE gpu=$G cpu=$C2"; fi
  done
  echo "  agree=$AGREE disagree=$DISAGREE"
done
