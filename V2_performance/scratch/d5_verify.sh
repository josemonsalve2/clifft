#!/bin/bash
# Post-fix verification, two independent questions.
#
# Q1 (LOGIC -- must be exact). Force both sides onto an identical 256-bit
# xoshiro state and compare shot for shot. The CPU seeds ONE prng and streams
# across shots; the GPU derives a stream per shot as
# splitmix64(seed ^ 0x9e3779b97f4a7c15 * (shot_id + 1)). So for shot N the CPU
# seed that reproduces the GPU's stream is seed ^ (0x9e3779b97f4a7c15*(N+1)),
# and with --shots 1 both sides run exactly that one shot. Any disagreement
# here is a real logic bug. Before the V2_DUST_EPS fix d5 failed this at
# shot 0 (GPU=0, CPU=1); it must now pass on every probed shot.
#
# Q2 (STATISTICS -- can never be exact). run_v2's "match" compares
# cpu_observable_ones to v2_observable_ones over a multi-shot batch. Those are
# samples from two DIFFERENT streams, so they agree in distribution, not
# shot-for-shot. Exact equality is impossible by construction and is not a
# target: the GPU runs shots in parallel, which rules out a single sequential
# stream. Verify convergence instead -- the relative gap must shrink as
# 1/sqrt(shots) and the z-scores must look like standard normal draws.
set -u
export CLIFFT=/shared/jmonsalv/quantum/clifft_rl/clifft
cd "$CLIFFT"
echo "NODE=$(hostname)"
grep -n "^#define V2_DUST_EPS" src/clifft/gpu/mlir/v2/v2_ops.h

D5=tests/fixtures/large/circuit_d5_p0.001.stim
D3=tests/fixtures/large/circuit_d3_p0.001.stim
D5b=tests/fixtures/large/circuit_d5_p0.005.stim
for f in "$D5" "$D3"; do [ -f "$f" ] || { echo "MISSING $f"; exit 1; }; done
[ -f "$D5b" ] || D5b=""

CACHE=V2_performance/scratch/verify_cache; rm -rf $CACHE; mkdir -p $CACHE
export V2_SPECIALIZE=1 V2_SPEC_CACHE_DIR=$CACHE

echo
echo "################ Q1: same-stream shot-for-shot (must be EXACT) ################"
# Only shot 0 is directly addressable: the GPU's per-shot seed derivation is
# internal, so the CPU can be aimed at shot 0 (via seed ^ K) but not at shot N
# without also replaying shots 0..N-1 through its streaming prng. Probing shot 0
# across many SEEDS covers the same ground -- each seed is an independent draw of
# the whole circuit -- so widen on seeds rather than on shot index.
K=$(python3 -c "print(0x9e3779b97f4a7c15)")
FAIL=0; TOT=0
for C in "$D5" "$D3" ${D5b:+$D5b}; do
  B=$(basename "$C" .stim)
  for SEED in 42 7 1234 99 2718 5 17 31 64 100 256 999; do
    G=$(python3 -c "print((($SEED) ^ $K) & 0xffffffffffffffff)")
    gpu=$(./build-v2-nohip/run_v2 --circuit "$C" --shots 1 --seed $SEED 2>/dev/null \
          | grep -oE '"v2_observable_ones": *\[[^]]*\]')
    cpu=$(./build-v2-nohip/run_v2 --circuit "$C" --shots 1 --seed $G 2>/dev/null \
          | grep -oE '"cpu_observable_ones": *\[[^]]*\]')
    gv=${gpu##*[}; gv=${gv%%]*}; cv=${cpu##*[}; cv=${cv%%]*}
    TOT=$((TOT+1))
    if [ "$gv" = "$cv" ]; then st=OK; else st="MISMATCH"; FAIL=$((FAIL+1)); fi
    printf '  %-22s seed=%-5s gpu=[%s] cpu=[%s]  %s\n' "$B" "$SEED" "$gv" "$cv" "$st"
  done
done
echo "  Q1: $((TOT-FAIL))/$TOT exact"

echo
echo "################ Q2: statistical convergence (gap ~ 1/sqrt(shots)) ################"
for C in "$D5" "$D3"; do
  B=$(basename "$C" .stim)
  echo "--- $B ---"
  printf '  %8s %10s %10s %8s %9s %7s\n' shots cpu gpu diff rel_gap z
  for S in 2000 10000 50000 200000; do
    ./build-v2-nohip/run_v2 --circuit "$C" --shots $S --seed 42 2>/dev/null \
      | python3 -c "
import sys,json,math
d=json.load(sys.stdin)
c=d['cpu_observable_ones'][0]; g=d['v2_observable_ones'][0]; n=d['shots']
p=(c+g)/(2*n); se=math.sqrt(2*p*(1-p)*n) if 0<p<1 else float('nan')
z=(g-c)/se if se==se and se>0 else float('nan')
print('  %8d %10d %10d %8d %8.3f%% %7.2f'%(n,c,g,g-c,100*abs(g-c)/max(c,1),z))
"
  done
done
echo ALL_DONE
