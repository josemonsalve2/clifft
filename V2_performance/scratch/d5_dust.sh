#!/bin/bash
# ROOT CAUSE TEST: V2_DUST_EPS was calibrated for the WRONG precision.
#
# kDustEpsilon = 1e-18 was chosen for the SVM, whose amplitudes are
# std::complex<double>. Its own comment says so: "squared amplitudes from
# analytically-zero Clifford+T interference sit around 1e-30 to 1e-24; this
# threshold safely swallows that dust". V2 copied the constant verbatim, but
# V2's amplitudes are CV2Complex = {float re; float im;}. In fp32 the same
# analytically-zero interference concentrates on fp32_eps^2 = 1.4e-14 -- four
# decades ABOVE 1e-18 -- so on the GPU the dust branch NEVER clamps.
#
# (The floor is rank-INDEPENDENT: the rounding error is relative to each
# amplitude, so summing more terms averages the residuals rather than
# accumulating them. Measured median p1/total is 9.7e-15 at rank 1 and 1.42e-14
# at rank 26, with the spread tightening as rank grows. One constant covers the
# whole range; 1e-11 sits ~2 decades above the widest tail.)
#
# Why a threshold that never fires is a correctness bug: sample_branch() returns
# WITHOUT drawing when a branch is dust. Clamp on CPU, no clamp on GPU => the
# GPU consumes a PRNG draw the CPU never did. Both still pick the same outcome
# (p0/total is 1-1e-15, so any uniform draw lands the same way), but every
# subsequent draw is shifted by one and the streams never resynchronize. In a
# d5 surface code almost every stabilizer measurement is deterministic, so this
# fires constantly.
#
# Reference (H;T^4;H;M, the exact case tests/test_svm.cc:1995 pins):
#   fp64: p0 = 2.465e-32  <= 1e-18  -> CLAMP,    no draw
#   fp32: p0 = 8.882e-16  >  1e-18  -> NO CLAMP, draw consumed
#
# ARM A rewrites the constant back to 1e-18, ARM B restores the committed
# 1e-11. Same node, same seeds, same shots. The sed targets only the #define
# line, so the explanatory comment block above it is preserved either way, and
# the tree is left on the committed value at exit.
set -u
export CLIFFT=/shared/jmonsalv/quantum/clifft_rl/clifft
cd "$CLIFFT"
echo "NODE=$(hostname)"

D5=tests/fixtures/large/circuit_d5_p0.001.stim
D3=tests/fixtures/large/circuit_d3_p0.001.stim
for f in "$D5" "$D3"; do [ -f "$f" ] || { echo "MISSING $f"; exit 1; }; done

# GPU shot-0 z for seed 42 -- feeding this as the CPU seed puts both sides on
# an IDENTICAL 256-bit xoshiro state, so shot 0 is a controlled comparison.
G=11400714819323198527

probe() {   # $1=label
  echo "===== $1 ====="
  for C in "$D5" "$D3"; do
    B=$(basename "$C" .stim)
    echo "--- $B same-stream shots=1 (GPU seed 42 vs CPU seed $G) ---"
    printf '  gpu: '; ./build-v2-nohip/run_v2 --circuit "$C" --shots 1 --seed 42 2>/dev/null \
      | grep -oE '"v2_observable_ones": *\[[^]]*\]'
    printf '  cpu: '; ./build-v2-nohip/run_v2 --circuit "$C" --shots 1 --seed $G 2>/dev/null \
      | grep -oE '"cpu_observable_ones": *\[[^]]*\]'
    for S in 42 7 1234; do
      printf '  %s shots=2000 seed=%-5s ' "$B" "$S"
      ./build-v2-nohip/run_v2 --circuit "$C" --shots 2000 --seed $S 2>/dev/null \
        | grep -oE '"(cpu|v2)_observable_ones": *\[[0-9]*\]|"match": *[a-z]*' | tr '\n' ' '
      echo
    done
  done
}

CACHE=V2_performance/scratch/dust_cache
export V2_SPECIALIZE=1 V2_SPEC_CACHE_DIR=$CACHE

# A .hsaco is only regenerated on a header edit because 72aee12 added the
# headers to the cmake DEPENDS list; wipe them anyway so neither arm can
# possibly measure a stale kernel, and give each arm its own spec cache
# (the gate verdict is cached on disk keyed by the generated C, which does
# NOT hash v2_ops.h).
HDR=src/clifft/gpu/mlir/v2/v2_ops.h
run_arm() {  # $1=label  $2=eps value
  sed -i "s/^#define V2_DUST_EPS .*$/#define V2_DUST_EPS    $2/" "$HDR"
  echo "  -> $(grep -n '^#define V2_DUST_EPS' $HDR)"
  rm -f build-v2-nohip/*.hsaco
  rm -rf $CACHE; mkdir -p $CACHE
  bash V2_performance/scratch/build_v2.sh 2>&1 | grep -E "BUILD_DONE|error:|Error" | head -20
  probe "$1"
}

echo "########## ARM A: old V2_DUST_EPS = 1e-18 (fp64-calibrated) ##########"
run_arm "ARM A / 1e-18" "1e-18"

echo
echo "########## ARM B: committed V2_DUST_EPS = 1e-11 (fp32-calibrated) ##########"
run_arm "ARM B / 1e-11" "1e-11"

grep -n "define V2_DUST_EPS" src/clifft/gpu/mlir/v2/v2_ops.h
echo ALL_DONE
