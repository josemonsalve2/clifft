#!/usr/bin/env bash
# d5_race.sh -- discriminate RACE vs ROUNDING for the coop_r10_n1720 gate failure.
#
# Background: all six fallback circuits compile to one shape, coop_r10_n1720,
# which fails the gate by 1-3 shots out of 5000. The standing explanation was an
# "irreducible ~1 ULP codegen difference". Two facts contradict that:
#
#   1. The build uses -O2 -ffp-contract=off with NO fast-math. Under those flags
#      LLVM may not reassociate or contract FP, so inlining cannot legally change
#      a floating-point result. Yet flipping V2_SPEC_NOISE_INLINE changed WHICH
#      seeds diverge (seed 42 went match->DIVERGE, seed 99 flipped sign). Code
#      motion that is FP-neutral by construction cannot do that.
#   2. Rate. ~1 diverging shot per 5000, at ~200 branch decisions per shot, is
#      ~1e-6 per decision. For a 1-ULP f64 perturbation to flip a comparison the
#      two sides must land within 1e-16 relative -- a ~1e-16 event. The observed
#      rate is ~10 orders of magnitude too high to be rounding.
#
# So the divergence is almost certainly nondeterminism, not codegen. These arms
# test that directly:
#
#   selftest  -- runs the INTERPRETER against ITSELF twice, same seed, same
#                shots, identical machine code. Rounding CANNOT differ here. Any
#                mismatch proves nondeterminism outright and moves the bug out of
#                the specializer entirely.
#   bisect    -- if the interpreter is self-consistent, find the exact diverging
#                shot on seed 99, turning a statistic into one reproducible case.
#
# Interpretation:
#   selftest NONDETERMINISTIC -> a race in the shared coop code; the specializer
#     is a victim, not the cause, and the gate has been blaming the wrong thing.
#   selftest deterministic + bisect names a stable shot -> genuinely spec-vs-interp;
#     that single shot is then the thing to inspect.
set -uo pipefail
cd "${CLIFFT:?}"

export V2_NO_CPU=1 V2_SPECIALIZE=1 V2_SPECIALIZE_VERBOSE=1
CIRC=tests/fixtures/large/circuit_d5_p0.001.stim
BASE="$PWD/V2_performance/scratch/race_cache"
rm -rf "$BASE"; mkdir -p "$BASE"
export V2_SPEC_CACHE_DIR="$BASE"

echo "########## ARM 1: interpreter vs itself (rounding is impossible here) ##########"
# 2000 shots keeps it quick; a race shows up at any shot count.
V2_GATE_SELFTEST=1 V2_GATE_SHOTS=2000 \
  ./build-v2-nohip/run_v2 --circuit "$CIRC" --shots 500 --seed 42 2>&1 \
  | grep -E 'v2-selftest|v2-spec|error|Error' || true
echo

echo "########## ARM 2: bisect seed 99 to the exact shot ##########"
# Fresh cache dir: the .gate verdict from arm 1 would otherwise short-circuit
# the comparison and skip the bisect entirely.
rm -rf "$BASE"; mkdir -p "$BASE"
V2_GATE_BISECT=99 V2_GATE_SHOTS=5000 V2_GATE_VERBOSE=1 \
  ./build-v2-nohip/run_v2 --circuit "$CIRC" --shots 500 --seed 42 2>&1 \
  | grep -E 'v2-bisect|v2-gate|v2-spec|error|Error' || true
echo
echo "RACE_DONE"
