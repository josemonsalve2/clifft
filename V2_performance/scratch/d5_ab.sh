#!/bin/bash
# d5_ab.sh -- A/B the specializer-vs-interpreter divergence on the ONE compiled
# shape that fails the correctness gate: coop_r10_n1720 (circuit_d5_p*, and
# cultivation_d5 -- all six fixtures compile to the identical 1720-instruction
# program, so this is one bug, not six).
#
# The standing explanation in the code is "irreducible ~1 ULP codegen
# difference". That was never actually tested. Each arm below isolates one
# candidate cause; a gate verdict that FLIPS under an arm identifies the cause,
# and one that does not rules it out.
set -u
cd "$CLIFFT"
for ROCM in /opt/rocm-7.2.3 /opt/rocm; do
  [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; \
     export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
export LLVM_PREFIX=/shared/jmonsalv/software/modules/llvm/upstream_05082025
export V2_NO_CPU=1 V2_SPECIALIZE=1 V2_SPECIALIZE_VERBOSE=1 V2_GATE_VERBOSE=1

CIRC=tests/fixtures/large/circuit_d5_p0.001.stim
# Isolate each arm's compiled artifacts so arms never share a cache or a gate
# verdict (the verdict is cached to <hsaco>.gate on disk and would otherwise
# leak across arms and silently fake a "no change" result).
BASE=$PWD/V2_performance/scratch/ab_cache
rm -rf "$BASE"; mkdir -p "$BASE"

arm() {  # arm <name> <env assignments...>
  local name=$1; shift
  export V2_SPEC_CACHE_DIR="$BASE/$name"
  mkdir -p "$V2_SPEC_CACHE_DIR"
  echo "########## ARM: $name ($*) ##########"
  # 5000 shots x 6 seeds is what the real gate uses; keep it identical so a
  # verdict here means the same thing as a verdict in production.
  # Keep the FULL output: an arm that errors out must not look like an arm that
  # simply produced no gate line. (An earlier version grepped this and every arm
  # came back empty, which said nothing about the hypothesis being tested.)
  env "$@" ./build-v2-nohip/run_v2 --circuit "$CIRC" --shots 2000 --seed 42 2>&1 | tail -25
  local g
  g=$(cat "$V2_SPEC_CACHE_DIR"/*.gate 2>/dev/null | tr -d '\n')
  echo "  --> GATE VERDICT: ${g:-<none>}"
  echo
}

echo "node=$(hostname)"
echo "Fixture: $CIRC (compiles to coop_r10_n1720; 42 qubits, 230 meas slots)"
echo

# Arm 1: baseline -- reproduce the known failure with an isolated cache.
arm baseline V2_ARM=baseline

# Arm 2: drop the noinline fence on the noise ops. The fence is the current
# explanation for byte-exactness. If the verdict is UNCHANGED (still 0), the
# fence is irrelevant to THIS circuit and the real cause is elsewhere.
arm noise_inline V2_SPEC_NOISE_INLINE=1

# Arm 3: force the coop path for a circuit small enough to also run on the
# register tier is not possible here (rank 10), but we CAN vary shot count to
# test whether the divergence is a rare-event tail (a handful of shots) or
# systematic. A systematic bug fails at 50 shots too; a 1-ULP tail needs many.
arm shots50 V2_GATE_SHOTS=50
arm shots500 V2_GATE_SHOTS=500

echo "AB_DONE"
