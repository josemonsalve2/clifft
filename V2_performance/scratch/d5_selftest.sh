#!/usr/bin/env bash
# d5_selftest.sh -- run BOTH kernels against THEMSELVES, twice, per seed.
# The spec kernel already produced different obs0 counts from the same binary
# across job 50345 vs 50356 (seed 99: 1719 then 1721; seed 2718: 1682 then 1681)
# while the interpreter reproduced exactly in both. That is nondeterminism, not
# codegen. This makes it a single-command, in-process demonstration.
set -uo pipefail
cd "${CLIFFT:?}"
export V2_NO_CPU=1 V2_SPECIALIZE=1 V2_SPECIALIZE_VERBOSE=1
CIRC=tests/fixtures/large/circuit_d5_p0.001.stim
BASE="$PWD/V2_performance/scratch/self_cache"; rm -rf "$BASE"; mkdir -p "$BASE"
export V2_SPEC_CACHE_DIR="$BASE"
V2_GATE_SELFTEST=1 V2_GATE_SHOTS=5000 \
  ./build-v2-nohip/run_v2 --circuit "$CIRC" --shots 500 --seed 42 2>&1 \
  | grep -E 'v2-selftest|v2-spec|rror' || true
echo SELFTEST_DONE
