#!/bin/bash
#SBATCH --job-name=wgsheur
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:40:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/V2_performance/tools/wgsheur_%j.log
#
# Verify the device-derived pool heuristic (v2_kernel.cc) against the measured
# optima from jobs 51166/51171, and confirm correctness is unaffected.
#
# The change: budget = 4/9 of the HSA-reported device pool, replacing a
# hardcoded 32 GB. On a 288 GB MI350X that is 128 GB, which reproduces the
# measured optimum at ranks 20, 21, 23 and 24 and lands within noise at 22.
#
# This job must show three things:
#   1. The kernel picks the predicted grid (dumped by V2_DUMP_WGS=1).
#   2. Kernel time matches the hand-swept optimum, not the old default.
#   3. Correctness is untouched -- the pool size changes only how many shots
#      are in flight, never the arithmetic, so byte-exactness must hold.
set -u
BASE=/shared/jmonsalv/quantum/clifft_rl/clifft
cd "$BASE" || exit 1

for ROCM in /opt/rocm-7.2.3 /opt/rocm; do
  [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"
                          export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
export LLVM_PREFIX=/shared/jmonsalv/software/modules/llvm/upstream_05082025
export V2_SPECIALIZE=1   # MANDATORY (project_v2_specialize_env)

echo "=== node ==="
echo "host=$(hostname -s) job=${SLURM_JOB_ID:-0}"
echo "=== build ==="
cmake --build build-v2-nohip -j"$(nproc)" --target run_v2 2>&1 \
  | grep -iE "error:|warning: unused|Built target run_v2" | head -20
V2=$BASE/build-v2-nohip/run_v2
[ -x "$V2" ] || { echo "*** BUILD FAILED -- no run_v2 ***" >&2; exit 1; }

export V2_SPEC_CACHE_DIR=$BASE/V2_performance/scratch/wgsheur_cache
mkdir -p "$V2_SPEC_CACHE_DIR"

echo
echo "=== device pool as HSA reports it (the heuristic's input) ==="
V2_DUMP_WGS=1 V2_NO_CPU=1 "$V2" --circuit tests/fixtures/large/qv22_L6_seed42.stim \
  --shots 10 --seed 42 2>&1 | grep -iE "v2-global|pool|CUs=" | head -5

run_kt() { "$V2" --circuit "$1" --shots "$2" --seed 42 2>/dev/null \
             | grep -oP '"v2_kernel_seconds": \K[0-9.e+-]+'; }

echo
echo "=========================================================="
echo "1+2. New default vs the hand-swept optimum and old default"
echo "     3 runs each, v2_kernel_seconds. 'new' = no env var."
echo "=========================================================="
printf "%-14s %-6s %-9s %-9s | %s\n" circuit rank "old(32G)" "opt(swept)" "new (heuristic)"
export V2_NO_CPU=1
while read -r c s rank old opt; do
  [ -f "$c" ] || { echo "*** MISSING $c ***" >&2; exit 1; }
  n=$(basename "$c" .stim)
  unset V2_GLOBAL_WGS; "$V2" --circuit "$c" --shots "$s" --seed 42 >/dev/null 2>&1  # warm
  printf "%-14s %-6s %-9s %-9s |" "$n" "$rank" "$old" "$opt"
  for i in 1 2 3; do printf " %s" "$(run_kt "$c" "$s")"; done
  echo
done <<'EOF'
tools/bench/fixtures/qv20_seed42.stim 2000 20 1.805 1.797
tests/fixtures/large/qv21_L8_seed42.stim 2000 21 2.980 2.719
tests/fixtures/large/qv22_L6_seed42.stim 1000 22 3.177 2.213
tests/fixtures/large/qv23_L5_seed42.stim 1000 23 6.165 3.098
tests/fixtures/large/qv24_L4_seed42.stim 500 24 7.242 3.468
EOF

echo
echo "=========================================================="
echo "3. Correctness: the pool changes concurrency, not arithmetic."
echo "   Shot results must be IDENTICAL across pool sizes, and the"
echo "   f64 CPU reference must still match. A mismatch here means"
echo "   the work-stealing loop is order-dependent -- a real bug."
echo "=========================================================="
unset V2_NO_CPU
C=tests/fixtures/large/qv22_L6_seed42.stim
for wgs in dflt 336 680 1360; do
  if [ "$wgs" = dflt ]; then unset V2_GLOBAL_WGS; else export V2_GLOBAL_WGS=$wgs; fi
  printf "  wgs=%-6s " "$wgs"
  V2_NO_CPU=1 "$V2" --circuit "$C" --shots 500 --seed 42 2>/dev/null \
    | grep -E '"v2_passed"|"v2_observable_ones"' | tr -d '\n ' ; echo
done
unset V2_GLOBAL_WGS

echo
echo "  --- vs f64 CPU reference (match must be true) ---"
"$V2" --circuit "$C" --shots 200 --seed 42 2>/dev/null | grep -E '"match"|"cpu_passed"|"v2_passed"'

echo
echo "DONE"
