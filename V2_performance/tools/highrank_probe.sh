#!/bin/bash
#SBATCH --job-name=hirank
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:45:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/V2_performance/tools/hirank_%j.log
#
# Ranks 25 and 26: the largest predicted beneficiaries of the pool fix, and the
# only ones with NO measurement behind them.
#
# Evaluating the old and new budgets across the global tier:
#
#     rank  MB/wg   old wgs  new wgs   change
#     11-20   <=12     2048     2048   same (the 2048 cap binds first)
#     21        24     1360     2048   1.5x
#     22        48      680     2048   3.0x
#     23        96      336     1360   4.0x
#     24       192      168      680   4.0x
#     25       384       80      336   4.2x   <-- no fixture existed
#     26       768       40      168   4.2x   <-- no fixture existed
#
# The corpus stopped at 24, so 25 and 26 were extrapolation. They are also
# where the old constant hurt most (40 resident workgroups on a 256-CU device)
# and where the risk lives: 768 MB per workgroup x 168 is 126 GB resident, by
# far the largest allocation this backend has ever attempted.
#
# Two things must be established before the numbers mean anything:
#   1. The fixtures really COMPILE to rank 25/26. The generator's own docstring
#      warns that StatevectorSqueezePass sees through most high-rank circuits
#      (rank19_q100_d10_wide.stim compiles to peak_rank=1). QV circuits are
#      supposed to resist that, but "supposed to" is not a measurement.
#   2. The allocation succeeds. If it OOMs, that is a finding, not a failure --
#      it means 4/9 is too aggressive at the top of the range and the fraction
#      needs a floor.
#
# Shot counts are low: at 768 MB/wg these are the most expensive circuits in
# the corpus and the point is the grid, not statistics.
set -u

BASE=/shared/jmonsalv/quantum/clifft_rl/clifft
cd "$BASE" || exit 1

for ROCM in /opt/rocm-7.2.3 /opt/rocm; do
  [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"
                          export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
export LLVM_PREFIX=/shared/jmonsalv/software/modules/llvm/upstream_05082025
export V2_SPECIALIZE=1   # MANDATORY (project_v2_specialize_env)
export V2_NO_CPU=1

V2=$BASE/build-v2-nohip/run_v2
export V2_SPEC_CACHE_DIR=$BASE/V2_performance/scratch/hirank_cache
mkdir -p "$V2_SPEC_CACHE_DIR"

echo "=== node ==="; echo "host=$(hostname -s) job=${SLURM_JOB_ID:-0}"
echo

echo "=========================================================="
echo "Q1. Do the new fixtures actually reach rank 25/26?"
echo "    StatevectorSqueezePass defeats most 'high rank'"
echo "    circuits; if it defeats these, Q2 is meaningless."
echo "=========================================================="
for f in qv24_L4 qv25_L3 qv26_L3; do
  c=tests/fixtures/large/${f}_seed42.stim
  [ -f "$c" ] || { echo "  $f: MISSING"; continue; }
  r=$(V2_DUMP_OPCODES=1 "$V2" --circuit "$c" --shots 1 --seed 42 2>&1 >/dev/null \
        | grep -m1 -oP 'PEAK_RANK \K[0-9]+')
  echo "  $f -> peak_rank=${r:-?}"
done

echo
echo "=========================================================="
echo "Q2. Grid chosen, and does the allocation hold?"
echo "    V2_DUMP_WGS prints the budget derivation, since a"
echo "    device-derived number is not readable from source."
echo "=========================================================="
for f in qv25_L3 qv26_L3; do
  c=tests/fixtures/large/${f}_seed42.stim
  [ -f "$c" ] || continue
  echo "--- $f ---"
  V2_DUMP_WGS=1 "$V2" --circuit "$c" --shots 8 --seed 42 2>&1 >/dev/null \
    | grep -E "v2-global|error|OOM|out of memory" | head -4
done

echo
echo "=========================================================="
echo "Q3. Old budget vs new, at the ranks never measured."
echo "    Emulating the old constant via V2_GLOBAL_WGS so both"
echo "    arms run the same binary -- the only difference is"
echo "    the grid, which is the variable under test."
echo "=========================================================="
kt() {
  "$V2" --circuit "$1" --shots "$2" --seed 42 2>/dev/null \
    | grep -oP '"v2_kernel_seconds": \K[0-9.e+-]+'
}
for spec in "qv25_L3 8 80" "qv26_L3 4 40"; do
  set -- $spec; f=$1; shots=$2; oldwgs=$3
  c=tests/fixtures/large/${f}_seed42.stim
  [ -f "$c" ] || continue
  echo "--- $f (shots=$shots) ---"
  "$V2" --circuit "$c" --shots "$shots" --seed 42 >/dev/null 2>&1  # warm compile+gate
  for arm in "old:$oldwgs" "new:dflt"; do
    name=${arm%%:*}; w=${arm##*:}
    if [ "$w" = dflt ]; then unset V2_GLOBAL_WGS; else export V2_GLOBAL_WGS=$w; fi
    printf "  %-4s wgs=%-5s " "$name" "$w"
    for i in 1 2 3; do printf "%s " "$(kt "$c" "$shots" || echo FAIL)"; done
    echo
  done
  unset V2_GLOBAL_WGS
done

echo
echo "=== done ==="
