#!/bin/bash
#SBATCH --job-name=r21dump
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/V2_performance/tools/r21dump_%j.log
#
# Why rank 21 did not move under the heuristic (job 51179).
#
# Ranks 22/23/24 landed on their swept optima. Rank 21 measured 2.976 s -- which
# is not the 2.719 s optimum at wgs=2048, it is the wgs=1360 time to three
# decimals (sweep 51166: 1360 -> 2.978/2.979/2.980). So the run behaved as if it
# had the OLD grid, even though the budget arithmetic says 2048 under both the
# 288 GB total and the 251 GB actually observed:
#
#     bytes/wg at rank 21 = 24 MB;  111.6 GB / 24 MB = 4759 -> capped to 2048
#
# The verify script does not leak V2_GLOBAL_WGS (it unsets before each warm-up),
# so this is not harness contamination. Three candidates remain, and they are
# distinguishable by observation rather than argument:
#
#   A. qv21_L8 does not actually compile to peak_rank 21. Every prediction here
#      is a function of peak_rank, so if the compiler disagrees with the
#      filename, the arithmetic was never about the grid that ran.
#   B. The grid IS 2048 and the time genuinely is ~2.98 s on this node right
#      now -- i.e. the 2.719 s was measured without a co-tenant, and rodas's
#      12-hour training job is eating the difference. That would make it a
#      contention artifact, not a heuristic miss.
#   C. Something clamps the grid between the budget and the dispatch.
#
# V2_DUMP_WGS settles A and C directly. For B, re-running the swept arms
# side-by-side under today's contention is the control: if 2048 and 1360 now
# measure the same, the optimum moved, and the heuristic is not what changed.
set -u

BASE=/shared/jmonsalv/quantum/clifft_rl/clifft
cd "$BASE" || exit 1
for ROCM in /opt/rocm-7.2.3 /opt/rocm; do
  [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"
                          export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
export LLVM_PREFIX=/shared/jmonsalv/software/modules/llvm/upstream_05082025
export V2_SPECIALIZE=1
export V2_NO_CPU=1
V2=$BASE/build-v2-nohip/run_v2
export V2_SPEC_CACHE_DIR=$BASE/V2_performance/scratch/wgsheur_cache

echo "=== node ==="; echo "host=$(hostname -s) job=${SLURM_JOB_ID:-0}"
echo "=== co-tenants (context for hypothesis B) ==="
rocm-smi --showmemuse 2>/dev/null | grep -i "VRAM%" | head -2
echo

echo "=========================================================="
echo "A/C. Compiled peak_rank and the grid actually chosen."
echo "=========================================================="
for f in tools/bench/fixtures/qv20_seed42.stim \
         tests/fixtures/large/qv21_L8_seed42.stim \
         tests/fixtures/large/qv22_L6_seed42.stim \
         tests/fixtures/large/qv23_L5_seed42.stim \
         tests/fixtures/large/qv24_L4_seed42.stim; do
  [ -f "$f" ] || { echo "  MISSING $f"; continue; }
  n=$(basename "$f" .stim)
  r=$(V2_DUMP_OPCODES=1 "$V2" --circuit "$f" --shots 1 --seed 42 2>&1 >/dev/null \
        | grep -m1 -oP 'PEAK_RANK \K[0-9]+')
  g=$(V2_DUMP_WGS=1 "$V2" --circuit "$f" --shots 8 --seed 42 2>&1 >/dev/null \
        | grep -m1 "v2-global")
  printf "  %-16s peak_rank=%-3s %s\n" "$n" "${r:-?}" "${g:-<no global dispatch>}"
done

echo
echo "=========================================================="
echo "B. Is 2048 still better than 1360 at rank 21 TODAY?"
echo "   Same node, same shots as sweep 51166, 3 runs. If these"
echo "   are now equal, the optimum moved under contention and"
echo "   the heuristic is exonerated."
echo "=========================================================="
C=tests/fixtures/large/qv21_L8_seed42.stim
"$V2" --circuit "$C" --shots 2000 --seed 42 >/dev/null 2>&1   # warm
for w in dflt 1360 2048 4096; do
  if [ "$w" = dflt ]; then unset V2_GLOBAL_WGS; else export V2_GLOBAL_WGS=$w; fi
  printf "  wgs=%-5s " "$w"
  for i in 1 2 3; do
    printf "%s " "$("$V2" --circuit "$C" --shots 2000 --seed 42 2>/dev/null \
      | grep -oP '"v2_kernel_seconds": \K[0-9.e+-]+')"
  done
  echo
done
unset V2_GLOBAL_WGS
echo
echo "  (sweep 51166, uncontended: dflt/1360 = 2.980, 2048 = 2.719)"
echo "=== done ==="
