#!/bin/bash
#SBATCH --job-name=wgsbw2
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:35:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/V2_performance/tools/wgsbw2_%j.log
#
# Continuation of job 51166, which was preempted (SIGTERM) 8 minutes in, after
# completing rank 20, 21 and most of 22. Shorter walltime here to be less
# preemptable, and the arms already answered are not repeated.
#
# What 51166 established, and why the remaining arms matter:
#
#   rank 20 (dflt 2048): saturates at 1360; the default is already past it.
#   rank 21 (dflt 1360): 2.980s -> 2.719s at wgs=2048. Default leaves 1.10x.
#   rank 22 (dflt  680): 3.177s -> 2.213s at wgs=1360. Default leaves 1.44x.
#
# So the 32 GB budget (v2_kernel.cc:438) is UNDERSIZED from rank 21 up, and the
# deficit grows with rank. Ranks 23 and 24 have defaults of 336 and 168 -- even
# further down the steep part of the curve -- so they are the arms most likely
# to show a large win, and they are the two weakest results in the corpus
# (0.990 and 0.882). Also finishes rank 22's tail (4096, 8192) to confirm the
# peak-then-degrade shape seen at rank 21.
#
# Then the bandwidth counters, which are the actual gate on the "HBM stack
# placement / spatial locality" design question: placement only matters if the
# global tier is near bandwidth saturation. bench_all.sh:78 records
# FETCH_SIZE/WRITE_SIZE as uncollectable, but that was a 6-counter pass.
#
# Reports v2_kernel_seconds (kernel time, not host wall). Same node for all
# arms; never compare across node types on this heterogeneous partition.
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
export V2_SPEC_CACHE_DIR=$BASE/V2_performance/scratch/wgsbw_cache
mkdir -p "$V2_SPEC_CACHE_DIR"

echo "=== node ==="
echo "host=$(hostname -s)  job=${SLURM_JOB_ID:-0}"
# 51166 grepped the FIRST "Compute Unit:" in rocminfo, which is the CPU agent
# (EPYC 9575F, 64c/128t -> 128). Walk agent blocks and report the GPU's.
rocminfo 2>/dev/null | awk '
  /^Agent [0-9]+/            { name=""; cu=""; isgpu=0 }
  /Device Type:.*GPU/        { isgpu=1 }
  /^ *Name: *gfx/            { name=$2 }
  /Compute Unit: *[0-9]+/    { cu=$3 }
  /Internal Node ID/ && isgpu && name!="" && cu!="" { print "  GPU agent " name "  Compute Units = " cu; isgpu=0 }
'
echo "  (SS14.5 of the report reasons against 256 CUs -- this is the check)"
rocminfo 2>/dev/null | grep -m1 "Marketing Name.*Instinct" || true
echo

run_kt() {  # $1=circuit $2=shots
  local t
  t=$("$V2" --circuit "$1" --shots "$2" --seed 42 2>/dev/null \
        | grep -oP '"v2_kernel_seconds": \K[0-9.e+-]+')
  echo "${t:-FAIL}"
}

sweep() {   # $1=circuit $2=shots $3=label $4...=wgs arms
  local c="$1" s="$2" n="$3"; shift 3
  [ -f "$c" ] || { echo "*** FIXTURE MISSING: $c ***" >&2; return 1; }
  echo "--- $n (shots=$s) ---"
  unset V2_GLOBAL_WGS
  "$V2" --circuit "$c" --shots "$s" --seed 42 >/dev/null 2>&1   # warm cache+gate
  for wgs in "$@"; do
    if [ "$wgs" = dflt ]; then unset V2_GLOBAL_WGS; else export V2_GLOBAL_WGS=$wgs; fi
    printf "  wgs=%-5s " "$wgs"
    for i in 1 2 3; do printf "%s " "$(run_kt "$c" "$s")"; done
    echo
  done
  unset V2_GLOBAL_WGS
}

echo "=========================================================="
echo "Q1 (cont). V2_GLOBAL_WGS sweep, ranks 22-24"
echo "    3 runs per arm, v2_kernel_seconds, lower is better."
echo "=========================================================="
# rank 22 tail only -- 168/336/680/1360/2048 already measured by 51166.
sweep tests/fixtures/large/qv22_L6_seed42.stim 1000 "qv22_L6 (dflt 680) TAIL" 4096 8192
# rank 23 and 24 in full. Defaults 336 and 168 sit low on the curve.
sweep tests/fixtures/large/qv23_L5_seed42.stim 1000 "qv23_L5 (dflt 336)" dflt 336 680 1360 2048 4096
sweep tests/fixtures/large/qv24_L4_seed42.stim  500 "qv24_L4 (dflt 168)" dflt 168 336 680 1360 2048

echo
echo "=========================================================="
echo "Q2. HBM bandwidth counters -- collectable at all?"
echo "    Gates the placement question: if the tier is far from"
echo "    saturation it is latency-bound and placement is moot."
echo "=========================================================="
BWC=tests/fixtures/large/qv22_L6_seed42.stim
BWS=1000
"$V2" --circuit "$BWC" --shots "$BWS" --seed 42 >/dev/null 2>&1   # warm

OUT=$BASE/V2_performance/scratch/wgsbw_pmc
rm -rf "$OUT"; mkdir -p "$OUT"
i=0
for PMC in "FETCH_SIZE,WRITE_SIZE" "FETCH_SIZE" \
           "TCC_EA_RDREQ_sum,TCC_EA_WRREQ_sum" "TCC_EA_RDREQ_sum" \
           "TCC_REQ_sum,TCC_MISS_sum"; do
  i=$((i+1))
  echo "--- pmc set $i: $PMC ---"
  if timeout 240 rocprofv3 --pmc "$PMC" --output-format csv -d "$OUT/set$i" \
       -- "$V2" --circuit "$BWC" --shots "$BWS" --seed 42 >"$OUT/set$i.out" 2>"$OUT/set$i.err"; then
    echo "  COLLECTED"
    f=$(find "$OUT/set$i" -name "*counter_collection.csv" 2>/dev/null | head -1)
    [ -n "$f" ] && python3 - "$f" <<'PY'
import csv, sys, collections
tot = collections.defaultdict(float)
with open(sys.argv[1]) as fh:
    for r in csv.DictReader(fh):
        tot[(r.get("Kernel_Name","")[:38], r.get("Counter_Name",""))] += float(r.get("Counter_Value",0) or 0)
for (k,c),v in sorted(tot.items()):
    print(f"    {k:40s} {c:22s} {v:,.0f}")
PY
  else
    echo "  UNAVAILABLE: $(head -3 "$OUT/set$i.err" 2>/dev/null | tr '\n' ' ' | cut -c1-150)"
  fi
done

echo
echo "--- kernel duration, same circuit (turns bytes into GB/s) ---"
timeout 240 rocprofv3 --kernel-trace --stats --output-format csv -d "$OUT/kt" \
  -- "$V2" --circuit "$BWC" --shots "$BWS" --seed 42 >/dev/null 2>&1
f=$(find "$OUT/kt" -name "*kernel_stats.csv" 2>/dev/null | head -1)
[ -n "$f" ] && head -4 "$f"

echo
echo "DONE. raw pmc under $OUT"
