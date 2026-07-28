#!/bin/bash
#SBATCH --job-name=wgsbw
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/V2_performance/tools/wgsbw_%j.log
#
# Two questions, one job, no code changes. Both gate the "spatial locality /
# HBM-stack placement" design work: there is no point placing slices if the
# global tier is latency-bound rather than bandwidth-bound, and no point
# reasoning about pool underfill if the pool is not actually the limiter.
#
# Q1. Is the global tier's 32 GB pool budget (v2_kernel.cc:438) leaving the
#     machine underfilled on the QV circuits? wgsweep_50152 answered this for
#     the surface family (saturates at 1024) but never touched QV, which is
#     where V2 is weakest (0.732-0.990). V2_GLOBAL_WGS (v2_kernel.cc:446) is
#     already an override, so this is pure measurement.
#
# Q2. What fraction of peak HBM bandwidth does the global tier achieve?
#     bench_all.sh records FETCH_SIZE/WRITE_SIZE as "not collectable on this
#     gfx950 config" -- but that was in a 6-counter pass. This retries them
#     ALONE, which may fit in the hardware's counter budget. If they are truly
#     unavailable, TCC_EA_RDREQ_sum/WRREQ_sum x 64B/32B is the fallback.
#
# Also settles a discrepancy: wgsweep_50152.log reports "Compute Unit: 128" on
# this partition, while report SS14.5 reasons against 256 CUs. If it is 128, the
# rank-24 pool of 168 wgs is OVERSUBSCRIBED, not underfilled, and that section's
# mechanism is wrong. Recorded either way.
#
# Reports v2_kernel_seconds (kernel time, not host wall) per feedback_numa_bind.
# All arms run back-to-back on the SAME node; never compare across node types.
set -u

BASE=/shared/jmonsalv/quantum/clifft_rl/clifft
cd "$BASE" || exit 1

for ROCM in /opt/rocm-7.2.3 /opt/rocm; do
  [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"
                          export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
export LLVM_PREFIX=/shared/jmonsalv/software/modules/llvm/upstream_05082025

# V2_SPECIALIZE=1 is MANDATORY: unset, run_v2 measures its own bytecode
# interpreter and every number below is meaningless (project_v2_specialize_env).
export V2_SPECIALIZE=1
export V2_NO_CPU=1

V2=$BASE/build-v2-nohip/run_v2
SPEC_CACHE=$BASE/V2_performance/scratch/wgsbw_cache
mkdir -p "$SPEC_CACHE"
export V2_SPEC_CACHE_DIR="$SPEC_CACHE"

echo "=== node ==="
echo "host=$(hostname -s)  job=${SLURM_JOB_ID:-0}  partition=${SLURM_JOB_PARTITION:-?}"
rocminfo 2>/dev/null | grep -m1 -E "gfx[0-9a-z]+"
echo "--- CU count (SS14.5 assumes 256; wgsweep_50152 logged 128) ---"
rocminfo 2>/dev/null | grep -m2 "Compute Unit"
rocminfo 2>/dev/null | grep -m1 -i "Marketing Name" || true
echo "--- HBM pool size (the 32 GB budget is a fraction of this) ---"
rocminfo 2>/dev/null | grep -A2 -m1 "Pool 1" | head -5 || true
echo

# The five circuits above rank 19 -- the only ones where the pool formula
# produces fewer than the 2048 cap, and the weakest results in the corpus.
# rank 20 -> 2048 (capped), 21 -> 1360, 22 -> 680, 23 -> 336, 24 -> 168.
CIRCS=(
"tools/bench/fixtures/qv20_seed42.stim 2000 2048"
"tests/fixtures/large/qv21_L8_seed42.stim 2000 1360"
"tests/fixtures/large/qv22_L6_seed42.stim 1000 680"
"tests/fixtures/large/qv23_L5_seed42.stim 1000 336"
"tests/fixtures/large/qv24_L4_seed42.stim 500 168"
)

# A missing fixture must not look like a clean result (job 50785 lost 8 of 26
# circuits that way). Fail loudly, before any measurement.
for e in "${CIRCS[@]}"; do
  c=$(echo "$e" | awk '{print $1}')
  [ -f "$c" ] || { echo "*** FIXTURE MISSING: $c -- aborting ***" >&2; exit 1; }
done

run_kt() {  # $1=circuit $2=shots  -> prints v2_kernel_seconds or "FAIL"
  local t
  t=$("$V2" --circuit "$1" --shots "$2" --seed 42 2>/dev/null \
        | grep -oP '"v2_kernel_seconds": \K[0-9.e+-]+')
  echo "${t:-FAIL}"
}

echo "=========================================================="
echo "Q1. V2_GLOBAL_WGS sweep on the five rank>=20 circuits"
echo "    3 runs per arm, kernel time in seconds, lower is better."
echo "    'dflt' = the 32 GB budget's own choice (3rd column above)."
echo "=========================================================="
for e in "${CIRCS[@]}"; do
  c=$(echo "$e" | awk '{print $1}'); s=$(echo "$e" | awk '{print $2}')
  d=$(echo "$e" | awk '{print $3}'); n=$(basename "$c" .stim)
  echo "--- $n (shots=$s, default wgs=$d) ---"
  # Warm the specialization cache + correctness gate OFF the timed path.
  unset V2_GLOBAL_WGS
  "$V2" --circuit "$c" --shots "$s" --seed 42 >/dev/null 2>&1

  for wgs in dflt 168 336 680 1360 2048 4096 8192; do
    if [ "$wgs" = dflt ]; then unset V2_GLOBAL_WGS; else export V2_GLOBAL_WGS=$wgs; fi
    printf "  wgs=%-5s " "$wgs"
    for i in 1 2 3; do printf "%s " "$(run_kt "$c" "$s")"; done
    echo
  done
  unset V2_GLOBAL_WGS
done

echo
echo "=========================================================="
echo "Q2. HBM bandwidth counters -- are they collectable at all?"
echo "    bench_all.sh:78 says no, but that was a 6-counter pass."
echo "    Retried one set at a time on a single circuit."
echo "=========================================================="
BWC=tests/fixtures/large/qv22_L6_seed42.stim
BWS=1000
"$V2" --circuit "$BWC" --shots "$BWS" --seed 42 >/dev/null 2>&1   # warm

OUT=$BASE/V2_performance/scratch/wgsbw_pmc
rm -rf "$OUT"; mkdir -p "$OUT"
i=0
for PMC in "FETCH_SIZE,WRITE_SIZE" \
           "FETCH_SIZE" \
           "TCC_EA_RDREQ_sum,TCC_EA_WRREQ_sum" \
           "TCC_EA_RDREQ_32B_sum" \
           "TCC_REQ_sum,TCC_HIT_sum,TCC_MISS_sum"; do
  i=$((i+1))
  echo "--- pmc set $i: $PMC ---"
  if timeout 300 rocprofv3 --pmc "$PMC" --output-format csv -d "$OUT/set$i" \
       -- "$V2" --circuit "$BWC" --shots "$BWS" --seed 42 >"$OUT/set$i.out" 2>"$OUT/set$i.err"; then
    echo "  COLLECTED"
    find "$OUT/set$i" -name "*counter_collection.csv" | head -1 | while read -r f; do
      python3 - "$f" <<'PY'
import csv, sys, collections
tot = collections.defaultdict(float)
with open(sys.argv[1]) as fh:
    for r in csv.DictReader(fh):
        k = r.get("Kernel_Name","")[:40]
        tot[(k, r.get("Counter_Name",""))] += float(r.get("Counter_Value",0) or 0)
for (k,c),v in sorted(tot.items()):
    print(f"    {k:42s} {c:24s} {v:,.0f}")
PY
    done
  else
    echo "  UNAVAILABLE: $(head -2 "$OUT/set$i.err" | tr '\n' ' ' | cut -c1-160)"
  fi
done

echo
echo "--- kernel duration for the same circuit (to turn bytes into GB/s) ---"
timeout 300 rocprofv3 --kernel-trace --stats --output-format csv -d "$OUT/kt" \
  -- "$V2" --circuit "$BWC" --shots "$BWS" --seed 42 >/dev/null 2>&1
find "$OUT/kt" -name "*kernel_stats.csv" | head -1 | xargs -r cat | head -5

echo
echo "DONE. raw pmc under $OUT"
