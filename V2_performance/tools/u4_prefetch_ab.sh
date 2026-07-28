#!/bin/bash
#SBATCH --job-name=u4pf
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:40:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/V2_performance/tools/u4pf_%j.log
#
# A/B the software-pipelined U4 butterfly (V2_U4_PREFETCH, commit 1134ffc).
#
# Why this experiment exists. The global tier moves ~750 GB/s against ~8 TB/s
# of peak: it is LATENCY-bound, not bandwidth-bound. gfx950 ISA shows why --
# the U4 loop issues four global_loads and immediately s_waitcnts on them, so
# the vector pipe idles for a whole HBM round trip with nothing in flight.
# Pipelining the next quad's loads ahead of the current quad's arithmetic, per
# loop body at rank 22 / k=(3,9):
#
#     baseline    152 instr  12 loads  12 vmwait  42 VGPR  0 spills
#     pipelined   142 instr   8 loads   8 vmwait  54 VGPR  0 spills
#
# The catch, and the actual question this job answers: +12 VGPRs. The rank>=22
# global kernels already sit at the 128 VGPR / 64 AGPR cap, so the overlap is
# bought with occupancy, and fewer waves in flight is itself a latency-hiding
# loss. Static ISA cannot settle which side wins. Hardware can.
#
# Pinned to d13-21: every wgs number this builds on was measured there, and
# mi350x-es is heterogeneous (feedback_numa_bind).
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
[ -x "$V2" ] || { echo "*** run_v2 missing -- build first ***"; exit 1; }

# Separate cache dirs per arm. The key already hashes the generated source
# (which carries the #define) so a collision is impossible, but keeping them
# apart makes a stale-cache mistake visibly impossible rather than argued.
echo "=== node ==="
echo "host=$(hostname -s)  job=${SLURM_JOB_ID:-0}"
echo

kt() {  # $1=circuit $2=shots
  local t
  t=$("$V2" --circuit "$1" --shots "$2" --seed 42 2>/dev/null \
        | grep -oP '"v2_kernel_seconds": \K[0-9.e+-]+')
  echo "${t:-FAIL}"
}
# Full JSON for the correctness comparison.
res() {
  "$V2" --circuit "$1" --shots "$2" --seed 42 2>/dev/null \
    | grep -oP '"(v2_passed|v2_observable_ones|match)": *[^,}]*' | tr '\n' ' '
}

CIRCUITS="
tests/fixtures/large/qv20_seed42.stim:1000
tests/fixtures/large/qv21_L8_seed42.stim:1000
tests/fixtures/large/qv22_L6_seed42.stim:1000
tests/fixtures/large/qv23_L5_seed42.stim:1000
tests/fixtures/large/qv24_L4_seed42.stim:500
"

echo "=========================================================="
echo "Q1. Kernel time, prefetch OFF vs ON (3 runs, lower better)"
echo "    Default (device-derived) pool in both arms."
echo "=========================================================="
for entry in $CIRCUITS; do
  c=${entry%%:*}; s=${entry##*:}
  [ -f "$c" ] || { echo "*** MISSING: $c ***"; continue; }
  echo "--- $(basename "$c" .stim) shots=$s ---"
  for arm in off on; do
    if [ "$arm" = on ]; then export V2_U4_PREFETCH=1; else unset V2_U4_PREFETCH; fi
    export V2_SPEC_CACHE_DIR=$BASE/V2_performance/scratch/u4pf_cache_$arm
    mkdir -p "$V2_SPEC_CACHE_DIR"
    "$V2" --circuit "$c" --shots "$s" --seed 42 >/dev/null 2>&1  # warm compile+gate
    printf "  pf=%-3s " "$arm"
    for i in 1 2 3; do printf "%s " "$(kt "$c" "$s")"; done
    echo
  done
  unset V2_U4_PREFETCH
done

echo
echo "=========================================================="
echo "Q2. Correctness -- prefetch must be byte-exact."
echo "    Only the fetch point moves; the arithmetic and its"
echo "    order are unchanged, so results must be IDENTICAL."
echo "    Any difference here falsifies the aliasing argument."
echo "=========================================================="
for entry in $CIRCUITS; do
  c=${entry%%:*}; s=${entry##*:}
  [ -f "$c" ] || continue
  echo "--- $(basename "$c" .stim) ---"
  for arm in off on; do
    if [ "$arm" = on ]; then export V2_U4_PREFETCH=1; else unset V2_U4_PREFETCH; fi
    export V2_SPEC_CACHE_DIR=$BASE/V2_performance/scratch/u4pf_cache_$arm
    printf "  pf=%-3s %s\n" "$arm" "$(res "$c" "$s")"
  done
  unset V2_U4_PREFETCH
done

echo
echo "=========================================================="
echo "Q3. Occupancy -- did the +12 VGPRs actually cost waves?"
echo "    This is the mechanism check behind whatever Q1 shows."
echo "=========================================================="
PFC=tests/fixtures/large/qv22_L6_seed42.stim
for arm in off on; do
  if [ "$arm" = on ]; then export V2_U4_PREFETCH=1; else unset V2_U4_PREFETCH; fi
  export V2_SPEC_CACHE_DIR=$BASE/V2_performance/scratch/u4pf_cache_$arm
  echo "--- pf=$arm ---"
  h=$(ls -t "$V2_SPEC_CACHE_DIR"/*.hsaco 2>/dev/null | head -1)
  if [ -n "$h" ]; then
    "$LLVM_PREFIX/bin/llvm-readelf" --notes "$h" 2>/dev/null \
      | grep -iE "vgpr|sgpr|spill|lds|occupancy" | head -12 \
      || echo "  (no metadata)"
  else
    echo "  (no hsaco found)"
  fi
done
unset V2_U4_PREFETCH

echo
echo "=========================================================="
echo "Q4. Bandwidth + L2 at rank 22, both arms."
echo "    If prefetch works, FETCH_SIZE/second should RISE"
echo "    (same bytes, less time) -- that is the direct"
echo "    evidence the stalls were the limiter."
echo "=========================================================="
OUT=$BASE/V2_performance/scratch/u4pf_pmc
rm -rf "$OUT"; mkdir -p "$OUT"
for arm in off on; do
  if [ "$arm" = on ]; then export V2_U4_PREFETCH=1; else unset V2_U4_PREFETCH; fi
  export V2_SPEC_CACHE_DIR=$BASE/V2_performance/scratch/u4pf_cache_$arm
  echo "--- pf=$arm : FETCH_SIZE ---"
  if timeout 300 rocprofv3 --pmc "FETCH_SIZE" --output-format csv -d "$OUT/$arm" \
       -- "$V2" --circuit "$PFC" --shots 1000 --seed 42 >"$OUT/$arm.out" 2>"$OUT/$arm.err"; then
    f=$(find "$OUT/$arm" -name "*counter_collection.csv" 2>/dev/null | head -1)
    [ -n "$f" ] && python3 - "$f" <<'PY'
import csv, sys, collections
tot = collections.defaultdict(float)
with open(sys.argv[1]) as fh:
    for r in csv.DictReader(fh):
        tot[r.get("Counter_Name","")] += float(r.get("Counter_Value",0) or 0)
for c, v in sorted(tot.items()):
    print(f"    {c:22s} {v:,.0f}")
PY
  else
    echo "  UNAVAILABLE: $(head -3 "$OUT/$arm.err" 2>/dev/null | tr '\n' ' ' | cut -c1-150)"
  fi
done
unset V2_U4_PREFETCH

echo
echo "=== done ==="
