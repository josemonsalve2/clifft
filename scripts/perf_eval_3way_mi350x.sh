#!/bin/bash
#SBATCH --job-name=eval-mi350x
#SBATCH --partition=mi350x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=01:50:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/eval_3way_mi350x_%j.log


# Full 3-Way Performance Evaluation on MI350X (gfx950)
set +e
BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS="$BASE/results/eval_3way_${TIMESTAMP}_mi350x"
mkdir -p "$RESULTS"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$ROCM/lib/llvm/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
# Ensure cmake is available
which cmake >/dev/null 2>&1 || {
    for cm in /usr/bin/cmake /opt/cmake/bin/cmake /usr/local/bin/cmake; do
        [ -x "$cm" ] && { export PATH="$(dirname $cm):$PATH"; break; }
    done
}
# Try module system
which cmake >/dev/null 2>&1 || { module load cmake 2>/dev/null || true; }

echo "=== MI350X 3-Way Evaluation ==="
echo "Node: $(hostname)  Job: ${SLURM_JOB_ID:-n/a}  Date: $(date)"
rocm-smi --showid --showproductname 2>/dev/null | head -5 || true
GPU_ARCH=$(rocminfo 2>/dev/null | grep -m1 "Name:.*gfx" | grep -oE "gfx[0-9]+" || echo "gfx950")
ROCM_VER=$(cat "$ROCM_PATH/.info/version" 2>/dev/null || echo "unknown")

# Use pre-built binary (cross-compiled on MI300X node)
BINARY="$BASE/build-gpu-mi350x-cross/run_gpu"
[ -x "$BINARY" ] || BINARY="$BASE/build-gpu-eval-mi350x/run_gpu"
[ -x "$BINARY" ] || { echo "ERROR: No gfx950 binary found. Run build_mi350x_on_mi300x.sh first."; exit 1; }

SHOTS=1000000; RUNS=20; WARMUP=3
CSV="$RESULTS/bench.csv"
echo "mode,label,tier,peak_rank,qubits,run,shots,actual_rank,passed,sample_sec,kernel_sec,shots_per_sec,node,gpu_arch,rocm_version" > "$CSV"

extract_field() { echo "$1" | grep "\"$2\"" | head -1 | grep -oE '[0-9]+\.?[0-9eE+\-]*' || true; }

run_mode() {
    local MODE=$1 FLAG=$2 CIRCUIT=$3 LABEL=$4 TIER=$5 RANK=$6 QUBITS=$7
    [ -f "$CIRCUIT" ] || return
    for _w in $(seq 1 $WARMUP); do "$BINARY" --circuit "$CIRCUIT" --shots 100000 $FLAG >/dev/null 2>&1 || true; done
    for i in $(seq 1 $RUNS); do
        OUT=$("$BINARY" --circuit "$CIRCUIT" --shots $SHOTS $FLAG 2>&1); RC=$?
        if [ $RC -ne 0 ]; then
            echo "$MODE,$LABEL,$TIER,$RANK,$QUBITS,$i,$SHOTS,-1,0,0,0,0,$(hostname),$GPU_ARCH,$ROCM_VER" >> "$CSV"; continue; fi
        SPS=$(extract_field "$OUT" "shots_per_second_sampling_only"); SPS="${SPS:-0}"
        SEC=$(extract_field "$OUT" "sample_seconds"); SEC="${SEC:-0}"
        KSEC=$(extract_field "$OUT" "kernel_seconds"); KSEC="${KSEC:-$SEC}"
        PASSED=$(extract_field "$OUT" "passed_shots"); PASSED="${PASSED:-0}"
        ARANK=$(extract_field "$OUT" "peak_rank"); ARANK="${ARANK:-0}"
        printf "  %-8s %-22s run %2d: %12s shots/s\n" "$MODE" "$LABEL" "$i" "$SPS"
        echo "$MODE,$LABEL,$TIER,$RANK,$QUBITS,$i,$SHOTS,$ARANK,$PASSED,$SEC,$KSEC,$SPS,$(hostname),$GPU_ARCH,$ROCM_VER" >> "$CSV"
    done
}

bench() {
    local LABEL=$1 CIRCUIT=$2 TIER=$3 RANK=$4 QUBITS=$5
    [ -f "$CIRCUIT" ] || return
    echo "=== $LABEL tier=$TIER ==="
    run_mode "svm" "" "$CIRCUIT" "$LABEL" "$TIER" "$RANK" "$QUBITS"
    run_mode "compiled" "--hybrid" "$CIRCUIT" "$LABEL" "$TIER" "$RANK" "$QUBITS"
}

# Same circuit set as MI300X for direct comparison
bench "target_qec"      "$BASE/tests/fixtures/target_qec.stim"                  "0" "0" "0"
bench "color_d7_r7"     "$BASE/tests/fixtures/large/color_d7_r7.stim"           "0" "0" "55"
bench "surface_d13_r13" "$BASE/tests/fixtures/large/surface_d13_r13.stim"       "0" "0" "337"
bench "cultivation_d5"  "$BASE/tests/fixtures/cultivation_d5.stim"              "1" "4" "0"
bench "surface_d7_t5"   "$BASE/tests/fixtures/large/surface_d7_t5.stim"         "1" "5" "97"
bench "rank_r1_q33_d3"  "$BASE/tests/fixtures/rank_sweep/rank_q33_r1_d3.stim"   "1" "1" "33"
bench "rank_r4_q33_d3"  "$BASE/tests/fixtures/rank_sweep/rank_q33_r4_d3.stim"   "1" "4" "33"
bench "surface_d7_t10"  "$BASE/tests/fixtures/large/surface_d7_t10.stim"        "2" "10" "97"
bench "rank_r5_q33_d3"  "$BASE/tests/fixtures/rank_sweep/rank_q33_r5_d3.stim"   "2" "5" "33"
bench "rank_r10_q33_d3" "$BASE/tests/fixtures/rank_sweep/rank_q33_r10_d3.stim"  "2" "10" "33"
bench "rank11_q50"      "$BASE/tests/fixtures/large/rank11_q50_d5_tier3.stim"   "3" "11" "50"
bench "rank_r15_q33_d3" "$BASE/tests/fixtures/rank_sweep/rank_q33_r15_d3.stim"  "3" "15" "33"
bench "rank_r19_q33_d3" "$BASE/tests/fixtures/rank_sweep/rank_q33_r19_d3.stim"  "3" "19" "33"

echo "CSV: $CSV"
echo "Done: $(date)"
