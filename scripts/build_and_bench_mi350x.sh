#!/bin/bash
#SBATCH --job-name=mi350x-full
#SBATCH --partition=mi350x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/mi350x_full_%j.log

set -euo pipefail

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
cd "$BASE"

export HOME="/home/jmonsalv"
export PATH="/home/jmonsalv/miniforge3/bin:/opt/rocm/bin:$PATH"

echo "=== MI350X Full Pipeline ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "cmake: $(which cmake 2>/dev/null || echo 'not found')"
echo ""

echo "=== GPU Info ==="
rocm-smi --showid --showproductname 2>/dev/null | head -8 || true
rocminfo 2>/dev/null | grep -E "Marketing Name|gfx" | head -5 || true
echo ""

# Build for gfx950
echo "=== Phase 1: Build ==="
BUILD_DIR="build-gpu-mi350x"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. \
    -DCLIFFT_ENABLE_HIP=ON \
    -DCMAKE_HIP_ARCHITECTURES=gfx950 \
    -DCMAKE_BUILD_TYPE=Release \
    2>&1 | tail -10

cmake --build . -j$(nproc) 2>&1 | tail -20

cd "$BASE"
BINARY="$BUILD_DIR/run_gpu"

if [ ! -f "$BINARY" ]; then
    echo "BUILD FAILED"
    exit 1
fi
echo "=== Build successful ==="
echo ""

# GPU Tests
echo "=== Phase 2: GPU Tests ==="
$BUILD_DIR/tests/clifft_tests "[gpu]" 2>&1 || echo "GPU tests had issues"
echo ""

# Benchmark
SHOTS=1000000
RUNS=20
WARMUP_RUNS=3

OUTDIR="$BASE/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="$OUTDIR/bench_mi350x_${TIMESTAMP}.csv"
SUMMARY="$OUTDIR/bench_mi350x_summary_${TIMESTAMP}.txt"

echo "mode,circuit,run,shots,peak_rank,passed,sample_seconds,shots_per_second" > "$OUTFILE"

extract_field() {
    echo "$1" | grep "\"$2\"" | head -1 | grep -oE '[0-9]+\.?[0-9eE+\-]*'
}

run_circuit() {
    local MODE=$1
    local CIRCUIT=$2
    local LABEL=$3
    local FLAG=""
    [ "$MODE" = "compiled" ] && FLAG="--hybrid"

    for i in $(seq 1 $WARMUP_RUNS); do
        $BINARY --circuit "$CIRCUIT" --shots 100000 $FLAG > /dev/null 2>&1 || true
    done

    for i in $(seq 1 $RUNS); do
        OUTPUT=$($BINARY --circuit "$CIRCUIT" --shots "$SHOTS" $FLAG 2>&1) || {
            echo "  $MODE run $i: FAILED"
            echo "$MODE,$LABEL,$i,$SHOTS,-1,0,0,0" >> "$OUTFILE"
            continue
        }
        SPS=$(extract_field "$OUTPUT" "shots_per_second_sampling_only")
        RANK=$(extract_field "$OUTPUT" "peak_rank")
        PASSED=$(extract_field "$OUTPUT" "passed_shots")
        SEC=$(extract_field "$OUTPUT" "sample_seconds")
        printf "  %s run %2d: %s shots/s (rank=%s)\n" "$MODE" "$i" "$SPS" "$RANK"
        echo "$MODE,$LABEL,$i,$SHOTS,$RANK,$PASSED,$SEC,$SPS" >> "$OUTFILE"
    done
}

echo "=== Phase 3: Rigorous Benchmark (MI350X) ==="
echo "Runs=$RUNS, Shots=$SHOTS, Warmup=$WARMUP_RUNS"

for LABEL_CIRCUIT in \
    "target_qec|$BASE/tests/fixtures/target_qec.stim" \
    "surface_d5_r5|$BASE/tests/fixtures/large/surface_d5_r5.stim" \
    "surface_d7_r7|$BASE/tests/fixtures/large/surface_d7_r7.stim" \
    "surface_d7_r14|$BASE/tests/fixtures/large/surface_d7_r14.stim" \
    "color_d7|$BASE/tests/fixtures/large/color_d7.stim" \
    "rep_d5_r100|$BASE/tests/fixtures/large/rep_d5_r100.stim" \
    "cultivation_d5|$BASE/tests/fixtures/cultivation_d5.stim" \
    "circuit_d7|$BASE/tests/fixtures/circuit_d7_p0.0005.stim" \
    "sweep_q17_t0_d3|$BASE/tests/fixtures/sweep/sweep_q17_t0_d3.stim" \
    "sweep_q17_t2_d5|$BASE/tests/fixtures/sweep/sweep_q17_t2_d5.stim" \
    "sweep_q17_t5_d5|$BASE/tests/fixtures/sweep/sweep_q17_t5_d5.stim" \
    "sweep_q17_t10_d5|$BASE/tests/fixtures/sweep/sweep_q17_t10_d5.stim" \
    "sweep_q33_t0_d3|$BASE/tests/fixtures/sweep/sweep_q33_t0_d3.stim" \
    "sweep_q33_t2_d5|$BASE/tests/fixtures/sweep/sweep_q33_t2_d5.stim" \
    "sweep_q33_t5_d5|$BASE/tests/fixtures/sweep/sweep_q33_t5_d5.stim" \
    "sweep_q33_t10_d5|$BASE/tests/fixtures/sweep/sweep_q33_t10_d5.stim" \
; do
    LABEL="${LABEL_CIRCUIT%%|*}"
    CIRCUIT="${LABEL_CIRCUIT#*|}"
    [ ! -f "$CIRCUIT" ] && { echo "SKIP: $LABEL"; continue; }
    echo ""
    echo "--- $LABEL ---"
    run_circuit "svm" "$CIRCUIT" "$LABEL"
    run_circuit "compiled" "$CIRCUIT" "$LABEL"
done

# Summary
echo "" | tee "$SUMMARY"
echo "MI350X Summary — $(date)" | tee -a "$SUMMARY"
echo "Host: $(hostname)" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"
printf "%-22s | %10s %8s %5s | %10s %8s %5s | %7s\n" \
    "Circuit" "SVM Mean" "StdDev" "CV" "Comp Mean" "StdDev" "CV" "Delta" | tee -a "$SUMMARY"

LABELS=$(awk -F',' 'NR>1 && $8+0>0 {print $2}' "$OUTFILE" | sort -u)
for LABEL in $LABELS; do
    SM=$(grep "^svm,$LABEL," "$OUTFILE" | awk -F',' '$8+0>0{print $8}' | awk 'BEGIN{n=0;s=0}{v=$1+0;a[n]=v;s+=v;n++}END{if(!n)exit;m=s/n;ss=0;for(i=0;i<n;i++)ss+=(a[i]-m)^2;printf "%.0f %.0f %.1f",m,sqrt(ss/(n>1?n-1:1)),100*sqrt(ss/(n>1?n-1:1))/m}')
    CM=$(grep "^compiled,$LABEL," "$OUTFILE" | awk -F',' '$8+0>0{print $8}' | awk 'BEGIN{n=0;s=0}{v=$1+0;a[n]=v;s+=v;n++}END{if(!n)exit;m=s/n;ss=0;for(i=0;i<n;i++)ss+=(a[i]-m)^2;printf "%.0f %.0f %.1f",m,sqrt(ss/(n>1?n-1:1)),100*sqrt(ss/(n>1?n-1:1))/m}')
    S1=$(echo "$SM" | awk '{print $1}'); C1=$(echo "$CM" | awk '{print $1}')
    [ "$S1" = "" ] || [ "$C1" = "" ] && continue
    D=$(awk "BEGIN{printf \"%+.1f\",100*($C1-$S1)/$S1}" 2>/dev/null)
    printf "%-22s | %s | %s | %6s%%\n" "$LABEL" "$SM" "$CM" "$D" | tee -a "$SUMMARY"
done

echo "" | tee -a "$SUMMARY"
echo "Data: $OUTFILE" | tee -a "$SUMMARY"
echo "=== COMPLETE ===" | tee -a "$SUMMARY"
