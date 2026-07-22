#!/bin/bash
#SBATCH --job-name=bench-mi350x
#SBATCH --partition=mi350x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/bench_mi350x_%j.log

set -euo pipefail

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
cd "$BASE"

SHOTS=1000000
RUNS=20
WARMUP_RUNS=3

echo "=== MI350X Rigorous Benchmark ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo ""

# Phase 1: Build for gfx950
echo "=== Phase 1: Build for gfx950 (MI350X) ==="
BUILD_DIR="build-gpu-mi350x"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Check ROCm version
echo "ROCm version:"
cat /opt/rocm/.info/version 2>/dev/null || hipcc --version 2>/dev/null | head -3 || true

cmake .. \
    -DCLIFFT_ENABLE_HIP=ON \
    -DCMAKE_HIP_ARCHITECTURES=gfx950 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_HIP_COMPILER=/opt/rocm/bin/hipcc \
    2>&1 | tail -10

cmake --build . -j$(nproc) 2>&1 | tail -20

cd "$BASE"
BINARY="$BUILD_DIR/run_gpu"

if [ ! -f "$BINARY" ]; then
    echo "BUILD FAILED - binary not found"
    exit 1
fi

echo ""
echo "=== Build successful ==="
echo ""

# Phase 2: GPU Info
echo "=== GPU Info ==="
rocm-smi --showid --showproductname 2>/dev/null | head -8 || true
rocminfo 2>/dev/null | grep -A2 "Marketing Name\|gfx" | head -10 || true
echo ""

# Phase 3: GPU Tests
echo "=== Phase 2: GPU Tests ==="
TEST_BINARY="$BUILD_DIR/tests/clifft_tests"
if [ -f "$TEST_BINARY" ]; then
    $TEST_BINARY "[gpu]" 2>&1 || echo "GPU TESTS FAILED (may be expected on new arch)"
else
    echo "Test binary not found, skipping"
fi
echo ""

# Phase 4: Rigorous Benchmark
OUTDIR="$BASE/results"
mkdir -p "$OUTDIR"
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
    if [ "$MODE" = "compiled" ]; then
        FLAG="--hybrid"
    fi

    # Warmup
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
        SEC=$(extract_field "$OUTPUT" "sample_seconds")
        PASSED=$(extract_field "$OUTPUT" "passed_shots")
        RANK=$(extract_field "$OUTPUT" "peak_rank")
        printf "  %s run %2d: %s shots/s (rank=%s)\n" "$MODE" "$i" "$SPS" "$RANK"
        echo "$MODE,$LABEL,$i,$SHOTS,$RANK,$PASSED,$SEC,$SPS" >> "$OUTFILE"
    done
}

echo "=== Phase 3: Rigorous Benchmark ==="
echo "Runs=$RUNS, Shots=$SHOTS, Warmup=$WARMUP_RUNS"
echo ""

# All interesting circuits
declare -A CIRCUITS
CIRCUITS["target_qec"]="$BASE/tests/fixtures/target_qec.stim"
CIRCUITS["surface_d5_r5"]="$BASE/tests/fixtures/large/surface_d5_r5.stim"
CIRCUITS["surface_d7_r7"]="$BASE/tests/fixtures/large/surface_d7_r7.stim"
CIRCUITS["surface_d7_r14"]="$BASE/tests/fixtures/large/surface_d7_r14.stim"
CIRCUITS["color_d7"]="$BASE/tests/fixtures/large/color_d7.stim"
CIRCUITS["rep_d5_r100"]="$BASE/tests/fixtures/large/rep_d5_r100.stim"
CIRCUITS["cultivation_d5"]="$BASE/tests/fixtures/cultivation_d5.stim"
CIRCUITS["circuit_d7"]="$BASE/tests/fixtures/circuit_d7_p0.0005.stim"

SWEEP_CIRCUITS=(
    "sweep_q17_t0_d3" "sweep_q17_t2_d5" "sweep_q17_t5_d5" "sweep_q17_t10_d5"
    "sweep_q33_t0_d3" "sweep_q33_t2_d5" "sweep_q33_t5_d5" "sweep_q33_t10_d5"
)

COUNT=0
for LABEL in target_qec surface_d5_r5 surface_d7_r7 surface_d7_r14 color_d7 rep_d5_r100 cultivation_d5 circuit_d7; do
    CIRCUIT="${CIRCUITS[$LABEL]}"
    if [ ! -f "$CIRCUIT" ]; then
        echo "SKIP: $LABEL not found"
        continue
    fi
    COUNT=$((COUNT + 1))
    echo ""
    echo "--- [$COUNT] $LABEL ---"
    run_circuit "svm" "$CIRCUIT" "$LABEL"
    run_circuit "compiled" "$CIRCUIT" "$LABEL"
done

for SWEEP in "${SWEEP_CIRCUITS[@]}"; do
    CIRCUIT="$BASE/tests/fixtures/sweep/${SWEEP}.stim"
    if [ ! -f "$CIRCUIT" ]; then
        echo "SKIP: $SWEEP not found"
        continue
    fi
    COUNT=$((COUNT + 1))
    echo ""
    echo "--- [$COUNT] $SWEEP ---"
    run_circuit "svm" "$CIRCUIT" "$SWEEP"
    run_circuit "compiled" "$CIRCUIT" "$SWEEP"
done

# Statistical summary
echo "" | tee "$SUMMARY"
echo "MI350X Rigorous Benchmark Summary — $(date)" | tee -a "$SUMMARY"
echo "Host: $(hostname), GPU: MI355X (gfx950), Runs=$RUNS, Shots=$SHOTS" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"
printf "%-22s | %10s %8s %5s | %10s %8s %5s | %7s\n" \
    "Circuit" "SVM Mean" "StdDev" "CV" "Comp Mean" "StdDev" "CV" "Delta" | tee -a "$SUMMARY"
echo "------------------------------------------------------------------------------------------------------" | tee -a "$SUMMARY"

LABELS=$(awk -F',' 'NR>1 && $8+0>0 {print $2}' "$OUTFILE" | sort -u)
for LABEL in $LABELS; do
    SVM_STATS=$(grep "^svm,$LABEL," "$OUTFILE" | awk -F',' '$8+0>0{print $8}' | awk '
    BEGIN{n=0;sum=0}{v=$1+0;vals[n]=v;sum+=v;n++}
    END{if(n==0){printf "0 0 0";exit};mean=sum/n;ss=0;for(i=0;i<n;i++)ss+=(vals[i]-mean)^2;sd=sqrt(ss/(n>1?n-1:1));printf "%.0f %.0f %.1f",mean,sd,100*sd/mean}')
    COMP_STATS=$(grep "^compiled,$LABEL," "$OUTFILE" | awk -F',' '$8+0>0{print $8}' | awk '
    BEGIN{n=0;sum=0}{v=$1+0;vals[n]=v;sum+=v;n++}
    END{if(n==0){printf "0 0 0";exit};mean=sum/n;ss=0;for(i=0;i<n;i++)ss+=(vals[i]-mean)^2;sd=sqrt(ss/(n>1?n-1:1));printf "%.0f %.0f %.1f",mean,sd,100*sd/mean}')

    SM=$(echo "$SVM_STATS" | awk '{print $1}')
    SS=$(echo "$SVM_STATS" | awk '{print $2}')
    SC=$(echo "$SVM_STATS" | awk '{print $3}')
    CM=$(echo "$COMP_STATS" | awk '{print $1}')
    CS=$(echo "$COMP_STATS" | awk '{print $2}')
    CC=$(echo "$COMP_STATS" | awk '{print $3}')

    if [ "$SM" != "0" ] && [ "$CM" != "0" ]; then
        DELTA=$(awk "BEGIN{printf \"%+.1f\", 100*($CM-$SM)/$SM}")
        printf "%-22s | %10s %8s %4s%% | %10s %8s %4s%% | %6s%%\n" \
            "$LABEL" "$SM" "$SS" "$SC" "$CM" "$CS" "$CC" "$DELTA" | tee -a "$SUMMARY"
    fi
done

echo "" | tee -a "$SUMMARY"
echo "Raw data: $OUTFILE" | tee -a "$SUMMARY"
echo "=== Benchmark complete ===" | tee -a "$SUMMARY"
