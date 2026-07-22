#!/bin/bash
#SBATCH --job-name=bench-mi350x
#SBATCH --partition=mi350x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/bench_mi350x_%j.log

# Run pre-built gfx950 binary on MI350X node
set -euo pipefail

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
cd "$BASE"

BINARY="$BASE/build-gpu-mi350x/run_gpu"
SHOTS=1000000
RUNS=20
WARMUP_RUNS=3

if [ ! -f "$BINARY" ]; then
    echo "ERROR: Binary not found at $BINARY"
    echo "Run build_mi350x.sh on mi300x partition first"
    exit 1
fi

echo "=== MI350X Rigorous Benchmark ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Binary: $BINARY"
echo ""

echo "=== GPU Info ==="
rocm-smi --showid --showproductname 2>/dev/null | head -8 || true
rocminfo 2>/dev/null | grep -A2 "Marketing Name\|gfx" | head -10 || true
echo ""

# GPU Tests
echo "=== GPU Tests ==="
TEST_BINARY="$BASE/build-gpu-mi350x/tests/clifft_tests"
if [ -f "$TEST_BINARY" ]; then
    $TEST_BINARY "[gpu]" 2>&1 || echo "GPU TESTS FAILED"
else
    echo "Test binary not found, skipping"
fi
echo ""

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

echo "=== Rigorous Benchmark ==="
echo "Runs=$RUNS, Shots=$SHOTS, Warmup=$WARMUP_RUNS"
echo ""

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

    if [ ! -f "$CIRCUIT" ]; then
        echo "SKIP: $LABEL not found"
        continue
    fi
    echo ""
    echo "--- $LABEL ---"
    run_circuit "svm" "$CIRCUIT" "$LABEL"
    run_circuit "compiled" "$CIRCUIT" "$LABEL"
done

# Statistical summary
echo "" | tee "$SUMMARY"
echo "MI350X Rigorous Benchmark Summary — $(date)" | tee -a "$SUMMARY"
echo "Host: $(hostname), Runs=$RUNS, Shots=$SHOTS" | tee -a "$SUMMARY"
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
    CM=$(echo "$COMP_STATS" | awk '{print $1}')
    SS=$(echo "$SVM_STATS" | awk '{print $2}')
    CS=$(echo "$COMP_STATS" | awk '{print $2}')
    SC=$(echo "$SVM_STATS" | awk '{print $3}')
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
