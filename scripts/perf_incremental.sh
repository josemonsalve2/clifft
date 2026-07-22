#!/bin/bash
#SBATCH --job-name=incr-eval
#SBATCH --partition=mi300x
#SBATCH --nodelist=rad-mi300x-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/incr_eval_%j.log

# Incremental Single-Gate Performance Evaluation
#
# Tests each gate type individually, printing explicit PASS/FAIL for each.
# No silent fallbacks — if a mode fails on a circuit, it says so.

set +e
BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS="$BASE/results/incr_eval_${TIMESTAMP}"
mkdir -p "$RESULTS"

# ROCm
for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done

echo "=== Incremental Performance Evaluation ==="
echo "Node: $(hostname)  Date: $(date)"
GPU_ARCH=$(rocminfo 2>/dev/null | grep -m1 "Name:.*gfx" | grep -oE "gfx[0-9]+" || echo "gfx942")

# Build if needed
BINARY="$BASE/build-gpu-hsa/run_gpu"
if [ ! -x "$BINARY" ]; then
    BINARY="$BASE/build-gpu-node/run_gpu"
fi
if [ ! -x "$BINARY" ]; then
    echo "Building..."
    BUILD="$BASE/build-gpu-incr"
    mkdir -p "$BUILD" && cd "$BUILD"
    cmake "$BASE" -DCLIFFT_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 \
        -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
    make -j$(nproc) run_gpu 2>&1 | tail -5
    cd "$BASE"
    BINARY="$BUILD/run_gpu"
fi
echo "Binary: $BINARY"

SHOTS=1000000
RUNS=5
CSV="$RESULTS/incr.csv"
echo "level,circuit,mode,run,shots,peak_rank,passed,sample_sec,shots_per_sec,status" > "$CSV"

INC_DIR="$BASE/tests/fixtures/incremental"

test_circuit() {
    local LEVEL=$1 CIRCUIT=$2 MODE=$3 FLAG=$4
    local CIRC_NAME=$(basename "$CIRCUIT" .stim)

    echo -n "  $MODE $CIRC_NAME: "

    # One warmup
    "$BINARY" --circuit "$CIRCUIT" --shots 1000 $FLAG >/dev/null 2>&1

    # Timed runs
    local PASS=0 FAIL=0 SPS_SUM=0
    for i in $(seq 1 $RUNS); do
        OUT=$("$BINARY" --circuit "$CIRCUIT" --shots $SHOTS $FLAG 2>&1)
        RC=$?

        if [ $RC -ne 0 ]; then
            FAIL=$((FAIL + 1))
            echo "$LEVEL,$CIRC_NAME,$MODE,$i,$SHOTS,-1,0,0,0,FAIL:rc=$RC" >> "$CSV"
            continue
        fi

        # Check for stderr error messages (compiled kernel compilation failures)
        ERRS=$(echo "$OUT" | grep -c "failed\|error\|Error" || true)
        if [ "$ERRS" -gt 0 ]; then
            ERR_MSG=$(echo "$OUT" | grep -i "failed\|error" | head -1 | tr ',' ';')
            echo "$LEVEL,$CIRC_NAME,$MODE,$i,$SHOTS,-1,0,0,0,WARN:$ERR_MSG" >> "$CSV"
        fi

        SPS=$(echo "$OUT" | grep '"shots_per_second_sampling_only"' | head -1 | grep -oE '[0-9]+\.?[0-9eE+\-]*' || true)
        SEC=$(echo "$OUT" | grep '"sample_seconds"' | head -1 | grep -oE '[0-9]+\.?[0-9eE+\-]*' || true)
        RANK=$(echo "$OUT" | grep '"peak_rank"' | head -1 | grep -oE '[0-9]+' || true)
        PASSED=$(echo "$OUT" | grep '"passed_shots"' | head -1 | grep -oE '[0-9]+' || true)
        SPS="${SPS:-0}"; SEC="${SEC:-0}"; RANK="${RANK:-0}"; PASSED="${PASSED:-0}"
        PASS=$((PASS + 1))
        echo "$LEVEL,$CIRC_NAME,$MODE,$i,$SHOTS,$RANK,$PASSED,$SEC,$SPS,PASS" >> "$CSV"

        # Accumulate for average
        SPS_SUM=$(python3 -c "print(float('$SPS_SUM') + float('$SPS'))" 2>/dev/null || echo "$SPS_SUM")
    done

    if [ $FAIL -gt 0 ]; then
        echo "FAIL ($FAIL/$RUNS failed)"
    else
        AVG=$(python3 -c "print(f'{float(\"$SPS_SUM\")/$RUNS/1e6:.2f}')" 2>/dev/null || echo "?")
        echo "PASS  ${AVG}M shots/s (avg of $RUNS)"
    fi
}

run_level() {
    local LEVEL=$1 DIR=$2
    echo ""
    echo "=============================="
    echo "Level: $LEVEL ($DIR)"
    echo "=============================="

    for CIRCUIT in "$INC_DIR/$DIR"/*.stim; do
        [ -f "$CIRCUIT" ] || continue
        local NAME=$(basename "$CIRCUIT" .stim)
        echo ""
        echo "--- $NAME ---"
        test_circuit "$LEVEL" "$CIRCUIT" "svm" ""
        test_circuit "$LEVEL" "$CIRCUIT" "compiled" "--hybrid"
    done
}

# Run levels in order
run_level "1_frame"     "01_frame_only"
run_level "2_expand"    "02_single_expand"
run_level "3_array"     "03_single_array"
run_level "4_measure"   "04_measure"
run_level "5_combo"     "05_combinations"
run_level "6_circuit"   "06_small_circuits"

# Summary table
echo ""
echo "=============================="
echo "SUMMARY"
echo "=============================="
python3 -c "
import csv, sys
from collections import defaultdict

data = defaultdict(lambda: defaultdict(lambda: {'pass': 0, 'fail': 0, 'sps': []}))
with open('$CSV') as f:
    for row in csv.DictReader(f):
        key = (row['level'], row['circuit'])
        mode = row['mode']
        if row['status'].startswith('PASS'):
            data[key][mode]['pass'] += 1
            sps = float(row.get('shots_per_sec', 0) or 0)
            if sps > 0: data[key][mode]['sps'].append(sps)
        elif row['status'].startswith('FAIL'):
            data[key][mode]['fail'] += 1

print(f\"{'Level':<12} {'Circuit':<22} {'SVM':>12} {'Compiled':>12} {'Delta':>8} {'Status'}\")
print('-'*75)
for (level, circuit) in sorted(data.keys()):
    svm = data[(level, circuit)]['svm']
    comp = data[(level, circuit)]['compiled']
    svm_avg = sum(svm['sps'])/len(svm['sps'])/1e6 if svm['sps'] else 0
    comp_avg = sum(comp['sps'])/len(comp['sps'])/1e6 if comp['sps'] else 0
    delta = f'{(comp_avg-svm_avg)/svm_avg*100:+.1f}%' if svm_avg > 0 and comp_avg > 0 else '—'
    svm_s = 'PASS' if svm['fail']==0 and svm['pass']>0 else f'FAIL({svm[\"fail\"]})'
    comp_s = 'PASS' if comp['fail']==0 and comp['pass']>0 else f'FAIL({comp[\"fail\"]})'
    status = 'OK' if svm_s == 'PASS' and comp_s == 'PASS' else f'{svm_s}/{comp_s}'
    print(f'{level:<12} {circuit:<22} {svm_avg:>11.2f}M {comp_avg:>11.2f}M {delta:>8} {status}')
" 2>/dev/null

echo ""
echo "CSV: $CSV"
echo "Done: $(date)"
