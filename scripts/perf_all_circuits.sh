#!/bin/bash
#SBATCH --job-name=all-circuits
#SBATCH --partition=mi300x
#SBATCH --nodelist=rad-mi300x-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=01:50:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/all_circuits_%j.log

# Run ALL circuit fixtures through SVM and compiled paths
# 5 runs each (not 20 — time-constrained), all available .stim files.
set +e
BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS="$BASE/results/all_circuits_${TIMESTAMP}"
mkdir -p "$RESULTS"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done

echo "=== All Circuits Evaluation ==="
echo "Node: $(hostname)  Date: $(date)"
GPU_ARCH=$(rocminfo 2>/dev/null | grep -m1 "Name:.*gfx" | grep -oE "gfx[0-9]+" || echo "gfx942")

BINARY="$BASE/build-gpu-hsa/run_gpu"
[ -x "$BINARY" ] || BINARY="$BASE/build-gpu-node/run_gpu"
[ -x "$BINARY" ] || {
    BUILD="$BASE/build-gpu-allcirc"
    mkdir -p "$BUILD" && cd "$BUILD"
    cmake "$BASE" -DCLIFFT_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
    make -j$(nproc) run_gpu 2>&1 | tail -3
    cd "$BASE"
    BINARY="$BUILD/run_gpu"
}
echo "Binary: $BINARY"

SHOTS=100000
RUNS=5
CSV="$RESULTS/all.csv"
echo "source,circuit,mode,run,shots,peak_rank,passed,sample_sec,shots_per_sec,status" > "$CSV"

extract_field() { echo "$1" | grep "\"$2\"" | head -1 | grep -oE '[0-9]+\.?[0-9eE+\-]*' || true; }

run_circuit() {
    local SOURCE=$1 CIRCUIT=$2 MODE=$3 FLAG=$4
    local NAME=$(basename "$CIRCUIT" .stim)
    [ -f "$CIRCUIT" ] || return

    "$BINARY" --circuit "$CIRCUIT" --shots 1000 $FLAG >/dev/null 2>&1 || true

    for i in $(seq 1 $RUNS); do
        OUT=$("$BINARY" --circuit "$CIRCUIT" --shots $SHOTS $FLAG 2>&1)
        RC=$?
        if [ $RC -ne 0 ]; then
            echo "$SOURCE,$NAME,$MODE,$i,$SHOTS,-1,0,0,0,FAIL:rc=$RC" >> "$CSV"
            continue
        fi
        SPS=$(extract_field "$OUT" "shots_per_second_sampling_only"); SPS="${SPS:-0}"
        SEC=$(extract_field "$OUT" "sample_seconds"); SEC="${SEC:-0}"
        RANK=$(extract_field "$OUT" "peak_rank"); RANK="${RANK:-0}"
        PASSED=$(extract_field "$OUT" "passed_shots"); PASSED="${PASSED:-0}"
        echo "$SOURCE,$NAME,$MODE,$i,$SHOTS,$RANK,$PASSED,$SEC,$SPS,PASS" >> "$CSV"
    done
}

COUNT=0
TOTAL=$(find "$BASE/tests/fixtures" -name "*.stim" | wc -l)
echo "Total circuits: $TOTAL"
echo ""

# Process all .stim files
for CIRCUIT in $(find "$BASE/tests/fixtures" -name "*.stim" | sort); do
    COUNT=$((COUNT + 1))
    REL=$(echo "$CIRCUIT" | sed "s|$BASE/tests/fixtures/||")
    SOURCE=$(dirname "$REL")
    NAME=$(basename "$CIRCUIT" .stim)
    printf "[%3d/%d] %-50s " "$COUNT" "$TOTAL" "$REL"

    run_circuit "$SOURCE" "$CIRCUIT" "svm" ""
    run_circuit "$SOURCE" "$CIRCUIT" "compiled" "--hybrid"

    # Quick inline summary
    SVM_SPS=$(grep ",$NAME,svm," "$CSV" | tail -1 | cut -d',' -f9)
    COMP_SPS=$(grep ",$NAME,compiled," "$CSV" | tail -1 | cut -d',' -f9)
    SVM_SPS="${SVM_SPS:-0}"; COMP_SPS="${COMP_SPS:-0}"
    printf "svm=%s comp=%s\n" "$SVM_SPS" "$COMP_SPS"
done

echo ""
echo "=== Summary ==="
wc -l "$CSV"
echo "CSV: $CSV"
echo "Done: $(date)"
