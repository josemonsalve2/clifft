#!/bin/bash
#SBATCH --job-name=3tier-sh
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/3tier_svm_hybrid_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BIN="$BASE/build-gpu-mlir-mi355x/run_gpu"
SHOTS=10000

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done

echo "=== 3-TIER: SVM vs Hybrid (no MLIR — fast run) ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Shots: $SHOTS"
echo ""

# All circuits spanning 3 tiers
CIRCUITS=(
    # Register tier
    "tests/fixtures/incremental/01_frame_only/frame_h.stim"
    "tests/fixtures/large/circuit_d3_p0.001.stim"
    "tests/fixtures/large/surface_d7_t5.stim"
    # Coop tier
    "tests/fixtures/qv10.stim"
    "tests/fixtures/cultivation_d5.stim"
    "tests/fixtures/large/circuit_d5_p0.001.stim"
    "tests/fixtures/large/surface_d7_t10.stim"
    "tests/fixtures/large/surface_d7_t15.stim"
    "tests/fixtures/large/surface_d9_t10.stim"
    # Global tier
    "tools/bench/fixtures/qv20_seed42.stim"
    "tests/fixtures/large/surface_d7_t19.stim"
    "tests/fixtures/large/surface_d9_t15.stim"
    "tests/fixtures/large/surface_d9_t19.stim"
    "tests/fixtures/large/surface_d11_t15.stim"
    "tests/fixtures/large/surface_d11_t19.stim"
)

printf "%-30s %5s %6s %10s %10s %10s %10s\n" "CIRCUIT" "RANK" "QUBITS" "SVM_kern" "HYB_kern" "SVM_total" "HYB_total"
printf "%-30s %5s %6s %10s %10s %10s %10s\n" "-------" "----" "------" "--------" "--------" "---------" "---------"

for stim in "${CIRCUITS[@]}"; do
    full="$BASE/$stim"
    [ -f "$full" ] || { echo "SKIP: $stim"; continue; }
    name=$(basename "$stim" .stim)

    # SVM (warm up + measure)
    $BIN --circuit "$full" --shots 100 --seed 42 >/dev/null 2>&1
    svm_out=$($BIN --circuit "$full" --shots $SHOTS --seed 42 2>&1)
    svm_rank=$(echo "$svm_out" | grep -o '"peak_rank": [0-9]*' | grep -o '[0-9]*')
    svm_kern=$(echo "$svm_out" | grep "kernel_seconds" | grep -oP '[0-9.]+' | head -1)
    svm_total=$(echo "$svm_out" | grep "TOTAL wall" | grep -oP '[0-9.]+' | head -1)
    svm_passed=$(echo "$svm_out" | grep -o '"passed_shots": [0-9]*' | grep -o '[0-9]*')
    svm_nq=$(echo "$svm_out" | grep -o '"measurements": [0-9]*' | grep -o '[0-9]*')

    # Hybrid (warm up + measure)
    timeout 120 $BIN --circuit "$full" --shots 100 --seed 42 --hybrid >/dev/null 2>&1
    hyb_out=$(timeout 120 $BIN --circuit "$full" --shots $SHOTS --seed 42 --hybrid 2>&1)
    hyb_kern=$(echo "$hyb_out" | grep "kernel_seconds" | grep -oP '[0-9.]+' | head -1)
    hyb_total=$(echo "$hyb_out" | grep "TOTAL wall" | grep -oP '[0-9.]+' | head -1)
    hyb_passed=$(echo "$hyb_out" | grep -o '"passed_shots": [0-9]*' | grep -o '[0-9]*')

    # Correctness check
    correct="OK"
    if [ -n "$svm_passed" ] && [ -n "$hyb_passed" ] && [ "$svm_passed" != "$hyb_passed" ]; then
        correct="MISMATCH(S=$svm_passed,H=$hyb_passed)"
    fi

    printf "%-30s %5s %6s %8sms %8sms %8sms %8sms  %s\n" \
        "$name" "${svm_rank:-?}" "${svm_nq:-?}" \
        "${svm_kern:-FAIL}" "${hyb_kern:-FAIL}" \
        "${svm_total:-FAIL}" "${hyb_total:-FAIL}" \
        "$correct"
done

echo ""
echo "Done: $(date)"
