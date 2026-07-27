#!/bin/bash
#SBATCH --job-name=correct
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/correctness_final_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BIN="$BASE/build-gpu-mlir-mi355x/run_gpu"
SHOTS=10000

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

echo "=== CORRECTNESS TEST: SVM vs MLIR ==="
echo "Node: $(hostname), Shots: $SHOTS, Date: $(date)"
echo ""

CIRCUITS=(
    "tests/fixtures/incremental/01_frame_only/frame_h.stim"
    "tests/fixtures/incremental/02_single_expand/one_t.stim"
    "tests/fixtures/incremental/02_single_expand/two_t.stim"
    "tests/fixtures/incremental/02_single_expand/four_t.stim"
    "tests/fixtures/incremental/05_combinations/h_then_t.stim"
    "tests/fixtures/incremental/06_small_circuits/rank4_mixed.stim"
    "tests/fixtures/qv10.stim"
    "tests/fixtures/large/circuit_d3_p0.001.stim"
)

printf "%-25s %5s %8s %8s %8s %s\n" "CIRCUIT" "RANK" "SVM" "MLIR" "KERN_MS" "STATUS"
printf "%-25s %5s %8s %8s %8s %s\n" "-------" "----" "---" "----" "-------" "------"

for stim in "${CIRCUITS[@]}"; do
    full="$BASE/$stim"
    [ -f "$full" ] || continue
    name=$(basename "$stim" .stim)

    svm_json=$($BIN --circuit "$full" --shots $SHOTS --seed 42 2>/dev/null)
    svm_p=$(echo "$svm_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
    svm_r=$(echo "$svm_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['peak_rank'])" 2>/dev/null)

    tmpf=$(mktemp)
    # Warmup (compilation)
    timeout 120 $BIN --circuit "$full" --shots 10 --seed 42 --mlir 2>/dev/null 1>/dev/null
    # Actual run with stderr separated
    mlir_json=$($BIN --circuit "$full" --shots $SHOTS --seed 42 --mlir 2>"$tmpf")
    mlir_p=$(echo "$mlir_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
    mlir_k=$(grep kernel_seconds "$tmpf" | grep -oP '[0-9.]+' | head -1)
    rm -f "$tmpf"

    status="MISMATCH"
    [ "$svm_p" = "$mlir_p" ] && status="OK"
    [ -z "$mlir_p" ] && status="PARSE_FAIL"

    printf "%-25s %5s %8s %8s %8s %s\n" "$name" "$svm_r" "$svm_p" "${mlir_p:-?}" "${mlir_k:-?}ms" "$status"
done

echo ""
echo "Done: $(date)"
