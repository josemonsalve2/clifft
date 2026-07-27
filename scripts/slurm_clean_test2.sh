#!/bin/bash
#SBATCH --job-name=clean-t2
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/clean_test2_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BIN="$BASE/build-gpu-mlir-mi355x/run_gpu"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

rm -rf ~/.clifft/kernel_cache 2>/dev/null

CIRCUITS=(
    "tests/fixtures/incremental/01_frame_only/frame_h.stim"
    "tests/fixtures/incremental/02_single_expand/four_t.stim"
    "tests/fixtures/incremental/05_combinations/h_then_t.stim"
    "tests/fixtures/qv10.stim"
    "tests/fixtures/large/circuit_d3_p0.001.stim"
)

for stim in "${CIRCUITS[@]}"; do
    full="$BASE/$stim"
    name=$(basename "$stim" .stim)
    echo "=== $name ==="

    svm_json=$($BIN --circuit "$full" --shots 10000 --seed 42 2>/dev/null)
    svm_p=$(echo "$svm_json" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['passed_shots'])" 2>/dev/null)
    svm_r=$(echo "$svm_json" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['peak_rank'])" 2>/dev/null)

    # MLIR: redirect stderr to a temp file to separate from stdout
    tmpf=$(mktemp)
    mlir_json=$($BIN --circuit "$full" --shots 10000 --seed 42 --mlir 2>"$tmpf")
    mlir_p=$(echo "$mlir_json" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['passed_shots'])" 2>/dev/null)
    mlir_r=$(echo "$mlir_json" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['peak_rank'])" 2>/dev/null)
    mlir_kern=$(grep kernel_seconds "$tmpf" | grep -oP '[0-9.]+' | head -1)
    rm -f "$tmpf"

    match="MISMATCH"
    [ "$svm_p" = "$mlir_p" ] && match="OK"
    printf "%-20s rank=%s  SVM=%s  MLIR=%s  kern=%sms  %s\n" "$name" "$svm_r" "$svm_p" "$mlir_p" "$mlir_kern" "$match"
done

echo ""
echo "Done: $(date)"
