#!/bin/bash
#SBATCH --job-name=clean-test
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/clean_test_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BIN="$BASE/build-gpu-mlir-mi355x/run_gpu"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

# FULL cache clean
rm -rf ~/.clifft/kernel_cache 2>/dev/null
echo "Cache fully cleaned"
echo ""

CIRCUITS=(
    "tests/fixtures/incremental/01_frame_only/frame_h.stim"
    "tests/fixtures/incremental/02_single_expand/four_t.stim"
    "tests/fixtures/incremental/02_single_expand/one_t.stim"
    "tests/fixtures/incremental/02_single_expand/two_t.stim"
    "tests/fixtures/incremental/05_combinations/h_then_t.stim"
)

for stim in "${CIRCUITS[@]}"; do
    full="$BASE/$stim"
    name=$(basename "$stim" .stim)
    echo "=== $name ==="
    svm_out=$($BIN --circuit "$full" --shots 10000 --seed 42 2>/dev/null)
    svm_p=$(echo "$svm_out" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['passed_shots'], d['peak_rank'])" 2>/dev/null)
    mlir_out=$($BIN --circuit "$full" --shots 10000 --seed 42 --mlir 2>&1)
    mlir_p=$(echo "$mlir_out" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['passed_shots'], d['peak_rank'])" 2>/dev/null)
    mlir_err=$(echo "$mlir_out" | grep -E "error|panic|abort|Unsupported" | head -3)
    echo "SVM: $svm_p | MLIR: $mlir_p $mlir_err"
    [ -n "$mlir_err" ] && echo "  STDERR: $mlir_err"
done

echo ""
echo "Done: $(date)"
