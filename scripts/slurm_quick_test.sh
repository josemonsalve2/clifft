#!/bin/bash
#SBATCH --job-name=quick-test
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:05:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/quick_test_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BUILD="$BASE/build-gpu-mlir-mi355x"
BIN="$BUILD/run_gpu"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

# Clean kernel cache
rm -rf "$HOME/.clifft/kernel_cache/mlir" 2>/dev/null

echo "=== SVM test ==="
$BIN --circuit "$BASE/tests/fixtures/incremental/01_frame/frame_h.stim" --shots 100 --seed 42 2>&1
echo "EXIT: $?"

echo ""
echo "=== MLIR test ==="
$BIN --circuit "$BASE/tests/fixtures/incremental/01_frame/frame_h.stim" --shots 100 --seed 42 --mlir 2>&1
echo "EXIT: $?"
