#!/bin/bash
#SBATCH --job-name=dbg-u2
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/dbg_u2_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BUILD="$BASE/build-gpu-mlir-mi355x"
BIN="$BUILD/run_gpu"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

echo "Node: $(hostname)"
echo "Binary: $(file $BIN)"
echo "Binary date: $(ls -la $BIN)"

# Clear MLIR cache
rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null

# Quick test with full stderr
echo "=== frame_h SVM ==="
$BIN --circuit "$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim" --shots 100 --seed 42 2>&1
echo ""
echo "=== frame_h MLIR ==="
$BIN --circuit "$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim" --shots 100 --seed 42 --mlir 2>&1
echo ""
echo "=== four_t MLIR ==="
$BIN --circuit "$BASE/tests/fixtures/incremental/02_single_expand/four_t.stim" --shots 100 --seed 42 --mlir 2>&1
echo ""
echo "=== qv10 MLIR (rank 10) ==="
timeout 120 $BIN --circuit "$BASE/tests/fixtures/qv10.stim" --shots 100 --seed 42 --mlir 2>&1

echo ""
echo "Done: $(date)"
