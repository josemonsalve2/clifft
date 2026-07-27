#!/bin/bash
#SBATCH --job-name=qperf
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/qperf_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BUILD="$BASE/build-gpu-mlir-mi355x"
BIN="$BUILD/run_gpu"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

STIM="$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim"
rm -rf "$HOME/.clifft/kernel_cache/mlir" 2>/dev/null

echo "=== SVM 100k ==="
timeout 30 $BIN --circuit "$STIM" --shots 100000 --seed 42 2>&1 | grep "sample_seconds"

echo ""
echo "=== MLIR warmup + 100 shots ==="
timeout 120 $BIN --circuit "$STIM" --shots 100 --seed 42 --mlir 2>&1 | tail -5

echo ""
echo "=== MLIR 100k ==="
timeout 30 $BIN --circuit "$STIM" --shots 100000 --seed 42 --mlir 2>&1 | grep "sample_seconds"

echo ""
echo "=== MLIR 1M ==="
timeout 30 $BIN --circuit "$STIM" --shots 1000000 --seed 42 --mlir 2>&1 | grep "sample_seconds"

echo ""
echo "=== SVM 1M ==="
timeout 30 $BIN --circuit "$STIM" --shots 1000000 --seed 42 2>&1 | grep "sample_seconds"

echo ""
echo "=== Kernel metadata ==="
HSACO=$(ls -t $HOME/.clifft/kernel_cache/mlir/*.hsaco 2>/dev/null | head -1)
[ -f "$HSACO" ] && $LLVM_PREFIX/bin/llvm-readelf -n "$HSACO" 2>/dev/null | grep -E "vgpr_count|sgpr_count|private_segment|flat_work_group|dynamic_stack"

echo ""
echo "Done: $(date)"
