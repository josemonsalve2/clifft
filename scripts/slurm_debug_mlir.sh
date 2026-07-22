#!/bin/bash
#SBATCH --job-name=debug-mlir
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/debug_mlir_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BUILD="$BASE/build-gpu-mlir-mi355x"
BIN="$BUILD/run_gpu"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

# Clear cache
rm -rf /tmp/clifft_mlir_cache "$HOME/.cache/clifft/mlir_*" 2>/dev/null

echo "=== Debug: rank_q17_r2_d1 (rank 2, should use register tier) ==="
echo "--- With stderr ---"
$BIN --circuit "$BASE/tests/fixtures/rank_sweep/rank_q17_r2_d1.stim" --shots 100 --seed 42 --mlir 2>&1 || echo "EXIT CODE: $?"

echo ""
echo "=== Debug: rank_q17_r1_d3 (rank 1 + noise, should use register tier) ==="
echo "--- With stderr ---"
$BIN --circuit "$BASE/tests/fixtures/rank_sweep/rank_q17_r1_d3.stim" --shots 100 --seed 42 --mlir 2>&1 || echo "EXIT CODE: $?"

echo ""
echo "=== Debug: rank_q17_r5_d1 (rank 5, should use coop tier) ==="
echo "--- With stderr ---"
$BIN --circuit "$BASE/tests/fixtures/rank_sweep/rank_q17_r5_d1.stim" --shots 100 --seed 42 --mlir 2>&1 || echo "EXIT CODE: $?"
