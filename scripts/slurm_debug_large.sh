#!/bin/bash
#SBATCH --job-name=debug-large
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/debug_large_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BIN="$BASE/build-gpu-mlir-mi355x/run_gpu"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

echo "=== circuit_d3_p0.001 ==="
timeout 120 $BIN --circuit "$BASE/tests/fixtures/large/circuit_d3_p0.001.stim" --shots 100 --seed 42 --mlir 2>&1 || echo "EXIT: $?"
echo ""
echo "=== rank13_q50_d20_deep ==="
timeout 120 $BIN --circuit "$BASE/tests/fixtures/large/rank13_q50_d20_deep.stim" --shots 100 --seed 42 --mlir 2>&1 || echo "EXIT: $?"
