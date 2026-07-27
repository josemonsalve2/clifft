#!/bin/bash
#SBATCH --job-name=qv10err
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/qv10_err_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BIN="$BASE/build-gpu-mlir-mi355x/run_gpu"
for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }; done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }; done
rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null

echo "=== qv10 MLIR full stderr ==="
timeout 120 "$BIN" --circuit "$BASE/tests/fixtures/qv10.stim" --shots 100 --seed 42 --mlir 2>&1 | grep -iE "error|dominate|abort|FATAL|does not|compiled|loaded|generated|verif" | head -20
echo ""
echo "=== validate saved coop MLIR through mlir-opt ==="
for f in "$BASE/results/debug_coop_140ops.mlir"; do
    [ -f "$f" ] || continue
    "$LLVM_PREFIX/bin/mlir-opt" --canonicalize --cse --convert-func-to-llvm "$f" 2>&1 | grep -iE "error|dominate|note" | head -10
done
echo ""
echo "Done: $(date)"
