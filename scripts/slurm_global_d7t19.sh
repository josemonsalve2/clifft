#!/bin/bash
#SBATCH --job-name=gd7t19
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/global_d7t19_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BIN="$BASE/build-gpu-mlir-mi355x/run_gpu"
for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }; done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }; done
MLIR_OPT="$LLVM_PREFIX/bin/mlir-opt"
rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null
rm -f "$BASE/results/debug_global_"*.mlir 2>/dev/null

T19="$BASE/tests/fixtures/large/surface_d7_t19.stim"
echo "=== surface_d7_t19 GLOBAL compile (rank 12) ==="
timeout 200 "$BIN" --circuit "$T19" --shots 10 --seed 42 --mlir 2>&1 | grep -iE "error|dominate|generated|compiled|FATAL|global|note:|redefinition" | head -25

echo ""
echo "=== validate saved global MLIR ==="
for f in "$BASE/results/debug_global_"*ops.mlir; do
    [ -f "$f" ] || continue
    echo "--- $f ($(wc -l <"$f") lines) ---"
    "$MLIR_OPT" --canonicalize --cse --convert-func-to-llvm "$f" 2>&1 | grep -iE "error|dominate|note|redefinition" | head -15
done
echo ""
echo "Done: $(date)"
