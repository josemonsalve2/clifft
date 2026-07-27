#!/bin/bash
#SBATCH --job-name=gdiag
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/global_diag_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BIN="$BASE/build-gpu-mlir-mi355x/run_gpu"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done
MLIR_OPT="$LLVM_PREFIX/bin/mlir-opt"

echo "=== BUILD ==="
cd "$BASE/build-gpu-mlir-mi355x" && make -j$(nproc) 2>&1 | tail -3
[ ${PIPESTATUS[0]} -ne 0 ] && { echo "BUILD FAILED"; make -j1 2>&1 | grep error: | head; exit 1; }

rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null
rm -f "$BASE/results/debug_global_"*.mlir 2>/dev/null

echo "=== qv20 GLOBAL tier compile (rank 20) ==="
# The codegen saves debug MLIR (coop path already does). Trigger compile:
timeout 120 "$BIN" --circuit "$BASE/tools/bench/fixtures/qv20_seed42.stim" --shots 10 --seed 42 --mlir 2>&1 | grep -iE "error|dominate|generated|compiled|FATAL|global|note:" | head -30

echo ""
echo "=== Locate any global MLIR temp ==="
ls -la /tmp/clifft_mlir_global_*.mlir 2>/dev/null
ls -la "$BASE/results/debug_global_"*.mlir 2>/dev/null
ls -la "$BASE/results/debug_coop_"*.mlir 2>/dev/null | tail -3

# If a global .mlir survived, validate it directly and show the failing region
for f in /tmp/clifft_mlir_global_*.mlir "$BASE/results/debug_global_"*ops.mlir; do
    [ -f "$f" ] || continue
    echo ""
    echo "=== mlir-opt validate $f ($(wc -l <"$f") lines) ==="
    "$MLIR_OPT" --canonicalize --cse --convert-func-to-llvm "$f" 2>&1 | grep -iE "error|dominate|note" | head -20
done

echo ""
echo "Done: $(date)"
