#!/bin/bash
#SBATCH --job-name=dbg-raw
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/debug_raw_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BUILD="$BASE/build-gpu-mlir-mi355x"
BIN="$BUILD/run_gpu"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

cd "$BUILD"
make -j$(nproc) 2>&1 | tail -3
echo "BUILD: $?"

rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null
rm -f /tmp/clifft_debug_raw.mlir

$BIN --circuit "$BASE/tests/fixtures/incremental/02_single_expand/four_t.stim" --shots 10 --seed 42 --mlir 2>&1 | head -5

echo "=== RAW MLIR (instruction comments) ==="
grep "// op\[" /tmp/clifft_debug_raw.mlir 2>/dev/null
echo ""
echo "=== DISCARD/UNSUPPORTED ==="
grep -i "discard\|unsupported" /tmp/clifft_debug_raw.mlir 2>/dev/null | head -10
echo ""
echo "=== POSTSELECT/DETECTOR ==="
grep -i "postselect\|detector\|OP_DETECTOR" /tmp/clifft_debug_raw.mlir 2>/dev/null | head -10
echo ""
echo "=== Line count ==="
wc -l /tmp/clifft_debug_raw.mlir 2>/dev/null
echo ""
echo "=== Active_k at each op ==="
grep "active_k=" /tmp/clifft_debug_raw.mlir 2>/dev/null | head -20

echo ""
echo "Done: $(date)"
