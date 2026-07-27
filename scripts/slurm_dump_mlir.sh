#!/bin/bash
#SBATCH --job-name=dump-mlir
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/dump_mlir_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BIN="$BASE/build-gpu-mlir-mi355x/run_gpu"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

# Set CLIFFT_DUMP_MLIR to save MLIR text
export CLIFFT_DUMP_MLIR=1
rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null

echo "=== Dump MLIR for frame_h ==="
$BIN --circuit "$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim" --shots 10 --seed 42 --mlir 2>&1 | head -5

echo "=== Check cache ==="
ls -la ~/.clifft/kernel_cache/mlir/ 2>/dev/null | head -10

# Also check if there's a .mlir or .ll file saved
find ~/.clifft/ -name "*.mlir" -o -name "*.ll" 2>/dev/null | head -10

# Check ISA
HSACO=$(ls -t ~/.clifft/kernel_cache/mlir/*.hsaco 2>/dev/null | head -1)
if [ -f "$HSACO" ]; then
    echo "=== ISA metadata ==="
    $LLVM_PREFIX/bin/llvm-readelf -n "$HSACO" 2>/dev/null | grep -E "vgpr|sgpr|private|group|flat_work|kernarg"
    echo ""
    echo "=== ISA disassembly (first 50 lines) ==="
    $LLVM_PREFIX/bin/llvm-objdump -d "$HSACO" 2>/dev/null | head -50
fi

echo ""
echo "Done: $(date)"
