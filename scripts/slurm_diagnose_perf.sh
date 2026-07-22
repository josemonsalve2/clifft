#!/bin/bash
#SBATCH --job-name=diag-perf
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/diag_perf_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BUILD="$BASE/build-gpu-mlir-mi355x"
BIN="$BUILD/run_gpu"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

rm -rf "$HOME/.clifft/kernel_cache/mlir" 2>/dev/null

echo "=== Register tier perf diagnosis ==="
echo "Circuit: frame_h (simplest possible — rank 0, 4 instructions)"
echo ""

# Run MLIR to generate the .mlir and .ll files (they stay in /tmp)
echo "--- Generating MLIR kernel ---"
$BIN --circuit "$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim" --shots 100 --seed 42 --mlir 2>&1 | head -5

echo ""
echo "--- Generated MLIR file size ---"
ls -la /tmp/clifft_mlir_*.mlir 2>/dev/null | tail -1

echo ""
echo "--- Generated LLVM-IR file size ---"
ls -la /tmp/clifft_mlir_*.ll 2>/dev/null | tail -1

echo ""
echo "--- LLVM-IR instruction count ---"
wc -l /tmp/clifft_mlir_*.ll 2>/dev/null | tail -1

echo ""
echo "--- HSACO size ---"
ls -la $HOME/.clifft/kernel_cache/mlir/*.hsaco 2>/dev/null | tail -1

echo ""
echo "--- Register usage (from llc verbose) ---"
LLIR=$(ls /tmp/clifft_mlir_*.ll 2>/dev/null | tail -1)
if [ -f "$LLIR" ]; then
    $LLVM_PREFIX/bin/llc --march=amdgcn --mcpu=gfx950 -mattr=+wavefrontsize64 -O3 \
        --print-after-all "$LLIR" -o /dev/null 2>&1 | grep "NumSgpr\|NumVgpr\|ScratchSize\|spill" | head -10
fi

echo ""
echo "=== Timing breakdown ==="
echo "--- SVM: 10k, 100k, 1M shots ---"
for s in 10000 100000 1000000; do
    t=$($BIN --circuit "$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim" --shots $s --seed 42 2>&1 | grep sample_seconds | grep -oP '[0-9.]+')
    echo "  SVM $s shots: ${t}s"
done

echo "--- MLIR: 10k, 100k, 1M shots ---"
for s in 10000 100000 1000000; do
    t=$($BIN --circuit "$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim" --shots $s --seed 42 --mlir 2>&1 | grep sample_seconds | grep -oP '[0-9.]+')
    echo "  MLIR $s shots: ${t}s"
done

echo ""
echo "=== Throughput comparison ==="
echo "--- SVM: shots/second ---"
$BIN --circuit "$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim" --shots 1000000 --seed 42 2>&1 | grep "shots_per_second"

echo "--- MLIR: shots/second ---"
$BIN --circuit "$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim" --shots 1000000 --seed 42 --mlir 2>&1 | grep "shots_per_second"

echo ""
echo "Done: $(date)"
