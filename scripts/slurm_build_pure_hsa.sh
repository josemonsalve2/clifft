#!/bin/bash
#SBATCH --job-name=pure-hsa
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/pure_hsa_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BUILD="$BASE/build-gpu-mlir-mi355x"
BIN="$BUILD/run_gpu"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

echo "=== Building Pure HSA MLIR Dispatch ==="
cd "$BUILD"
make -j1 2>&1 | grep "error:" | head -10
echo "---"
make -j$(nproc) 2>&1 | tail -3
echo "RC=$?"
echo ""

STIM="$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim"
rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null

echo "=== SVM baseline ==="
timeout 30 $BIN --circuit "$STIM" --shots 100000 --seed 42 2>&1 | grep "sample_seconds"

echo ""
echo "=== MLIR warmup (pure HSA) ==="
timeout 180 $BIN --circuit "$STIM" --shots 100 --seed 42 --mlir 2>&1 | tail -10

echo ""
echo "=== MLIR 100k ==="
timeout 30 $BIN --circuit "$STIM" --shots 100000 --seed 42 --mlir 2>&1 | grep "sample_seconds"

echo ""
echo "=== MLIR 1M ==="
timeout 60 $BIN --circuit "$STIM" --shots 1000000 --seed 42 --mlir 2>&1 | grep "sample_seconds"

echo ""
echo "=== SVM 1M ==="
timeout 60 $BIN --circuit "$STIM" --shots 1000000 --seed 42 2>&1 | grep "sample_seconds"

echo ""
echo "=== Correctness ==="
svm_p=$(timeout 30 $BIN --circuit "$STIM" --shots 10000 --seed 42 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
mlir_p=$(timeout 30 $BIN --circuit "$STIM" --shots 10000 --seed 42 --mlir 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
echo "SVM=$svm_p MLIR=$mlir_p"

echo ""
echo "=== Kernel metadata ==="
HSACO=$(ls -t ~/.clifft/kernel_cache/mlir/*.hsaco 2>/dev/null | head -1)
[ -f "$HSACO" ] && $LLVM_PREFIX/bin/llvm-readelf -n "$HSACO" 2>/dev/null | grep -E "vgpr|sgpr|private|group|flat_work|dynamic|kernarg"

echo ""
echo "Done: $(date)"
