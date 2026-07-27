#!/bin/bash
#SBATCH --job-name=bld-fix
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/bld_fix_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BUILD="$BASE/build-gpu-mlir-mi355x"
BIN="$BUILD/run_gpu"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

echo "=== Building with all fixes ==="
cd "$BUILD"
make -j1 2>&1 | grep -E "error:" | head -10
echo "---"
make -j$(nproc) 2>&1 | tail -5
echo "RC=$?"
echo ""

rm -rf "$HOME/.clifft/kernel_cache/mlir" 2>/dev/null
STIM="$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim"

echo "=== MLIR warmup ==="
timeout 120 $BIN --circuit "$STIM" --shots 100 --seed 42 --mlir 2>&1 | tail -5

echo ""
echo "=== Performance ==="
svm_t=$(timeout 30 $BIN --circuit "$STIM" --shots 100000 --seed 42 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['sample_seconds'])" 2>/dev/null)
mlir_t=$(timeout 30 $BIN --circuit "$STIM" --shots 100000 --seed 42 --mlir 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['sample_seconds'])" 2>/dev/null)
echo "SVM 100k:  ${svm_t}s"
echo "MLIR 100k: ${mlir_t}s"

echo ""
echo "=== Kernel metadata ==="
HSACO=$(ls -t $HOME/.clifft/kernel_cache/mlir/*.hsaco 2>/dev/null | head -1)
[ -f "$HSACO" ] && $LLVM_PREFIX/bin/llvm-readelf -n "$HSACO" 2>/dev/null | grep -E "vgpr|sgpr|private|group|flat_work|dynamic|kernarg"

echo ""
echo "Done: $(date)"
