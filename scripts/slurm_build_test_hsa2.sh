#!/bin/bash
#SBATCH --job-name=build-hsa2
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/build_hsa2_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BUILD="$BASE/build-gpu-mlir-mi355x"
BIN="$BUILD/run_gpu"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

echo "=== Building ==="
cd "$BUILD"
make -j$(nproc) 2>&1 | tail -10
RC=${PIPESTATUS[0]}
if [ $RC -ne 0 ]; then
    echo "BUILD FAILED (exit $RC)"
    make -j1 2>&1 | tail -30
    exit 1
fi
echo "BUILD OK"
echo ""

rm -rf "$HOME/.clifft/kernel_cache/mlir" 2>/dev/null

STIM="$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim"

echo "=== Test 1: SVM register tier (should work) ==="
$BIN --circuit "$STIM" --shots 1000 --seed 42 2>&1 | tail -5
echo ""

echo "=== Test 2: MLIR register tier (the one that crashes) ==="
# Enable HSA error reporting
export HSA_TOOLS_LIB=""
$BIN --circuit "$STIM" --shots 100 --seed 42 --mlir 2>&1 | tail -20
echo "Exit code: $?"
echo ""

echo "=== Test 3: MLIR with hybrid instead (uses same HSA alloc) ==="
$BIN --circuit "$STIM" --shots 100 --seed 42 --hybrid 2>&1 | tail -20
echo "Exit code: $?"
echo ""

echo "=== Test 4: Checking compiled .hsaco exists ==="
ls -la $HOME/.clifft/kernel_cache/mlir/*.hsaco 2>/dev/null | tail -3
echo ""

echo "=== Test 5: Checking MLIR/LLVM-IR for array alloca ==="
ls -la /tmp/clifft_mlir_*.mlir 2>/dev/null | tail -1
if ls /tmp/clifft_mlir_*.mlir 2>/dev/null | tail -1 | read f; then
    grep -c "llvm.array" "$f" 2>/dev/null
    grep "llvm.alloca" "$f" | head -5
fi
echo ""

echo "=== Test 6: MLIR IR snippet (allocas) ==="
MLIR_FILE=$(ls -t /tmp/clifft_mlir_*.mlir 2>/dev/null | head -1)
if [ -f "$MLIR_FILE" ]; then
    grep -A0 "llvm.alloca" "$MLIR_FILE" | head -10
fi
echo ""

echo "=== Test 7: LLVM-IR snippet (allocas) ==="
LL_FILE=$(ls -t /tmp/clifft_mlir_*.ll 2>/dev/null | head -1)
if [ -f "$LL_FILE" ]; then
    grep "alloca" "$LL_FILE" | head -10
fi
echo ""

echo "=== Test 8: kernel descriptor (uses_dynamic_stack) ==="
HSACO=$(ls -t $HOME/.clifft/kernel_cache/mlir/*.hsaco 2>/dev/null | head -1)
if [ -f "$HSACO" ]; then
    $LLVM_PREFIX/bin/llvm-readelf -n "$HSACO" 2>/dev/null | head -30
fi
echo ""

echo "Done: $(date)"
