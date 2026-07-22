#!/bin/bash
#SBATCH --job-name=test-codegen
#SBATCH --partition=mi300x
#SBATCH --nodelist=rad-mi300x-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/test_codegen_%j.log

# Test that the AMDGCN-native codegen compiles and runs correctly
set +e
BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done

echo "=== Test AMDGCN-native codegen ==="
echo "Node: $(hostname)  Date: $(date)"

# Rebuild with the fixed codegen
BUILD="$BASE/build-gpu-codegen-test"
rm -rf "$BUILD"
mkdir -p "$BUILD" && cd "$BUILD"
cmake "$BASE" -DCLIFFT_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 \
    -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
make -j$(nproc) run_gpu 2>&1 | tail -5
RC=$?
echo "Build exit: $RC"
[ $RC -ne 0 ] && { echo "BUILD FAILED"; exit 1; }

BINARY="$BUILD/run_gpu"
echo "Binary: $BINARY"

# Clear kernel cache to force recompilation
rm -rf ~/.clifft/kernel_cache/*.hsaco

echo ""
echo "=== Test 1: SVM baseline (should always work) ==="
$BINARY --circuit "$BASE/tests/fixtures/target_qec.stim" --shots 1000 2>&1 | head -5

echo ""
echo "=== Test 2: Compiled kernel (--hybrid, should now compile) ==="
$BINARY --circuit "$BASE/tests/fixtures/target_qec.stim" --shots 1000 --hybrid 2>&1

echo ""
echo "=== Test 3: Check if .hsaco was created ==="
ls -la ~/.clifft/kernel_cache/*.hsaco 2>/dev/null || echo "No .hsaco found (compilation still failing)"

echo ""
echo "=== Test 4: Quick benchmark comparison (if both work) ==="
echo "SVM (5 runs):"
for i in 1 2 3 4 5; do
    OUT=$($BINARY --circuit "$BASE/tests/fixtures/target_qec.stim" --shots 1000000 2>&1)
    SPS=$(echo "$OUT" | grep "shots_per_second_sampling_only" | grep -oE '[0-9]+\.?[0-9eE+\-]*')
    echo "  run $i: $SPS shots/s"
done

echo ""
echo "Compiled (5 runs):"
for i in 1 2 3 4 5; do
    OUT=$($BINARY --circuit "$BASE/tests/fixtures/target_qec.stim" --shots 1000000 --hybrid 2>&1)
    SPS=$(echo "$OUT" | grep "shots_per_second_sampling_only" | grep -oE '[0-9]+\.?[0-9eE+\-]*')
    echo "  run $i: $SPS shots/s"
done

echo ""
echo "Done: $(date)"
