#!/bin/bash
#SBATCH --job-name=clang-bld
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/clang_build_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BUILD="$BASE/build-gpu-mlir-mi355x"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

echo "GCC version:"
/usr/bin/c++ --version | head -1
echo "Clang version:"
$ROCM_PATH/lib/llvm/bin/clang++ --version | head -1

# Force rebuild mlir_emit.cc with clang
cd "$BUILD"

# Set CXX to clang and rebuild only the changed files
export CXX="$ROCM_PATH/lib/llvm/bin/clang++"
export CC="$ROCM_PATH/lib/llvm/bin/clang"

# Delete the object file to force recompile
rm -f src/clifft/CMakeFiles/clifft_core.dir/gpu/mlir/mlir_emit.cc.o

# Rebuild with clang for C++ files
cmake "$BASE" -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_C_COMPILER="$CC" 2>&1 | tail -5
make -j$(nproc) 2>&1 | tail -10
echo "Build exit: $?"

# Clean kernel cache and test
rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null

echo ""
echo "=== Test: frame_h ==="
svm_p=$($BIN --circuit "$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim" --shots 10000 --seed 42 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
mlir_p=$($BUILD/run_gpu --circuit "$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim" --shots 10000 --seed 42 --mlir 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
echo "SVM=$svm_p MLIR=$mlir_p $([ "$svm_p" = "$mlir_p" ] && echo OK || echo MISMATCH)"

echo "=== Test: qv10 ==="
timeout 120 $BUILD/run_gpu --circuit "$BASE/tests/fixtures/qv10.stim" --shots 10 --seed 42 --mlir 2>/dev/null 1>/dev/null
mlir_p=$($BUILD/run_gpu --circuit "$BASE/tests/fixtures/qv10.stim" --shots 10000 --seed 42 --mlir 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
svm_p=$($BUILD/run_gpu --circuit "$BASE/tests/fixtures/qv10.stim" --shots 10000 --seed 42 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
echo "SVM=$svm_p MLIR=$mlir_p $([ "$svm_p" = "$mlir_p" ] && echo OK || echo MISMATCH)"

echo ""
echo "Done: $(date)"
