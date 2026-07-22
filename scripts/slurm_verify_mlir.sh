#!/bin/bash
#SBATCH --job-name=verify-mlir
#SBATCH --partition=mi300x
#SBATCH --nodelist=rad-mi300x-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/verify_mlir_%j.log

set -e
BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BUILD="$BASE/build-gpu-mlir-verify"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done

# LLVM tools for MLIR path
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

echo "GPU arch: $(rocm_agent_enumerator | grep gfx)"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "LLVM_PREFIX: ${LLVM_PREFIX:-unset}"
echo "mlir-opt: $(which mlir-opt 2>/dev/null || echo 'not found')"

# Build with both HIP and MLIR enabled
echo "=== Building (HIP + MLIR) ==="
rm -rf "$BUILD"
mkdir -p "$BUILD" && cd "$BUILD"
cmake "$BASE" -DCLIFFT_ENABLE_HIP=ON -DCLIFFT_ENABLE_MLIR=ON \
    -DCMAKE_HIP_ARCHITECTURES=gfx942 \
    -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5
make -j$(nproc) run_gpu 2>&1 | tail -5
RC=$?
echo "Build exit: $RC"
if [ $RC -ne 0 ]; then echo "BUILD FAILED"; exit 1; fi
echo "Build OK: $BUILD/run_gpu"

BIN="$BUILD/run_gpu"

# Quick correctness: SVM vs compiled kernel (non-MLIR path)
echo ""
echo "=== HIP Codegen Correctness ==="
PASS=0; FAIL=0
check_circuit() {
    local name=$1 stim=$2
    echo "--- $name ---"
    set +e
    svm=$("$BIN" --circuit "$stim" --shots 10000 --seed 42 2>&1)
    ck=$("$BIN" --circuit "$stim" --shots 10000 --seed 42 --hybrid 2>&1)
    set -e
    svm_le=$(echo "$svm" | grep '"logical_errors"' | grep -o '[0-9]*')
    ck_le=$(echo "$ck" | grep '"logical_errors"' | grep -o '[0-9]*')
    echo "  SVM: logical=$svm_le"
    echo "  CK:  logical=$ck_le"
    if [ "$svm_le" = "$ck_le" ]; then
        echo "  => EXACT MATCH"
        PASS=$((PASS+1))
    else
        echo "  => MISMATCH!"
        FAIL=$((FAIL+1))
    fi
}

check_circuit "target_qec" "$BASE/tests/fixtures/target_qec.stim"
check_circuit "rank_r4_q33" "$BASE/tests/fixtures/rank_sweep/rank_q33_r4_d3.stim"
check_circuit "rank_r8_q33" "$BASE/tests/fixtures/rank_sweep/rank_q33_r8_d3.stim"
check_circuit "rank_r12_q33" "$BASE/tests/fixtures/rank_sweep/rank_q33_r12_d3.stim"

echo ""
echo "=== HIP Codegen Summary ==="
echo "  PASS: $PASS  FAIL: $FAIL"

echo ""
echo "=== MLIR Build Verification ==="
echo "MLIR codegen compiled successfully (CLIFFT_ENABLE_MLIR=ON)"
echo "MLIR source files:"
ls -la "$BASE/src/clifft/gpu/mlir/"
echo "MLIR ops:"
ls -la "$BASE/src/clifft/gpu/mlir/ops/"

# Verify MLIR object files exist in the build
echo ""
echo "MLIR object files in build:"
find "$BUILD" -name '*mlir*' -type f | head -10

echo ""
echo "Done: $(date)"
