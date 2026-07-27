#!/bin/bash
#SBATCH --job-name=u2-fix
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/u2_fix_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BUILD="$BASE/build-gpu-mlir-mi355x"
BIN="$BUILD/run_gpu"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

echo "=== BUILD + TEST: U2/U4 FIX ==="
echo "Node: $(hostname)"
echo "Date: $(date)"

# Build
cd "$BUILD"
make -j$(nproc) 2>&1 | tail -5
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "BUILD FAILED (parallel), retrying single-threaded..."
    make -j1 2>&1 | grep -E "error:|warning:" | head -30
    exit 1
fi
echo "BUILD OK"
echo ""

# Clear MLIR cache
rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null

# Test 1: frame_h (rank 0, should still work)
echo "=== Test: frame_h (rank 0) ==="
svm_out=$($BIN --circuit "$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim" --shots 10000 --seed 42 2>/dev/null)
svm_p=$(echo "$svm_out" | python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
mlir_out=$($BIN --circuit "$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim" --shots 10000 --seed 42 --mlir 2>/dev/null)
mlir_p=$(echo "$mlir_out" | python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
echo "SVM=$svm_p MLIR=$mlir_p $([ "$svm_p" = "$mlir_p" ] && echo OK || echo MISMATCH)"

# Test 2: qv10 (rank 10, uses U2/U4 — the bug)
echo "=== Test: qv10 (rank 10, uses U2/U4) ==="
svm_out=$($BIN --circuit "$BASE/tests/fixtures/qv10.stim" --shots 10000 --seed 42 2>/dev/null)
svm_p=$(echo "$svm_out" | python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
echo "SVM passed_shots=$svm_p"
# MLIR: first run compiles
echo "MLIR compile..."
timeout 120 $BIN --circuit "$BASE/tests/fixtures/qv10.stim" --shots 10 --seed 42 --mlir 2>&1 | grep -E "compiled|error|panic" | head -3
echo "MLIR run..."
mlir_out=$(timeout 60 $BIN --circuit "$BASE/tests/fixtures/qv10.stim" --shots 10000 --seed 42 --mlir 2>&1)
mlir_p=$(echo "$mlir_out" | python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
mlir_err=$(echo "$mlir_out" | grep -E "clifft-timing|error|abort" | head -5)
echo "MLIR passed_shots=$mlir_p"
echo "$mlir_err"
echo "$([ "$svm_p" = "$mlir_p" ] && echo CORRECTNESS_OK || echo CORRECTNESS_MISMATCH)"

# Test 3: circuit_d3_p0.001 (rank 4, detectors+postselection)
echo "=== Test: circuit_d3 (rank 4, postselection) ==="
svm_out=$($BIN --circuit "$BASE/tests/fixtures/large/circuit_d3_p0.001.stim" --shots 10000 --seed 42 2>/dev/null)
svm_p=$(echo "$svm_out" | python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
echo "SVM passed_shots=$svm_p"
timeout 300 $BIN --circuit "$BASE/tests/fixtures/large/circuit_d3_p0.001.stim" --shots 10 --seed 42 --mlir 2>&1 | grep -E "compiled|error" | head -3
mlir_out=$(timeout 60 $BIN --circuit "$BASE/tests/fixtures/large/circuit_d3_p0.001.stim" --shots 10000 --seed 42 --mlir 2>&1)
mlir_p=$(echo "$mlir_out" | python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
echo "MLIR passed_shots=$mlir_p"
echo "$([ "$svm_p" = "$mlir_p" ] && echo CORRECTNESS_OK || echo CORRECTNESS_MISMATCH)"

# Test 4: four_t (rank 0, should still work)
echo "=== Test: four_t (rank 0) ==="
svm_out=$($BIN --circuit "$BASE/tests/fixtures/incremental/02_single_expand/four_t.stim" --shots 10000 --seed 42 2>/dev/null)
svm_p=$(echo "$svm_out" | python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
mlir_out=$($BIN --circuit "$BASE/tests/fixtures/incremental/02_single_expand/four_t.stim" --shots 10000 --seed 42 --mlir 2>/dev/null)
mlir_p=$(echo "$mlir_out" | python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
echo "SVM=$svm_p MLIR=$mlir_p $([ "$svm_p" = "$mlir_p" ] && echo OK || echo MISMATCH)"

# Test 5: surface_d7_t5 (rank 3, large, no U2/U4)
echo "=== Test: surface_d7_t5 (rank 3) ==="
svm_out=$($BIN --circuit "$BASE/tests/fixtures/large/surface_d7_t5.stim" --shots 10000 --seed 42 2>/dev/null)
svm_p=$(echo "$svm_out" | python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
echo "SVM passed_shots=$svm_p"
timeout 300 $BIN --circuit "$BASE/tests/fixtures/large/surface_d7_t5.stim" --shots 10 --seed 42 --mlir 2>&1 | grep -E "compiled|error" | head -3
mlir_out=$(timeout 60 $BIN --circuit "$BASE/tests/fixtures/large/surface_d7_t5.stim" --shots 10000 --seed 42 --mlir 2>&1)
mlir_p=$(echo "$mlir_out" | python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
echo "MLIR passed_shots=$mlir_p"
echo "$([ "$svm_p" = "$mlir_p" ] && echo CORRECTNESS_OK || echo CORRECTNESS_MISMATCH)"

echo ""
echo "Done: $(date)"
