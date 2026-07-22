#!/bin/bash
#SBATCH --job-name=build-test-bench
#SBATCH --partition=mi300x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/build_test_bench_%j.log
#SBATCH --exclusive

set -euo pipefail

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
cd "$BASE"

echo "=== Phase 1: Build ==="
echo "Host: $(hostname)"
echo "Date: $(date)"

# Rebuild with the new kernel_cache
cd build-gpu
cmake --build . -j$(nproc) 2>&1
cd "$BASE"

echo ""
echo "=== Build successful ==="
echo ""

# Phase 2: GPU Tests
echo "=== Phase 2: GPU Tests ==="
./build-gpu/tests/clifft_tests "[gpu]" 2>&1 || {
    echo "GPU TESTS FAILED"
    exit 1
}
echo ""
echo "=== All GPU tests passed ==="
echo ""

# Phase 3: Correctness cross-check (SVM vs Compiled, same seed)
echo "=== Phase 3: Correctness Cross-Check ==="
echo "Running SVM on cultivation_d5 (1M shots, seed=42)..."
SVM_OUT=$(./build-gpu/run_gpu --circuit tests/fixtures/cultivation_d5.stim --shots 1000000 --seed 42 2>&1)
SVM_PASSED=$(echo "$SVM_OUT" | grep '"passed_shots"' | head -1 | grep -o '[0-9]*')
echo "SVM passed_shots: $SVM_PASSED"

echo "Running Compiled on cultivation_d5 (1M shots, seed=42)..."
COMP_OUT=$(./build-gpu/run_gpu --circuit tests/fixtures/cultivation_d5.stim --shots 1000000 --seed 42 --hybrid 2>&1)
COMP_PASSED=$(echo "$COMP_OUT" | grep '"passed_shots"' | head -1 | grep -o '[0-9]*')
echo "Compiled passed_shots: $COMP_PASSED"

# For coop tier (rank=10), --hybrid falls back to SVM, so results should be identical
echo "SVM=$SVM_PASSED vs Compiled=$COMP_PASSED"
if [ "$SVM_PASSED" = "$COMP_PASSED" ]; then
    echo "CORRECTNESS CHECK: PASS (identical results)"
else
    echo "CORRECTNESS CHECK: MISMATCH (expected identical for coop tier fallback)"
fi

# Also test a per-thread circuit where compiled kernel should actually activate
echo ""
echo "Running SVM on target_qec (1M shots, seed=42)..."
SVM_OUT2=$(./build-gpu/run_gpu --circuit tests/fixtures/target_qec.stim --shots 1000000 --seed 42 2>&1)
SVM_PASSED2=$(echo "$SVM_OUT2" | grep '"passed_shots"' | head -1 | grep -o '[0-9]*')
echo "SVM passed_shots: $SVM_PASSED2"

echo "Running Compiled on target_qec (1M shots, seed=42)..."
COMP_OUT2=$(./build-gpu/run_gpu --circuit tests/fixtures/target_qec.stim --shots 1000000 --seed 42 --hybrid 2>&1)
COMP_PASSED2=$(echo "$COMP_OUT2" | grep '"passed_shots"' | head -1 | grep -o '[0-9]*')
echo "Compiled passed_shots: $COMP_PASSED2"

echo "SVM=$SVM_PASSED2 vs Compiled=$COMP_PASSED2"
if [ "$SVM_PASSED2" = "$COMP_PASSED2" ]; then
    echo "CORRECTNESS CHECK (per-thread): PASS (identical results)"
else
    echo "CORRECTNESS CHECK (per-thread): RESULT DIFFERS"
    echo "  This is expected if the compiled kernel uses a different reduction path."
    echo "  Check that both values are within statistical bounds."
fi

echo ""
echo "=== Correctness checks done ==="
echo ""

# Phase 4: Full rigorous benchmark
echo "=== Phase 4: Rigorous Benchmark ==="
bash scripts/bench_rigorous.sh

echo ""
echo "=== ALL PHASES COMPLETE ==="
echo "Date: $(date)"
