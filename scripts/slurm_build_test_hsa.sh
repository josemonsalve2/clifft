#!/bin/bash
#SBATCH --job-name=build-hsa
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/build_hsa_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BUILD="$BASE/build-gpu-mlir-mi355x"
BIN="$BUILD/run_gpu"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

echo "=== Building with pure HSA dispatch ==="
cd "$BUILD"
make -j$(nproc) 2>&1 | tail -30
RC=${PIPESTATUS[0]}
if [ $RC -ne 0 ]; then
    echo "BUILD FAILED (exit $RC)"
    exit 1
fi
echo "BUILD OK"
echo ""

# Clear kernel cache to force recompile with new alloca patterns
rm -rf "$HOME/.clifft/kernel_cache/mlir" 2>/dev/null

echo "=== Quick correctness test (register tier, simple circuit) ==="
STIM="$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim"

echo "--- CPU reference ---"
CPU_OUT=$($BASE/build-cpu/run_cpu --circuit "$STIM" --shots 10000 --seed 42 2>/dev/null)
echo "$CPU_OUT" | head -5

echo "--- MLIR GPU ---"
MLIR_OUT=$($BIN --circuit "$STIM" --shots 10000 --seed 42 --mlir 2>&1)
echo "$MLIR_OUT" | head -10

echo ""
echo "=== Correctness sweep: first 20 circuits ==="
PASS=0
FAIL=0
TOTAL=0
for stim in "$BASE"/tests/fixtures/incremental/*/frame_h.stim \
            "$BASE"/tests/fixtures/incremental/*/rep_d3.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q*.stim; do
    [ -f "$stim" ] || continue
    TOTAL=$((TOTAL + 1))
    [ $TOTAL -gt 20 ] && break
    name=$(basename "$(dirname "$stim")")/$(basename "$stim")
    cpu_passed=$($BASE/build-cpu/run_cpu --circuit "$stim" --shots 1000 --seed 42 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
    mlir_passed=$($BIN --circuit "$stim" --shots 1000 --seed 42 --mlir 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
    if [ "$cpu_passed" = "$mlir_passed" ]; then
        echo "  PASS: $name (passed=$cpu_passed)"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $name (CPU=$cpu_passed MLIR=$mlir_passed)"
        FAIL=$((FAIL + 1))
    fi
done
echo ""
echo "=== Results: $PASS/$TOTAL passed, $FAIL failed ==="

echo ""
echo "=== Performance comparison (100k shots, register tier) ==="
STIM="$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim"

# Warm up
$BIN --circuit "$STIM" --shots 100 --seed 42 >/dev/null 2>&1
$BIN --circuit "$STIM" --shots 100 --seed 42 --mlir >/dev/null 2>&1

echo "--- SVM 100k ---"
$BIN --circuit "$STIM" --shots 100000 --seed 42 2>&1 | grep "sample_seconds"
echo "--- MLIR 100k ---"
$BIN --circuit "$STIM" --shots 100000 --seed 42 --mlir 2>&1 | grep "sample_seconds"
echo "--- SVM 1M ---"
$BIN --circuit "$STIM" --shots 1000000 --seed 42 2>&1 | grep "sample_seconds"
echo "--- MLIR 1M ---"
$BIN --circuit "$STIM" --shots 1000000 --seed 42 --mlir 2>&1 | grep "sample_seconds"

echo ""
echo "=== Performance: coop tier (rank 5) ==="
STIM_COOP="$BASE/tests/fixtures/rank_sweep/rank_q17_r5_d1.stim"
$BIN --circuit "$STIM_COOP" --shots 100 --seed 42 --mlir >/dev/null 2>&1
echo "--- SVM 100k ---"
$BIN --circuit "$STIM_COOP" --shots 100000 --seed 42 2>&1 | grep "sample_seconds"
echo "--- MLIR 100k ---"
$BIN --circuit "$STIM_COOP" --shots 100000 --seed 42 --mlir 2>&1 | grep "sample_seconds"

echo ""
echo "Done: $(date)"
