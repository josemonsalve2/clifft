#!/bin/bash
#SBATCH --job-name=diag2
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/diag2_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BIN="$BASE/build-gpu-mlir-mi355x/run_gpu"
STIM="$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

echo "=== Warm cache timing (5 runs each) ==="
echo ""
echo "--- SVM warm cache ---"
$BIN --circuit "$STIM" --shots 100 --seed 42 >/dev/null 2>&1
for i in 1 2 3 4 5; do
    t=$($BIN --circuit "$STIM" --shots 100000 --seed 42 2>&1 | grep sample_seconds | grep -oP '[0-9.]+')
    echo "  Run $i: ${t}s"
done

echo ""
echo "--- MLIR warm cache ---"
$BIN --circuit "$STIM" --shots 100 --seed 42 --mlir >/dev/null 2>&1
for i in 1 2 3 4 5; do
    t=$($BIN --circuit "$STIM" --shots 100000 --seed 42 --mlir 2>&1 | grep sample_seconds | grep -oP '[0-9.]+')
    echo "  Run $i: ${t}s"
done

echo ""
echo "--- SVM 1M shots warm ---"
$BIN --circuit "$STIM" --shots 1000000 --seed 42 2>&1 | grep "sample_seconds\|shots_per_second"

echo ""
echo "--- MLIR 1M shots warm ---"
$BIN --circuit "$STIM" --shots 1000000 --seed 42 --mlir 2>&1 | grep "sample_seconds\|shots_per_second"

echo ""
echo "--- SVM 10M shots ---"
$BIN --circuit "$STIM" --shots 10000000 --seed 42 2>&1 | grep "sample_seconds\|shots_per_second"

echo ""
echo "--- MLIR 10M shots ---"
$BIN --circuit "$STIM" --shots 10000000 --seed 42 --mlir 2>&1 | grep "sample_seconds\|shots_per_second"

echo ""
echo "=== Higher rank circuits ==="
echo "--- rep_d3 (rank 4, more ops) ---"
STIM2="$BASE/tests/fixtures/incremental/06_small_circuits/rep_d3.stim"
$BIN --circuit "$STIM2" --shots 100 --seed 42 --mlir >/dev/null 2>&1
svm_t=$($BIN --circuit "$STIM2" --shots 1000000 --seed 42 2>&1 | grep sample_seconds | grep -oP '[0-9.]+')
mlir_t=$($BIN --circuit "$STIM2" --shots 1000000 --seed 42 --mlir 2>&1 | grep sample_seconds | grep -oP '[0-9.]+')
echo "  SVM: ${svm_t}s  MLIR: ${mlir_t}s"

echo ""
echo "Done: $(date)"
