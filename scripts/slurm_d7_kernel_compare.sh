#!/bin/bash
#SBATCH --job-name=d7-cmp
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/d7_compare_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BIN="$BASE/build-gpu-mlir-mi355x/run_gpu"
STIM="$BASE/tests/fixtures/large/surface_d7_r7.stim"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

echo "=== surface_d7_r7 Kernel Time Comparison ==="
echo "97 qubits, 239 lines"
echo ""

# Warmup
$BIN --circuit "$STIM" --shots 100 --seed 42 >/dev/null 2>&1
timeout 300 $BIN --circuit "$STIM" --shots 100 --seed 42 --mlir >/dev/null 2>&1

# Timing breakdown — SVM
echo "=== SVM timing ==="
$BIN --circuit "$STIM" --shots 10000 --seed 42 2>&1 | grep -E "clifft-timing|sample_seconds"
echo ""

# Timing breakdown — MLIR
echo "=== MLIR timing ==="
$BIN --circuit "$STIM" --shots 10000 --seed 42 --mlir 2>&1 | grep -E "clifft-timing|sample_seconds"
echo ""

# Timing breakdown — Hybrid
echo "=== Hybrid timing ==="
$BIN --circuit "$STIM" --shots 10000 --seed 42 --hybrid 2>&1 | grep -E "clifft-timing|sample_seconds"
echo ""

# rocprofv3 kernel trace — SVM
echo "=== rocprofv3: SVM ==="
PROF="/tmp/d7_prof_$$"
mkdir -p "$PROF"
rocprofv3 --kernel-trace --stats -d "$PROF/svm" \
    -- $BIN --circuit "$STIM" --shots 10000 --seed 42 2>&1 | tail -3
cat "$PROF"/svm/*/*kernel_stats* 2>/dev/null
echo ""

# rocprofv3 kernel trace — MLIR
echo "=== rocprofv3: MLIR ==="
rocprofv3 --kernel-trace --stats -d "$PROF/mlir" \
    -- $BIN --circuit "$STIM" --shots 10000 --seed 42 --mlir 2>&1 | tail -3
cat "$PROF"/mlir/*/*kernel_stats* 2>/dev/null
echo ""

# rocprofv3 kernel trace — Hybrid
echo "=== rocprofv3: Hybrid ==="
rocprofv3 --kernel-trace --stats -d "$PROF/hybrid" \
    -- $BIN --circuit "$STIM" --shots 10000 --seed 42 --hybrid 2>&1 | tail -3
cat "$PROF"/hybrid/*/*kernel_stats* 2>/dev/null
echo ""

echo "Done: $(date)"
