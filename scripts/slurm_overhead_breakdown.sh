#!/bin/bash
#SBATCH --job-name=overhead
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/overhead_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BUILD="$BASE/build-gpu-mlir-mi355x"
BIN="$BUILD/run_gpu"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

# Use miniforge for rocprof-compute deps
export PATH="/home/jmonsalv/miniforge3/bin:$PATH"
export PYTHONPATH="/home/jmonsalv/miniforge3/lib/python3.12/site-packages:${PYTHONPATH:-}"

echo "=== Host Overhead Breakdown ==="
echo "Node: $(hostname)"
echo ""

# Build
cd "$BUILD"
make -j$(nproc) 2>&1 | tail -3
echo ""

STIM="$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim"

# Warm caches
$BIN --circuit "$STIM" --shots 100 --seed 42 >/dev/null 2>&1
timeout 120 $BIN --circuit "$STIM" --shots 100 --seed 42 --mlir >/dev/null 2>&1

echo "=== SVM 100k shots (timing breakdown) ==="
$BIN --circuit "$STIM" --shots 100000 --seed 42 2>&1 | grep -E "clifft-timing|sample_seconds"
echo ""

echo "=== MLIR 100k shots (timing breakdown) ==="
$BIN --circuit "$STIM" --shots 100000 --seed 42 --mlir 2>&1 | grep -E "clifft-timing|sample_seconds"
echo ""

echo "=== SVM 1M shots ==="
$BIN --circuit "$STIM" --shots 1000000 --seed 42 2>&1 | grep -E "clifft-timing|sample_seconds"
echo ""

echo "=== MLIR 1M shots ==="
$BIN --circuit "$STIM" --shots 1000000 --seed 42 --mlir 2>&1 | grep -E "clifft-timing|sample_seconds"
echo ""

echo "=== Hybrid 100k shots ==="
$BIN --circuit "$STIM" --shots 100000 --seed 42 --hybrid 2>&1 | grep -E "clifft-timing|sample_seconds"
echo ""

# Try GEAK profile_kernel.sh with miniforge python for rocprofv3
echo "=== GEAK profiling (with rocprofv3) ==="
GEAK_HOME="/shared/jmonsalv/quantum/GEAK"
if [ -f "$GEAK_HOME/kernel_workflow/scripts/profile_kernel.sh" ]; then
    export PROFILER_PRIORITY="rocprofv3 rocprof"
    export WARMUP_RUNS=1
    PROF_DIR="$BASE/results/geak_overhead_$$"

    echo "--- GEAK: SVM ---"
    bash "$GEAK_HOME/kernel_workflow/scripts/profile_kernel.sh" 0 \
        "$BIN --circuit $STIM --shots 100000 --seed 42" \
        "$PROF_DIR/svm" 2>&1 | tail -5
    echo "--- SVM profile report ---"
    head -30 "$PROF_DIR/svm/profile_report.txt" 2>/dev/null
    echo ""
    echo "--- SVM kernel stats ---"
    cat "$PROF_DIR"/svm/rocprofv3/*kernel_stats* 2>/dev/null
    echo ""

    echo "--- GEAK: MLIR ---"
    bash "$GEAK_HOME/kernel_workflow/scripts/profile_kernel.sh" 0 \
        "$BIN --circuit $STIM --shots 100000 --seed 42 --mlir" \
        "$PROF_DIR/mlir" 2>&1 | tail -5
    echo "--- MLIR profile report ---"
    head -30 "$PROF_DIR/mlir/profile_report.txt" 2>/dev/null
    echo ""
    echo "--- MLIR kernel stats ---"
    cat "$PROF_DIR"/mlir/rocprofv3/*kernel_stats* 2>/dev/null
    echo ""
fi

# Try rocprof-compute with miniforge python
echo "=== rocprof-compute check ==="
PYTHONPATH="/home/jmonsalv/miniforge3/lib/python3.12/site-packages" \
    rocprof-compute --version 2>&1 | head -3
echo ""

echo "Done: $(date)"
