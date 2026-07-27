#!/bin/bash
#SBATCH --job-name=perf-rank
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/perf_rank_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BUILD="$BASE/build-gpu-mlir-mi355x"
BIN="$BUILD/run_gpu"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

echo "=== Multi-Rank Performance: SVM vs MLIR (post-fixes) ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

# Build
cd "$BUILD"
make -j$(nproc) 2>&1 | tail -3
echo ""

# Warm up SVM
echo "=== SVM Warmup ==="
$BIN --circuit "$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim" --shots 100 --seed 42 >/dev/null 2>&1

# Pre-compile ALL MLIR kernels (so timing excludes compilation)
echo "=== MLIR Pre-compilation ==="
rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null
for stim in "$BASE"/tests/fixtures/rank_sweep/rank_q17_r{0,2,4}_d1.stim \
            "$BASE"/tests/fixtures/incremental/01_frame_only/frame_h.stim \
            "$BASE"/tests/fixtures/incremental/06_small_circuits/rep_d3.stim; do
    [ -f "$stim" ] || continue
    name=$(basename "$stim" .stim)
    echo -n "  Compiling $name... "
    timeout 180 $BIN --circuit "$stim" --shots 10 --seed 42 --mlir 2>&1 | grep -oP "compiled in [0-9.]+ ms" || echo "FAIL"
done
echo ""

# Benchmark: warm cache, register tier circuits
echo "=== Performance Comparison (100k shots, warm cache) ==="
printf "%-30s %12s %12s %8s\n" "Circuit" "SVM(s)" "MLIR(s)" "Ratio"
printf "%-30s %12s %12s %8s\n" "--------" "------" "-------" "-----"

for stim in "$BASE"/tests/fixtures/incremental/01_frame_only/frame_h.stim \
            "$BASE"/tests/fixtures/incremental/06_small_circuits/rep_d3.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q17_r0_d1.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q17_r2_d1.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q17_r4_d1.stim; do
    [ -f "$stim" ] || continue
    name=$(basename "$stim" .stim)

    svm_t=$(timeout 30 $BIN --circuit "$stim" --shots 100000 --seed 42 2>/dev/null | \
        python3 -c "import json,sys; print(json.load(sys.stdin)['sample_seconds'])" 2>/dev/null)
    mlir_t=$(timeout 30 $BIN --circuit "$stim" --shots 100000 --seed 42 --mlir 2>/dev/null | \
        python3 -c "import json,sys; print(json.load(sys.stdin)['sample_seconds'])" 2>/dev/null)

    if [ -n "$svm_t" ] && [ -n "$mlir_t" ]; then
        ratio=$(python3 -c "s=float('$svm_t');m=float('$mlir_t');print(f'{m/s:.2f}x')" 2>/dev/null)
        printf "%-30s %12s %12s %8s\n" "$name" "${svm_t}s" "${mlir_t}s" "$ratio"
    else
        printf "%-30s %12s %12s %8s\n" "$name" "${svm_t:-FAIL}" "${mlir_t:-FAIL}" "-"
    fi
done

# Shot scaling test: is time constant (host-bound) or scaling (kernel-bound)?
echo ""
echo "=== Shot Scaling: frame_h (is kernel time scaling?) ==="
printf "%-12s %12s %12s\n" "Shots" "SVM(s)" "MLIR(s)"
for shots in 1000 10000 100000 1000000; do
    svm_t=$(timeout 30 $BIN --circuit "$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim" \
        --shots $shots --seed 42 2>/dev/null | \
        python3 -c "import json,sys; print(json.load(sys.stdin)['sample_seconds'])" 2>/dev/null)
    mlir_t=$(timeout 30 $BIN --circuit "$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim" \
        --shots $shots --seed 42 --mlir 2>/dev/null | \
        python3 -c "import json,sys; print(json.load(sys.stdin)['sample_seconds'])" 2>/dev/null)
    printf "%-12s %12s %12s\n" "$shots" "${svm_t:-FAIL}" "${mlir_t:-FAIL}"
done

# Kernel metadata for register tier
echo ""
echo "=== Kernel Metadata ==="
echo "--- MLIR ---"
HSACO=$(ls -t ~/.clifft/kernel_cache/mlir/*.hsaco 2>/dev/null | head -1)
[ -f "$HSACO" ] && $LLVM_PREFIX/bin/llvm-readelf -n "$HSACO" 2>/dev/null | \
    grep -E "\.name|vgpr|sgpr|private|group|flat_work|dynamic|kernarg"

# rocprofv3 on both paths
echo ""
echo "=== rocprofv3 Kernel Trace ==="
PROF_DIR="/tmp/rocprof_perf_$$"
mkdir -p "$PROF_DIR"

echo "--- SVM ---"
rocprofv3 --kernel-trace --stats -d "$PROF_DIR/svm" \
    -- $BIN --circuit "$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim" \
    --shots 100000 --seed 42 2>&1 | tail -1
cat "$PROF_DIR"/svm/*kernel_stats* 2>/dev/null

echo ""
echo "--- MLIR ---"
rocprofv3 --kernel-trace --stats -d "$PROF_DIR/mlir" \
    -- $BIN --circuit "$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim" \
    --shots 100000 --seed 42 --mlir 2>&1 | tail -1
cat "$PROF_DIR"/mlir/*kernel_stats* 2>/dev/null

echo ""
echo "Done: $(date)"
