#!/bin/bash
#SBATCH --job-name=geak-300
#SBATCH --partition=mi300x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/geak_prof_mi300x_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
GEAK_HOME="/shared/jmonsalv/quantum/GEAK"

for ROCM in /opt/rocm-6.4.0 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

echo "=== MI300X GEAK Profiling ==="
echo "Node: $(hostname)"
echo "ROCm: $ROCM_PATH"
echo ""

# Build for MI300X
BUILD="$BASE/build-gpu-hsa"
BIN="$BUILD/run_gpu"
cd "$BUILD"
make -j$(nproc) 2>&1 | tail -3
echo ""

STIM="$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim"

# Warm up SVM + Hybrid (MLIR may fail on this build)
$BIN --circuit "$STIM" --shots 100 --seed 42 >/dev/null 2>&1
timeout 120 $BIN --circuit "$STIM" --shots 100 --seed 42 --hybrid 2>&1 | grep -E "compiled|loaded|bound"
echo ""

# Profile SVM with GEAK
echo "=== GEAK Profile: SVM ==="
PROF_SVM="$BASE/results/geak_svm_mi300x_$$"
export PROFILER_PRIORITY="rocprofv3 rocprof"
export WARMUP_RUNS=2
if [ -f "$GEAK_HOME/kernel_workflow/scripts/profile_kernel.sh" ]; then
    bash "$GEAK_HOME/kernel_workflow/scripts/profile_kernel.sh" 0 \
        "$BIN --circuit $STIM --shots 100000 --seed 42" \
        "$PROF_SVM" 2>&1
    echo ""
    echo "--- SVM profile report (first 100 lines) ---"
    head -100 "$PROF_SVM/profile_report.txt" 2>/dev/null
fi
echo ""

# Profile Hybrid with GEAK
echo "=== GEAK Profile: Hybrid ==="
PROF_HYB="$BASE/results/geak_hybrid_mi300x_$$"
bash "$GEAK_HOME/kernel_workflow/scripts/profile_kernel.sh" 0 \
    "$BIN --circuit $STIM --shots 100000 --seed 42 --hybrid" \
    "$PROF_HYB" 2>&1
echo ""
echo "--- Hybrid profile report (first 100 lines) ---"
head -100 "$PROF_HYB/profile_report.txt" 2>/dev/null
echo ""

# ISA comparison
echo "=== ISA Comparison ==="
echo "--- SVM kernel ---"
SVM_HSACO=$(ls -t ~/.clifft/kernel_cache/*_gfx942.hsaco 2>/dev/null | grep -v mlir | head -1)
if [ -f "$SVM_HSACO" ]; then
    echo "File: $SVM_HSACO ($(stat -c%s "$SVM_HSACO") bytes)"
    $LLVM_PREFIX/bin/llvm-readelf -n "$SVM_HSACO" 2>/dev/null | grep -E "\.name|vgpr|sgpr|private|group|flat_work|dynamic|kernarg|wavefront"
    echo "Instructions: $($LLVM_PREFIX/bin/llvm-objdump -d "$SVM_HSACO" 2>/dev/null | grep -c '^\s')"
    echo "Top instructions:"
    $LLVM_PREFIX/bin/llvm-objdump -d "$SVM_HSACO" 2>/dev/null | awk '{for(i=3;i<=NF;i++) print $i}' | grep -E '^[vs]_' | sort | uniq -c | sort -rn | head -20
fi

echo ""
echo "--- Hybrid kernel ---"
HYB_HSACO=$(ls -t ~/.clifft/kernel_cache/*_gfx942.hsaco 2>/dev/null | grep -v mlir | tail -1)
if [ -f "$HYB_HSACO" ] && [ "$HYB_HSACO" != "$SVM_HSACO" ]; then
    echo "File: $HYB_HSACO ($(stat -c%s "$HYB_HSACO") bytes)"
    $LLVM_PREFIX/bin/llvm-readelf -n "$HYB_HSACO" 2>/dev/null | grep -E "\.name|vgpr|sgpr|private|group|flat_work|dynamic|kernarg|wavefront"
fi

# Performance comparison
echo ""
echo "=== Performance ==="
for shots in 10000 100000 1000000; do
    svm_t=$(timeout 30 $BIN --circuit "$STIM" --shots $shots --seed 42 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['sample_seconds'])" 2>/dev/null)
    hyb_t=$(timeout 30 $BIN --circuit "$STIM" --shots $shots --seed 42 --hybrid 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['sample_seconds'])" 2>/dev/null)
    echo "$shots shots: SVM=${svm_t}s Hybrid=${hyb_t}s"
done

echo ""
echo "Done: $(date)"
