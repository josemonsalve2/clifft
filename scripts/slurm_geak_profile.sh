#!/bin/bash
#SBATCH --job-name=geak-prof
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/geak_prof_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BUILD="$BASE/build-gpu-mlir-mi355x"
BIN="$BUILD/run_gpu"
GEAK_HOME="/shared/jmonsalv/quantum/GEAK"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

echo "=== GEAK-Style Performance Profiling ==="
echo "Node: $(hostname)"
echo "ROCm: $ROCM_PATH"
echo "GPU: $(rocminfo 2>/dev/null | grep -m1 'Marketing Name' || echo 'unknown')"
echo ""

STIM="$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim"

# Step 1: Build verification
echo "=== Step 1: Build ==="
cd "$BUILD"
make -j$(nproc) 2>&1 | tail -3
echo ""

# Step 2: Warm up all paths
echo "=== Step 2: Warmup ==="
rm -rf /tmp/clifft_mlir_* 2>/dev/null
$BIN --circuit "$STIM" --shots 100 --seed 42 >/dev/null 2>&1
timeout 120 $BIN --circuit "$STIM" --shots 100 --seed 42 --mlir 2>&1 | grep -E "compiled|loaded|bound|FATAL|error"
timeout 120 $BIN --circuit "$STIM" --shots 100 --seed 42 --hybrid 2>&1 | grep -E "compiled|loaded|bound|FATAL|error"
echo ""

# Step 3: GEAK profile_kernel.sh for SVM
echo "=== Step 3: GEAK profiling — SVM ==="
PROF_DIR="$BASE/results/geak_svm_$$"
mkdir -p "$PROF_DIR"
if [ -f "$GEAK_HOME/kernel_workflow/scripts/profile_kernel.sh" ]; then
    export PROFILER_PRIORITY="rocprofv3 rocprof"
    export WARMUP_RUNS=2
    bash "$GEAK_HOME/kernel_workflow/scripts/profile_kernel.sh" 0 \
        "$BIN --circuit $STIM --shots 100000 --seed 42" \
        "$PROF_DIR" 2>&1 | tail -20
    echo ""
    echo "--- GEAK SVM profile report ---"
    cat "$PROF_DIR/profile_report.txt" 2>/dev/null | head -80
else
    echo "GEAK profile_kernel.sh not found, using rocprofv3 directly"

    echo "--- rocprofv3 SVM kernel trace ---"
    mkdir -p "$PROF_DIR"
    rocprofv3 --kernel-trace --stats -d "$PROF_DIR/svm_trace" \
        -- $BIN --circuit "$STIM" --shots 100000 --seed 42 2>&1 | tail -5

    echo ""
    echo "--- SVM kernel trace results ---"
    for f in "$PROF_DIR"/svm_trace/*.csv; do
        [ -f "$f" ] || continue
        echo "File: $(basename $f)"
        cat "$f"
        echo ""
    done
fi
echo ""

# Step 4: GEAK profiling — MLIR
echo "=== Step 4: GEAK profiling — MLIR ==="
PROF_DIR_MLIR="$BASE/results/geak_mlir_$$"
mkdir -p "$PROF_DIR_MLIR"
if [ -f "$GEAK_HOME/kernel_workflow/scripts/profile_kernel.sh" ]; then
    bash "$GEAK_HOME/kernel_workflow/scripts/profile_kernel.sh" 0 \
        "$BIN --circuit $STIM --shots 100000 --seed 42 --mlir" \
        "$PROF_DIR_MLIR" 2>&1 | tail -20
    echo ""
    echo "--- GEAK MLIR profile report ---"
    cat "$PROF_DIR_MLIR/profile_report.txt" 2>/dev/null | head -80
else
    echo "--- rocprofv3 MLIR kernel trace ---"
    rocprofv3 --kernel-trace --stats -d "$PROF_DIR_MLIR/mlir_trace" \
        -- $BIN --circuit "$STIM" --shots 100000 --seed 42 --mlir 2>&1 | tail -5

    echo ""
    echo "--- MLIR kernel trace results ---"
    for f in "$PROF_DIR_MLIR"/mlir_trace/*.csv; do
        [ -f "$f" ] || continue
        echo "File: $(basename $f)"
        cat "$f"
        echo ""
    done
fi
echo ""

# Step 5: GEAK profiling — Hybrid
echo "=== Step 5: GEAK profiling — Hybrid ==="
PROF_DIR_HYB="$BASE/results/geak_hybrid_$$"
mkdir -p "$PROF_DIR_HYB"
if [ -f "$GEAK_HOME/kernel_workflow/scripts/profile_kernel.sh" ]; then
    bash "$GEAK_HOME/kernel_workflow/scripts/profile_kernel.sh" 0 \
        "$BIN --circuit $STIM --shots 100000 --seed 42 --hybrid" \
        "$PROF_DIR_HYB" 2>&1 | tail -20
    echo ""
    echo "--- GEAK Hybrid profile report ---"
    cat "$PROF_DIR_HYB/profile_report.txt" 2>/dev/null | head -80
else
    echo "--- rocprofv3 Hybrid kernel trace ---"
    rocprofv3 --kernel-trace --stats -d "$PROF_DIR_HYB/hybrid_trace" \
        -- $BIN --circuit "$STIM" --shots 100000 --seed 42 --hybrid 2>&1 | tail -5

    echo ""
    echo "--- Hybrid kernel trace results ---"
    for f in "$PROF_DIR_HYB"/hybrid_trace/*.csv; do
        [ -f "$f" ] || continue
        echo "File: $(basename $f)"
        cat "$f"
        echo ""
    done
fi
echo ""

# Step 6: Static ISA analysis of compiled .hsaco files
echo "=== Step 6: ISA analysis ==="
echo "--- MLIR kernel descriptor ---"
MLIR_HSACO=$(ls -t ~/.clifft/kernel_cache/mlir/*.hsaco 2>/dev/null | head -1)
if [ -f "$MLIR_HSACO" ]; then
    echo "File: $MLIR_HSACO ($(stat -c%s "$MLIR_HSACO") bytes)"
    $LLVM_PREFIX/bin/llvm-readelf -n "$MLIR_HSACO" 2>/dev/null | grep -E "\.name|vgpr|sgpr|private|group|flat_work|dynamic|kernarg|wavefront"
    echo ""
    echo "--- MLIR ISA instruction count ---"
    $LLVM_PREFIX/bin/llvm-objdump -d "$MLIR_HSACO" 2>/dev/null | wc -l
    echo ""
    echo "--- MLIR ISA instruction mix ---"
    $LLVM_PREFIX/bin/llvm-objdump -d "$MLIR_HSACO" 2>/dev/null | awk '{print $NF}' | sort | uniq -c | sort -rn | head -30
fi

echo ""
echo "--- Hybrid kernel descriptor ---"
HYB_HSACO=$(ls -t ~/.clifft/kernel_cache/*_gfx950.hsaco 2>/dev/null | grep -v mlir | head -1)
if [ -f "$HYB_HSACO" ]; then
    echo "File: $HYB_HSACO ($(stat -c%s "$HYB_HSACO") bytes)"
    $LLVM_PREFIX/bin/llvm-readelf -n "$HYB_HSACO" 2>/dev/null | grep -E "\.name|vgpr|sgpr|private|group|flat_work|dynamic|kernarg|wavefront"
    echo ""
    echo "--- Hybrid ISA instruction count ---"
    $LLVM_PREFIX/bin/llvm-objdump -d "$HYB_HSACO" 2>/dev/null | wc -l
    echo ""
    echo "--- Hybrid ISA instruction mix ---"
    $LLVM_PREFIX/bin/llvm-objdump -d "$HYB_HSACO" 2>/dev/null | awk '{print $NF}' | sort | uniq -c | sort -rn | head -30
fi

# Step 7: Performance comparison
echo ""
echo "=== Step 7: Performance Comparison ==="
for shots in 10000 100000 1000000; do
    svm_t=$(timeout 30 $BIN --circuit "$STIM" --shots $shots --seed 42 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['sample_seconds'])" 2>/dev/null)
    mlir_t=$(timeout 30 $BIN --circuit "$STIM" --shots $shots --seed 42 --mlir 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['sample_seconds'])" 2>/dev/null)
    hyb_t=$(timeout 30 $BIN --circuit "$STIM" --shots $shots --seed 42 --hybrid 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['sample_seconds'])" 2>/dev/null)
    echo "$shots shots: SVM=${svm_t}s MLIR=${mlir_t}s Hybrid=${hyb_t}s"
done

# Step 8: Host vs kernel time isolation
echo ""
echo "=== Step 8: Host vs Kernel Time ==="
echo "If sample_seconds is constant across 10k-10M shots, host overhead dominates."
echo "If sample_seconds scales linearly, kernel compute dominates."
for shots in 1000 10000 100000 1000000 10000000; do
    t=$(timeout 60 $BIN --circuit "$STIM" --shots $shots --seed 42 --mlir 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['sample_seconds'])" 2>/dev/null)
    echo "  MLIR $shots shots: ${t}s"
done

echo ""
echo "Done: $(date)"
