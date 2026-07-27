#!/bin/bash
#SBATCH --job-name=perf-hsa
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/perf_hsa_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BUILD="$BASE/build-gpu-mlir-mi355x"
BIN="$BUILD/run_gpu"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

echo "=== MI350X Performance: SVM vs MLIR vs Hybrid ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

STIM_REG="$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim"
STIM_COOP="$BASE/tests/fixtures/rank_sweep/rank_q17_r5_d1.stim"
STIM_REP="$BASE/tests/fixtures/incremental/06_small_circuits/rep_d3.stim"

# Warm up all paths
$BIN --circuit "$STIM_REG" --shots 100 --seed 42 >/dev/null 2>&1
$BIN --circuit "$STIM_REG" --shots 100 --seed 42 --mlir >/dev/null 2>&1
$BIN --circuit "$STIM_REG" --shots 100 --seed 42 --hybrid >/dev/null 2>&1

echo "=== Register tier: frame_h (rank 0, 4 ops) ==="
for shots in 10000 100000 1000000; do
    echo "  --- ${shots} shots ---"
    svm=$($BIN --circuit "$STIM_REG" --shots $shots --seed 42 2>&1)
    svm_t=$(echo "$svm" | python3 -c "import json,sys; print(json.load(sys.stdin)['sample_seconds'])" 2>/dev/null)
    mlir=$($BIN --circuit "$STIM_REG" --shots $shots --seed 42 --mlir 2>&1)
    mlir_t=$(echo "$mlir" | python3 -c "import json,sys; print(json.load(sys.stdin)['sample_seconds'])" 2>/dev/null)
    hyb=$($BIN --circuit "$STIM_REG" --shots $shots --seed 42 --hybrid 2>&1)
    hyb_t=$(echo "$hyb" | python3 -c "import json,sys; print(json.load(sys.stdin)['sample_seconds'])" 2>/dev/null)
    echo "    SVM:    ${svm_t}s"
    echo "    MLIR:   ${mlir_t}s"
    echo "    Hybrid: ${hyb_t}s"
    if [ -n "$svm_t" ] && [ -n "$mlir_t" ]; then
        ratio=$(python3 -c "print(f'{float(\"$mlir_t\")/float(\"$svm_t\"):.2f}x')" 2>/dev/null)
        echo "    MLIR/SVM ratio: $ratio"
    fi
done

echo ""
echo "=== Register tier: rep_d3 (rank 4, more ops) ==="
$BIN --circuit "$STIM_REP" --shots 100 --seed 42 --mlir >/dev/null 2>&1
for shots in 10000 100000; do
    echo "  --- ${shots} shots ---"
    svm=$($BIN --circuit "$STIM_REP" --shots $shots --seed 42 2>&1)
    svm_t=$(echo "$svm" | python3 -c "import json,sys; print(json.load(sys.stdin)['sample_seconds'])" 2>/dev/null)
    mlir=$($BIN --circuit "$STIM_REP" --shots $shots --seed 42 --mlir 2>&1)
    mlir_t=$(echo "$mlir" | python3 -c "import json,sys; print(json.load(sys.stdin)['sample_seconds'])" 2>/dev/null)
    echo "    SVM:  ${svm_t}s"
    echo "    MLIR: ${mlir_t}s"
    if [ -n "$svm_t" ] && [ -n "$mlir_t" ]; then
        ratio=$(python3 -c "print(f'{float(\"$mlir_t\")/float(\"$svm_t\"):.2f}x')" 2>/dev/null)
        echo "    MLIR/SVM ratio: $ratio"
    fi
done

echo ""
echo "=== Coop tier: rank_q17_r5_d1 (rank 5) ==="
$BIN --circuit "$STIM_COOP" --shots 100 --seed 42 --mlir >/dev/null 2>&1
for shots in 10000 100000; do
    echo "  --- ${shots} shots ---"
    svm=$($BIN --circuit "$STIM_COOP" --shots $shots --seed 42 2>&1)
    svm_t=$(echo "$svm" | python3 -c "import json,sys; print(json.load(sys.stdin)['sample_seconds'])" 2>/dev/null)
    mlir=$($BIN --circuit "$STIM_COOP" --shots $shots --seed 42 --mlir 2>&1)
    mlir_t=$(echo "$mlir" | python3 -c "import json,sys; print(json.load(sys.stdin)['sample_seconds'])" 2>/dev/null)
    echo "    SVM:  ${svm_t}s"
    echo "    MLIR: ${mlir_t}s"
    if [ -n "$svm_t" ] && [ -n "$mlir_t" ]; then
        ratio=$(python3 -c "print(f'{float(\"$mlir_t\")/float(\"$svm_t\"):.2f}x')" 2>/dev/null)
        echo "    MLIR/SVM ratio: $ratio"
    fi
done

echo ""
echo "=== Kernel metadata comparison ==="
echo "--- MLIR .hsaco ---"
MLIR_HSACO=$(ls -t $HOME/.clifft/kernel_cache/mlir/*.hsaco 2>/dev/null | head -1)
if [ -f "$MLIR_HSACO" ]; then
    $LLVM_PREFIX/bin/llvm-readelf -n "$MLIR_HSACO" 2>/dev/null | grep -E "vgpr_count|sgpr_count|private_segment|group_segment|kernarg_segment|flat_work_group|wavefront_size|dynamic_stack"
fi
echo "--- Hybrid .hsaco ---"
HYB_HSACO=$(ls -t $HOME/.clifft/kernel_cache/*_gfx950.hsaco 2>/dev/null | head -1)
if [ -f "$HYB_HSACO" ]; then
    $LLVM_PREFIX/bin/llvm-readelf -n "$HYB_HSACO" 2>/dev/null | grep -E "vgpr_count|sgpr_count|private_segment|group_segment|kernarg_segment|flat_work_group|wavefront_size|dynamic_stack"
fi

echo ""
echo "Done: $(date)"
