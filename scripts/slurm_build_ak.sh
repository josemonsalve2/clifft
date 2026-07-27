#!/bin/bash
#SBATCH --job-name=build-ak
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/build_ak_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BUILD="$BASE/build-gpu-mlir-mi355x"
BIN="$BUILD/run_gpu"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

echo "=== Building with active_k constant folding ==="
cd "$BUILD"
make -j$(nproc) 2>&1 | tail -5
RC=${PIPESTATUS[0]}
if [ $RC -ne 0 ]; then
    echo "BUILD FAILED"
    make -j1 2>&1 | tail -30
    exit 1
fi
echo "BUILD OK"
echo ""

rm -rf "$HOME/.clifft/kernel_cache/mlir" 2>/dev/null
STIM="$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim"

echo "=== MLIR warmup ==="
timeout 120 $BIN --circuit "$STIM" --shots 100 --seed 42 --mlir 2>&1 | tail -5
echo ""

echo "=== Performance: SVM vs MLIR (100k shots, register tier) ==="
svm=$($BIN --circuit "$STIM" --shots 100000 --seed 42 2>&1)
svm_t=$(echo "$svm" | python3 -c "import json,sys; print(json.load(sys.stdin)['sample_seconds'])" 2>/dev/null)
mlir=$($BIN --circuit "$STIM" --shots 100000 --seed 42 --mlir 2>&1)
mlir_t=$(echo "$mlir" | python3 -c "import json,sys; print(json.load(sys.stdin)['sample_seconds'])" 2>/dev/null)
echo "SVM:  ${svm_t}s"
echo "MLIR: ${mlir_t}s"
if [ -n "$svm_t" ] && [ -n "$mlir_t" ]; then
    echo "Ratio: $(python3 -c "print(f'{float(\"$mlir_t\")/float(\"$svm_t\"):.2f}x')" 2>/dev/null)"
fi

echo ""
echo "=== 1M shots ==="
svm=$($BIN --circuit "$STIM" --shots 1000000 --seed 42 2>&1)
svm_t=$(echo "$svm" | python3 -c "import json,sys; print(json.load(sys.stdin)['sample_seconds'])" 2>/dev/null)
mlir=$($BIN --circuit "$STIM" --shots 1000000 --seed 42 --mlir 2>&1)
mlir_t=$(echo "$mlir" | python3 -c "import json,sys; print(json.load(sys.stdin)['sample_seconds'])" 2>/dev/null)
echo "SVM:  ${svm_t}s"
echo "MLIR: ${mlir_t}s"
if [ -n "$svm_t" ] && [ -n "$mlir_t" ]; then
    echo "Ratio: $(python3 -c "print(f'{float(\"$mlir_t\")/float(\"$svm_t\"):.2f}x')" 2>/dev/null)"
fi

echo ""
echo "=== Correctness ==="
cpu_p=$($BASE/build-cpu/run_cpu --circuit "$STIM" --shots 10000 --seed 42 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['passed_shots'])" 2>/dev/null)
mlir_p=$($BIN --circuit "$STIM" --shots 10000 --seed 42 --mlir 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['passed_shots'])" 2>/dev/null)
svm_p=$($BIN --circuit "$STIM" --shots 10000 --seed 42 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['passed_shots'])" 2>/dev/null)
echo "CPU=$cpu_p SVM=$svm_p MLIR=$mlir_p"

echo ""
echo "=== Kernel metadata ==="
HSACO=$(ls -t $HOME/.clifft/kernel_cache/mlir/*.hsaco 2>/dev/null | head -1)
if [ -f "$HSACO" ]; then
    $LLVM_PREFIX/bin/llvm-readelf -n "$HSACO" 2>/dev/null | grep -E "vgpr_count|sgpr_count|private_segment|group_segment|flat_work_group|dynamic_stack"
fi

echo ""
echo "Done: $(date)"
