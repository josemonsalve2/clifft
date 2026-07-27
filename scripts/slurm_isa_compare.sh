#!/bin/bash
#SBATCH --job-name=isa-cmp
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/isa_compare_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BIN="$BASE/build-gpu-mlir-mi355x/run_gpu"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

cd "$BASE/build-gpu-mlir-mi355x" && make -j$(nproc) 2>&1 | tail -3

rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null

# Compile frame_h (register tier) and qv10 (coop tier)
echo "=== Compiling frame_h (register) ==="
$BIN --circuit "$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim" --shots 10 --seed 42 --mlir 2>&1 | grep -E "compiled|loaded|private|group"
echo ""

echo "=== Compiling qv10 (coop) ==="
timeout 120 $BIN --circuit "$BASE/tests/fixtures/qv10.stim" --shots 10 --seed 42 --mlir 2>&1 | grep -E "compiled|loaded|private|group"
echo ""

echo "=== ISA metadata ==="
for hsaco in ~/.clifft/kernel_cache/mlir/*.hsaco; do
    [ -f "$hsaco" ] || continue
    echo "--- $(basename $hsaco) ---"
    $LLVM_PREFIX/bin/llvm-readelf -n "$hsaco" 2>/dev/null | grep -E "vgpr|sgpr|private|group|flat_work|kernarg|spill|dynamic"
    echo ""
    echo "ISA instruction count:"
    $LLVM_PREFIX/bin/llvm-objdump -d "$hsaco" 2>/dev/null | grep -c "^\s*[0-9a-f]*:"
    echo ""
    echo "ds_read/write patterns:"
    $LLVM_PREFIX/bin/llvm-objdump -d "$hsaco" 2>/dev/null | grep -c "ds_read_b32"
    echo "ds_read_b32 count: $(!!)"
    $LLVM_PREFIX/bin/llvm-objdump -d "$hsaco" 2>/dev/null | grep -c "ds_read_b64"
    echo "ds_read_b64 count: $(!!)"
    $LLVM_PREFIX/bin/llvm-objdump -d "$hsaco" 2>/dev/null | grep -c "ds_write_b32"
    echo "ds_write_b32 count: $(!!)"
    $LLVM_PREFIX/bin/llvm-objdump -d "$hsaco" 2>/dev/null | grep -c "ds_write_b64"
    echo "ds_write_b64 count: $(!!)"
done

# Performance comparison
echo ""
echo "=== Performance: qv10 SVM vs MLIR ==="
tmpf=$(mktemp)

svm_json=$($BIN --circuit "$BASE/tests/fixtures/qv10.stim" --shots 10000 --seed 42 2>"$tmpf")
svm_kern=$(grep kernel_seconds "$tmpf" | grep -oP '[0-9.]+')
svm_p=$(echo "$svm_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
echo "SVM:    passed=$svm_p  kernel=${svm_kern}ms"

mlir_json=$($BIN --circuit "$BASE/tests/fixtures/qv10.stim" --shots 10000 --seed 42 --mlir 2>"$tmpf")
mlir_kern=$(grep kernel_seconds "$tmpf" | grep -oP '[0-9.]+')
mlir_p=$(echo "$mlir_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
echo "MLIR:   passed=$mlir_p  kernel=${mlir_kern}ms"

rm -f "$tmpf"

echo ""
echo "Done: $(date)"
