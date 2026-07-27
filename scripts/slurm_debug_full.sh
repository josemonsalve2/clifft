#!/bin/bash
#SBATCH --job-name=dbg-full
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/debug_full_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BUILD="$BASE/build-gpu-mlir-mi355x"
BIN="$BUILD/run_gpu"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

cd "$BUILD" && make -j$(nproc) 2>&1 | tail -3
echo "BUILD: ${PIPESTATUS[0]}"

rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null

# Test frame_h and four_t
echo ""
echo "=== frame_h ==="
$BIN --circuit "$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim" --shots 100 --seed 42 --mlir 2>&1 | grep -E "passed_shots|debug|error"
echo ""
echo "=== four_t ==="
$BIN --circuit "$BASE/tests/fixtures/incremental/02_single_expand/four_t.stim" --shots 100 --seed 42 --mlir 2>&1 | grep -E "passed_shots|debug|error"
echo ""

# Check saved MLIR files
echo "=== Saved MLIR files ==="
ls -la $BASE/results/debug_raw_*.mlir 2>/dev/null
echo ""

# Examine four_t MLIR
for f in $BASE/results/debug_raw_*ops.mlir; do
    [ -f "$f" ] || continue
    ops=$(basename "$f" | grep -oP '\d+(?=ops)')
    echo "=== $f ($ops ops, $(wc -l < "$f") lines) ==="
    grep "// op\[" "$f"
    echo "---"
    grep -c "discarded" "$f"
    echo "discard stores: $(grep -c 'store.*c1_i8.*discarded' "$f")"
    echo "---"
done

echo "Done: $(date)"
