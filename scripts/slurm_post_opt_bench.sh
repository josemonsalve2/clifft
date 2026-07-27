#!/bin/bash
#SBATCH --job-name=post-opt
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:45:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/post_opt_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BUILD="$BASE/build-gpu-mlir-mi355x"
BIN="$BUILD/run_gpu"
SHOTS=10000

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

echo "=== POST-OPTIMIZATION: P1.9 Alignment + P1.13 Atomics ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

cd "$BUILD"
make -j$(nproc) 2>&1 | tail -3
RC=$?
if [ $RC -ne 0 ]; then
    echo "BUILD FAILED"
    make -j1 2>&1 | grep "error:" | head -10
    exit 1
fi
echo "BUILD OK"
echo ""

# Pre-compile
echo "=== Pre-compilation ==="
rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null
for stim in "$BASE"/tests/fixtures/rank_sweep/rank_q17_r{0,4,10,17}_d1.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q33_r{0,15}_d1.stim \
            "$BASE"/tests/fixtures/incremental/01_frame_only/frame_h.stim; do
    [ -f "$stim" ] || continue
    name=$(basename "$stim" .stim)
    echo -n "  $name: "
    timeout 300 $BIN --circuit "$stim" --shots 10 --seed 42 --mlir 2>&1 | \
        grep -oP "compiled in [0-9.]+ ms" || echo "timeout"
done
echo ""

# Correctness
echo "=== Correctness ==="
for stim in "$BASE"/tests/fixtures/rank_sweep/rank_q17_r{0,4,10,17}_d1.stim; do
    [ -f "$stim" ] || continue
    name=$(basename "$stim" .stim)
    svm_p=$(timeout 30 $BIN --circuit "$stim" --shots $SHOTS --seed 42 2>/dev/null | \
        python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
    mlir_p=$(timeout 30 $BIN --circuit "$stim" --shots $SHOTS --seed 42 --mlir 2>/dev/null | \
        python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
    echo "  $name: SVM=$svm_p MLIR=$mlir_p $([ "$svm_p" = "$mlir_p" ] && echo OK || echo MISMATCH)"
done
echo ""

# Performance
echo "=== Performance ($SHOTS shots, cached) ==="
printf "%-25s %4s %10s %10s %10s | %8s %8s\n" "Circuit" "Rank" "SVM(s)" "MLIR(s)" "Hybrid(s)" "kern_ms" "disp_ms"
printf "%-25s %4s %10s %10s %10s | %8s %8s\n" "---" "---" "---" "---" "---" "---" "---"
for stim in "$BASE"/tests/fixtures/incremental/01_frame_only/frame_h.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q17_r{0,4,10,17}_d1.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q33_r{0,15}_d1.stim; do
    [ -f "$stim" ] || continue
    name=$(basename "$stim" .stim)
    rank=$(echo "$name" | grep -oP 'r\K[0-9]+' || echo "0")

    svm_t=$(timeout 30 $BIN --circuit "$stim" --shots $SHOTS --seed 42 2>/dev/null | \
        python3 -c "import json,sys; print(f\"{json.load(sys.stdin)['sample_seconds']:.6f}\")" 2>/dev/null)
    # MLIR: warm run (second invocation)
    timeout 60 $BIN --circuit "$stim" --shots $SHOTS --seed 42 --mlir 2>/dev/null 1>/dev/null
    mlir_t=$(timeout 30 $BIN --circuit "$stim" --shots $SHOTS --seed 42 --mlir 2>/dev/null | \
        python3 -c "import json,sys; print(f\"{json.load(sys.stdin)['sample_seconds']:.6f}\")" 2>/dev/null)
    mlir_err=$(timeout 30 $BIN --circuit "$stim" --shots $SHOTS --seed 42 --mlir 2>&1 1>/dev/null)
    kern_ms=$(echo "$mlir_err" | grep "kernel_seconds" | grep -oP '[0-9.]+' | head -1)
    disp_ms=$(echo "$mlir_err" | grep "dispatch_loop" | grep -oP '[0-9.]+' | head -1)

    hyb_t=$(timeout 60 $BIN --circuit "$stim" --shots $SHOTS --seed 42 --hybrid 2>/dev/null | \
        python3 -c "import json,sys; print(f\"{json.load(sys.stdin)['sample_seconds']:.6f}\")" 2>/dev/null)

    printf "%-25s %4s %10s %10s %10s | %8s %8s\n" \
        "$name" "$rank" "${svm_t:-FAIL}s" "${mlir_t:-FAIL}s" "${hyb_t:-FAIL}s" "${kern_ms:-?}" "${disp_ms:-?}"
done

# ISA metadata comparison
echo ""
echo "=== Kernel Metadata (post-optimization) ==="
HSACO=$(ls -t ~/.clifft/kernel_cache/mlir/*.hsaco 2>/dev/null | head -1)
if [ -f "$HSACO" ]; then
    echo "File: $(basename $HSACO)"
    $LLVM_PREFIX/bin/llvm-readelf -n "$HSACO" 2>/dev/null | grep -E "vgpr|sgpr|private|group|flat_work|dynamic"
fi

echo ""
echo "Done: $(date)"
