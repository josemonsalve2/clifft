#!/bin/bash
#SBATCH --job-name=final-bl
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/final_baseline_%j.log

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

echo "=== FINAL BASELINE: Correctness + Performance ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Shots: $SHOTS"
echo ""

# Build
cd "$BUILD"
make -j$(nproc) 2>&1 | tail -3
echo ""

# Build CPU binary
CPU_BIN="$BASE/build-cpu/run_cpu"
if [ ! -f "$CPU_BIN" ]; then
    echo "Building CPU binary..."
    mkdir -p "$BASE/build-cpu"
    cd "$BASE/build-cpu"
    cmake "$BASE" -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
    make -j$(nproc) run_cpu 2>&1 | tail -3
    cd "$BUILD"
fi
echo "CPU: $CPU_BIN"
echo ""

# Step 1: Pre-compile ALL MLIR kernels (cached to disk)
echo "=== Step 1: MLIR Pre-compilation (excluded from timing) ==="
rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null
for stim in "$BASE"/tests/fixtures/rank_sweep/rank_q17_r{0,2,4}_d1.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q17_r{5,8,10}_d1.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q17_r{12,15,17}_d1.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q33_r{0,4,10,15}_d1.stim \
            "$BASE"/tests/fixtures/incremental/01_frame_only/frame_h.stim \
            "$BASE"/tests/fixtures/large/surface_d5_r5.stim \
            "$BASE"/tests/fixtures/large/surface_d7_r7.stim; do
    [ -f "$stim" ] || continue
    name=$(basename "$stim" .stim)
    echo -n "  $name: "
    timeout 300 $BIN --circuit "$stim" --shots 10 --seed 42 --mlir 2>&1 | \
        grep -oP "compiled in [0-9.]+ ms|cache hit|FATAL" || echo "timeout"
done
echo ""

# Step 2: Correctness — CPU vs SVM vs MLIR (separate JSON capture, no stderr mixing)
echo "=== Step 2: Correctness Verification ==="
printf "%-30s %4s %8s %8s %8s %6s\n" "Circuit" "Rank" "CPU" "SVM" "MLIR" "Match"
printf "%-30s %4s %8s %8s %8s %6s\n" "---" "---" "---" "---" "---" "---"

for stim in "$BASE"/tests/fixtures/incremental/01_frame_only/frame_h.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q17_r{0,2,4,5,8,10,12,15,17}_d1.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q33_r{0,4,10,15}_d1.stim \
            "$BASE"/tests/fixtures/large/surface_d5_r5.stim; do
    [ -f "$stim" ] || continue
    name=$(basename "$stim" .stim)
    rank=$(echo "$name" | grep -oP 'r\K[0-9]+' || echo "0")

    cpu_p=$(timeout 60 $CPU_BIN --circuit "$stim" --shots $SHOTS --seed 42 2>/dev/null | \
        python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
    svm_p=$(timeout 60 $BIN --circuit "$stim" --shots $SHOTS --seed 42 2>/dev/null | \
        python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
    mlir_p=$(timeout 60 $BIN --circuit "$stim" --shots $SHOTS --seed 42 --mlir 2>/dev/null | \
        python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)

    match="?"
    if [ -n "$cpu_p" ] && [ -n "$svm_p" ] && [ -n "$mlir_p" ]; then
        if [ "$cpu_p" = "$svm_p" ] && [ "$cpu_p" = "$mlir_p" ]; then match="ALL_OK"
        elif [ "$cpu_p" = "$svm_p" ]; then match="C=S"
        elif [ "$cpu_p" = "$mlir_p" ]; then match="C=M"
        else match="NONE"; fi
    elif [ -z "$mlir_p" ]; then match="M_FAIL"
    fi

    printf "%-30s %4s %8s %8s %8s %6s\n" "$name" "$rank" "${cpu_p:-?}" "${svm_p:-?}" "${mlir_p:-?}" "$match"
done
echo ""

# Step 3: Performance with CACHED compilation and timing breakdown
# Run each benchmark TWICE — first populates kernel cache + persistent pool, second is the real measurement
echo "=== Step 3: Performance (cached compilation, warm pool) ==="
printf "%-30s %4s %10s %10s %10s | %8s %8s %8s\n" \
    "Circuit" "Rank" "SVM(s)" "MLIR(s)" "Hybrid(s)" "kern_ms" "alloc_ms" "disp_ms"
printf "%-30s %4s %10s %10s %10s | %8s %8s %8s\n" \
    "---" "---" "---" "---" "---" "---" "---" "---"

for stim in "$BASE"/tests/fixtures/incremental/01_frame_only/frame_h.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q17_r{0,2,4,5,8,10,12,15,17}_d1.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q33_r{0,4,10,15}_d1.stim \
            "$BASE"/tests/fixtures/large/surface_d5_r5.stim \
            "$BASE"/tests/fixtures/large/surface_d7_r7.stim; do
    [ -f "$stim" ] || continue
    name=$(basename "$stim" .stim)
    rank=$(echo "$name" | grep -oP 'r\K[0-9]+' || echo "0")

    # SVM (stdout only, stderr suppressed)
    svm_t=$(timeout 60 $BIN --circuit "$stim" --shots $SHOTS --seed 42 2>/dev/null | \
        python3 -c "import json,sys; print(f\"{json.load(sys.stdin)['sample_seconds']:.6f}\")" 2>/dev/null)

    # MLIR — run once to populate pool, capture timing from stderr
    mlir_stderr=$(timeout 60 $BIN --circuit "$stim" --shots $SHOTS --seed 42 --mlir 2>&1 1>/dev/null)
    # Run again to get JSON from stdout (pool warm, kernel cached)
    mlir_t=$(timeout 60 $BIN --circuit "$stim" --shots $SHOTS --seed 42 --mlir 2>/dev/null | \
        python3 -c "import json,sys; print(f\"{json.load(sys.stdin)['sample_seconds']:.6f}\")" 2>/dev/null)
    # Extract timing from the SECOND run's stderr
    mlir_stderr2=$(timeout 60 $BIN --circuit "$stim" --shots $SHOTS --seed 42 --mlir 2>&1 1>/dev/null)
    kern_ms=$(echo "$mlir_stderr2" | grep "kernel_seconds" | grep -oP '[0-9.]+' | head -1)
    alloc_ms=$(echo "$mlir_stderr2" | grep "buffer_alloc" | grep -oP '[0-9.]+' | head -1)
    disp_ms=$(echo "$mlir_stderr2" | grep "dispatch_loop" | grep -oP '[0-9.]+' | head -1)

    # Hybrid
    hyb_t=$(timeout 120 $BIN --circuit "$stim" --shots $SHOTS --seed 42 --hybrid 2>/dev/null | \
        python3 -c "import json,sys; print(f\"{json.load(sys.stdin)['sample_seconds']:.6f}\")" 2>/dev/null)

    printf "%-30s %4s %10s %10s %10s | %8s %8s %8s\n" \
        "$name" "$rank" "${svm_t:-FAIL}s" "${mlir_t:-FAIL}s" "${hyb_t:-FAIL}s" \
        "${kern_ms:-?}" "${alloc_ms:-?}" "${disp_ms:-?}"
done

echo ""
echo "=== Step 4: Shot Scaling (frame_h, cached) ==="
printf "%-12s %10s %10s %10s\n" "Shots" "SVM(s)" "MLIR(s)" "Hybrid(s)"
for shots in 1000 10000 100000 1000000; do
    svm_t=$(timeout 30 $BIN --circuit "$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim" \
        --shots $shots --seed 42 2>/dev/null | \
        python3 -c "import json,sys; print(f\"{json.load(sys.stdin)['sample_seconds']:.6f}\")" 2>/dev/null)
    mlir_t=$(timeout 30 $BIN --circuit "$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim" \
        --shots $shots --seed 42 --mlir 2>/dev/null | \
        python3 -c "import json,sys; print(f\"{json.load(sys.stdin)['sample_seconds']:.6f}\")" 2>/dev/null)
    hyb_t=$(timeout 30 $BIN --circuit "$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim" \
        --shots $shots --seed 42 --hybrid 2>/dev/null | \
        python3 -c "import json,sys; print(f\"{json.load(sys.stdin)['sample_seconds']:.6f}\")" 2>/dev/null)
    printf "%-12s %10s %10s %10s\n" "$shots" "${svm_t:-FAIL}s" "${mlir_t:-FAIL}s" "${hyb_t:-FAIL}s"
done

echo ""
echo "Done: $(date)"
