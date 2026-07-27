#!/bin/bash
#SBATCH --job-name=valid-bl
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/valid_baseline_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BUILD="$BASE/build-gpu-mlir-mi355x"
BIN="$BUILD/run_gpu"
CPU_BIN="$BASE/build-cpu/run_cpu"
SHOTS=10000

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

echo "=== VALIDATED BASELINE: CPU vs SVM vs MLIR vs Hybrid ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Shots: $SHOTS"
echo ""

# Build
cd "$BUILD"
make -j$(nproc) 2>&1 | tail -3
echo ""

# Check CPU binary
if [ ! -f "$CPU_BIN" ]; then
    echo "CPU binary not found at $CPU_BIN"
    echo "Trying to build CPU version..."
    mkdir -p "$BASE/build-cpu"
    cd "$BASE/build-cpu"
    cmake "$BASE" -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
    make -j$(nproc) run_cpu 2>&1 | tail -3
    CPU_BIN="$BASE/build-cpu/run_cpu"
fi
echo "CPU binary: $(ls -la $CPU_BIN 2>/dev/null | awk '{print $NF}')"
echo ""

# Pre-compile ALL MLIR kernels
echo "=== MLIR Pre-compilation ==="
rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null
for stim in "$BASE"/tests/fixtures/rank_sweep/rank_q17_r{0,2,4,5,8,10,12,15,17}_d1.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q33_r{0,4,10,15}_d1.stim \
            "$BASE"/tests/fixtures/incremental/01_frame_only/frame_h.stim; do
    [ -f "$stim" ] || continue
    name=$(basename "$stim" .stim)
    echo -n "  $name: "
    timeout 300 $BIN --circuit "$stim" --shots 10 --seed 42 --mlir 2>&1 | grep -oP "compiled in [0-9.]+ ms|cache hit" || echo "timeout/error"
done
echo ""

# Function to get passed_shots, sample_seconds, and timing breakdown
get_result() {
    local bin="$1" stim="$2" shots="$3" extra="$4"
    local out=$(timeout 120 $bin --circuit "$stim" --shots $shots --seed 42 $extra 2>&1)
    local passed=$(echo "$out" | python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
    local secs=$(echo "$out" | python3 -c "import json,sys; print(f\"{json.load(sys.stdin)['sample_seconds']:.6f}\")" 2>/dev/null)
    local kernel=$(echo "$out" | grep "clifft-timing.*kernel_seconds" | grep -oP '[0-9.]+' | head -1)
    local alloc=$(echo "$out" | grep "clifft-timing.*buffer_alloc" | grep -oP '[0-9.]+' | head -1)
    local dispatch=$(echo "$out" | grep "clifft-timing.*dispatch_loop" | grep -oP '[0-9.]+' | head -1)
    echo "$passed|$secs|${kernel:-?}|${alloc:-?}|${dispatch:-?}"
}

echo "=== Validated Baseline ($SHOTS shots) ==="
printf "%-25s %4s %10s %10s %10s %10s | %7s %7s | %7s %7s %7s | %5s\n" \
    "Circuit" "Rank" "CPU(s)" "SVM(s)" "MLIR(s)" "Hybrid(s)" "CPU_p" "SVM_p" "kern_ms" "alloc" "disp" "Match"
printf "%-25s %4s %10s %10s %10s %10s | %7s %7s | %7s %7s %7s | %5s\n" \
    "---" "---" "---" "---" "---" "---" "---" "---" "---" "---" "---" "---"

for stim in "$BASE"/tests/fixtures/incremental/01_frame_only/frame_h.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q17_r0_d1.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q17_r2_d1.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q17_r4_d1.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q17_r5_d1.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q17_r8_d1.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q17_r10_d1.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q17_r12_d1.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q17_r15_d1.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q17_r17_d1.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q33_r0_d1.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q33_r4_d1.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q33_r10_d1.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q33_r15_d1.stim; do
    [ -f "$stim" ] || continue
    name=$(basename "$stim" .stim)
    rank=$(echo "$name" | grep -oP 'r\K[0-9]+' || echo "0")

    # CPU reference
    cpu_result=$(get_result "$CPU_BIN" "$stim" "$SHOTS" "")
    cpu_passed=$(echo "$cpu_result" | cut -d'|' -f1)
    cpu_secs=$(echo "$cpu_result" | cut -d'|' -f2)

    # SVM
    svm_result=$(get_result "$BIN" "$stim" "$SHOTS" "")
    svm_passed=$(echo "$svm_result" | cut -d'|' -f1)
    svm_secs=$(echo "$svm_result" | cut -d'|' -f2)

    # MLIR (kernel already pre-compiled)
    mlir_result=$(get_result "$BIN" "$stim" "$SHOTS" "--mlir")
    mlir_passed=$(echo "$mlir_result" | cut -d'|' -f1)
    mlir_secs=$(echo "$mlir_result" | cut -d'|' -f2)
    mlir_kern=$(echo "$mlir_result" | cut -d'|' -f3)
    mlir_alloc=$(echo "$mlir_result" | cut -d'|' -f4)
    mlir_disp=$(echo "$mlir_result" | cut -d'|' -f5)

    # Hybrid
    hyb_result=$(get_result "$BIN" "$stim" "$SHOTS" "--hybrid")
    hyb_secs=$(echo "$hyb_result" | cut -d'|' -f2)

    # Correctness check: CPU vs SVM vs MLIR
    match="?"
    if [ -n "$cpu_passed" ] && [ -n "$svm_passed" ] && [ -n "$mlir_passed" ]; then
        if [ "$cpu_passed" = "$svm_passed" ] && [ "$cpu_passed" = "$mlir_passed" ]; then
            match="ALL"
        elif [ "$cpu_passed" = "$svm_passed" ]; then
            match="C=S"
        elif [ "$cpu_passed" = "$mlir_passed" ]; then
            match="C=M"
        else
            match="NONE"
        fi
    elif [ -z "$mlir_passed" ]; then
        match="M:FL"
    fi

    printf "%-25s %4s %10s %10s %10s %10s | %7s %7s | %7s %7s %7s | %5s\n" \
        "$name" "$rank" "${cpu_secs:-FAIL}s" "${svm_secs:-FAIL}s" "${mlir_secs:-FAIL}s" "${hyb_secs:-FAIL}s" \
        "${cpu_passed:-?}" "${svm_passed:-?}" "${mlir_kern:-?}" "${mlir_alloc:-?}" "${mlir_disp:-?}" "$match"
done

echo ""
echo "Legend: kern_ms=GPU kernel time, alloc=buffer alloc(ms), disp=dispatch loop(ms)"
echo "Match: ALL=CPU=SVM=MLIR, C=S=CPU matches SVM only, M:FL=MLIR failed"
echo ""
echo "Done: $(date)"
