#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/baseline_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BUILD="$BASE/build-gpu-mlir-mi355x"
BIN="$BUILD/run_gpu"
SHOTS=100000

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

echo "=== BASELINE: SVM vs MLIR All Tiers ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Shots: $SHOTS"
echo ""

# Build
cd "$BUILD"
make -j$(nproc) 2>&1 | tail -3
echo ""

# Warm SVM
$BIN --circuit "$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim" --shots 100 --seed 42 >/dev/null 2>&1

# Pre-compile ALL MLIR kernels first (so timing excludes compilation)
echo "=== MLIR Pre-compilation Phase ==="
rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null
for stim in "$BASE"/tests/fixtures/rank_sweep/rank_q17_r{0,2,4,5,8,10,12,15,17}_d1.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q33_r{0,4,10,15}_d1.stim \
            "$BASE"/tests/fixtures/incremental/01_frame_only/frame_h.stim \
            "$BASE"/tests/fixtures/incremental/06_small_circuits/rep_d3.stim; do
    [ -f "$stim" ] || continue
    name=$(basename "$stim" .stim)
    echo -n "  $name: "
    result=$(timeout 300 $BIN --circuit "$stim" --shots 10 --seed 42 --mlir 2>&1)
    echo "$result" | grep -oP "compiled in [0-9.]+ ms" || echo "$result" | grep -oP "cache hit|FATAL|error" || echo "timeout/unknown"
done
echo ""

# Function to run benchmark and extract sample_seconds
bench() {
    local stim="$1" mode="$2"
    local args="--circuit $stim --shots $SHOTS --seed 42"
    [ "$mode" = "mlir" ] && args="$args --mlir"
    [ "$mode" = "hybrid" ] && args="$args --hybrid"
    timeout 60 $BIN $args 2>/dev/null | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print(f\"{d['sample_seconds']:.6f}\")
except:
    print('FAIL')
" 2>/dev/null
}

# Print header
echo "=== Performance Baseline ($SHOTS shots) ==="
printf "%-35s %5s %4s %10s %10s %10s %8s %8s\n" \
    "Circuit" "Qbits" "Rank" "SVM(s)" "MLIR(s)" "Hybrid(s)" "MLIR/SVM" "Corr"
printf "%-35s %5s %4s %10s %10s %10s %8s %8s\n" \
    "---" "---" "---" "---" "---" "---" "---" "---"

# Register tier (rank 0-4)
echo ""
echo "--- REGISTER TIER (rank 0-4) ---"
for stim in "$BASE"/tests/fixtures/incremental/01_frame_only/frame_h.stim \
            "$BASE"/tests/fixtures/incremental/06_small_circuits/rep_d3.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q17_r0_d1.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q17_r2_d1.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q17_r4_d1.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q17_r4_d3.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q33_r0_d1.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q33_r4_d1.stim; do
    [ -f "$stim" ] || continue
    name=$(basename "$stim" .stim)
    # Extract qubits and rank from filename
    qbits=$(echo "$name" | grep -oP 'q\K[0-9]+' || echo "?")
    rank=$(echo "$name" | grep -oP 'r\K[0-9]+' || echo "?")
    [ "$qbits" = "?" ] && qbits=$(grep -c "QUBIT_COORDS" "$stim" 2>/dev/null || echo "?")

    svm_t=$(bench "$stim" svm)
    mlir_t=$(bench "$stim" mlir)
    hyb_t=$(bench "$stim" hybrid)

    # Correctness check
    svm_p=$(timeout 30 $BIN --circuit "$stim" --shots 1000 --seed 42 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
    mlir_p=$(timeout 30 $BIN --circuit "$stim" --shots 1000 --seed 42 --mlir 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
    corr="?"
    [ -n "$svm_p" ] && [ -n "$mlir_p" ] && { [ "$svm_p" = "$mlir_p" ] && corr="EXACT" || corr="DIFF"; }
    [ -z "$mlir_p" ] && corr="FAIL"

    ratio="-"
    if [ "$svm_t" != "FAIL" ] && [ "$mlir_t" != "FAIL" ] && [ -n "$svm_t" ] && [ -n "$mlir_t" ]; then
        ratio=$(python3 -c "print(f'{float(\"$mlir_t\")/float(\"$svm_t\"):.2f}x')" 2>/dev/null)
    fi

    printf "%-35s %5s %4s %10s %10s %10s %8s %8s\n" \
        "$name" "$qbits" "$rank" "${svm_t}s" "${mlir_t}s" "${hyb_t}s" "$ratio" "$corr"
done

# Coop tier (rank 5-10)
echo ""
echo "--- COOP TIER (rank 5-10) ---"
for stim in "$BASE"/tests/fixtures/rank_sweep/rank_q17_r5_d1.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q17_r8_d1.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q17_r8_d3.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q17_r10_d1.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q33_r5_d1.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q33_r10_d1.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q33_r10_d3.stim; do
    [ -f "$stim" ] || continue
    name=$(basename "$stim" .stim)
    qbits=$(echo "$name" | grep -oP 'q\K[0-9]+')
    rank=$(echo "$name" | grep -oP 'r\K[0-9]+')

    svm_t=$(bench "$stim" svm)
    mlir_t=$(bench "$stim" mlir)
    hyb_t=$(bench "$stim" hybrid)

    svm_p=$(timeout 30 $BIN --circuit "$stim" --shots 1000 --seed 42 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
    mlir_p=$(timeout 60 $BIN --circuit "$stim" --shots 1000 --seed 42 --mlir 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
    corr="?"
    [ -n "$svm_p" ] && [ -n "$mlir_p" ] && { [ "$svm_p" = "$mlir_p" ] && corr="EXACT" || corr="DIFF"; }
    [ -z "$mlir_p" ] && corr="FAIL"

    ratio="-"
    if [ "$svm_t" != "FAIL" ] && [ "$mlir_t" != "FAIL" ] && [ -n "$svm_t" ] && [ -n "$mlir_t" ]; then
        ratio=$(python3 -c "print(f'{float(\"$mlir_t\")/float(\"$svm_t\"):.2f}x')" 2>/dev/null)
    fi

    printf "%-35s %5s %4s %10s %10s %10s %8s %8s\n" \
        "$name" "$qbits" "$rank" "${svm_t}s" "${mlir_t}s" "${hyb_t}s" "$ratio" "$corr"
done

# Global tier (rank 11-19)
echo ""
echo "--- GLOBAL TIER (rank 11-19) ---"
for stim in "$BASE"/tests/fixtures/rank_sweep/rank_q17_r12_d1.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q17_r15_d1.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q17_r17_d1.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q17_r17_d3.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q33_r12_d1.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q33_r15_d1.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q33_r15_d3.stim; do
    [ -f "$stim" ] || continue
    name=$(basename "$stim" .stim)
    qbits=$(echo "$name" | grep -oP 'q\K[0-9]+')
    rank=$(echo "$name" | grep -oP 'r\K[0-9]+')

    svm_t=$(bench "$stim" svm)
    mlir_t=$(bench "$stim" mlir)
    hyb_t=$(bench "$stim" hybrid)

    svm_p=$(timeout 30 $BIN --circuit "$stim" --shots 1000 --seed 42 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
    mlir_p=$(timeout 120 $BIN --circuit "$stim" --shots 1000 --seed 42 --mlir 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
    corr="?"
    [ -n "$svm_p" ] && [ -n "$mlir_p" ] && { [ "$svm_p" = "$mlir_p" ] && corr="EXACT" || corr="DIFF"; }
    [ -z "$mlir_p" ] && corr="FAIL"

    ratio="-"
    if [ "$svm_t" != "FAIL" ] && [ "$mlir_t" != "FAIL" ] && [ -n "$svm_t" ] && [ -n "$mlir_t" ]; then
        ratio=$(python3 -c "print(f'{float(\"$mlir_t\")/float(\"$svm_t\"):.2f}x')" 2>/dev/null)
    fi

    printf "%-35s %5s %4s %10s %10s %10s %8s %8s\n" \
        "$name" "$qbits" "$rank" "${svm_t}s" "${mlir_t}s" "${hyb_t}s" "$ratio" "$corr"
done

# Shot scaling test
echo ""
echo "=== Shot Scaling (frame_h register tier) ==="
printf "%-12s %10s %10s %10s\n" "Shots" "SVM(s)" "MLIR(s)" "Hybrid(s)"
for shots in 1000 10000 100000 1000000; do
    svm_t=$(timeout 30 $BIN --circuit "$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim" --shots $shots --seed 42 2>/dev/null | python3 -c "import json,sys; print(f\"{json.load(sys.stdin)['sample_seconds']:.6f}\")" 2>/dev/null)
    mlir_t=$(timeout 30 $BIN --circuit "$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim" --shots $shots --seed 42 --mlir 2>/dev/null | python3 -c "import json,sys; print(f\"{json.load(sys.stdin)['sample_seconds']:.6f}\")" 2>/dev/null)
    hyb_t=$(timeout 30 $BIN --circuit "$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim" --shots $shots --seed 42 --hybrid 2>/dev/null | python3 -c "import json,sys; print(f\"{json.load(sys.stdin)['sample_seconds']:.6f}\")" 2>/dev/null)
    printf "%-12s %10s %10s %10s\n" "$shots" "${svm_t:-FAIL}s" "${mlir_t:-FAIL}s" "${hyb_t:-FAIL}s"
done

echo ""
echo "Done: $(date)"
