#!/bin/bash
#SBATCH --job-name=3tier-bench
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/3tier_bench_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BIN="$BASE/build-gpu-mlir-mi355x/run_gpu"
SHOTS=10000

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

echo "=== 3-TIER BENCHMARK: SVM vs MLIR vs Hybrid ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Shots: $SHOTS"
echo ""

# ---- REGISTER TIER (rank 0-4) ----
echo "================================================================"
echo "=== REGISTER TIER (rank 0-4, stack alloca, ≤16 amplitudes) ==="
echo "================================================================"
echo ""

REG_CIRCUITS=(
    "$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim"
    "$BASE/tests/fixtures/incremental/02_single_expand/four_t.stim"
    "$BASE/tests/fixtures/large/circuit_d3_p0.001.stim"
    "$BASE/tests/fixtures/large/surface_d7_t5.stim"
)

for stim in "${REG_CIRCUITS[@]}"; do
    [ -f "$stim" ] || { echo "SKIP: $stim not found"; continue; }
    name=$(basename "$stim" .stim)
    echo "--- $name ---"

    # Warmup (SVM always, MLIR cached from prior run or compile here)
    $BIN --circuit "$stim" --shots 10 --seed 42 >/dev/null 2>&1
    timeout 120 $BIN --circuit "$stim" --shots 10 --seed 42 --mlir >/dev/null 2>&1

    # SVM
    echo "SVM:"
    svm_out=$($BIN --circuit "$stim" --shots $SHOTS --seed 42 2>&1)
    echo "$svm_out" | grep -E "peak_rank|sample_seconds|passed_shots" | head -3
    svm_err=$(echo "$svm_out" | grep "clifft-timing")
    echo "$svm_err"

    # MLIR (second run = cached)
    echo "MLIR:"
    $BIN --circuit "$stim" --shots $SHOTS --seed 42 --mlir >/dev/null 2>&1  # ensure cached
    mlir_out=$($BIN --circuit "$stim" --shots $SHOTS --seed 42 --mlir 2>&1)
    echo "$mlir_out" | grep -E "peak_rank|sample_seconds|passed_shots" | head -3
    mlir_err=$(echo "$mlir_out" | grep "clifft-timing")
    echo "$mlir_err"

    # Hybrid
    echo "Hybrid:"
    hyb_out=$(timeout 120 $BIN --circuit "$stim" --shots $SHOTS --seed 42 --hybrid 2>&1)
    echo "$hyb_out" | grep -E "peak_rank|sample_seconds|passed_shots" | head -3
    hyb_err=$(echo "$hyb_out" | grep "clifft-timing")
    echo "$hyb_err"

    # Correctness check
    svm_p=$(echo "$svm_out" | python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
    mlir_p=$(echo "$mlir_out" | python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
    if [ "$svm_p" = "$mlir_p" ]; then
        echo "CORRECTNESS: OK (SVM=$svm_p, MLIR=$mlir_p)"
    else
        echo "CORRECTNESS: MISMATCH (SVM=$svm_p, MLIR=$mlir_p)"
    fi
    echo ""
done

# ---- COOP TIER (rank 5-10) ----
echo "================================================================"
echo "=== COOP TIER (rank 5-10, LDS, ≤1024 amplitudes) ==="
echo "================================================================"
echo ""

COOP_CIRCUITS=(
    "$BASE/tests/fixtures/qv10.stim"
    "$BASE/tests/fixtures/cultivation_d5.stim"
    "$BASE/tests/fixtures/large/circuit_d5_p0.001.stim"
    "$BASE/tests/fixtures/large/surface_d7_t10.stim"
    "$BASE/tests/fixtures/large/surface_d7_t15.stim"
    "$BASE/tests/fixtures/large/surface_d9_t10.stim"
)

for stim in "${COOP_CIRCUITS[@]}"; do
    [ -f "$stim" ] || { echo "SKIP: $stim not found"; continue; }
    name=$(basename "$stim" .stim)
    echo "--- $name ---"

    # Warmup
    $BIN --circuit "$stim" --shots 10 --seed 42 >/dev/null 2>&1
    timeout 300 $BIN --circuit "$stim" --shots 10 --seed 42 --mlir >/dev/null 2>&1

    # SVM
    echo "SVM:"
    svm_out=$($BIN --circuit "$stim" --shots $SHOTS --seed 42 2>&1)
    echo "$svm_out" | grep -E "peak_rank|sample_seconds|passed_shots" | head -3
    svm_err=$(echo "$svm_out" | grep "clifft-timing")
    echo "$svm_err"

    # MLIR (cached)
    echo "MLIR:"
    timeout 300 $BIN --circuit "$stim" --shots $SHOTS --seed 42 --mlir >/dev/null 2>&1
    mlir_out=$(timeout 300 $BIN --circuit "$stim" --shots $SHOTS --seed 42 --mlir 2>&1)
    echo "$mlir_out" | grep -E "peak_rank|sample_seconds|passed_shots" | head -3
    mlir_err=$(echo "$mlir_out" | grep "clifft-timing")
    echo "$mlir_err"

    # Hybrid
    echo "Hybrid:"
    hyb_out=$(timeout 300 $BIN --circuit "$stim" --shots $SHOTS --seed 42 --hybrid 2>&1)
    echo "$hyb_out" | grep -E "peak_rank|sample_seconds|passed_shots" | head -3
    hyb_err=$(echo "$hyb_out" | grep "clifft-timing")
    echo "$hyb_err"

    # Correctness
    svm_p=$(echo "$svm_out" | python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
    mlir_p=$(echo "$mlir_out" | python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
    if [ "$svm_p" = "$mlir_p" ]; then
        echo "CORRECTNESS: OK (SVM=$svm_p, MLIR=$mlir_p)"
    else
        echo "CORRECTNESS: MISMATCH (SVM=$svm_p, MLIR=$mlir_p)"
    fi
    echo ""
done

# ---- GLOBAL TIER (rank 11+) ----
echo "================================================================"
echo "=== GLOBAL TIER (rank 11+, HBM, ≤524K amplitudes) ==="
echo "================================================================"
echo ""

GLOBAL_CIRCUITS=(
    "$BASE/tools/bench/fixtures/qv20_seed42.stim"
    "$BASE/tests/fixtures/large/surface_d7_t19.stim"
    "$BASE/tests/fixtures/large/surface_d9_t15.stim"
    "$BASE/tests/fixtures/large/surface_d9_t19.stim"
    "$BASE/tests/fixtures/large/surface_d11_t15.stim"
    "$BASE/tests/fixtures/large/surface_d11_t19.stim"
)

for stim in "${GLOBAL_CIRCUITS[@]}"; do
    [ -f "$stim" ] || { echo "SKIP: $stim not found"; continue; }
    name=$(basename "$stim" .stim)
    echo "--- $name ---"

    # Warmup
    $BIN --circuit "$stim" --shots 10 --seed 42 >/dev/null 2>&1
    timeout 600 $BIN --circuit "$stim" --shots 10 --seed 42 --mlir >/dev/null 2>&1

    # SVM
    echo "SVM:"
    svm_out=$($BIN --circuit "$stim" --shots $SHOTS --seed 42 2>&1)
    echo "$svm_out" | grep -E "peak_rank|sample_seconds|passed_shots" | head -3
    svm_err=$(echo "$svm_out" | grep "clifft-timing")
    echo "$svm_err"

    # MLIR (cached)
    echo "MLIR:"
    timeout 600 $BIN --circuit "$stim" --shots $SHOTS --seed 42 --mlir >/dev/null 2>&1
    mlir_out=$(timeout 600 $BIN --circuit "$stim" --shots $SHOTS --seed 42 --mlir 2>&1)
    echo "$mlir_out" | grep -E "peak_rank|sample_seconds|passed_shots" | head -3
    mlir_err=$(echo "$mlir_out" | grep "clifft-timing")
    echo "$mlir_err"

    # Hybrid
    echo "Hybrid:"
    hyb_out=$(timeout 600 $BIN --circuit "$stim" --shots $SHOTS --seed 42 --hybrid 2>&1)
    echo "$hyb_out" | grep -E "peak_rank|sample_seconds|passed_shots" | head -3
    hyb_err=$(echo "$hyb_out" | grep "clifft-timing")
    echo "$hyb_err"

    # Correctness
    svm_p=$(echo "$svm_out" | python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
    mlir_p=$(echo "$mlir_out" | python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
    if [ "$svm_p" = "$mlir_p" ]; then
        echo "CORRECTNESS: OK (SVM=$svm_p, MLIR=$mlir_p)"
    else
        echo "CORRECTNESS: MISMATCH (SVM=$svm_p, MLIR=$mlir_p)"
    fi
    echo ""
done

echo "================================================================"
echo "Done: $(date)"
