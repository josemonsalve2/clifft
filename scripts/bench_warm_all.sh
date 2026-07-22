#!/bin/bash
#SBATCH --job-name=bench-warm-all
#SBATCH --partition=mi300x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/bench_warm_all_%j.log
set +e

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
RESULTS="$BASE/results"
mkdir -p "$RESULTS"
export PATH="/shared/jmonsalv/software/cmake-3.31.7-linux-x86_64/bin:$PATH"
for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done

GPU_ARCH=$(rocminfo 2>/dev/null | grep -m1 'Name:.*gfx' | grep -oE 'gfx[0-9]+')
echo "=== Warm-Cache Full Benchmark ==="
echo "Node: $(hostname)  GPU: $GPU_ARCH"
echo "Date: $(date)"
echo ""

# Build
BUILD="$BASE/build-gpu-perf"
[ -x "$BUILD/run_gpu" ] || {
    rm -rf "$BUILD" && mkdir -p "$BUILD" && cd "$BUILD"
    cmake "$BASE" -DCLIFFT_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=$GPU_ARCH -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5
    make -j$(nproc) run_gpu 2>&1 | tail -5
    cd "$BASE"
}
BINARY="$BUILD/run_gpu"
[ -x "$BINARY" ] || { echo "BUILD FAILED"; exit 1; }

# Collect all circuits
CIRCUITS=$(find "$BASE/tests/fixtures" -name "*.stim" -type f | sort)
TOTAL=$(echo "$CIRCUITS" | wc -l)
echo "Total circuits: $TOTAL"

SHOTS=1000000
RUNS=3

# ====================================================================
# Phase 1: Warm kernel cache — compile every circuit's megakernel once
# ====================================================================
echo ""
echo "=== Phase 1: Warming kernel cache ($TOTAL circuits) ==="
echo ""

WARM_OK=0
WARM_FAIL=0
WARM_SKIP=0

for CIRC in $CIRCUITS; do
    NAME=$(basename "$CIRC" .stim)
    printf "  Warming %-40s " "$NAME"
    OUT=$($BINARY --circuit "$CIRC" --shots 100 --hybrid 2>&1)
    RC=$?
    if [ $RC -eq 0 ]; then
        echo "OK"
        WARM_OK=$((WARM_OK + 1))
    else
        # Check if it's a known limitation (kMaxQubits, kMaxMeas)
        if echo "$OUT" | grep -qE "exceeds|too many|unsupported|limit"; then
            echo "SKIP (limit)"
            WARM_SKIP=$((WARM_SKIP + 1))
        else
            echo "FAIL (rc=$RC)"
            WARM_FAIL=$((WARM_FAIL + 1))
        fi
    fi
done

CACHED=$(ls ~/.clifft/kernel_cache/*.hsaco 2>/dev/null | wc -l)
echo ""
echo "Phase 1 done: OK=$WARM_OK SKIP=$WARM_SKIP FAIL=$WARM_FAIL cached_hsaco=$CACHED"

# ====================================================================
# Phase 2: Benchmark — warm cache, measure pure execution time
# ====================================================================
echo ""
echo "=== Phase 2: Execution-only benchmark ($RUNS runs x $SHOTS shots) ==="
echo ""

# CSV output for easy analysis
CSV="$RESULTS/bench_warm_all_$(date +%Y%m%d_%H%M%S).csv"
echo "circuit,category,peak_rank,svm_sps,compiled_sps,delta_pct,status" > "$CSV"

printf "%-40s %6s %12s %12s %8s %6s\n" "Circuit" "Rank" "SVM(sps)" "Compiled(sps)" "Delta" "Status"
echo "--------------------------------------------------------------------------------------------"

extract_field() { echo "$1" | grep "\"$2\"" | head -1 | grep -oE '[0-9]+\.?[0-9eE+\-]*' | head -1 || true; }

TOTAL_OK=0
TOTAL_SLOW=0
TOTAL_FAIL=0
TOTAL_SKIP=0

for CIRC in $CIRCUITS; do
    NAME=$(basename "$CIRC" .stim)
    DIR=$(basename "$(dirname "$CIRC")")

    # Quick check: can this circuit run at all?
    CHECK=$($BINARY --circuit "$CIRC" --shots 100 2>&1)
    CHECK_RC=$?
    if [ $CHECK_RC -ne 0 ]; then
        printf "%-40s %6s %12s %12s %8s %6s\n" "$NAME" "-" "-" "-" "-" "SKIP"
        echo "$NAME,$DIR,-,-,-,-,SKIP" >> "$CSV"
        TOTAL_SKIP=$((TOTAL_SKIP + 1))
        continue
    fi
    RANK=$(extract_field "$CHECK" "peak_rank")
    RANK="${RANK:-?}"

    # Run SVM (no --hybrid)
    SVM_SUM=0
    SVM_OK=true
    for i in $(seq 1 $RUNS); do
        OUT=$($BINARY --circuit "$CIRC" --shots $SHOTS 2>&1)
        SPS=$(extract_field "$OUT" "shots_per_second_sampling_only")
        SPS="${SPS:-0}"
        if [ "$SPS" = "0" ]; then SVM_OK=false; fi
        SVM_SUM=$(python3 -c "print(float('$SVM_SUM')+float('$SPS'))")
    done

    # Run Compiled (--hybrid, cache already warm)
    COMP_SUM=0
    COMP_OK=true
    for i in $(seq 1 $RUNS); do
        OUT=$($BINARY --circuit "$CIRC" --shots $SHOTS --hybrid 2>&1)
        SPS=$(extract_field "$OUT" "shots_per_second_sampling_only")
        SPS="${SPS:-0}"
        if [ "$SPS" = "0" ]; then COMP_OK=false; fi
        COMP_SUM=$(python3 -c "print(float('$COMP_SUM')+float('$SPS'))")
    done

    if ! $SVM_OK || ! $COMP_OK; then
        printf "%-40s %6s %12s %12s %8s %6s\n" "$NAME" "$RANK" "-" "-" "-" "FAIL"
        echo "$NAME,$DIR,$RANK,-,-,-,FAIL" >> "$CSV"
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
        continue
    fi

    SVM_AVG=$(python3 -c "print(f'{float(\"$SVM_SUM\")/$RUNS:.0f}')")
    COMP_AVG=$(python3 -c "print(f'{float(\"$COMP_SUM\")/$RUNS:.0f}')")
    DELTA=$(python3 -c "
s=float('$SVM_SUM'); c=float('$COMP_SUM')
print(f'{(c-s)/s*100:+.1f}' if s > 0 and c > 0 else 'N/A')
")
    STATUS=$(python3 -c "
s=float('$SVM_SUM'); c=float('$COMP_SUM')
r = c/s if s > 0 else 0
print('OK' if r > 0.85 else 'SLOW')
")

    SVM_M=$(python3 -c "print(f'{float(\"$SVM_AVG\")/1e6:.2f}M')")
    COMP_M=$(python3 -c "print(f'{float(\"$COMP_AVG\")/1e6:.2f}M')")

    printf "%-40s %6s %12s %12s %7s%% %6s\n" "$NAME" "$RANK" "$SVM_M" "$COMP_M" "$DELTA" "$STATUS"
    echo "$NAME,$DIR,$RANK,$SVM_AVG,$COMP_AVG,$DELTA,$STATUS" >> "$CSV"

    if [ "$STATUS" = "OK" ]; then
        TOTAL_OK=$((TOTAL_OK + 1))
    else
        TOTAL_SLOW=$((TOTAL_SLOW + 1))
    fi
done

echo ""
echo "=== Summary ==="
echo "OK (compiled >= 85% of SVM): $TOTAL_OK"
echo "SLOW (compiled < 85% of SVM): $TOTAL_SLOW"
echo "FAIL: $TOTAL_FAIL"
echo "SKIP: $TOTAL_SKIP"
echo ""
echo "CSV: $CSV"
echo "Done: $(date)"
