#!/bin/bash
#SBATCH --job-name=perf-compiled
#SBATCH --partition=mi300x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:45:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/perf_compiled_%j.log
set +e

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
export PATH="/shared/jmonsalv/software/cmake-3.31.7-linux-x86_64/bin:$PATH"
for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done

echo "=== Compiled Kernel Performance (with warmup) ==="
echo "Node: $(hostname)  GPU: $(rocminfo 2>/dev/null | grep -m1 'Name:.*gfx' | grep -oE 'gfx[0-9]+')"

# Build
BUILD="$BASE/build-gpu-perf"
[ -x "$BUILD/run_gpu" ] || {
    rm -rf "$BUILD" && mkdir -p "$BUILD" && cd "$BUILD"
    cmake "$BASE" -DCLIFFT_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=$(rocminfo 2>/dev/null | grep -m1 'Name:.*gfx' | grep -oE 'gfx[0-9]+') -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
    make -j$(nproc) run_gpu 2>&1 | tail -3
    cd "$BASE"
}
BINARY="$BUILD/run_gpu"
[ -x "$BINARY" ] || { echo "BUILD FAILED"; exit 1; }

# Clear cache to start fresh
rm -rf ~/.clifft/kernel_cache/*.hsaco*
SHOTS=1000000
RUNS=5

echo ""
echo "=== Phase 1: Warm up kernel cache (compile all circuits once) ==="
for CIRC in "$BASE/tests/fixtures/incremental"/{01,02,03,04,05,06}_*/*.stim; do
    [ -f "$CIRC" ] || continue
    NAME=$(basename "$CIRC" .stim)
    printf "Warming: %-24s " "$NAME"
    $BINARY --circuit "$CIRC" --shots 100 --hybrid >/dev/null 2>&1
    echo "done"
done
echo ""
echo "Cached .hsaco files: $(ls ~/.clifft/kernel_cache/*.hsaco 2>/dev/null | wc -l)"

echo ""
echo "=== Phase 2: Benchmark (cache warm, $RUNS runs x $SHOTS shots) ==="
echo ""
printf "%-24s %12s %12s %8s %6s\n" "Circuit" "SVM" "Compiled" "Delta" "OK?"
echo "------------------------------------------------------------------------"

extract_field() { echo "$1" | grep "\"$2\"" | head -1 | grep -oE '[0-9]+\.?[0-9eE+\-]*' || true; }

ALL_OK=true
for CIRC in "$BASE/tests/fixtures/incremental"/{01,02,03,04,05,06}_*/*.stim; do
    [ -f "$CIRC" ] || continue
    NAME=$(basename "$CIRC" .stim)

    SVM_SUM=0; COMP_SUM=0
    SVM_OK=true; COMP_OK=true

    for i in $(seq 1 $RUNS); do
        OUT=$($BINARY --circuit "$CIRC" --shots $SHOTS 2>&1)
        SPS=$(extract_field "$OUT" "shots_per_second_sampling_only"); SPS="${SPS:-0}"
        SVM_SUM=$(python3 -c "print(float('$SVM_SUM')+float('$SPS'))")

        OUT=$($BINARY --circuit "$CIRC" --shots $SHOTS --hybrid 2>&1)
        SPS=$(extract_field "$OUT" "shots_per_second_sampling_only"); SPS="${SPS:-0}"
        COMP_SUM=$(python3 -c "print(float('$COMP_SUM')+float('$SPS'))")
    done

    SVM_AVG=$(python3 -c "print(f'{float(\"$SVM_SUM\")/$RUNS/1e6:.2f}')")
    COMP_AVG=$(python3 -c "print(f'{float(\"$COMP_SUM\")/$RUNS/1e6:.2f}')")
    DELTA=$(python3 -c "
s=float('$SVM_SUM'); c=float('$COMP_SUM')
print(f'{(c-s)/s*100:+.1f}%' if s > 0 and c > 0 else 'N/A')
")
    # Check if within 15% of SVM (allowing some overhead)
    RATIO=$(python3 -c "
s=float('$SVM_SUM'); c=float('$COMP_SUM')
r = c/s if s > 0 else 0
print('OK' if r > 0.85 else 'SLOW')
")
    [ "$RATIO" = "SLOW" ] && ALL_OK=false

    printf "%-24s %10sM %10sM %8s %6s\n" "$NAME" "$SVM_AVG" "$COMP_AVG" "$DELTA" "$RATIO"
done

echo ""
if $ALL_OK; then
    echo "ALL CIRCUITS: COMPILED KERNEL WITHIN 15% OF SVM"
else
    echo "SOME CIRCUITS STILL SLOW — check for compilation failures"
fi

# Also run the interesting circuits
echo ""
echo "=== Phase 3: Interesting circuits ==="
for CIRC in \
    "$BASE/tests/fixtures/target_qec.stim" \
    "$BASE/tests/fixtures/cultivation_d5.stim" \
    "$BASE/tests/fixtures/large/surface_d7_t5.stim" \
    "$BASE/tests/fixtures/large/surface_d7_t10.stim" \
    "$BASE/tests/fixtures/large/color_d7_r7.stim" \
    "$BASE/tests/fixtures/large/rep_d5_r100.stim"; do
    [ -f "$CIRC" ] || continue
    NAME=$(basename "$CIRC" .stim)
    # Warmup (compiles kernel)
    $BINARY --circuit "$CIRC" --shots 100 --hybrid >/dev/null 2>&1
    SVM_SUM=0; COMP_SUM=0
    for i in $(seq 1 $RUNS); do
        OUT=$($BINARY --circuit "$CIRC" --shots $SHOTS 2>&1)
        SPS=$(extract_field "$OUT" "shots_per_second_sampling_only"); SPS="${SPS:-0}"
        SVM_SUM=$(python3 -c "print(float('$SVM_SUM')+float('$SPS'))")
        OUT=$($BINARY --circuit "$CIRC" --shots $SHOTS --hybrid 2>&1)
        SPS=$(extract_field "$OUT" "shots_per_second_sampling_only"); SPS="${SPS:-0}"
        COMP_SUM=$(python3 -c "print(float('$COMP_SUM')+float('$SPS'))")
    done
    SVM_AVG=$(python3 -c "print(f'{float(\"$SVM_SUM\")/$RUNS/1e6:.2f}')")
    COMP_AVG=$(python3 -c "print(f'{float(\"$COMP_SUM\")/$RUNS/1e6:.2f}')")
    DELTA=$(python3 -c "s=float('$SVM_SUM');c=float('$COMP_SUM');print(f'{(c-s)/s*100:+.1f}%' if s>0 and c>0 else 'N/A')")
    printf "%-24s %10sM %10sM %8s\n" "$NAME" "$SVM_AVG" "$COMP_AVG" "$DELTA"
done

echo ""
echo "Done: $(date)"
