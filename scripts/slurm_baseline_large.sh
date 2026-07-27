#!/bin/bash
#SBATCH --job-name=base-lg
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/baseline_large_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BUILD="$BASE/build-gpu-mlir-mi355x"
BIN="$BUILD/run_gpu"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

echo "=== BASELINE: Large QEC Circuits ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

cd "$BUILD"
make -j$(nproc) 2>&1 | tail -3
echo ""

bench() {
    local stim="$1" mode="$2" shots="$3"
    local args="--circuit $stim --shots $shots --seed 42"
    [ "$mode" = "mlir" ] && args="$args --mlir"
    [ "$mode" = "hybrid" ] && args="$args --hybrid"
    timeout 300 $BIN $args 2>/dev/null | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print(f\"{d['sample_seconds']:.6f}\")
except:
    print('FAIL')
" 2>/dev/null
}

echo "=== Large Circuit Performance (10k shots) ==="
printf "%-40s %5s %4s %5s %10s %10s %10s %8s\n" \
    "Circuit" "Qbits" "Rank" "Lines" "SVM(s)" "MLIR(s)" "Hybrid(s)" "MLIR/SVM"
printf "%-40s %5s %4s %5s %10s %10s %10s %8s\n" \
    "---" "---" "---" "---" "---" "---" "---" "---"

for stim in "$BASE"/tests/fixtures/large/surface_d5_r5.stim \
            "$BASE"/tests/fixtures/large/surface_d5_r10.stim \
            "$BASE"/tests/fixtures/large/color_d3.stim \
            "$BASE"/tests/fixtures/large/surface_d7_r7.stim \
            "$BASE"/tests/fixtures/large/surface_d7_r21.stim \
            "$BASE"/tests/fixtures/large/surface_d7_t19.stim \
            "$BASE"/tests/fixtures/large/color_d9_r9.stim \
            "$BASE"/tests/fixtures/large/color_d9_r18.stim \
            "$BASE"/tests/fixtures/large/cultivation_d7_p0001.stim \
            "$BASE"/tests/fixtures/large/cultivation_d7_p0005.stim \
            "$BASE"/tests/fixtures/large/rank15_q100_d5_wide.stim \
            "$BASE"/tests/fixtures/large/rank17_q100_d8_wide.stim \
            "$BASE"/tests/fixtures/large/rank19_q100_d10_wide.stim \
            "$BASE"/tests/fixtures/target_qec.stim \
            "$BASE"/tests/fixtures/sweep/sweep_q65_t10_d3.stim \
            "$BASE"/tests/fixtures/sweep/sweep_q65_t15_d3.stim; do
    [ -f "$stim" ] || continue
    name=$(basename "$stim" .stim)
    nq=$(grep -c "QUBIT_COORDS" "$stim" 2>/dev/null)
    ni=$(wc -l < "$stim" 2>/dev/null)
    # Get rank from first SVM run
    rank=$(timeout 30 $BIN --circuit "$stim" --shots 10 --seed 42 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin).get('peak_rank','?'))" 2>/dev/null)

    svm_t=$(bench "$stim" svm 10000)
    mlir_t=$(bench "$stim" mlir 10000)
    hyb_t=$(bench "$stim" hybrid 10000)

    ratio="-"
    if [ "$svm_t" != "FAIL" ] && [ "$mlir_t" != "FAIL" ] && [ -n "$svm_t" ] && [ -n "$mlir_t" ]; then
        ratio=$(python3 -c "print(f'{float(\"$mlir_t\")/float(\"$svm_t\"):.2f}x')" 2>/dev/null)
    fi

    printf "%-40s %5s %4s %5s %10s %10s %10s %8s\n" \
        "$name" "$nq" "$rank" "$ni" "${svm_t:-FAIL}s" "${mlir_t:-FAIL}s" "${hyb_t:-FAIL}s" "$ratio"
done

echo ""
echo "=== Memory-Pushing Test (rank19, 100 qubits) ==="
echo "Memory per block: 4MB, 512 blocks = 2GB HBM"
stim="$BASE/tests/fixtures/large/rank19_q100_d10_wide.stim"
if [ -f "$stim" ]; then
    echo "--- SVM 1000 shots ---"
    timeout 120 $BIN --circuit "$stim" --shots 1000 --seed 42 2>&1 | grep -E "sample_seconds|peak_rank|shots_per_second"
    echo "--- MLIR 1000 shots ---"
    timeout 300 $BIN --circuit "$stim" --shots 1000 --seed 42 --mlir 2>&1 | grep -E "sample_seconds|peak_rank|shots_per_second|FATAL|error"
fi

echo ""
echo "Done: $(date)"
