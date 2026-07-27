#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/baseline_all_%j.log

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

echo "=== BUILD ==="
cd "$BUILD"
make -j$(nproc) 2>&1 | tail -3
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "BUILD FAILED"
    make -j1 2>&1 | grep -E "error:" | head -20
    exit 1
fi
echo "BUILD OK"
echo ""

rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null

# Format: name | stim path | mlir_compile_timeout
# Ordered by expected compile time (fast first)
# qv20 excluded: peak_rank=20 exceeds kGlobalMaxPeakRank=19 (hard cap for ALL
# backends incl SVM/Hybrid, not an MLIR bug). Focus: coop (5-10) + global (11-19).
CIRCUITS=(
    "qv10|tests/fixtures/qv10.stim|180"
    "cultivation_d5|tests/fixtures/cultivation_d5.stim|600"
    "circuit_d5_p0.001|tests/fixtures/large/circuit_d5_p0.001.stim|600"
    "surface_d7_t10|tests/fixtures/large/surface_d7_t10.stim|600"
    "surface_d7_t15|tests/fixtures/large/surface_d7_t15.stim|600"
    "surface_d9_t10|tests/fixtures/large/surface_d9_t10.stim|900"
    "surface_d7_t19|tests/fixtures/large/surface_d7_t19.stim|600"
    "surface_d9_t15|tests/fixtures/large/surface_d9_t15.stim|900"
    "surface_d9_t19|tests/fixtures/large/surface_d9_t19.stim|900"
    "surface_d11_t15|tests/fixtures/large/surface_d11_t15.stim|1200"
    "surface_d11_t19|tests/fixtures/large/surface_d11_t19.stim|1200"
)

printf "%-22s %5s %5s %8s %8s %10s %10s %10s %s\n" \
    "CIRCUIT" "RANK" "TIER" "SVM_p" "MLIR_p" "SVM_ms" "MLIR_ms" "HYB_ms" "STATUS"
printf "%-22s %5s %5s %8s %8s %10s %10s %10s %s\n" \
    "-------" "----" "----" "-----" "------" "------" "-------" "------" "------"

run_json() {  # circuit, extra_flags, stderr_file
    "$BIN" --circuit "$1" --shots $SHOTS --seed 42 $2 2>"$3"
}

for entry in "${CIRCUITS[@]}"; do
    IFS='|' read -r name stim timeo <<< "$entry"
    full="$BASE/$stim"
    [ -f "$full" ] || { printf "%-22s  MISSING\n" "$name"; continue; }

    # Rank/tier
    rank=$("$BIN" --circuit "$full" --cpu-reference --shots 1 --seed 42 2>/dev/null | \
        python3 -c "import json,sys;print(json.load(sys.stdin)['peak_rank'])" 2>/dev/null)
    rank=${rank:-?}
    if [ "$rank" = "?" ]; then tier="?"; elif [ "$rank" -le 4 ]; then tier="REG"; elif [ "$rank" -le 10 ]; then tier="COOP"; else tier="GLOB"; fi

    # SVM
    sf=$(mktemp)
    svm_json=$(run_json "$full" "" "$sf")
    svm_p=$(echo "$svm_json" | python3 -c "import json,sys;print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
    svm_ms=$(grep kernel_seconds "$sf" | grep -oP '[0-9.]+' | head -1)
    rm -f "$sf"

    # MLIR: warmup compile (cached), then timed run
    cf=$(mktemp)
    timeout "$timeo" "$BIN" --circuit "$full" --shots 10 --seed 42 --mlir >/dev/null 2>"$cf"
    compile_rc=$?
    compile_err=$(grep -iE "error|does not dominate|FATAL" "$cf" | head -1)
    rm -f "$cf"

    if [ $compile_rc -ne 0 ] || [ -n "$compile_err" ]; then
        mlir_p="FAIL"; mlir_ms="-"
        [ $compile_rc -eq 124 ] && { mlir_p="TIMEOUT"; }
        [ -n "$compile_err" ] && mlir_p="CERR"
    else
        mf=$(mktemp)
        mlir_json=$("$BIN" --circuit "$full" --shots $SHOTS --seed 42 --mlir 2>"$mf")
        mlir_p=$(echo "$mlir_json" | python3 -c "import json,sys;print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
        mlir_ms=$(grep kernel_seconds "$mf" | grep -oP '[0-9.]+' | head -1)
        rm -f "$mf"
        mlir_p=${mlir_p:-PARSEFAIL}
    fi

    # Hybrid
    hf=$(mktemp)
    timeout 180 "$BIN" --circuit "$full" --shots 10 --seed 42 --hybrid >/dev/null 2>"$hf"
    hyb_rc=$?
    if [ $hyb_rc -eq 0 ]; then
        hyb_json=$("$BIN" --circuit "$full" --shots $SHOTS --seed 42 --hybrid 2>"$hf")
        hyb_ms=$(grep kernel_seconds "$hf" | grep -oP '[0-9.]+' | head -1)
    else
        hyb_ms="-"
    fi
    rm -f "$hf"

    # Status
    if [ "$mlir_p" = "$svm_p" ]; then status="OK"
    elif [ "$mlir_p" = "TIMEOUT" ]; then status="MLIR_TIMEOUT"
    elif [ "$mlir_p" = "CERR" ]; then status="MLIR_COMPILE_ERR"
    else status="MISMATCH"; fi

    printf "%-22s %5s %5s %8s %8s %10s %10s %10s %s\n" \
        "$name" "$rank" "$tier" "${svm_p:-?}" "${mlir_p:-?}" \
        "${svm_ms:-?}" "${mlir_ms:-?}" "${hyb_ms:-?}" "$status"
done

echo ""
echo "Done: $(date)"
