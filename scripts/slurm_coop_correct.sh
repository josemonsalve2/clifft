#!/bin/bash
#SBATCH --job-name=coopfix
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:40:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/coop_correct_%j.log

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
cd "$BUILD" && make -j$(nproc) 2>&1 | tail -3
[ ${PIPESTATUS[0]} -ne 0 ] && { echo "BUILD FAILED"; make -j1 2>&1 | grep error: | head -20; exit 1; }
echo "BUILD OK"; echo ""

rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null

# Coop correctness targets: qv10 (baseline OK), surface_d7_t10/t15 (were MLIR=0)
CIRCUITS=(
    "circuit_d3_p0.001|tests/fixtures/large/circuit_d3_p0.001.stim|180"
    "qv10|tests/fixtures/qv10.stim|180"
    "surface_d7_t10|tests/fixtures/large/surface_d7_t10.stim|600"
    "surface_d7_t15|tests/fixtures/large/surface_d7_t15.stim|600"
)

printf "%-18s %5s %8s %8s %10s %s\n" "CIRCUIT" "RANK" "SVM_p" "MLIR_p" "MLIR_ms" "STATUS"
printf "%-18s %5s %8s %8s %10s %s\n" "-------" "----" "-----" "------" "-------" "------"

for entry in "${CIRCUITS[@]}"; do
    IFS='|' read -r name stim timeo <<< "$entry"
    full="$BASE/$stim"
    [ -f "$full" ] || { echo "$name MISSING"; continue; }
    rank=$("$BIN" --circuit "$full" --cpu-reference --shots 1 --seed 42 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin)['peak_rank'])" 2>/dev/null)
    svm=$("$BIN" --circuit "$full" --shots $SHOTS --seed 42 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
    cf=$(mktemp)
    timeout "$timeo" "$BIN" --circuit "$full" --shots 10 --seed 42 --mlir >/dev/null 2>"$cf"; rc=$?
    cerr=$(grep -iE "error|dominate|FATAL" "$cf" | head -1); rm -f "$cf"
    if [ $rc -ne 0 ] || [ -n "$cerr" ]; then
        mp="FAIL"; mms="-"; [ $rc -eq 124 ] && mp="TIMEOUT"; [ -n "$cerr" ] && mp="CERR:$cerr"
    else
        mf=$(mktemp)
        mp=$("$BIN" --circuit "$full" --shots $SHOTS --seed 42 --mlir 2>"$mf" | python3 -c "import json,sys;print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
        mms=$(grep kernel_seconds "$mf" | grep -oP '[0-9.]+' | head -1); rm -f "$mf"
        mp=${mp:-PARSEFAIL}
    fi
    st="MISMATCH"; [ "$mp" = "$svm" ] && st="OK"
    printf "%-18s %5s %8s %8s %10s %s\n" "$name" "${rank:-?}" "${svm:-?}" "${mp:-?}" "${mms:-?}" "$st"
done
echo ""; echo "Done: $(date)"
