#!/bin/bash
#SBATCH --job-name=dustfix
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:40:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/dust_fix_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BUILD="$BASE/build-gpu-mlir-mi355x"
BIN="$BUILD/run_gpu"
for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }; done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }; done

echo "=== BUILD ==="
cd "$BUILD" && make -j$(nproc) 2>&1 | tail -3
[ ${PIPESTATUS[0]} -ne 0 ] && { echo BUILD_FAIL; make -j1 2>&1 | grep error: | head -20; exit 1; }
echo "BUILD OK"; echo ""
rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null

getp(){ python3 -c "import json,sys;d=json.load(sys.stdin);print(d['passed_shots'])" 2>/dev/null; }
getobs(){ python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('observable_ones'))" 2>/dev/null; }

D3="$BASE/tests/fixtures/large/circuit_d3_p0.001.stim"
echo "=== circuit_d3 observable check (--no-postselection, 2000 shots) ==="
echo -n "SVM  obs="; "$BIN" --circuit "$D3" --shots 2000 --seed 42 --no-postselection 2>/dev/null | getobs
"$BIN" --circuit "$D3" --shots 10 --seed 42 --mlir --no-postselection >/dev/null 2>&1
echo -n "MLIR obs="; "$BIN" --circuit "$D3" --shots 2000 --seed 42 --mlir --no-postselection 2>/dev/null | getobs
echo ""

echo "=== correctness: passed_shots MLIR vs SVM (10000 shots) ==="
printf "%-20s %5s %8s %8s %s\n" CIRCUIT RANK SVM MLIR STATUS
for entry in "circuit_d3_p0.001|$D3|180" "qv10|$BASE/tests/fixtures/qv10.stim|180" "surface_d7_t10|$BASE/tests/fixtures/large/surface_d7_t10.stim|600" "surface_d7_t15|$BASE/tests/fixtures/large/surface_d7_t15.stim|600"; do
    IFS='|' read -r name stim timeo <<< "$entry"
    rank=$("$BIN" --circuit "$stim" --cpu-reference --shots 1 --seed 42 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin)['peak_rank'])" 2>/dev/null)
    svm=$("$BIN" --circuit "$stim" --shots 10000 --seed 42 2>/dev/null | getp)
    timeout "$timeo" "$BIN" --circuit "$stim" --shots 10 --seed 42 --mlir >/dev/null 2>&1
    ml=$("$BIN" --circuit "$stim" --shots 10000 --seed 42 --mlir 2>/dev/null | getp)
    printf "%-20s %5s %8s %8s %s\n" "$name" "${rank:-?}" "${svm:-?}" "${ml:-?}" "$([ "$svm" = "$ml" ] && echo OK || echo MISMATCH)"
done
echo ""; echo "Done: $(date)"
