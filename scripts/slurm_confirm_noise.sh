#!/bin/bash
#SBATCH --job-name=confnoise
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:45:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/confirm_noise_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BUILD="$BASE/build-gpu-mlir-mi355x"
BIN="$BUILD/run_gpu"
SHOTS=10000
for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }; done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }; done

# Ensure current source is built (Codex edited it)
cd "$BUILD" && make -j$(nproc) 2>&1 | tail -2
[ ${PIPESTATUS[0]} -ne 0 ] && { echo BUILD_FAIL; make -j1 2>&1 | grep error: | head; exit 1; }
echo "BUILD OK"; echo ""
rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null

getp(){ python3 -c "import json,sys;print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null; }
printf "%-20s %5s %8s %8s %s\n" CIRCUIT RANK SVM MLIR STATUS
for entry in \
  "circuit_d3_p0.001|tests/fixtures/large/circuit_d3_p0.001.stim|180" \
  "qv10|tests/fixtures/qv10.stim|180" \
  "circuit_d5_p0.001|tests/fixtures/large/circuit_d5_p0.001.stim|700" \
  "surface_d7_t10|tests/fixtures/large/surface_d7_t10.stim|700" \
  "surface_d7_t15|tests/fixtures/large/surface_d7_t15.stim|700"; do
    IFS='|' read -r name stim timeo <<< "$entry"
    full="$BASE/$stim"
    rank=$("$BIN" --circuit "$full" --cpu-reference --shots 1 --seed 42 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin)['peak_rank'])" 2>/dev/null)
    svm=$("$BIN" --circuit "$full" --shots $SHOTS --seed 42 2>/dev/null | getp)
    timeout "$timeo" "$BIN" --circuit "$full" --shots 10 --seed 42 --mlir >/dev/null 2>&1
    ml=$("$BIN" --circuit "$full" --shots $SHOTS --seed 42 --mlir 2>/dev/null | getp)
    printf "%-20s %5s %8s %8s %s\n" "$name" "${rank:-?}" "${svm:-?}" "${ml:-?}" "$([ "$svm" = "$ml" ] && echo OK || echo MISMATCH)"
done
echo ""; echo "Done: $(date)"
