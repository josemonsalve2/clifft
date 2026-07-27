#!/bin/bash
#SBATCH --job-name=noiseiso
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/noise_isolate_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BIN="$BASE/build-gpu-mlir-mi355x/run_gpu"
for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }; done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }; done
rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null

# Generate a NOISELESS variant of surface_d7_t10 by stripping noise lines
NOISELESS=/tmp/surface_d7_t10_noiseless.stim
grep -viE "DEPOLARIZE|X_ERROR|Z_ERROR|Y_ERROR|PAULI_CHANNEL" "$BASE/tests/fixtures/large/surface_d7_t10.stim" > "$NOISELESS"
echo "orig lines: $(wc -l < $BASE/tests/fixtures/large/surface_d7_t10.stim), noiseless: $(wc -l < $NOISELESS)"
echo ""

for label in NOISY NOISELESS; do
    [ "$label" = NOISY ] && stim="$BASE/tests/fixtures/large/surface_d7_t10.stim" || stim="$NOISELESS"
    echo "=== $label ($stim) ==="
    svm=$("$BIN" --circuit "$stim" --shots 2000 --seed 42 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
    "$BIN" --circuit "$stim" --shots 10 --seed 42 --mlir >/dev/null 2>&1
    ml=$("$BIN" --circuit "$stim" --shots 2000 --seed 42 --mlir 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
    echo "SVM=$svm  MLIR=$ml  $([ "$svm" = "$ml" ] && echo OK || echo MISMATCH)"
    echo ""
done
echo "Done: $(date)"
