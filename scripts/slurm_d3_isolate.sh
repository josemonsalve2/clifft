#!/bin/bash
#SBATCH --job-name=d3iso
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/d3_isolate_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BIN="$BASE/build-gpu-mlir-mi355x/run_gpu"
for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }; done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }; done
rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null

D3="$BASE/tests/fixtures/large/circuit_d3_p0.001.stim"
NOISELESS=/tmp/circuit_d3_noiseless.stim
grep -viE "DEPOLARIZE|X_ERROR|Z_ERROR|Y_ERROR|PAULI_CHANNEL" "$D3" > "$NOISELESS"

getp(){ python3 -c "import json,sys;d=json.load(sys.stdin);print(d['passed_shots'])" 2>/dev/null; }

echo "=== circuit_d3 WITH postselection (default 'all') ==="
svm=$("$BIN" --circuit "$D3" --shots 2000 --seed 42 2>/dev/null | getp)
"$BIN" --circuit "$D3" --shots 10 --seed 42 --mlir >/dev/null 2>&1
ml=$("$BIN" --circuit "$D3" --shots 2000 --seed 42 --mlir 2>/dev/null | getp)
echo "SVM=$svm MLIR=$ml $([ "$svm" = "$ml" ] && echo OK || echo MISMATCH)"

echo ""
echo "=== circuit_d3 NO postselection ==="
svm=$("$BIN" --circuit "$D3" --shots 2000 --seed 42 --no-postselection 2>/dev/null | getp)
ml=$("$BIN" --circuit "$D3" --shots 2000 --seed 42 --mlir --no-postselection 2>/dev/null | getp)
echo "SVM=$svm MLIR=$ml $([ "$svm" = "$ml" ] && echo OK || echo MISMATCH)"

echo ""
echo "=== circuit_d3 NOISELESS + postselection ==="
svm=$("$BIN" --circuit "$NOISELESS" --shots 2000 --seed 42 2>/dev/null | getp)
"$BIN" --circuit "$NOISELESS" --shots 10 --seed 42 --mlir >/dev/null 2>&1
ml=$("$BIN" --circuit "$NOISELESS" --shots 2000 --seed 42 --mlir 2>/dev/null | getp)
echo "SVM=$svm MLIR=$ml $([ "$svm" = "$ml" ] && echo OK || echo MISMATCH)"

echo ""
echo "=== circuit_d3 NOISELESS + NO postselection ==="
svm=$("$BIN" --circuit "$NOISELESS" --shots 2000 --seed 42 --no-postselection 2>/dev/null | getp)
ml=$("$BIN" --circuit "$NOISELESS" --shots 2000 --seed 42 --mlir --no-postselection 2>/dev/null | getp)
echo "SVM=$svm MLIR=$ml $([ "$svm" = "$ml" ] && echo OK || echo MISMATCH)"

echo ""
echo "Done: $(date)"
