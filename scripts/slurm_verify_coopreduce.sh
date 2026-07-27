#!/bin/bash
#SBATCH --job-name=coopred
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:50:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/coopred_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BUILD="$BASE/build-gpu-mlir-mi355x"
BIN="$BUILD/run_gpu"
for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }; done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }; done

echo "=== BUILD (Codex coop-reduce port) ==="
cd "$BUILD" && make -j$(nproc) 2>&1 | tail -3
if [ ${PIPESTATUS[0]} -ne 0 ]; then
  echo "BUILD_FAIL — errors:"; make -j1 2>&1 | grep -iE "error:" | head -20; exit 1
fi
echo "BUILD OK"; echo ""
rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null

getobs(){ python3 -c "import json,sys;d=json.load(sys.stdin);print('obs=',d.get('observable_ones'),'passed=',d['passed_shots'])" 2>/dev/null; }
getp(){ python3 -c "import json,sys;print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null; }

C5="$BASE/tests/fixtures/large/circuit_d5_p0.001.stim"
NL=/tmp/c5_nl.stim
grep -viE "DEPOLARIZE|X_ERROR|Z_ERROR|Y_ERROR|PAULI_CHANNEL" "$C5" > "$NL"

echo "=== circuit_d5 NOISELESS coop determinism (isolates coop summation bug) ==="
"$BIN" --circuit "$NL" --shots 10 --seed 42 --mlir --no-postselection >/dev/null 2>&1
for N in 1 2 4 16; do
  s=$("$BIN" --circuit "$NL" --shots $N --seed 42 --no-postselection 2>/dev/null | getobs)
  m=$("$BIN" --circuit "$NL" --shots $N --seed 42 --mlir --no-postselection 2>/dev/null | getobs)
  echo "N=$N  SVM: $s | MLIR: $m"
done

echo ""
echo "=== full correctness (10000 shots, seed 42) ==="
printf "%-20s %8s %8s %s\n" CIRCUIT SVM MLIR STATUS
for entry in \
  "circuit_d3_p0.001|tests/fixtures/large/circuit_d3_p0.001.stim|180" \
  "qv10|tests/fixtures/qv10.stim|180" \
  "cultivation_d5|tests/fixtures/large/cultivation_d5.stim|300" \
  "circuit_d5_p0.001|tests/fixtures/large/circuit_d5_p0.001.stim|700" \
  "surface_d7_t10|tests/fixtures/large/surface_d7_t10.stim|700" \
  "surface_d7_t15|tests/fixtures/large/surface_d7_t15.stim|700"; do
    IFS='|' read -r name stim timeo <<< "$entry"
    full="$BASE/$stim"
    [ -f "$full" ] || { printf "%-20s %8s %8s %s\n" "$name" MISSING - -; continue; }
    svm=$("$BIN" --circuit "$full" --shots 10000 --seed 42 2>/dev/null | getp)
    timeout "$timeo" "$BIN" --circuit "$full" --shots 10 --seed 42 --mlir >/dev/null 2>&1
    ml=$("$BIN" --circuit "$full" --shots 10000 --seed 42 --mlir 2>/dev/null | getp)
    printf "%-20s %8s %8s %s\n" "$name" "${svm:-?}" "${ml:-?}" "$([ "$svm" = "$ml" ] && echo OK || echo MISMATCH)"
done
echo ""; echo "Done: $(date)"
