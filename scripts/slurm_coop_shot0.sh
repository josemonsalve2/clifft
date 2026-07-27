#!/bin/bash
#SBATCH --job-name=coop0
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/coop_shot0_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BIN="$BASE/build-gpu-mlir-mi355x/run_gpu"
for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }; done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }; done
rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null

C5="$BASE/tests/fixtures/large/circuit_d5_p0.001.stim"
NL=/tmp/c5_nl.stim
grep -viE "DEPOLARIZE|X_ERROR|Z_ERROR|Y_ERROR|PAULI_CHANNEL" "$C5" > "$NL"
getobs(){ python3 -c "import json,sys;d=json.load(sys.stdin);print('obs=',d.get('observable_ones'),'passed=',d['passed_shots'])" 2>/dev/null; }

echo "=== circuit_d5 (COOP rank 10) shot-0 determinism, --no-postselection ==="
echo "--- NOISELESS (isolates deterministic coop bug) ---"
"$BIN" --circuit "$NL" --shots 10 --seed 42 --mlir --no-postselection >/dev/null 2>&1
for N in 1 2 4; do
  s=$("$BIN" --circuit "$NL" --shots $N --seed 42 --no-postselection 2>/dev/null | getobs)
  m=$("$BIN" --circuit "$NL" --shots $N --seed 42 --mlir --no-postselection 2>/dev/null | getobs)
  echo "N=$N noiseless  SVM: $s | MLIR: $m"
done
echo "--- WITH NOISE ---"
"$BIN" --circuit "$C5" --shots 10 --seed 42 --mlir --no-postselection >/dev/null 2>&1
for N in 1 2 4; do
  s=$("$BIN" --circuit "$C5" --shots $N --seed 42 --no-postselection 2>/dev/null | getobs)
  m=$("$BIN" --circuit "$C5" --shots $N --seed 42 --mlir --no-postselection 2>/dev/null | getobs)
  echo "N=$N noisy  SVM: $s | MLIR: $m"
done
echo ""; echo "Done: $(date)"
