#!/bin/bash
#SBATCH --job-name=hybctl
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/hybctl_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BIN="$BASE/build-gpu-mlir-mi355x/run_gpu"
for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }; done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }; done
rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null

C5="$BASE/tests/fixtures/large/circuit_d5_p0.001.stim"
NL=/tmp/c5_nl.stim
grep -viE "DEPOLARIZE|X_ERROR|Z_ERROR|Y_ERROR|PAULI_CHANNEL" "$C5" > "$NL"
getobs(){ python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('observable_ones'))" 2>/dev/null; }

echo "=== NOISELESS circuit_d5: 3-way SVM vs HYBRID vs MLIR ==="
echo "(if HYBRID also diverges from SVM, the bug is NOT MLIR-specific)"
"$BIN" --circuit "$NL" --shots 10 --seed 42 --mlir --no-postselection >/dev/null 2>&1
"$BIN" --circuit "$NL" --shots 10 --seed 42 --hybrid --no-postselection >/dev/null 2>&1
for N in 1 2 4 8 16; do
  s=$("$BIN" --circuit "$NL" --shots $N --seed 42 --no-postselection 2>/dev/null | getobs)
  h=$("$BIN" --circuit "$NL" --shots $N --seed 42 --hybrid --no-postselection 2>/dev/null | getobs)
  m=$("$BIN" --circuit "$NL" --shots $N --seed 42 --mlir --no-postselection 2>/dev/null | getobs)
  echo "N=$N  SVM=$s  HYBRID=$h  MLIR=$m"
done

echo ""
echo "=== NOISELESS circuit_d5 WITH postselection, 3-way passed_shots ==="
getp(){ python3 -c "import json,sys;print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null; }
for N in 100 1000; do
  s=$("$BIN" --circuit "$NL" --shots $N --seed 42 2>/dev/null | getp)
  h=$("$BIN" --circuit "$NL" --shots $N --seed 42 --hybrid 2>/dev/null | getp)
  "$BIN" --circuit "$NL" --shots 10 --seed 42 --mlir >/dev/null 2>&1
  m=$("$BIN" --circuit "$NL" --shots $N --seed 42 --mlir 2>/dev/null | getp)
  echo "N=$N  SVM=$s  HYBRID=$h  MLIR=$m"
done
echo ""; echo "Done: $(date)"
