#!/bin/bash
#SBATCH --job-name=bisd5
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/bisd5_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BIN="$BASE/build-gpu-mlir-mi355x/run_gpu"
for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }; done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }; done
rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null

C5="$BASE/tests/fixtures/large/circuit_d5_p0.001.stim"
getobs(){ python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('observable_ones'))" 2>/dev/null; }

# Variant A: strip gate noise only (current 'noiseless') - keeps readout noise M(0.001)
A=/tmp/d5_A.stim
grep -viE "DEPOLARIZE|X_ERROR|Z_ERROR|Y_ERROR|PAULI_CHANNEL" "$C5" > "$A"
# Variant B: also strip readout noise probabilities M(0.001)->M, MX(0.001)->MX
B=/tmp/d5_B.stim
sed -E 's/\(0\.[0-9]+\)//g' "$A" > "$B"

echo "=== Variant A (gate noise stripped, readout noise KEPT) ==="
for N in 1 2 4 16; do
  s=$("$BIN" --circuit "$A" --shots $N --seed 42 --no-postselection 2>/dev/null | getobs)
  "$BIN" --circuit "$A" --shots 10 --seed 42 --mlir --no-postselection >/dev/null 2>&1
  m=$("$BIN" --circuit "$A" --shots $N --seed 42 --mlir --no-postselection 2>/dev/null | getobs)
  echo "N=$N  SVM=$s  MLIR=$m"
done
rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null
echo ""
echo "=== Variant B (gate AND readout noise fully stripped) ==="
for N in 1 2 4 16; do
  s=$("$BIN" --circuit "$B" --shots $N --seed 42 --no-postselection 2>/dev/null | getobs)
  "$BIN" --circuit "$B" --shots 10 --seed 42 --mlir --no-postselection >/dev/null 2>&1
  m=$("$BIN" --circuit "$B" --shots $N --seed 42 --mlir --no-postselection 2>/dev/null | getobs)
  echo "N=$N  SVM=$s  MLIR=$m"
done
echo ""; echo "Done: $(date)"
