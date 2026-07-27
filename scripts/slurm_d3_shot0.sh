#!/bin/bash
#SBATCH --job-name=d3shot0
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:12:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/d3_shot0_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BIN="$BASE/build-gpu-mlir-mi355x/run_gpu"
for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }; done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }; done
rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null

D3="$BASE/tests/fixtures/large/circuit_d3_p0.001.stim"
getobs(){ python3 -c "import json,sys;d=json.load(sys.stdin);print('obs=',d.get('observable_ones'),'passed=',d['passed_shots'])" 2>/dev/null; }

echo "=== single-shot determinism (--no-postselection, 1 shot each) ==="
for N in 1 2 4 8 16; do
  s=$("$BIN" --circuit "$D3" --shots $N --seed 42 --no-postselection 2>/dev/null | getobs)
  m=$("$BIN" --circuit "$D3" --shots $N --seed 42 --mlir --no-postselection 2>/dev/null | getobs)
  echo "N=$N  SVM: $s | MLIR: $m"
done

echo ""
echo "=== NOISELESS circuit_d3, small N ==="
NL=/tmp/d3_nl.stim
grep -viE "DEPOLARIZE|X_ERROR|Z_ERROR|Y_ERROR|PAULI_CHANNEL" "$D3" > "$NL"
for N in 1 2 4 8; do
  s=$("$BIN" --circuit "$NL" --shots $N --seed 42 --no-postselection 2>/dev/null | getobs)
  "$BIN" --circuit "$NL" --shots 1 --seed 42 --mlir --no-postselection >/dev/null 2>&1
  m=$("$BIN" --circuit "$NL" --shots $N --seed 42 --mlir --no-postselection 2>/dev/null | getobs)
  echo "N=$N  SVM: $s | MLIR: $m"
done
echo ""
echo "Done: $(date)"
