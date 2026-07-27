#!/bin/bash
#SBATCH --job-name=ffiso
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/ffiso_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BIN="$BASE/build-gpu-mlir-mi355x/run_gpu"
for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }; done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }; done
getobs(){ python3 -c "import json,sys;d=json.load(sys.stdin);print('obs=',d.get('observable_ones'),'rank=',d.get('peak_rank'))" 2>/dev/null; }

C=/tmp/mr_ff.stim
echo "=== $C (MR feedforward -> APPLY_PAULI, rank ~10) ==="
rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null
"$BIN" --circuit "$C" --shots 10 --seed 42 --mlir --no-postselection >/dev/null 2>&1
for N in 1 2 4 16 64; do
  s=$("$BIN" --circuit "$C" --shots $N --seed 42 --no-postselection 2>/dev/null | getobs)
  h=$("$BIN" --circuit "$C" --shots $N --seed 42 --hybrid --no-postselection 2>/dev/null | getobs)
  m=$("$BIN" --circuit "$C" --shots $N --seed 42 --mlir --no-postselection 2>/dev/null | getobs)
  echo "N=$N  SVM: $s | HYB: $h | MLIR: $m"
done
echo "Done: $(date)"
