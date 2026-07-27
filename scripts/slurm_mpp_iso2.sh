#!/bin/bash
#SBATCH --job-name=mppiso2
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/mppiso2_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BIN="$BASE/build-gpu-mlir-mi355x/run_gpu"
for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }; done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }; done

# GHZ Z-parity MPP: rank stays ~1 after squeeze (all stabilizers). Force higher rank
# with H on many qubits + MPP mixing bases so peak_rank is high enough for coop.
cat > /tmp/mpp_hi.stim <<'EOF'
R 0 1 2 3 4 5 6 7 8 9 10 11
H 0 1 2 3 4 5 6 7 8 9 10 11
T 0 1 2 3 4 5 6 7 8 9 10 11
MPP X0*X1*X2*X3
MPP X4*X5*X6*X7
MPP X8*X9*X10*X11
MPP Y0*Y1*Y2*Y3*Y4*Y5*Y6*Y7*Y8*Y9*Y10*Y11
OBSERVABLE_INCLUDE(0) rec[-1]
EOF

C=/tmp/mpp_hi.stim
echo "=== $C ==="
echo "--- CPU reference (rank probe) ---"
"$BIN" --circuit "$C" --shots 4 --seed 42 --cpu-reference 2>&1 | python3 -c "import json,sys;d=json.load(sys.stdin);print('rank=',d.get('peak_rank'),'obs=',d.get('observable_ones'))" 2>&1 | head
echo "--- MLIR compile attempt (capture error) ---"
rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null
"$BIN" --circuit "$C" --shots 10 --seed 42 --mlir --no-postselection 2>&1 | grep -iE "error|abort|fatal|rank|assert|discard" | head
getobs(){ python3 -c "import json,sys;d=json.load(sys.stdin);print('obs=',d.get('observable_ones'))" 2>/dev/null; }
for N in 1 4 16; do
  s=$("$BIN" --circuit "$C" --shots $N --seed 42 --no-postselection 2>/dev/null | getobs)
  h=$("$BIN" --circuit "$C" --shots $N --seed 42 --hybrid --no-postselection 2>/dev/null | getobs)
  m=$("$BIN" --circuit "$C" --shots $N --seed 42 --mlir --no-postselection 2>/dev/null | getobs)
  echo "N=$N  SVM: $s | HYB: $h | MLIR: $m"
done
echo "Done: $(date)"
