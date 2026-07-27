#!/bin/bash
#SBATCH --job-name=ffiso2
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/ffiso2_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BIN="$BASE/build-gpu-mlir-mi355x/run_gpu"
for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }; done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }; done
getobs(){ python3 -c "import json,sys;d=json.load(sys.stdin);print('obs=',d.get('observable_ones'),'rank=',d.get('peak_rank'))" 2>/dev/null; }

# MR feedforward test (creates file ON the compute node)
FF=/tmp/mr_ff_$$.stim
cat > "$FF" <<'EOF'
R 0 1 2 3 4 5 6 7 8 9 10 11
H 0 1 2 3 4 5 6 7 8 9 10 11
T 0 1 2 3 4 5 6 7 8 9 10 11
MR 3 7
CX 3 0 7 4
MPP Y0*Y1*Y2*Y3*Y4*Y5*Y6*Y7*Y8*Y9*Y10*Y11
OBSERVABLE_INCLUDE(0) rec[-1]
EOF

echo "=== MR feedforward (APPLY_PAULI), rank probe ==="
"$BIN" --circuit "$FF" --shots 4 --seed 42 --cpu-reference 2>&1 | python3 -c "import json,sys;d=json.load(sys.stdin);print('rank=',d.get('peak_rank'))" 2>&1 | head
rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null
"$BIN" --circuit "$FF" --shots 10 --seed 42 --mlir --no-postselection >/dev/null 2>&1
for N in 1 2 4 16 64; do
  s=$("$BIN" --circuit "$FF" --shots $N --seed 42 --no-postselection 2>/dev/null | getobs)
  h=$("$BIN" --circuit "$FF" --shots $N --seed 42 --hybrid --no-postselection 2>/dev/null | getobs)
  m=$("$BIN" --circuit "$FF" --shots $N --seed 42 --mlir --no-postselection 2>/dev/null | getobs)
  echo "N=$N  SVM: $s | HYB: $h | MLIR: $m"
done

# Also test MULTI_CNOT-heavy (star graph) + FRAME_SWAP via a GHZ with many CX from one control
echo ""
echo "=== star-graph (MULTI_CNOT) + T, rank probe ==="
ST=/tmp/star_$$.stim
cat > "$ST" <<'EOF'
R 0 1 2 3 4 5 6 7 8 9 10 11
H 0
CX 0 1 0 2 0 3 0 4 0 5 0 6 0 7 0 8 0 9 0 10 0 11
T 0 1 2 3 4 5 6 7 8 9 10 11
MPP Y0*Y1*Y2*Y3*Y4*Y5*Y6*Y7*Y8*Y9*Y10*Y11
OBSERVABLE_INCLUDE(0) rec[-1]
EOF
"$BIN" --circuit "$ST" --shots 4 --seed 42 --cpu-reference 2>&1 | python3 -c "import json,sys;d=json.load(sys.stdin);print('rank=',d.get('peak_rank'))" 2>&1 | head
rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null
"$BIN" --circuit "$ST" --shots 10 --seed 42 --mlir --no-postselection >/dev/null 2>&1
for N in 1 4 16; do
  s=$("$BIN" --circuit "$ST" --shots $N --seed 42 --no-postselection 2>/dev/null | getobs)
  m=$("$BIN" --circuit "$ST" --shots $N --seed 42 --mlir --no-postselection 2>/dev/null | getobs)
  echo "N=$N  SVM: $s | MLIR: $m"
done
echo "Done: $(date)"
