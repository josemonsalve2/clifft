#!/bin/bash
#SBATCH --job-name=s2only
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:25:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/s2only_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BIN="$BASE/build-gpu-mlir-mi355x/run_gpu"
for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }; done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }; done
getobs(){ python3 -c "import json,sys;d=json.load(sys.stdin);print('obs=',d.get('observable_ones'),'rank=',d.get('peak_rank'))" 2>/dev/null; }

S2=/tmp/shrink2_$$.stim
cat > "$S2" <<'EOF'
R 0 1 2 3 4 5 6 7 8 9 10 11
H 0 1 2 3 4 5 6 7 8 9 10 11
T 0 1 2 3 4 5 6 7 8 9 10 11
M 11
M 10
M 9
M 8
MPP Y0*Y1*Y2*Y3*Y4*Y5*Y6*Y7
OBSERVABLE_INCLUDE(0) rec[-1]
EOF

echo "=== S2 (M diagonal shrink) timing ==="
rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null
echo "MLIR compile+run (timeout 300s):"
t0=$(date +%s)
timeout 300 "$BIN" --circuit "$S2" --shots 10 --seed 42 --mlir --no-postselection 2>&1 | grep -iE "error|abort|obs|compiled|saved" | head
rc=$?
t1=$(date +%s)
echo "elapsed: $((t1-t0))s (rc=$rc; 124=timeout)"

if [ $rc -ne 124 ]; then
  for N in 1 2 4 16; do
    s=$("$BIN" --circuit "$S2" --shots $N --seed 42 --no-postselection 2>/dev/null | getobs)
    m=$(timeout 60 "$BIN" --circuit "$S2" --shots $N --seed 42 --mlir --no-postselection 2>/dev/null | getobs)
    echo "N=$N  SVM: $s | MLIR: $m"
  done
fi
echo "Done: $(date)"
