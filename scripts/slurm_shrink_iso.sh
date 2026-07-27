#!/bin/bash
#SBATCH --job-name=shrink
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/shrink_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BIN="$BASE/build-gpu-mlir-mi355x/run_gpu"
for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }; done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }; done
getobs(){ python3 -c "import json,sys;d=json.load(sys.stdin);print('obs=',d.get('observable_ones'),'rank=',d.get('peak_rank'))" 2>/dev/null; }

run3(){
  local C="$1"; local name="$2"
  echo "=== $name ==="
  "$BIN" --circuit "$C" --shots 4 --seed 42 --cpu-reference 2>&1 | python3 -c "import json,sys;d=json.load(sys.stdin);print('rank=',d.get('peak_rank'))" 2>&1 | head -1
  rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null
  "$BIN" --circuit "$C" --shots 10 --seed 42 --mlir --no-postselection >/dev/null 2>&1
  for N in 1 2 4 16 64; do
    s=$("$BIN" --circuit "$C" --shots $N --seed 42 --no-postselection 2>/dev/null | getobs)
    m=$("$BIN" --circuit "$C" --shots $N --seed 42 --mlir --no-postselection 2>/dev/null | getobs)
    tag=$([ "$s" = "$m" ] && echo OK || echo "<<< DIFF")
    echo "N=$N  SVM: $s | MLIR: $m  $tag"
  done
  echo ""
}

# Progressive shrink: build rank-10 GHZ+T, then measure qubits one by one (X-basis
# interfering measurements shrink active_k 10->...). Final observable on remaining.
S1=/tmp/shrink1_$$.stim
cat > "$S1" <<'EOF'
R 0 1 2 3 4 5 6 7 8 9 10 11
H 0 1 2 3 4 5 6 7 8 9 10 11
T 0 1 2 3 4 5 6 7 8 9 10 11
MX 11
MX 10
MX 9
MX 8
MPP Y0*Y1*Y2*Y3*Y4*Y5*Y6*Y7
OBSERVABLE_INCLUDE(0) rec[-1]
EOF
run3 "$S1" "shrink via MX (interfering meas, k 10->..)"

# Shrink via Z-basis (diagonal) measurements
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
run3 "$S2" "shrink via M (diagonal meas)"

# Mixed CX entangling between rounds (like real code)
S3=/tmp/shrink3_$$.stim
cat > "$S3" <<'EOF'
R 0 1 2 3 4 5 6 7 8 9 10 11
H 0 1 2 3 4 5 6 7 8 9 10 11
T 0 1 2 3 4 5 6 7 8 9 10 11
CX 0 11 1 10 2 9 3 8
M 11 10 9 8
CX 0 7 1 6 2 5 3 4
MX 7 6 5 4
MPP Y0*Y1*Y2*Y3
OBSERVABLE_INCLUDE(0) rec[-1]
EOF
run3 "$S3" "shrink + CX entangle rounds"
echo "Done: $(date)"
