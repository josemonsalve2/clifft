#!/bin/bash
#SBATCH --job-name=swocc
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/swocc_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BIN="$BASE/build-gpu-mlir-mi355x/run_gpu"
for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }; done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }; done
getobs(){ python3 -c "import json,sys;d=json.load(sys.stdin);print('obs=',d.get('observable_ones'))" 2>/dev/null; }

# qv10 (uses swap_meas, KNOWN GOOD at 10000) — is it good across shot counts?
QV="$BASE/tests/fixtures/qv10.stim"
echo "=== qv10 (swap_meas, control) shot sweep ==="
rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null
"$BIN" --circuit "$QV" --shots 10 --seed 42 --mlir >/dev/null 2>&1
for N in 4 8 16 100; do
  s=$("$BIN" --circuit "$QV" --shots $N --seed 42 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
  m=$(timeout 40 "$BIN" --circuit "$QV" --shots $N --seed 42 --mlir 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
  echo "N=$N  SVM_passed=$s  MLIR_passed=$m"
done

# A T-heavy circuit measured via MX (interfere, NOT diagonal) to match circuit_d5's swap_meas path
echo ""
echo "=== interfere-measured shrink (MX), occupancy sweep ==="
S=/tmp/sw_$$.stim
cat > "$S" <<'EOF'
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
rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null
"$BIN" --circuit "$S" --shots 4 --seed 42 --mlir --no-postselection >/dev/null 2>&1
for N in 1 4 8 16 32 64; do
  t0=$(date +%s)
  m=$(timeout 25 "$BIN" --circuit "$S" --shots $N --seed 42 --mlir --no-postselection 2>/dev/null | getobs)
  rc=$?; t1=$(date +%s)
  s=$("$BIN" --circuit "$S" --shots $N --seed 42 --no-postselection 2>/dev/null | getobs)
  htag=$([ $rc -eq 124 ] && echo HANG || echo ok)
  echo "N=$N [$htag $((t1-t0))s]  SVM: $s | MLIR: $m"
done
echo "Done: $(date)"
