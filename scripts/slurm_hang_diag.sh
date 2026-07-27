#!/bin/bash
#SBATCH --job-name=hangdiag
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:25:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/hangdiag_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BIN="$BASE/build-gpu-mlir-mi355x/run_gpu"
for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }; done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }; done
getobs(){ python3 -c "import json,sys;d=json.load(sys.stdin);print('obs=',d.get('observable_ones'))" 2>/dev/null; }

# S2 M-diagonal shrink circuit (known to hang at N>=10)
S2=/tmp/s2_$$.stim
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

echo "=== S2 M-diagonal: hang test per shot count (25s timeout each) ==="
rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null
"$BIN" --circuit "$S2" --shots 4 --seed 42 --mlir --no-postselection >/dev/null 2>&1  # warm cache
for N in 1 2 4 8 10 16 32; do
  t0=$(date +%s)
  m=$(timeout 25 "$BIN" --circuit "$S2" --shots $N --seed 42 --mlir --no-postselection 2>/dev/null | getobs)
  rc=$?; t1=$(date +%s)
  s=$("$BIN" --circuit "$S2" --shots $N --seed 42 --no-postselection 2>/dev/null | getobs)
  htag=$([ $rc -eq 124 ] && echo "HANG" || echo "ok(${rc})")
  echo "N=$N  [$htag $((t1-t0))s]  SVM: $s | MLIR: $m"
done

echo ""
echo "=== circuit_d5 correctness with OBSERVABLE fix ==="
C5="$BASE/tests/fixtures/large/circuit_d5_p0.001.stim"
NL=/tmp/c5nl_$$.stim
grep -viE "DEPOLARIZE|X_ERROR|Z_ERROR|Y_ERROR|PAULI_CHANNEL" "$C5" > "$NL"
rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null
"$BIN" --circuit "$NL" --shots 10 --seed 42 --mlir --no-postselection >/dev/null 2>&1
for N in 1 2 4 16; do
  s=$("$BIN" --circuit "$NL" --shots $N --seed 42 --no-postselection 2>/dev/null | getobs)
  m=$(timeout 40 "$BIN" --circuit "$NL" --shots $N --seed 42 --mlir --no-postselection 2>/dev/null | getobs)
  echo "N=$N noiseless  SVM: $s | MLIR: $m"
done
getp(){ python3 -c "import json,sys;print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null; }
echo "--- full circuit_d5 WITH noise+postsel, 10000 shots ---"
rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null
"$BIN" --circuit "$C5" --shots 10 --seed 42 --mlir >/dev/null 2>&1
svm=$("$BIN" --circuit "$C5" --shots 10000 --seed 42 2>/dev/null | getp)
ml=$(timeout 300 "$BIN" --circuit "$C5" --shots 10000 --seed 42 --mlir 2>/dev/null | getp)
echo "circuit_d5  SVM=$svm  MLIR=$ml"
echo "Done: $(date)"
