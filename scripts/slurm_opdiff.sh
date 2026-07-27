#!/bin/bash
#SBATCH --job-name=opdiff
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/opdiff_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BIN="$BASE/build-gpu-mlir-mi355x/run_gpu"
for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }; done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }; done
rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null
rm -f "$BASE/results/debug_coop_"*.mlir 2>/dev/null

C5="$BASE/tests/fixtures/large/circuit_d5_p0.001.stim"
NL=/tmp/c5_nl.stim
grep -viE "DEPOLARIZE|X_ERROR|Z_ERROR|Y_ERROR|PAULI_CHANNEL" "$C5" > "$NL"
QV="$BASE/tests/fixtures/qv10.stim"

echo "=== compile circuit_d5 (noiseless) coop MLIR ==="
"$BIN" --circuit "$NL" --shots 10 --seed 42 --mlir >/dev/null 2>&1
D5F=$(ls -t "$BASE/results/debug_coop_"*.mlir 2>/dev/null | head -1)
echo "d5 dump: $D5F"
cp "$D5F" /tmp/d5_coop.mlir 2>/dev/null

rm -f "$BASE/results/debug_coop_"*.mlir 2>/dev/null
rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null
echo "=== compile qv10 coop MLIR ==="
"$BIN" --circuit "$QV" --shots 10 --seed 42 --mlir >/dev/null 2>&1
QVF=$(ls -t "$BASE/results/debug_coop_"*.mlir 2>/dev/null | head -1)
echo "qv dump: $QVF"
cp "$QVF" /tmp/qv_coop.mlir 2>/dev/null

echo ""
echo "=== circuit_d5 op histogram (from // comments) ==="
grep -oE "// (array_u[24]|swap_meas_interfere|meas_active_(diagonal|interfere)|expand[a-z_]*|apply_pauli|noise[a-z_]*|multi_cnot|frame_swap|meas_dormant[a-z_]*)" /tmp/d5_coop.mlir 2>/dev/null | awk '{print $2}' | sort | uniq -c | sort -rn
echo ""
echo "=== qv10 op histogram ==="
grep -oE "// (array_u[24]|swap_meas_interfere|meas_active_(diagonal|interfere)|expand[a-z_]*|apply_pauli|noise[a-z_]*|multi_cnot|frame_swap|meas_dormant[a-z_]*)" /tmp/qv_coop.mlir 2>/dev/null | awk '{print $2}' | sort | uniq -c | sort -rn
echo ""
echo "=== ops in d5 but NOT in qv10 ==="
comm -23 \
  <(grep -oE "// (array_u[24]|swap_meas_interfere|meas_active_(diagonal|interfere)|expand[a-z_]*|apply_pauli|noise[a-z_]*|multi_cnot|frame_swap|meas_dormant[a-z_]*)" /tmp/d5_coop.mlir 2>/dev/null | awk '{print $2}' | sort -u) \
  <(grep -oE "// (array_u[24]|swap_meas_interfere|meas_active_(diagonal|interfere)|expand[a-z_]*|apply_pauli|noise[a-z_]*|multi_cnot|frame_swap|meas_dormant[a-z_]*)" /tmp/qv_coop.mlir 2>/dev/null | awk '{print $2}' | sort -u)
echo ""
echo "d5 total lines: $(wc -l </tmp/d5_coop.mlir 2>/dev/null), qv lines: $(wc -l </tmp/qv_coop.mlir 2>/dev/null)"
echo "Done: $(date)"
