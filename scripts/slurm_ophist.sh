#!/bin/bash
#SBATCH --job-name=ophist
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/ophist_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BIN="$BASE/build-gpu-mlir-mi355x/run_gpu"
for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }; done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }; done
rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null

C5="$BASE/tests/fixtures/large/circuit_d5_p0.001.stim"
NL=/tmp/c5_nl.stim
grep -viE "DEPOLARIZE|X_ERROR|Z_ERROR|Y_ERROR|PAULI_CHANNEL" "$C5" > "$NL"

echo "=== compile circuit_d5 (noiseless) ==="
"$BIN" --circuit "$NL" --shots 10 --seed 42 --mlir >/dev/null 2>&1
D5F=$(ls -t "$BASE/results/debug_coop_"*.mlir 2>/dev/null | head -1)
echo "d5 dump: $D5F ($(wc -l <"$D5F") lines)"
echo ""
echo "=== circuit_d5 opcode histogram (opcode=N) ==="
grep -oE "opcode=[0-9]+" "$D5F" | sort | uniq -c | sort -rn
echo ""
echo "=== circuit_d5 named-op comment histogram ==="
grep -oE "^  // [a-z_]+_?[a-z]+:" "$D5F" | sort | uniq -c | sort -rn
echo ""
echo "=== active_k range seen in swap_meas_interfere ==="
grep -oE "swap_meas_interfere.*active_k=[0-9]+" "$D5F" | grep -oE "active_k=[0-9]+|from=[0-9]+ to=[0-9]+" | paste - - | sort | uniq -c | head -30
echo ""
echo "Done: $(date)"
