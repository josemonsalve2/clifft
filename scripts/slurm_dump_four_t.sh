#!/bin/bash
#SBATCH --job-name=dump-4t
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:05:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/dump_four_t_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BIN="$BASE/build-gpu-mlir-mi355x/run_gpu"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null

# Run four_t MLIR (will compile and save .mlir to /tmp)
$BIN --circuit "$BASE/tests/fixtures/incremental/02_single_expand/four_t.stim" --shots 10 --seed 42 --mlir 2>&1 | head -5

# Find and dump the MLIR text
echo "=== MLIR text for four_t ==="
ls /tmp/clifft_mlir_*.mlir 2>/dev/null
for f in /tmp/clifft_mlir_*.mlir; do
    echo "--- $f ($(wc -l < "$f") lines) ---"
    grep "// op\[" "$f" | head -20
    echo "---"
    grep "discarded\|Unsupported\|discard" "$f" | head -10
    echo "---"
    head -200 "$f"
done

echo ""
echo "Done: $(date)"
