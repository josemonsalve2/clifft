#!/bin/bash
#SBATCH --job-name=dump-4t2
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:05:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/dump_four_t2_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BIN="$BASE/build-gpu-mlir-mi355x/run_gpu"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null

# Save the MLIR text before compilation deletes it
# Temporarily patch: instead of running, generate text only
# Actually let's just copy the .mlir before it's deleted:
# Run and check what .ll files survive
$BIN --circuit "$BASE/tests/fixtures/incremental/02_single_expand/four_t.stim" --shots 10 --seed 42 --mlir 2>&1 | head -5

echo "=== .ll files in /tmp ==="
ls -la /tmp/clifft_mlir_*.ll 2>/dev/null
echo ""

for f in /tmp/clifft_mlir_*.ll; do
    [ -f "$f" ] || continue
    echo "=== $f ==="
    # Show the module-level declarations and the instruction comments
    grep -n "define\|ret void\|op\[\|discard\|Unsupported\|atomicrmw\|active_k" "$f" | head -40
    echo ""
    # Show total line count
    echo "Total lines: $(wc -l < "$f")"
done

echo ""
echo "Done: $(date)"
