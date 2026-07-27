#!/bin/bash
#SBATCH --job-name=dbg-noise
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/dbg_noise_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BUILD="$BASE/build-gpu-mlir-mi355x"
BIN="$BUILD/run_gpu"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

cd "$BUILD" && make -j$(nproc) 2>&1 | tail -3
rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null

# circuit_d3 is register tier (rank 4) — tid0 guard shouldn't apply
echo "=== circuit_d3 rank/tier check ==="
$BIN --circuit "$BASE/tests/fixtures/large/circuit_d3_p0.001.stim" --cpu-reference --shots 1 --seed 42 2>/dev/null | grep "peak_rank"

echo ""
echo "=== circuit_d3 MLIR compile ==="
timeout 60 $BIN --circuit "$BASE/tests/fixtures/large/circuit_d3_p0.001.stim" --shots 10 --seed 42 --mlir 2>&1 | head -10

echo ""
echo "=== saved MLIR ==="
ls -la $BASE/results/debug_raw_*ops.mlir 2>/dev/null

# Check if tid0_guard appears in register tier MLIR
for f in $BASE/results/debug_raw_*344*ops.mlir; do
    [ -f "$f" ] || continue
    echo "--- $f ---"
    echo "tid0_guard count: $(grep -c t0_guard "$f")"
    echo "noise_guard count: $(grep -c noise "$f" | head -1)"
    grep "does not dominate" "$f" | head -3
done

echo ""
echo "Done: $(date)"
