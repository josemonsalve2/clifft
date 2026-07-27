#!/bin/bash
#SBATCH --job-name=r19-cmp
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:45:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/r19_compare_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BIN="$BASE/build-gpu-mlir-mi355x/run_gpu"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

echo "=== Rank 19 Kernel Time Comparison ==="
echo "Node: $(hostname)"
echo ""

# Test circuits at rank 19 (global tier, 524K amplitudes per shot)
for STIM in "$BASE/tests/fixtures/rank_sweep/rank_q17_r17_d1.stim" \
            "$BASE/tests/fixtures/rank_sweep/rank_q33_r19_d1.stim" \
            "$BASE/tests/fixtures/large/rank19_q50_d12_tier3.stim"; do
    [ -f "$STIM" ] || continue
    name=$(basename "$STIM" .stim)
    nq=$(grep -c "QUBIT_COORDS" "$STIM")
    nl=$(wc -l < "$STIM")

    echo "=== $name ($nq qubits, $nl lines) ==="

    # Warm up
    $BIN --circuit "$STIM" --shots 10 --seed 42 >/dev/null 2>&1
    timeout 300 $BIN --circuit "$STIM" --shots 10 --seed 42 --mlir >/dev/null 2>&1

    # SVM timing
    echo "--- SVM 1000 shots ---"
    $BIN --circuit "$STIM" --shots 1000 --seed 42 2>&1 | grep -E "clifft-timing|sample_seconds|peak_rank"
    echo ""

    # MLIR timing
    echo "--- MLIR 1000 shots ---"
    $BIN --circuit "$STIM" --shots 1000 --seed 42 --mlir 2>&1 | grep -E "clifft-timing|sample_seconds|peak_rank"
    echo ""

    # Hybrid timing
    echo "--- Hybrid 1000 shots ---"
    timeout 120 $BIN --circuit "$STIM" --shots 1000 --seed 42 --hybrid 2>&1 | grep -E "clifft-timing|sample_seconds|peak_rank"
    echo ""

    # Try higher shot counts for SVM to see if kernel time scales
    echo "--- SVM 10k shots ---"
    $BIN --circuit "$STIM" --shots 10000 --seed 42 2>&1 | grep -E "clifft-timing|sample_seconds"
    echo ""

    echo "--- MLIR 10k shots ---"
    $BIN --circuit "$STIM" --shots 10000 --seed 42 --mlir 2>&1 | grep -E "clifft-timing|sample_seconds"
    echo ""
done

echo "Done: $(date)"
