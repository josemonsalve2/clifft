#!/bin/bash
#SBATCH --job-name=rocprof
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/rocprof_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BIN="$BASE/build-gpu-mlir-mi355x/run_gpu"
STIM_REG="$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim"
STIM_COOP="$BASE/tests/fixtures/rank_sweep/rank_q17_r5_d1.stim"
OUTDIR="$BASE/results/rocprof_$$"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

ROCPROF=""
for rp in rocprofv3 rocprof; do
    if command -v $rp &>/dev/null; then ROCPROF=$rp; break; fi
done
echo "rocprof: ${ROCPROF:-NOT FOUND}"
echo "Node: $(hostname)"
echo ""

rm -rf "$HOME/.clifft/kernel_cache/mlir" 2>/dev/null

# Warm up both paths
$BIN --circuit "$STIM_REG" --shots 100 --seed 42 >/dev/null 2>&1
$BIN --circuit "$STIM_REG" --shots 100 --seed 42 --mlir >/dev/null 2>&1
$BIN --circuit "$STIM_COOP" --shots 100 --seed 42 --mlir >/dev/null 2>&1

mkdir -p "$OUTDIR"

if [ -n "$ROCPROF" ]; then
    echo "=== rocprof kernel trace: SVM register tier (100k shots) ==="
    $ROCPROF --kernel-trace -o "$OUTDIR/svm_reg" -- $BIN --circuit "$STIM_REG" --shots 100000 --seed 42 2>&1 | tail -3
    echo ""
    echo "=== rocprof kernel trace: MLIR register tier (100k shots) ==="
    $ROCPROF --kernel-trace -o "$OUTDIR/mlir_reg" -- $BIN --circuit "$STIM_REG" --shots 100000 --seed 42 --mlir 2>&1 | tail -3
    echo ""
    echo "=== rocprof kernel trace: MLIR coop tier (100k shots) ==="
    $ROCPROF --kernel-trace -o "$OUTDIR/mlir_coop" -- $BIN --circuit "$STIM_COOP" --shots 100000 --seed 42 --mlir 2>&1 | tail -3
    echo ""

    # Parse results
    echo "=== Kernel durations ==="
    for f in "$OUTDIR"/*kernel_trace*.csv "$OUTDIR"/*/*.csv; do
        [ -f "$f" ] || continue
        echo "--- $f ---"
        head -1 "$f"
        # Show kernel name and duration columns
        awk -F',' 'NR>1 {print $0}' "$f" | head -20
    done
else
    echo "No rocprof found, using manual timing"
fi

echo ""
echo "=== Manual timing comparison (PersistentDispatcher kernel_seconds) ==="
echo "--- Register tier ---"
echo "SVM 100k:"
$BIN --circuit "$STIM_REG" --shots 100000 --seed 42 2>&1 | grep "sample_seconds"
echo "MLIR 100k:"
$BIN --circuit "$STIM_REG" --shots 100000 --seed 42 --mlir 2>&1 | grep "sample_seconds"
echo "SVM 10M:"
$BIN --circuit "$STIM_REG" --shots 10000000 --seed 42 2>&1 | grep "sample_seconds"
echo "MLIR 10M:"
$BIN --circuit "$STIM_REG" --shots 10000000 --seed 42 --mlir 2>&1 | grep "sample_seconds"

echo ""
echo "--- Coop tier ---"
echo "SVM 100k:"
$BIN --circuit "$STIM_COOP" --shots 100000 --seed 42 2>&1 | grep "sample_seconds"
echo "MLIR 100k:"
$BIN --circuit "$STIM_COOP" --shots 100000 --seed 42 --mlir 2>&1 | grep "sample_seconds"
echo "SVM 10M:"
$BIN --circuit "$STIM_COOP" --shots 10000000 --seed 42 2>&1 | grep "sample_seconds"
echo "MLIR 10M:"
$BIN --circuit "$STIM_COOP" --shots 10000000 --seed 42 --mlir 2>&1 | grep "sample_seconds"

echo ""
echo "Done: $(date)"
