#!/bin/bash
#SBATCH --job-name=multi-call
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/multi_call_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BUILD="$BASE/build-gpu-mlir-mi355x"
BIN="$BUILD/run_gpu"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

echo "=== Multi-Call Benchmark (same process, persistent pool) ==="
echo "Node: $(hostname)"
echo ""

cd "$BUILD"
make -j$(nproc) 2>&1 | tail -3
echo ""

STIM="$BASE/tests/fixtures/rank_sweep/rank_q17_r4_d1.stim"

# Single-process test: run the SAME circuit 5 times via --repeat flag
# Since run_gpu doesn't have --repeat, we use a trick:
# Run with increasing shot counts to see if the second batch reuses the pool

echo "=== SVM: 5 separate invocations ==="
for i in 1 2 3 4 5; do
    t=$($BIN --circuit "$STIM" --shots 10000 --seed 42 2>/dev/null | \
        python3 -c "import json,sys; print(f\"{json.load(sys.stdin)['sample_seconds']:.6f}\")" 2>/dev/null)
    echo "  Run $i: ${t}s"
done

echo ""
echo "=== MLIR: 5 separate invocations (each cold) ==="
for i in 1 2 3 4 5; do
    t=$($BIN --circuit "$STIM" --shots 10000 --seed 42 --mlir 2>/dev/null | \
        python3 -c "import json,sys; print(f\"{json.load(sys.stdin)['sample_seconds']:.6f}\")" 2>/dev/null)
    echo "  Run $i: ${t}s"
done

echo ""
echo "=== Hybrid: 5 separate invocations ==="
for i in 1 2 3 4 5; do
    t=$($BIN --circuit "$STIM" --shots 10000 --seed 42 --hybrid 2>/dev/null | \
        python3 -c "import json,sys; print(f\"{json.load(sys.stdin)['sample_seconds']:.6f}\")" 2>/dev/null)
    echo "  Run $i: ${t}s"
done

echo ""
echo "=== MLIR timing breakdown (single invocation) ==="
$BIN --circuit "$STIM" --shots 10000 --seed 42 --mlir 2>&1 | grep "clifft-timing"

echo ""
echo "=== MLIR with kMaxBatchShots trick (2 batches in same process) ==="
echo "Running 200M shots forces 2 batch iterations in the same process"
echo "First batch pays alloc+load, second batch reuses pool"
$BIN --circuit "$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim" \
    --shots 200000001 --seed 42 --mlir 2>&1 | grep -E "clifft-timing|sample_seconds|sampled"

echo ""
echo "Done: $(date)"
