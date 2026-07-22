#!/bin/bash
#SBATCH --job-name=baseline-mi300x
#SBATCH --partition=mi300x
#SBATCH --nodelist=rad-mi300x-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/slurm_baseline_mi300x_%j.log
#SBATCH --exclusive

# Baseline benchmark on MI300X (rad-mi300x-2, gfx942)
# Runs SVM and compiled megakernel across all rank 1-19 circuits + interesting circuits.
# rad-mi300x-1 is faulty — always use rad-mi300x-2.

set -euo pipefail

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BINARY="$BASE/build-gpu/run_gpu"

# NUMA binding: pin CPU and memory to node 1 (nearest to GPU on MI300X server)
NUMA_CMD="numactl --cpunodebind=1 --membind=1"

echo "=== MI300X Baseline Job: $(date) ==="
echo "Node: $(hostname)"
echo "SLURM Job ID: ${SLURM_JOB_ID:-n/a}"

# Verify GPU
rocm-smi --showid --showproductname 2>/dev/null | head -10 || true
echo ""

# Check binary
if [ ! -x "$BINARY" ]; then
    echo "ERROR: binary not found: $BINARY"
    echo "Build with: cmake -DCLIFFT_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_BUILD_TYPE=Release .."
    exit 1
fi

# Run benchmark (SVM + compiled, all circuits, 20 runs, 1M shots)
exec $NUMA_CMD bash "$BASE/scripts/bench_rigorous.sh" \
    --modes "svm,compiled" \
    --circuits "all" \
    --shots 1000000 \
    --runs 20 \
    --binary "$BINARY" \
    --outdir "$BASE/results"
