#!/bin/bash
#SBATCH --job-name=baseline-mi350x
#SBATCH --partition=mi350x
#SBATCH --nodelist=smci350-rck-g03-d13-21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/slurm_baseline_mi350x_%j.log
#SBATCH --exclusive

# Baseline benchmark on MI350X (smci350-rck-g03-d13-21, gfx950)
# Requires a build targeting gfx950: build-gpu-mi350x/run_gpu
# Build: cmake -DCLIFFT_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx950 ..

set -euo pipefail

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
# MI350X uses a separate build with gfx950 arch
BINARY="$BASE/build-gpu-mi350x/run_gpu"

echo "=== MI350X Baseline Job: $(date) ==="
echo "Node: $(hostname)"
echo "SLURM Job ID: ${SLURM_JOB_ID:-n/a}"

# Verify GPU
rocm-smi --showid --showproductname 2>/dev/null | head -10 || true
echo ""

# Check binary — if gfx950 build doesn't exist, attempt to build it
if [ ! -x "$BINARY" ]; then
    echo "Binary not found: $BINARY"
    echo "Attempting to build for gfx950..."
    mkdir -p "$BASE/build-gpu-mi350x"
    cd "$BASE/build-gpu-mi350x"
    cmake .. \
        -DCLIFFT_ENABLE_HIP=ON \
        -DCMAKE_HIP_ARCHITECTURES=gfx950 \
        -DCMAKE_BUILD_TYPE=Release \
        -DCLIFFT_BUILD_PROFILER=ON \
        2>&1
    make -j$(nproc) run_gpu 2>&1
    cd "$BASE"
fi

if [ ! -x "$BINARY" ]; then
    echo "ERROR: build failed, binary still not found: $BINARY"
    exit 1
fi

# No numactl needed on MI350X (single-socket or different topology)
exec bash "$BASE/scripts/bench_rigorous.sh" \
    --modes "svm,compiled" \
    --circuits "all" \
    --shots 1000000 \
    --runs 20 \
    --binary "$BINARY" \
    --outdir "$BASE/results"
