#!/bin/bash
#SBATCH --job-name=build-gfx950
#SBATCH --partition=mi300x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/build_gfx950_%j.log

# Cross-compile for gfx950 (MI350X) on mi300x node (has cmake + ROCm)
set -euo pipefail

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
cd "$BASE"

echo "=== Cross-compiling for gfx950 ==="
echo "Host: $(hostname)"
echo "Date: $(date)"

BUILD_DIR="build-gpu-mi350x"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. \
    -DCLIFFT_ENABLE_HIP=ON \
    -DCMAKE_HIP_ARCHITECTURES=gfx950 \
    -DCMAKE_BUILD_TYPE=Release \
    2>&1

cmake --build . -j$(nproc) 2>&1

cd "$BASE"
echo ""
ls -la "$BUILD_DIR/run_gpu" "$BUILD_DIR/tests/clifft_tests" 2>/dev/null
echo ""
echo "=== Build complete. Ready to run on MI350X. ==="
