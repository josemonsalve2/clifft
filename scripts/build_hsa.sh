#!/bin/bash
#SBATCH --job-name=build-hsa
#SBATCH --partition=mi300x
#SBATCH --nodelist=rad-mi300x-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/build_hsa_%j.log

# Build clifft with HSA runtime layer on compute node
set +e
BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"

# Setup ROCm
for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    if [ -d "$ROCM/lib" ]; then
        export ROCM_PATH="$ROCM"
        export PATH="$ROCM/bin:$PATH"
        export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"
        echo "Using ROCm: $ROCM"
        break
    fi
done

echo "Node: $(hostname)  Date: $(date)"
echo "=== Build HSA runtime ==="

BUILD_DIR="$BASE/build-gpu-hsa"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake "$BASE" \
    -DCLIFFT_ENABLE_HIP=ON \
    -DCMAKE_HIP_ARCHITECTURES=gfx942 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCLIFFT_BUILD_PROFILER=ON \
    -DCMAKE_BUILD_PARALLEL_LEVEL=$(nproc) 2>&1

echo ""
echo "=== Compile ==="
make -j$(nproc) run_gpu 2>&1
BUILD_RC=$?

echo ""
echo "Build exit code: $BUILD_RC"

if [ $BUILD_RC -eq 0 ]; then
    echo ""
    echo "=== Quick test ==="
    ./run_gpu --diagnose 2>&1
    echo ""
    echo "=== Run target_qec (SVM) ==="
    ./run_gpu --circuit "$BASE/tests/fixtures/target_qec.stim" --shots 1000 2>&1
    echo ""
    echo "=== Run target_qec (compiled/HSA) ==="
    ./run_gpu --circuit "$BASE/tests/fixtures/target_qec.stim" --shots 1000 --hybrid 2>&1
    echo ""
    echo "=== Run Catch2 GPU tests ==="
    cd "$BUILD_DIR"
    make -j$(nproc) tests 2>&1 | tail -5
    ./tests --reporter compact "[gpu]" 2>&1 || echo "Test exit: $?"
fi

echo ""
echo "Done: $(date)"
