#!/bin/bash
#SBATCH --job-name=bld-dbg2
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/bld_dbg2_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BUILD="$BASE/build-gpu-mlir-mi355x"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done

cd "$BUILD"
echo "=== Full rebuild ==="
make -j1 2>&1 | grep -E "error:|Error|warning.*error" | head -20
echo ""
echo "=== Final make status ==="
make -j$(nproc) 2>&1 | tail -5
echo "RC=$?"
