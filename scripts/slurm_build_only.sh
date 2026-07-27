#!/bin/bash
#SBATCH --job-name=build
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/build_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BUILD="$BASE/build-gpu-mlir-mi355x"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done

cd "$BUILD"
make -j1 src/clifft/CMakeFiles/clifft_core.dir/gpu/mlir/mlir_emit.cc.o 2>&1 | tail -40
echo "EXIT: $?"
