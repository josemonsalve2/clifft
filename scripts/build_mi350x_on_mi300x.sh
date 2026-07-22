#!/bin/bash
#SBATCH --job-name=build-350x
#SBATCH --partition=mi300x
#SBATCH --nodelist=rad-mi300x-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/build_mi350x_%j.log

# Cross-compile for gfx950 on MI300X node (which has cmake)
set +e
BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done

BUILD="$BASE/build-gpu-mi350x-cross"
mkdir -p "$BUILD" && cd "$BUILD"
cmake "$BASE" -DCLIFFT_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx950 \
    -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5
make -j$(nproc) run_gpu 2>&1 | tail -5
echo "Build result: $?"
ls -la run_gpu 2>/dev/null
