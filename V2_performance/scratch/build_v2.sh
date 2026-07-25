#!/bin/bash
# build_v2.sh — rebuild the V2 (no-HIP) and SVM/MLIR (HIP) trees on a compute
# node. radha/login has no ROCm, so every build of GPU code must go through
# SLURM. Both trees are needed: run_v2 for the V2 backend, run_gpu for the
# GPU-SVM reference the correctness gate and the benchmark sweep compare against.
set -u
cd "$CLIFFT"
for ROCM in /opt/rocm-7.2.3 /opt/rocm; do
  [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; \
     export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
export LLVM_PREFIX=/shared/jmonsalv/software/modules/llvm/upstream_05082025
# cmake is reachable on the login node only via a $HOME symlink; compute nodes
# get a different $HOME, so resolve it to the shared install directly.
# Must be a COMPLETE cmake install on SHARED storage. The build trees were
# originally configured with a pip cmake under $HOME/.local, which no longer
# exists and was never visible from compute nodes anyway; the trees have been
# repointed at this shared 4.1.0 copy.
CMAKE=/shared/jmonsalv/software/cmake_pkg41/data/bin/cmake
[ -x "$CMAKE" ] || CMAKE=$(command -v cmake)
echo "ROCM_PATH=$ROCM_PATH  node=$(hostname)  cmake=$CMAKE"

echo "########## build-v2-nohip (run_v2) ##########"
"$CMAKE" --build build-v2-nohip -j"$(nproc)" --target run_v2 2>&1 | tail -40

# Some mi350x-es nodes (e.g. f13-21, but NOT d13-21 — the partition is
# heterogeneous) carry a stale /usr/include/hip from HIP 5.7. clang's HIP
# toolchain adds ROCm's own include via -idirafter, i.e. AFTER /usr/include,
# so the 5.7 headers win and __shfl_xor's warpSize default arg fails to
# compile. Passing -I/opt/rocm-7.2.3/include does NOT help: clang dedups it
# against the identical -idirafter entry and keeps only the late one. The fix
# is a shim directory of symlinks with a DIFFERENT path, which survives the
# dedup and lands ahead of /usr/include. CMAKE_HIP_FLAGS in the cache points
# at .rocm_include_shim; recreate it here (symlinks, cheap) and force a
# reconfigure so the flag lands in flags.make.
mkdir -p .rocm_include_shim
for d in hip hsa amd_comgr CL; do ln -sfn "$ROCM_PATH/include/$d" ".rocm_include_shim/$d"; done

echo "########## build-gpu-mlir-mi355x (run_gpu) ##########"
"$CMAKE" -S . -B build-gpu-mlir-mi355x 2>&1 | tail -5
"$CMAKE" --build build-gpu-mlir-mi355x -j"$(nproc)" --target run_gpu 2>&1 | grep -vE "^\s|warning:|note:|\^~*$" | tail -60

echo "BUILD_DONE"
ls -la build-v2-nohip/run_v2 build-gpu-mlir-mi355x/run_gpu 2>&1
