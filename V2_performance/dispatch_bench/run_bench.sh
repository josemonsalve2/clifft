#!/bin/bash
# run_bench.sh — SLURM body for the HIP-vs-HSA dispatch overhead measurement.
#
# Submit with:
#   sbatch --partition=mi350x-es --nodes=1 --gpus=1 --time=00:20:00 \
#          --output=<dir>/bench.log V2_performance/dispatch_bench/run_bench.sh
#
# --gpus=1 is REQUIRED: without it hsa_init fails HSA_STATUS_ERROR_OUT_OF_RESOURCES.
# radha (login) has no GPU and no ROCm, so this cannot run there.
set -u
# SLURM copies the batch script into a spool dir, so $0 is NOT the source path.
# BENCH_DIR must be exported by the submitter.
cd "${BENCH_DIR:?export BENCH_DIR to the dispatch_bench source dir}"
OUT="$PWD"

for ROCM in /opt/rocm-7.2.3 /opt/rocm; do
  [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; \
     export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
LLVM=${LLVM_PREFIX:-/shared/jmonsalv/software/modules/llvm/upstream_05082025}
ARCH=$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+')
echo "# node=$(hostname) arch=$ARCH rocm=$ROCM_PATH"

# Some mi350x-es nodes carry a stale /usr/include/hip that shadows ROCm 7.2.3.
mkdir -p .shim && for d in hip hsa amd_comgr CL; do ln -sfn "$ROCM_PATH/include/$d" ".shim/$d"; done

# --- build the empty amdgcn kernel with the EXACT V2 flags (v2_compile_cache.cc) ---
"$LLVM/bin/clang" --target=amdgcn-amd-amdhsa -mcpu="$ARCH" -ffreestanding -nostdlib \
    -nogpulib -std=c23 -O2 -ffp-contract=off -c -emit-llvm -o bench_empty.bc empty_kernel.c || exit 1
"$LLVM/bin/llc" -mtriple=amdgcn-amd-amdhsa -mcpu="$ARCH" -mattr=+wavefrontsize64 \
    -filetype=obj -O2 -o bench_empty.o bench_empty.bc || exit 1
"$LLVM/bin/ld.lld" -shared -o bench_empty.hsaco bench_empty.o || exit 1
echo "built bench_empty.hsaco"

# --- HSA benchmark ---
# Do NOT `exit 1` on a build failure here: the HIP arm below is independent and
# a partial result is worth more than none. Job 50474 lost both arms to one
# compile error in this file.
echo "=== HSA ==="
if g++ -O2 -std=c++17 -I"$ROCM_PATH/include" -I.shim -o hsa_dispatch_bench \
       hsa_dispatch_bench.cc -L"$ROCM_PATH/lib" -lhsa-runtime64; then
    ./hsa_dispatch_bench bench_empty.hsaco "${ITERS:-2000}" "${REPS:-5}" | tee "$OUT/hsa_results.csv"
else
    echo "g++ failed -- HSA arm unavailable" | tee "$OUT/hsa_results.csv"
fi

# --- HIP benchmark ---
echo "=== HIP ==="
"$ROCM_PATH/bin/hipcc" -O2 -std=c++17 --offload-arch="$ARCH" -o hip_dispatch_bench \
    hip_dispatch_bench.hip 2>&1 | grep -v '^$' | head -5
if [ -x ./hip_dispatch_bench ]; then
    ./hip_dispatch_bench "${ITERS:-2000}" "${REPS:-5}" | tee "$OUT/hip_results.csv"
else
    echo "hipcc failed -- HIP arm unavailable" | tee "$OUT/hip_results.csv"
fi
echo "DONE"
