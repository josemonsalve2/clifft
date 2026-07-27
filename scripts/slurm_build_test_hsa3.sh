#!/bin/bash
#SBATCH --job-name=build-hsa3
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/build_hsa3_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BUILD="$BASE/build-gpu-mlir-mi355x"
BIN="$BUILD/run_gpu"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

echo "=== Building ==="
cd "$BUILD"
make -j$(nproc) 2>&1 | tail -5
echo "BUILD RC=$?"
echo ""

rm -rf "$HOME/.clifft/kernel_cache/mlir" 2>/dev/null

STIM="$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim"

echo "=== SVM baseline ==="
$BIN --circuit "$STIM" --shots 1000 --seed 42 2>&1 | grep "sample_seconds"

echo ""
echo "=== MLIR register tier ==="
timeout 120 $BIN --circuit "$STIM" --shots 100 --seed 42 --mlir 2>&1 | tail -10
echo "RC=$?"

echo ""
echo "=== MLIR 100k shots (perf) ==="
timeout 120 $BIN --circuit "$STIM" --shots 100000 --seed 42 --mlir 2>&1 | grep "sample_seconds"

echo ""
echo "=== Hybrid register tier ==="
timeout 120 $BIN --circuit "$STIM" --shots 100 --seed 42 --hybrid 2>&1 | tail -10
echo "RC=$?"

echo ""
echo "=== Hybrid 100k shots (perf) ==="
timeout 120 $BIN --circuit "$STIM" --shots 100000 --seed 42 --hybrid 2>&1 | grep "sample_seconds"

echo ""
echo "=== Correctness vs CPU ==="
cpu_p=$($BASE/build-cpu/run_cpu --circuit "$STIM" --shots 10000 --seed 42 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
mlir_p=$(timeout 120 $BIN --circuit "$STIM" --shots 10000 --seed 42 --mlir 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null)
echo "CPU passed=$cpu_p MLIR passed=$mlir_p"
[ "$cpu_p" = "$mlir_p" ] && echo "MATCH" || echo "MISMATCH"

echo ""
echo "Done: $(date)"
