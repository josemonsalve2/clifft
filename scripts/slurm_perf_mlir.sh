#!/bin/bash
#SBATCH --job-name=perf-mlir
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:40:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/perf_mlir_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BUILD="$BASE/build-gpu-mlir-mi355x"
BIN="$BUILD/run_gpu"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

echo "Node: $(hostname)"
echo "Date: $(date)"
echo "GPU: $(rocm_agent_enumerator | grep gfx | head -1)"

# Build fresh
export PATH="/home/jmonsalv/software/bin:$HOME/.local/bin:$PATH"
if ! command -v cmake &>/dev/null; then pip install --user --break-system-packages cmake 2>&1 | tail -3; fi
GPU_ARCH=$(rocm_agent_enumerator | grep gfx | head -1)
echo "Building for $GPU_ARCH..."
rm -rf "$BUILD"
mkdir -p "$BUILD" && cd "$BUILD"
cmake "$BASE" -DCLIFFT_ENABLE_HIP=ON -DCLIFFT_ENABLE_MLIR=ON \
    -DCMAKE_HIP_ARCHITECTURES=$GPU_ARCH \
    -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
make -j$(nproc) run_gpu 2>&1 | tail -20
cd "$BASE"
rm -rf "$HOME/.clifft/kernel_cache/mlir" 2>/dev/null
if [ ! -x "$BIN" ]; then
    echo "FATAL: build failed"
    exit 1
fi
echo "Build OK"

SHOTS=100000

echo ""
echo "=== Performance: SVM vs MLIR (warm cache, ${SHOTS} shots) ==="
echo ""
printf "%-30s %12s %12s %10s\n" "Circuit" "SVM(s)" "MLIR(s)" "Ratio"
echo "------------------------------------------------------------------------"

perf_test() {
    local name=$1 stim=$2
    # Warm cache
    timeout 30 "$BIN" --circuit "$stim" --shots 1000 --seed 42 >/dev/null 2>&1
    timeout 30 "$BIN" --circuit "$stim" --shots 1000 --seed 42 --mlir >/dev/null 2>&1
    # Measured
    local svm_t=$(timeout 60 "$BIN" --circuit "$stim" --shots $SHOTS --seed 42 2>&1 | grep sample_seconds | grep -oP '[0-9.]+')
    local mlir_t=$(timeout 60 "$BIN" --circuit "$stim" --shots $SHOTS --seed 42 --mlir 2>&1 | grep sample_seconds | grep -oP '[0-9.]+')
    if [ -n "$svm_t" ] && [ -n "$mlir_t" ]; then
        local ratio=$(python3 -c "print(f'{float(\"$svm_t\")/float(\"$mlir_t\"):.2f}x')" 2>/dev/null || echo "N/A")
        printf "%-30s %12s %12s %10s\n" "$name" "$svm_t" "$mlir_t" "$ratio"
    else
        printf "%-30s %12s %12s %10s\n" "$name" "${svm_t:-ERR}" "${mlir_t:-ERR}" "N/A"
    fi
}

echo "--- Register Tier (rank <= 4) ---"
perf_test "frame_h" "$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim"
perf_test "frame_h_x10" "$BASE/tests/fixtures/incremental/01_frame_only/frame_h_x10.stim"
perf_test "h_then_t" "$BASE/tests/fixtures/incremental/05_combinations/h_then_t.stim"
perf_test "rank4_mixed" "$BASE/tests/fixtures/incremental/06_small_circuits/rank4_mixed.stim"
perf_test "rep_d3" "$BASE/tests/fixtures/incremental/06_small_circuits/rep_d3.stim"
perf_test "noise_measure" "$BASE/tests/fixtures/incremental/04_measure/noise_measure.stim"

echo ""
echo "--- Coop Tier (rank 5-10) ---"
for stim in "$BASE"/tests/fixtures/rank_sweep/rank_q17_r5_d1.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q17_r8_d3.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q33_r6_d3.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q33_r10_d3.stim; do
    [ -f "$stim" ] && perf_test "$(basename "$stim" .stim)" "$stim"
done

echo ""
echo "--- Global Tier (rank 11-19) ---"
for stim in "$BASE"/tests/fixtures/rank_sweep/rank_q17_r12_d1.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q17_r17_d3.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q33_r15_d3.stim \
            "$BASE"/tests/fixtures/rank_sweep/rank_q33_r19_d3.stim; do
    [ -f "$stim" ] && perf_test "$(basename "$stim" .stim)" "$stim"
done

echo ""
echo "--- Large Circuits ---"
for stim in "$BASE"/tests/fixtures/large/rank11_q50_d5_tier3.stim \
            "$BASE"/tests/fixtures/large/rank15_q100_d5_wide.stim \
            "$BASE"/tests/fixtures/large/rank19_q50_d12_tier3.stim; do
    [ -f "$stim" ] && perf_test "$(basename "$stim" .stim)" "$stim"
done

echo ""
echo "Done: $(date)"
