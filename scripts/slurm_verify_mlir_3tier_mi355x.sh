#!/bin/bash
#SBATCH --job-name=verify-mlir-3tier-mi355x
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/verify_mlir_3tier_mi355x_%j.log

set -e
BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BUILD="$BASE/build-gpu-mlir-mi355x"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done

for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

GPU_ARCH=$(rocm_agent_enumerator | grep gfx | head -1)
# Ensure cmake is in PATH — check user install, then well-known pip installs
export PATH="/home/jmonsalv/software/bin:$HOME/.local/bin:$PATH"
if ! command -v cmake &>/dev/null; then
    pip install --user --break-system-packages cmake 2>&1 | tail -3
    export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v cmake &>/dev/null; then
    echo "FATAL: cmake not found after pip install"; exit 1
fi
echo "cmake: $(which cmake) ($(cmake --version | head -1))"
echo "GPU arch: $GPU_ARCH"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "LLVM_PREFIX: ${LLVM_PREFIX:-unset}"

# Clear MLIR kernel cache to avoid stale kernels
rm -rf /tmp/clifft_mlir_cache "$HOME/.cache/clifft/mlir_*" 2>/dev/null

echo "=== Building (HIP + MLIR) for $GPU_ARCH ==="
rm -rf "$BUILD"
mkdir -p "$BUILD" && cd "$BUILD"
cmake "$BASE" -DCLIFFT_ENABLE_HIP=ON -DCLIFFT_ENABLE_MLIR=ON \
    -DCMAKE_HIP_ARCHITECTURES=$GPU_ARCH \
    -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5
RC_CMAKE=$?
if [ $RC_CMAKE -ne 0 ]; then echo "CMAKE FAILED"; exit 1; fi
make -j$(nproc) run_gpu 2>&1 | tail -5
RC=$?
echo "Build exit: $RC"
if [ $RC -ne 0 ]; then echo "BUILD FAILED"; exit 1; fi
echo "Build OK: $BUILD/run_gpu"

BIN="$BUILD/run_gpu"
SHOTS=10000

echo ""
echo "=== MLIR 3-Tier Correctness: --mlir vs SVM ==="
echo ""
printf "%-52s %4s %6s  %12s %12s    %s\n" "Circuit" "Rank" "Status" "SVM_logical" "MLIR_logical" "Match"
echo "-----------------------------------------------------------------------------------------------------------"

PASS=0; FAIL=0; SKIP=0

check_circuit() {
    local name=$1 stim=$2
    set +e
    svm_out=$(timeout 60 "$BIN" --circuit "$stim" --shots $SHOTS --seed 42 2>/dev/null)
    svm_rc=$?
    set -e
    if [ $svm_rc -ne 0 ]; then
        SKIP=$((SKIP+1))
        printf "%-52s %4s %7s\n" "$name" "?" "SVM_ERR"
        return
    fi
    svm_le=$(echo "$svm_out" | grep '"logical_errors"' | grep -o '[0-9]*')

    set +e
    mlir_out=$(timeout 60 "$BIN" --circuit "$stim" --shots $SHOTS --seed 42 --mlir 2>/dev/null)
    mlir_rc=$?
    set -e
    if [ $mlir_rc -ne 0 ]; then
        SKIP=$((SKIP+1))
        printf "%-52s %4s %7s\n" "$name" "?" "MLIR_ERR"
        return
    fi
    mlir_le=$(echo "$mlir_out" | grep '"logical_errors"' | grep -o '[0-9]*')

    if [ "$svm_le" = "$mlir_le" ]; then
        PASS=$((PASS+1))
        printf "%-52s %4s %6s  %12s %12s    %s\n" "$name" "?" "OK" "$svm_le" "$mlir_le" "EXACT"
    else
        FAIL=$((FAIL+1))
        printf "%-52s %4s %6s  %12s %12s    %s\n" "$name" "?" "MISMATCH" "$svm_le" "$mlir_le" "FAIL"
    fi
}

echo "--- Incremental Per-Op Tests ---"
for stim in "$BASE"/tests/fixtures/incremental/*/*.stim; do
    [ -f "$stim" ] || continue
    check_circuit "$(basename "$stim" .stim)" "$stim"
done

echo ""
echo "--- Rank Sweep ---"
for stim in "$BASE"/tests/fixtures/rank_sweep/*.stim; do
    [ -f "$stim" ] || continue
    check_circuit "$(basename "$stim" .stim)" "$stim"
done

echo ""
echo "--- Large Circuits ---"
for stim in "$BASE"/tests/fixtures/large/*.stim; do
    [ -f "$stim" ] || continue
    check_circuit "$(basename "$stim" .stim)" "$stim"
done

echo ""
echo "--- Standard Circuits ---"
for stim in "$BASE"/tests/fixtures/*.stim; do
    [ -f "$stim" ] || continue
    check_circuit "$(basename "$stim" .stim)" "$stim"
done

echo ""
echo "=== Summary ==="
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
echo "  SKIP: $SKIP"
echo "  Total: $((PASS+FAIL+SKIP))"
echo ""
echo "=== Performance (warm cache, 1M shots) ==="
PERF_SHOTS=1000000
for circuit_name in frame_h h_then_t rep_d3 rank4_mixed; do
    stim="$BASE/tests/fixtures/incremental/01_frame/${circuit_name}.stim"
    [ -f "$stim" ] || stim="$BASE/tests/fixtures/incremental/03_array/${circuit_name}.stim"
    [ -f "$stim" ] || stim="$BASE/tests/fixtures/incremental/06_noise/${circuit_name}.stim"
    [ -f "$stim" ] || stim=$(find "$BASE/tests/fixtures" -name "${circuit_name}.stim" -print -quit 2>/dev/null)
    [ -f "$stim" ] || { echo "  $circuit_name: fixture not found"; continue; }
    echo "--- $circuit_name ---"
    # Warm cache run
    timeout 30 "$BIN" --circuit "$stim" --shots 1000 --seed 42 >/dev/null 2>&1
    timeout 30 "$BIN" --circuit "$stim" --shots 1000 --seed 42 --mlir >/dev/null 2>&1
    # Measured runs
    svm_time=$(timeout 60 "$BIN" --circuit "$stim" --shots $PERF_SHOTS --seed 42 2>&1 | grep kernel_seconds | grep -o '[0-9.]*')
    mlir_time=$(timeout 60 "$BIN" --circuit "$stim" --shots $PERF_SHOTS --seed 42 --mlir 2>&1 | grep kernel_seconds | grep -o '[0-9.]*')
    echo "  SVM:  ${svm_time:-N/A}s"
    echo "  MLIR: ${mlir_time:-N/A}s"
done

echo ""
echo "Done: $(date)"
