#!/bin/bash
#SBATCH --job-name=verify-mlir-3tier
#SBATCH --partition=mi300x
#SBATCH --nodelist=rad-mi300x-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/verify_mlir_3tier_%j.log

set -e
BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BUILD="$BASE/build-gpu-mlir-3tier"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done

for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

echo "GPU arch: $(rocm_agent_enumerator | grep gfx)"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "LLVM_PREFIX: ${LLVM_PREFIX:-unset}"

echo "=== Building (HIP + MLIR) ==="
rm -rf "$BUILD"
mkdir -p "$BUILD" && cd "$BUILD"
cmake "$BASE" -DCLIFFT_ENABLE_HIP=ON -DCLIFFT_ENABLE_MLIR=ON \
    -DCMAKE_HIP_ARCHITECTURES=gfx942 \
    -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5
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

# Incremental per-op tests (register tier)
echo "--- Incremental Per-Op Tests (register tier) ---"
for stim in "$BASE"/tests/fixtures/incremental/*/*.stim; do
    [ -f "$stim" ] || continue
    name=$(basename "$stim" .stim)
    check_circuit "$name" "$stim"
done

# Rank sweep (covers all tiers)
echo ""
echo "--- Rank Sweep (register + coop + global tiers) ---"
for stim in "$BASE"/tests/fixtures/rank_sweep/*.stim; do
    [ -f "$stim" ] || continue
    name=$(basename "$stim" .stim)
    check_circuit "$name" "$stim"
done

# Large circuits (higher tiers)
echo ""
echo "--- Large Circuits (higher tiers) ---"
for stim in "$BASE"/tests/fixtures/large/*.stim; do
    [ -f "$stim" ] || continue
    name=$(basename "$stim" .stim)
    check_circuit "$name" "$stim"
done

# Standard circuits
echo ""
echo "--- Standard Circuits ---"
for stim in "$BASE"/tests/fixtures/*.stim; do
    [ -f "$stim" ] || continue
    name=$(basename "$stim" .stim)
    check_circuit "$name" "$stim"
done

echo ""
echo "=== Summary ==="
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
echo "  SKIP: $SKIP"
echo "  Total: $((PASS+FAIL+SKIP))"
echo ""
echo "Done: $(date)"
