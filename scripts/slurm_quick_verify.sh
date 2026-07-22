#!/bin/bash
#SBATCH --job-name=quick-verify
#SBATCH --partition=mi300x
#SBATCH --nodelist=rad-mi300x-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/quick_verify_%j.log

set -e
BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BUILD="$BASE/build-gpu-verify-reorg"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done

echo "GPU arch: $(rocm_agent_enumerator | grep gfx)"
echo "Node: $(hostname)"
echo "Date: $(date)"

# Always rebuild to verify latest source
echo "=== Building ==="
rm -rf "$BUILD"
mkdir -p "$BUILD" && cd "$BUILD"
{
    cmake "$BASE" -DCLIFFT_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 \
        -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
    make -j$(nproc) run_gpu 2>&1 | tail -3
}

BIN="$BUILD/run_gpu"
echo "Binary: $BIN"
echo ""

# Quick correctness: compare SVM vs compiled kernel on representative circuits
PASS=0; FAIL=0
check_circuit() {
    local name=$1 stim=$2
    echo "--- $name ---"
    set +e
    svm=$("$BIN" --circuit "$stim" --shots 10000 --seed 42 2>&1)
    ck=$("$BIN" --circuit "$stim" --shots 10000 --seed 42 --hybrid 2>&1)
    set -e
    svm_le=$(echo "$svm" | grep '"logical_errors"' | grep -o '[0-9]*')
    ck_le=$(echo "$ck" | grep '"logical_errors"' | grep -o '[0-9]*')
    svm_ps=$(echo "$svm" | grep '"passed_shots"' | grep -o '[0-9]*')
    ck_ps=$(echo "$ck" | grep '"passed_shots"' | grep -o '[0-9]*')
    echo "  SVM: passed=$svm_ps logical=$svm_le"
    echo "  CK:  passed=$ck_ps logical=$ck_le"
    if [ "$svm_le" = "$ck_le" ] && [ "$svm_ps" = "$ck_ps" ]; then
        echo "  => EXACT MATCH"
        PASS=$((PASS+1))
    else
        echo "  => MISMATCH!"
        FAIL=$((FAIL+1))
    fi
}

# Tier 0 (rank 0): frame-only
check_circuit "target_qec" "$BASE/tests/fixtures/target_qec.stim"
check_circuit "color_d5" "$BASE/tests/fixtures/large/color_d5.stim"

# Tier 1 (rank 1-4): register
check_circuit "cultivation_d5" "$BASE/tests/fixtures/cultivation_d5.stim"
check_circuit "rank_r4_q33" "$BASE/tests/fixtures/rank_sweep/rank_q33_r4_d3.stim"

# Tier 2 (rank 5-10): coop/LDS
check_circuit "rank_r5_q33" "$BASE/tests/fixtures/rank_sweep/rank_q33_r5_d3.stim"
check_circuit "rank_r8_q33" "$BASE/tests/fixtures/rank_sweep/rank_q33_r8_d3.stim"
check_circuit "rank_r10_q33" "$BASE/tests/fixtures/rank_sweep/rank_q33_r10_d3.stim"

# Tier 3 (rank 11-19): global/HBM
check_circuit "rank_r12_q33" "$BASE/tests/fixtures/rank_sweep/rank_q33_r12_d3.stim"
check_circuit "rank_r15_q33" "$BASE/tests/fixtures/rank_sweep/rank_q33_r15_d3.stim"

echo ""
echo "=== Summary ==="
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
echo "Done: $(date)"
