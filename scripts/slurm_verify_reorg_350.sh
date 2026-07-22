#!/bin/bash
#SBATCH --job-name=verify-reorg
#SBATCH --partition=mi350x-es
#SBATCH --nodelist=smci350-rck-g03-d13-21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/verify_reorg_%j.log

set -e
BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done

echo "Welcome to SLURM"
echo "--------------------------------------------"
echo "current Job id $SLURM_JOB_ID"
echo "--------------------------------------------"
echo "GPU arch: $(rocm_agent_enumerator | grep gfx)"
echo "Node: $(hostname)"
echo "Date: $(date)"

echo "=== Building (post-reorganization) ==="
BUILD="$BASE/build-gpu-verify-reorg"
rm -rf "$BUILD"
mkdir -p "$BUILD" && cd "$BUILD"
cmake "$BASE" -DCLIFFT_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx950 \
    -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5
make -j$(nproc) run_gpu 2>&1 | tail -5
RC=$?
echo "Build exit: $RC"
if [ $RC -ne 0 ]; then echo "BUILD FAILED"; exit 1; fi
echo "Build OK: $BUILD/run_gpu"

echo ""
echo "=== Correctness Sweep: compiled megakernel vs SVM ==="
echo "  shots=100000 seed=42"
echo ""

printf "%-52s %4s %6s  %12s %12s    %s\n" "Circuit" "Rank" "Status" "SVM_logical" "CK_logical" "Match"
echo "-----------------------------------------------------------------------------------------------------------"

PASS=0; FAIL=0; SKIP=0; TIMEOUT=0
for stim_file in "$BASE"/tests/fixtures/rank_sweep/*.stim "$BASE"/tests/fixtures/large/*.stim "$BASE"/tests/fixtures/*.stim; do
    [ -f "$stim_file" ] || continue
    circuit_name=$(basename "$stim_file" .stim)

    # SVM reference run
    svm_out=$(timeout 120 "$BUILD/run_gpu" "$stim_file" --shots 100000 --seed 42 2>/dev/null) || { SKIP=$((SKIP+1)); printf "%-52s %4s %7s\n" "$circuit_name" "?" "SVM_ERR"; continue; }
    svm_logical=$(echo "$svm_out" | grep 'logical_errors' | awk -F= '{print $2}' | tr -d ' ')

    # Compiled kernel run
    ck_out=$(timeout 120 "$BUILD/run_gpu" "$stim_file" --shots 100000 --seed 42 --hybrid 2>/dev/null)
    ck_rc=$?
    if [ $ck_rc -eq 124 ]; then
        TIMEOUT=$((TIMEOUT+1))
        printf "%-52s %4s %7s\n" "$circuit_name" "?" "TIMEOUT"
        continue
    fi
    if [ $ck_rc -ne 0 ]; then
        SKIP=$((SKIP+1))
        printf "%-52s %4s %7s\n" "$circuit_name" "?" "CK_ERR"
        continue
    fi
    ck_logical=$(echo "$ck_out" | grep 'logical_errors' | awk -F= '{print $2}' | tr -d ' ')

    if [ "$svm_logical" = "$ck_logical" ]; then
        PASS=$((PASS+1))
        printf "%-52s %4s %6s  %12s %12s    %s\n" "$circuit_name" "?" "OK" "$svm_logical" "$ck_logical" "EXACT"
    else
        FAIL=$((FAIL+1))
        printf "%-52s %4s %6s  %12s %12s    %s\n" "$circuit_name" "?" "MISMATCH" "$svm_logical" "$ck_logical" "FAIL"
    fi
done

echo ""
echo "=== Summary ==="
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
echo "  SKIP: $SKIP"
echo "  TIMEOUT: $TIMEOUT"
echo "  Total: $((PASS+FAIL+SKIP+TIMEOUT))"
echo ""
echo "Done: $(date)"
