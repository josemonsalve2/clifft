#!/bin/bash
#SBATCH --job-name=qv10-dbg
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/qv10_dbg_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BUILD="$BASE/build-gpu-mlir-mi355x"
BIN="$BUILD/run_gpu"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

cd "$BUILD" && make -j$(nproc) 2>&1 | tail -3
echo "BUILD: ${PIPESTATUS[0]}"

rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null

echo "=== qv10 SVM ==="
$BIN --circuit "$BASE/tests/fixtures/qv10.stim" --shots 100 --seed 42 2>/dev/null | python3 -c "
import json,sys
d = json.load(sys.stdin)
print(f'passed={d[\"passed_shots\"]} rank={d[\"peak_rank\"]} det={d[\"detectors\"]} obs={d[\"observables\"]}')
"

echo "=== qv10 MLIR (compiling) ==="
timeout 120 $BIN --circuit "$BASE/tests/fixtures/qv10.stim" --shots 10 --seed 42 --mlir 2>&1 | head -5

echo "=== qv10 MLIR (cached run) ==="
$BIN --circuit "$BASE/tests/fixtures/qv10.stim" --shots 100 --seed 42 --mlir 2>/dev/null | python3 -c "
import json,sys
d = json.load(sys.stdin)
print(f'passed={d[\"passed_shots\"]} rank={d[\"peak_rank\"]} discarded={d[\"discarded_shots\"]} logical={d[\"logical_errors\"]}')
" 2>/dev/null || echo "MLIR JSON parse failed"

echo ""
echo "=== Check coop MLIR debug ==="
ls -la $BASE/results/debug_coop_*.mlir 2>/dev/null
echo ""
for f in $BASE/results/debug_coop_*.mlir; do
    [ -f "$f" ] || continue
    echo "--- $f ---"
    grep "// op\[" "$f" | head -20
    echo "..."
    grep "// op\[" "$f" | tail -5
    echo ""
    echo "Discard stores: $(grep -c 'store.*c1_i8.*discarded' "$f")"
    echo "Unsupported: $(grep -c 'Unsupported' "$f")"
    echo "U2/U4: $(grep -c 'U2\|U4\|u2\|u4' "$f")"
    echo ""
done

echo "Done: $(date)"
