#!/bin/bash
#SBATCH --job-name=codegen-bench
#SBATCH --partition=mi300x
#SBATCH --nodelist=rad-mi300x-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/codegen_bench_%j.log

# Build with AMDGCN-native codegen, test ALL circuits, benchmark
set +e
BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS="$BASE/results/codegen_bench_${TIMESTAMP}"
mkdir -p "$RESULTS"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done

echo "=== AMDGCN-native codegen test ==="
echo "Node: $(hostname)  Date: $(date)"
GPU_ARCH=$(rocminfo 2>/dev/null | grep -m1 "Name:.*gfx" | grep -oE "gfx[0-9]+" || echo "gfx942")

# Build
BUILD="$BASE/build-gpu-amdgcn"
rm -rf "$BUILD"
mkdir -p "$BUILD" && cd "$BUILD"
cmake "$BASE" -DCLIFFT_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 \
    -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
make -j$(nproc) run_gpu 2>&1 | tail -5
RC=$?
echo "Build: $RC"
[ $RC -ne 0 ] && exit 1
BINARY="$BUILD/run_gpu"

# Clear kernel cache
rm -rf ~/.clifft/kernel_cache/*.hsaco

# Quick smoke test
echo ""
echo "=== Smoke test ==="
echo "SVM:"
$BINARY --circuit "$BASE/tests/fixtures/target_qec.stim" --shots 100 2>&1 | grep "passed_shots"
echo "Compiled:"
$BINARY --circuit "$BASE/tests/fixtures/target_qec.stim" --shots 100 --hybrid 2>&1 | grep -E "passed_shots|error|FAIL"

# Check if .hsaco was created
echo ""
ls ~/.clifft/kernel_cache/*.hsaco 2>/dev/null && echo "HSACO CREATED — compiled kernel is working!" || echo "NO HSACO — compilation still failing"

# If HSACO exists, run the incremental test suite
HSACO_EXISTS=$(ls ~/.clifft/kernel_cache/*.hsaco 2>/dev/null | wc -l)

echo ""
echo "=== Incremental tests (20 single-gate circuits) ==="
CSV="$RESULTS/incr.csv"
echo "circuit,mode,shots,passed,sample_sec,shots_per_sec,status" > "$CSV"

extract_field() { echo "$1" | grep "\"$2\"" | head -1 | grep -oE '[0-9]+\.?[0-9eE+\-]*' || true; }

for CIRCUIT in "$BASE/tests/fixtures/incremental"/{01,02,03,04,05,06}_*/*.stim; do
    [ -f "$CIRCUIT" ] || continue
    NAME=$(basename "$CIRCUIT" .stim)
    printf "%-24s " "$NAME"

    for MODE in svm compiled; do
        FLAG=""
        [ "$MODE" = "compiled" ] && FLAG="--hybrid"
        OUT=$($BINARY --circuit "$CIRCUIT" --shots 100000 $FLAG 2>&1)
        RC=$?
        if [ $RC -ne 0 ]; then
            printf "%-8s FAIL " "$MODE"
            echo "$NAME,$MODE,100000,0,0,0,FAIL:rc=$RC" >> "$CSV"
        else
            SPS=$(extract_field "$OUT" "shots_per_second_sampling_only"); SPS="${SPS:-0}"
            PASSED=$(extract_field "$OUT" "passed_shots"); PASSED="${PASSED:-0}"
            SEC=$(extract_field "$OUT" "sample_seconds"); SEC="${SEC:-0}"
            printf "%-8s %10s/s " "$MODE" "$SPS"
            echo "$NAME,$MODE,100000,$PASSED,$SEC,$SPS,PASS" >> "$CSV"
        fi
    done
    echo ""
done

echo ""
echo "=== Full circuit sweep (all .stim files, 100K shots) ==="
ALL_CSV="$RESULTS/all.csv"
echo "circuit,mode,shots,passed,sample_sec,shots_per_sec,status" > "$ALL_CSV"
COUNT=0
TOTAL=$(find "$BASE/tests/fixtures" -name "*.stim" | wc -l)

for CIRCUIT in $(find "$BASE/tests/fixtures" -name "*.stim" | sort); do
    COUNT=$((COUNT + 1))
    NAME=$(basename "$CIRCUIT" .stim)

    for MODE in svm compiled; do
        FLAG=""
        [ "$MODE" = "compiled" ] && FLAG="--hybrid"
        OUT=$($BINARY --circuit "$CIRCUIT" --shots 100000 $FLAG 2>&1)
        RC=$?
        if [ $RC -ne 0 ]; then
            echo "$NAME,$MODE,100000,0,0,0,FAIL:rc=$RC" >> "$ALL_CSV"
        else
            SPS=$(extract_field "$OUT" "shots_per_second_sampling_only"); SPS="${SPS:-0}"
            PASSED=$(extract_field "$OUT" "passed_shots"); PASSED="${PASSED:-0}"
            SEC=$(extract_field "$OUT" "sample_seconds"); SEC="${SEC:-0}"
            echo "$NAME,$MODE,100000,$PASSED,$SEC,$SPS,PASS" >> "$ALL_CSV"
        fi
    done

    [ $((COUNT % 50)) -eq 0 ] && echo "[$COUNT/$TOTAL] processed..."
done

echo ""
echo "=== Summary ==="
python3 -c "
import csv
from collections import defaultdict
data = defaultdict(lambda: defaultdict(list))
fails = defaultdict(lambda: defaultdict(int))
with open('$ALL_CSV') as f:
    for row in csv.DictReader(f):
        if row['status'].startswith('PASS'):
            sps = float(row.get('shots_per_sec',0) or 0)
            if sps > 0: data[row['circuit']][row['mode']].append(sps)
        else:
            fails[row['circuit']][row['mode']] += 1

total = len(set(list(data.keys()) + list(fails.keys())))
svm_pass = sum(1 for c in data if data[c].get('svm'))
comp_pass = sum(1 for c in data if data[c].get('compiled'))
svm_fail = sum(1 for c in fails if fails[c].get('svm',0)>0)
comp_fail = sum(1 for c in fails if fails[c].get('compiled',0)>0)

print(f'Total circuits: {total}')
print(f'SVM:      {svm_pass} pass, {svm_fail} fail')
print(f'Compiled: {comp_pass} pass, {comp_fail} fail')

# Per-circuit comparison
ratios = []
for c in data:
    s = data[c].get('svm',[])
    co = data[c].get('compiled',[])
    if s and co:
        sm = sum(s)/len(s); cm = sum(co)/len(co)
        if sm > 0 and cm > 0: ratios.append(cm/sm)

if ratios:
    print(f'Compiled/SVM ratio: mean={sum(ratios)/len(ratios):.3f}x  min={min(ratios):.3f}x  max={max(ratios):.3f}x')
    near = sum(1 for r in ratios if 0.9 <= r <= 1.1)
    print(f'Near parity (0.9-1.1x): {near}/{len(ratios)} ({100*near/len(ratios):.0f}%)')
" 2>/dev/null

echo ""
echo "CSV: $ALL_CSV"
echo "Done: $(date)"
