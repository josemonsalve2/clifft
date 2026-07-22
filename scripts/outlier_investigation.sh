#!/bin/bash
#SBATCH --job-name=outlier-188pct
#SBATCH --partition=mi300x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/outlier_investigation_%j.log
#SBATCH --exclusive

# Investigation: Is the 188% outlier at q=33, t=10, d=5 real?
#
# Original RESULTS.md reported:
#   SVM:      3.65M shots/s
#   Compiled: 10.5M shots/s  => +188%
#
# But the CSV data shows:
#   SVM:      10.32M shots/s
#   Compiled: 10.54M shots/s => +2.2%
#
# This suggests the 3.65M was a transient performance drop.
# We run 20 iterations of each to prove it statistically.

set -euo pipefail

BINARY="/shared/jmonsalv/quantum/clifft_rl/clifft/build-gpu/run_gpu"
CIRCUIT="/shared/jmonsalv/quantum/clifft_rl/clifft/tests/fixtures/sweep/sweep_q33_t10_d5.stim"
OUTDIR="/shared/jmonsalv/quantum/clifft_rl/clifft/results"
mkdir -p "$OUTDIR"
OUTFILE="$OUTDIR/outlier_investigation_$(date +%Y%m%d_%H%M%S).csv"

SHOTS=1000000
RUNS=20

echo "mode,circuit,run,shots_per_second,sample_seconds,passed" > "$OUTFILE"

echo "=== GPU Info ==="
rocm-smi --showid --showproductname 2>/dev/null | head -8 || true
echo ""
echo "=== Hostname: $(hostname) ==="
echo ""

# Warm up GPU (3 warmup runs)
echo "=== Warming up GPU (3 runs) ==="
for i in 1 2 3; do
    $BINARY --circuit "$CIRCUIT" --shots 100000 > /dev/null 2>&1
done
echo "Warmup done."
echo ""

extract_sps() {
    echo "$1" | grep '"shots_per_second_sampling_only"' | head -1 | grep -oE '[0-9]+\.[0-9eE+\-]+'
}
extract_sec() {
    echo "$1" | grep '"sample_seconds"' | head -1 | grep -oE '[0-9]+\.[0-9eE+\-]+'
}
extract_passed() {
    echo "$1" | grep '"passed_shots"' | head -1 | grep -o '[0-9]*'
}

# Phase 1: SVM (default mode, no --hybrid flag) x 20 runs
echo "=============================================="
echo "PHASE 1: SVM Interpreter x $RUNS runs"
echo "Circuit: sweep_q33_t10_d5.stim (q=33, t=10, d=5)"
echo "=============================================="
SVM_VALUES=""
for i in $(seq 1 $RUNS); do
    OUTPUT=$($BINARY --circuit "$CIRCUIT" --shots "$SHOTS" 2>&1)
    SPS=$(extract_sps "$OUTPUT")
    SEC=$(extract_sec "$OUTPUT")
    PASSED=$(extract_passed "$OUTPUT")
    printf "  SVM Run %2d: %s shots/s\n" "$i" "$SPS"
    echo "svm,sweep_q33_t10_d5,$i,$SPS,$SEC,$PASSED" >> "$OUTFILE"
    SVM_VALUES="$SVM_VALUES $SPS"
done
echo ""

# Phase 2: Compiled (--hybrid flag) x 20 runs
echo "=============================================="
echo "PHASE 2: Compiled Megakernel x $RUNS runs"
echo "=============================================="
COMP_VALUES=""
for i in $(seq 1 $RUNS); do
    OUTPUT=$($BINARY --circuit "$CIRCUIT" --shots "$SHOTS" --hybrid 2>&1)
    SPS=$(extract_sps "$OUTPUT")
    SEC=$(extract_sec "$OUTPUT")
    PASSED=$(extract_passed "$OUTPUT")
    printf "  Compiled Run %2d: %s shots/s\n" "$i" "$SPS"
    echo "compiled,sweep_q33_t10_d5,$i,$SPS,$SEC,$PASSED" >> "$OUTFILE"
    COMP_VALUES="$COMP_VALUES $SPS"
done
echo ""

# Phase 3: Context — same circuit, more runs to check for transient drops
echo "=============================================="
echo "PHASE 3: Interleaved SVM/Compiled x 10 (detect transient drops)"
echo "=============================================="
for i in $(seq 1 10); do
    OUTPUT_SVM=$($BINARY --circuit "$CIRCUIT" --shots "$SHOTS" 2>&1)
    SPS_SVM=$(extract_sps "$OUTPUT_SVM")

    OUTPUT_COMP=$($BINARY --circuit "$CIRCUIT" --shots "$SHOTS" --hybrid 2>&1)
    SPS_COMP=$(extract_sps "$OUTPUT_COMP")

    printf "  Pair %2d: SVM=%s  Compiled=%s\n" "$i" "$SPS_SVM" "$SPS_COMP"
    echo "svm_interleaved,sweep_q33_t10_d5,$i,$SPS_SVM,0,0" >> "$OUTFILE"
    echo "compiled_interleaved,sweep_q33_t10_d5,$i,$SPS_COMP,0,0" >> "$OUTFILE"
done
echo ""

# Phase 4: Context circuits (q=33, varying t, d=5) — 5 runs each SVM only
echo "=============================================="
echo "PHASE 4: Context circuits (q=33, d=5, varying t) — 5 runs SVM"
echo "=============================================="
for t in 1 3 5 8; do
    CTX_CIRCUIT="/shared/jmonsalv/quantum/clifft_rl/clifft/tests/fixtures/sweep/sweep_q33_t${t}_d5.stim"
    if [ ! -f "$CTX_CIRCUIT" ]; then
        echo "  SKIP: t=$t not found"
        continue
    fi
    echo "  --- t=$t ---"
    for i in $(seq 1 5); do
        OUTPUT=$($BINARY --circuit "$CTX_CIRCUIT" --shots "$SHOTS" 2>&1)
        SPS=$(extract_sps "$OUTPUT")
        printf "    Run %d: %s shots/s\n" "$i" "$SPS"
        echo "svm_context,sweep_q33_t${t}_d5,$i,$SPS,0,0" >> "$OUTFILE"
    done
done
echo ""

# Phase 5: Statistical summary
echo "=============================================="
echo "PHASE 5: STATISTICAL SUMMARY"
echo "=============================================="
echo ""

for MODE in svm compiled; do
    echo "$MODE runs on target circuit:"
    grep "^${MODE},sweep_q33_t10_d5," "$OUTFILE" | awk -F',' '{print $4}' | sort -g | awk '
    BEGIN { n=0; sum=0 }
    {
        v=$1+0; vals[n]=v; sum+=v; n++
    }
    END {
        if (n==0) { print "  No data"; exit }
        mean=sum/n
        for (i=0;i<n;i++) ss+=(vals[i]-mean)^2
        sd=sqrt(ss/(n>1 ? n-1 : 1))
        printf "  N=%d  Mean=%.2fM  StdDev=%.2fM  CV=%.1f%%\n", n, mean/1e6, sd/1e6, 100*sd/mean
        printf "  Min=%.2fM  Max=%.2fM  Range=%.2fM\n", vals[0]/1e6, vals[n-1]/1e6, (vals[n-1]-vals[0])/1e6
        printf "  Median=%.2fM  Q1=%.2fM  Q3=%.2fM\n", vals[int(n/2)]/1e6, vals[int(n*0.25)]/1e6, vals[int(n*0.75)]/1e6
    }'
    echo ""
done

echo "Interleaved runs:"
grep "svm_interleaved" "$OUTFILE" | awk -F',' '{print $4}' | sort -g | awk '
BEGIN { n=0; sum=0 }
{ v=$1+0; vals[n]=v; sum+=v; n++ }
END {
    mean=sum/n
    for (i=0;i<n;i++) ss+=(vals[i]-mean)^2
    sd=sqrt(ss/(n>1 ? n-1 : 1))
    printf "  SVM interleaved: N=%d  Mean=%.2fM  StdDev=%.2fM  CV=%.1f%%\n", n, mean/1e6, sd/1e6, 100*sd/mean
    printf "    Min=%.2fM  Max=%.2fM\n", vals[0]/1e6, vals[n-1]/1e6
}'
grep "compiled_interleaved" "$OUTFILE" | awk -F',' '{print $4}' | sort -g | awk '
BEGIN { n=0; sum=0 }
{ v=$1+0; vals[n]=v; sum+=v; n++ }
END {
    mean=sum/n
    for (i=0;i<n;i++) ss+=(vals[i]-mean)^2
    sd=sqrt(ss/(n>1 ? n-1 : 1))
    printf "  Compiled interleaved: N=%d  Mean=%.2fM  StdDev=%.2fM  CV=%.1f%%\n", n, mean/1e6, sd/1e6, 100*sd/mean
    printf "    Min=%.2fM  Max=%.2fM\n", vals[0]/1e6, vals[n-1]/1e6
}'

echo ""
echo "Conclusion:"
echo "If SVM mean ≈ Compiled mean ≈ 10-11M shots/s with CV < 5%,"
echo "then the 3.65M outlier was a transient performance drop."
echo "If any SVM run drops below 5M, the issue may be reproducible."
echo ""
echo "All results: $OUTFILE"
echo "=== Investigation complete ==="
