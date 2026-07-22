#!/bin/bash
#SBATCH --job-name=quick-hwc
#SBATCH --partition=mi300x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/quick_hwc_%j.log
set +e

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
export PATH="/shared/jmonsalv/software/cmake-3.31.7-linux-x86_64/bin:$PATH"
for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done

GPU_ARCH=$(rocminfo 2>/dev/null | grep -m1 'Name:.*gfx' | grep -oE 'gfx[0-9]+')
BINARY="$BASE/build-gpu-perf/run_gpu"
[ -x "$BINARY" ] || { echo "NO BINARY"; exit 1; }

echo "=== Quick Hardware Counter Collection ==="
echo "Node: $(hostname)  GPU: $GPU_ARCH"
echo "Date: $(date)"

RESULTS="$BASE/results/hwcounters_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS"

# 5 representative circuits: rank0, rank1, rank4, rank7, rank10
CIRCUITS=(
    "$BASE/tests/fixtures/target_qec.stim:rank0_target_qec"
    "$BASE/tests/fixtures/incremental/06_small_circuits/rank4_mixed.stim:rank1_rank4_mixed"
    "$BASE/tests/fixtures/large/circuit_d3_p0.001.stim:rank4_circuit_d3"
    "$BASE/tests/fixtures/large/surface_d7_t10.stim:rank7_surface_d7_t10"
    "$BASE/tests/fixtures/large/circuit_d5_p0.001.stim:rank10_circuit_d5"
)

# Check available profilers
echo ""
echo "Checking profilers..."
which rocprofv3 2>/dev/null && echo "rocprofv3: available" || echo "rocprofv3: not found"
which rocprof 2>/dev/null && echo "rocprof: available" || echo "rocprof: not found"
ls "$ROCM_PATH/libexec/rocprofiler-compute/rocprof-compute" 2>/dev/null && echo "rocprof-compute: available" || echo "rocprof-compute: not found"
echo ""

# Warmup all circuits first
echo "=== Warming cache ==="
for entry in "${CIRCUITS[@]}"; do
    CIRC="${entry%%:*}"
    LABEL="${entry##*:}"
    $BINARY --circuit "$CIRC" --shots 100 --hybrid >/dev/null 2>&1
    echo "Warmed: $LABEL"
done

SHOTS=1000000

for entry in "${CIRCUITS[@]}"; do
    CIRC="${entry%%:*}"
    LABEL="${entry##*:}"
    
    for MODE in svm compiled; do
        CMD="$BINARY --circuit $CIRC --shots $SHOTS"
        [ "$MODE" = "compiled" ] && CMD="$CMD --hybrid"
        
        OUTDIR="$RESULTS/${LABEL}_${MODE}"
        mkdir -p "$OUTDIR"
        
        echo ""
        echo "=== Profiling: $LABEL ($MODE) ==="
        
        # Try rocprofv3 first
        if command -v rocprofv3 &>/dev/null; then
            echo "--- rocprofv3 kernel trace + stats ---"
            rocprofv3 --kernel-trace --stats -d "$OUTDIR/trace" -- $CMD 2>"$OUTDIR/trace_stderr.log"
            
            echo "--- rocprofv3 instruction counters ---"
            cat > "$OUTDIR/counters_insts.txt" <<'COUNTERS'
pmc: SQ_INSTS_VALU SQ_INSTS_VMEM SQ_INSTS_SALU SQ_INSTS_LDS
COUNTERS
            rocprofv3 -i "$OUTDIR/counters_insts.txt" -d "$OUTDIR/insts" -- $CMD 2>"$OUTDIR/insts_stderr.log"
            
            echo "--- rocprofv3 wait/stall counters ---"
            cat > "$OUTDIR/counters_wait.txt" <<'COUNTERS'
pmc: SQ_WAIT_INST_LDS SQ_WAIT_INST_VMEM SQ_WAIT_INST_ANY SQ_BUSY_CU_CYCLES
COUNTERS
            rocprofv3 -i "$OUTDIR/counters_wait.txt" -d "$OUTDIR/wait" -- $CMD 2>"$OUTDIR/wait_stderr.log"
            
            echo "--- rocprofv3 cache counters ---"
            cat > "$OUTDIR/counters_cache.txt" <<'COUNTERS'
pmc: TCP_TOTAL_READ TCP_TOTAL_WRITE TCC_HIT TCC_MISS
COUNTERS
            rocprofv3 -i "$OUTDIR/counters_cache.txt" -d "$OUTDIR/cache" -- $CMD 2>"$OUTDIR/cache_stderr.log"
            
        elif command -v rocprof &>/dev/null; then
            echo "--- rocprof stats ---"
            rocprof --stats -o "$OUTDIR/stats.csv" $CMD 2>"$OUTDIR/stderr.log"
        fi
        
        echo "Done: $LABEL ($MODE)"
    done
done

# Register pressure from .hsaco files
echo ""
echo "=== Register Pressure (llvm-objdump) ==="
OBJDUMP="$ROCM_PATH/lib/llvm/bin/llvm-objdump"
if [ -x "$OBJDUMP" ]; then
    for hsaco in ~/.clifft/kernel_cache/*.hsaco; do
        [ -f "$hsaco" ] || continue
        HASH=$(basename "$hsaco" | sed 's/_gfx[0-9]*.hsaco//')
        INFO=$($OBJDUMP -t "$hsaco" 2>/dev/null | head -5)
        DISASM=$($OBJDUMP --disassemble "$hsaco" 2>/dev/null | grep -cE "v_|s_|ds_" || true)
        echo "$HASH: $DISASM instructions"
    done | sort -t: -k2 -rn | head -10
    echo "(top 10 by instruction count)"
fi

# Summary analysis
echo ""
echo "=== Hardware Counter Summary ==="
python3 - "$RESULTS" <<'PYEOF'
import sys, os, csv, glob

results_dir = sys.argv[1]

# Parse rocprofv3 kernel stats
for sdir in sorted(glob.glob(os.path.join(results_dir, "*/trace/"))):
    label = os.path.basename(os.path.dirname(sdir))
    for f in glob.glob(os.path.join(sdir, "**/*kernel_stats*"), recursive=True):
        print(f"\n--- {label} kernel stats ---")
        try:
            with open(f) as csvf:
                for line in csvf:
                    print(f"  {line.strip()}")
        except: pass

# Parse instruction counters
for idir in sorted(glob.glob(os.path.join(results_dir, "*/insts/"))):
    label = os.path.basename(os.path.dirname(idir))
    for f in glob.glob(os.path.join(idir, "**/*.csv"), recursive=True):
        try:
            with open(f) as csvf:
                reader = csv.DictReader(csvf)
                for row in reader:
                    valu = int(row.get('SQ_INSTS_VALU', 0) or 0)
                    vmem = int(row.get('SQ_INSTS_VMEM', 0) or 0)
                    salu = int(row.get('SQ_INSTS_SALU', 0) or 0)
                    lds = int(row.get('SQ_INSTS_LDS', 0) or 0)
                    total = valu + vmem + salu + lds
                    if total == 0: continue
                    valu_pct = valu * 100 / total
                    vmem_pct = vmem * 100 / total
                    salu_pct = salu * 100 / total
                    lds_pct = lds * 100 / total
                    if valu_pct > 60 and vmem_pct < 40: bn = "COMPUTE-BOUND"
                    elif vmem_pct > 60 and valu_pct < 40: bn = "MEMORY-BOUND"
                    elif lds_pct > 50: bn = "LDS-BOUND"
                    elif valu_pct < 40 and vmem_pct < 40: bn = "LATENCY-BOUND"
                    else: bn = "BALANCED"
                    print(f"  {label:<40} VALU={valu_pct:5.1f}% VMEM={vmem_pct:5.1f}% SALU={salu_pct:5.1f}% LDS={lds_pct:5.1f}%  => {bn}")
        except: pass

# Parse wait counters
for wdir in sorted(glob.glob(os.path.join(results_dir, "*/wait/"))):
    label = os.path.basename(os.path.dirname(wdir))
    for f in glob.glob(os.path.join(wdir, "**/*.csv"), recursive=True):
        try:
            with open(f) as csvf:
                reader = csv.DictReader(csvf)
                for row in reader:
                    wait_lds = int(row.get('SQ_WAIT_INST_LDS', 0) or 0)
                    wait_vmem = int(row.get('SQ_WAIT_INST_VMEM', 0) or 0)
                    wait_any = int(row.get('SQ_WAIT_INST_ANY', 0) or 0)
                    busy = int(row.get('SQ_BUSY_CU_CYCLES', 0) or 0)
                    if wait_any > 0:
                        print(f"  {label:<40} WaitLDS={wait_lds} WaitVMEM={wait_vmem} WaitAny={wait_any} BusyCU={busy}")
        except: pass

# Parse cache counters
for cdir in sorted(glob.glob(os.path.join(results_dir, "*/cache/"))):
    label = os.path.basename(os.path.dirname(cdir))
    for f in glob.glob(os.path.join(cdir, "**/*.csv"), recursive=True):
        try:
            with open(f) as csvf:
                reader = csv.DictReader(csvf)
                for row in reader:
                    tr = int(row.get('TCP_TOTAL_READ', 0) or 0)
                    tw = int(row.get('TCP_TOTAL_WRITE', 0) or 0)
                    l2h = int(row.get('TCC_HIT', 0) or 0)
                    l2m = int(row.get('TCC_MISS', 0) or 0)
                    if tr + tw > 0:
                        l2_total = l2h + l2m
                        l2_hr = l2h * 100 / l2_total if l2_total > 0 else 0
                        print(f"  {label:<40} L1Read={tr} L1Write={tw} L2Hit={l2h} L2Miss={l2m} L2HR={l2_hr:.1f}%")
        except: pass

PYEOF

echo ""
echo "Results in: $RESULTS"
echo "Done: $(date)"
