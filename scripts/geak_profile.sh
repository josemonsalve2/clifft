#!/bin/bash
#SBATCH --job-name=geak-prof
#SBATCH --partition=mi300x
#SBATCH --nodelist=rad-mi300x-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:45:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/geak_profile_%j.log

# Run GEAK's profiling pipeline on clifft SVM and compiled kernels
set +e
BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
GEAK="/shared/jmonsalv/quantum/clifft_rl/GEAK"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROFDIR="$BASE/results/geak_profile_${TIMESTAMP}"
mkdir -p "$PROFDIR"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done

echo "=== GEAK Profiling Pipeline ==="
echo "Node: $(hostname)  Date: $(date)"
echo "GEAK: $GEAK"
echo "ROCm: $ROCM_PATH"

BINARY="$BASE/build-gpu-hsa/run_gpu"
[ -x "$BINARY" ] || BINARY="$BASE/build-gpu-node/run_gpu"
[ -x "$BINARY" ] || { echo "No binary"; exit 1; }
echo "Binary: $BINARY"

PROFILE_SCRIPT="$GEAK/kernel_workflow/scripts/profile_kernel.sh"
GPU_LOCK="$GEAK/kernel_workflow/scripts/gpu_lock.sh"

# Circuits to profile (one representative per tier)
T0_CIRC="$BASE/tests/fixtures/target_qec.stim"
T1_CIRC="$BASE/tests/fixtures/cultivation_d5.stim"
T2_CIRC="$BASE/tests/fixtures/large/surface_d7_t10.stim"

SHOTS=500000

# Profile each mode on each tier circuit using GEAK's profiler
profile_with_geak() {
    local MODE=$1 FLAG=$2 CIRCUIT=$3 LABEL=$4
    local OUTDIR="$PROFDIR/${LABEL}_${MODE}"
    mkdir -p "$OUTDIR"
    echo ""
    echo "--- GEAK profile: $MODE $LABEL ---"

    local CMD="$BINARY --circuit $CIRCUIT --shots $SHOTS $FLAG"
    echo "CMD: $CMD"

    # Use GEAK's profile_kernel.sh
    if [ -x "$PROFILE_SCRIPT" ]; then
        echo "Using GEAK profile_kernel.sh"
        bash "$PROFILE_SCRIPT" 0 "$CMD" "$OUTDIR" 2>&1 | tee "$OUTDIR/geak_output.txt"
    else
        echo "GEAK script not executable, using rocprof-compute directly"
        # Warmup
        $CMD >/dev/null 2>&1 || true
        $CMD >/dev/null 2>&1 || true
        $CMD >/dev/null 2>&1 || true

        # Try rocprof-compute (rocprofiler-compute)
        if command -v rocprof-compute >/dev/null 2>&1; then
            echo "Using rocprof-compute"
            rocprof-compute profile --no-roof -o "$OUTDIR" -- $CMD 2>&1 | tee "$OUTDIR/rocprof_compute.txt"
        elif command -v omniperf >/dev/null 2>&1; then
            echo "Using omniperf"
            omniperf profile --no-roof -o "$OUTDIR" -- $CMD 2>&1 | tee "$OUTDIR/omniperf.txt"
        elif command -v rocprofv3 >/dev/null 2>&1; then
            echo "Using rocprofv3"
            rocprofv3 --kernel-trace --stats --output-format csv -o "$OUTDIR/rocprofv3" -- $CMD 2>&1 | tee "$OUTDIR/rocprofv3.txt"
        elif command -v rocprof >/dev/null 2>&1; then
            echo "Using rocprof (legacy)"
            # Counter sets
            cat > "$OUTDIR/counters.txt" << 'CEOF'
pmc: SQ_WAVES SQ_INSTS_VALU SQ_INSTS_VMEM SQ_INSTS_LDS SQ_BUSY_CYCLES
pmc: TCC_HIT_sum TCC_MISS_sum TCC_EA_RDREQ_32B_sum
pmc: SQ_LDS_BANK_CONFLICT SQ_WAVE_CYCLES
CEOF
            rocprof --stats -i "$OUTDIR/counters.txt" -o "$OUTDIR/counters.csv" $CMD 2>&1 | tee "$OUTDIR/rocprof.txt"
        else
            echo "No profiler available!"
        fi
    fi

    echo "Profile data: $OUTDIR"
}

# TIER 0: Frame-only (rank=0)
profile_with_geak "svm"      ""        "$T0_CIRC" "tier0_target_qec"
profile_with_geak "compiled" "--hybrid" "$T0_CIRC" "tier0_target_qec"

# TIER 1: Register (rank=4)
profile_with_geak "svm"      ""        "$T1_CIRC" "tier1_cultivation_d5"
profile_with_geak "compiled" "--hybrid" "$T1_CIRC" "tier1_cultivation_d5"

# TIER 2: LDS/coop (rank=10)
profile_with_geak "svm"      ""        "$T2_CIRC" "tier2_surface_d7_t10"
profile_with_geak "compiled" "--hybrid" "$T2_CIRC" "tier2_surface_d7_t10"

# =============================================
# GEAK-style bottleneck classification
# =============================================
echo ""
echo "=== GEAK Bottleneck Classification ==="
python3 -c "
import os, csv, glob

profdir = '$PROFDIR'
print('Per-kernel profiling results:')
print(f\"{'Config':<40} {'Profiler':<15} {'Data'}\")
print('-'*80)

for subdir in sorted(os.listdir(profdir)):
    full = os.path.join(profdir, subdir)
    if not os.path.isdir(full): continue
    # Check what profiler data exists
    files = os.listdir(full)
    profiler = 'none'
    data_summary = 'no data'
    if 'profile_report.txt' in files:
        profiler = 'geak'
        with open(os.path.join(full, 'profile_report.txt')) as f:
            lines = f.readlines()
            data_summary = f'{len(lines)} lines of profiler output'
    elif any('rocprof' in f for f in files):
        profiler = 'rocprof'
        csvfiles = glob.glob(f'{full}/*.csv')
        if csvfiles:
            with open(csvfiles[0]) as f:
                rows = list(csv.DictReader(f))
                data_summary = f'{len(rows)} counter rows'
                # GEAK classification from counters
                for row in rows:
                    valu = float(row.get('SQ_INSTS_VALU', 0) or 0)
                    vmem = float(row.get('SQ_INSTS_VMEM', 0) or 0)
                    lds = float(row.get('SQ_INSTS_LDS', 0) or 0)
                    total = valu + vmem + lds
                    if total > 0:
                        vp = 100*valu/total; mp = 100*vmem/total; lp = 100*lds/total
                        if vp > 60 and mp < 40: cls = 'COMPUTE-BOUND'
                        elif mp > 60 and vp < 40: cls = 'MEMORY-BOUND'
                        elif lp > 50: cls = 'LDS-BOUND'
                        elif vp < 40 and mp < 40: cls = 'LATENCY-BOUND'
                        else: cls = 'BALANCED'
                        data_summary = f'VALU={vp:.0f}% VMEM={mp:.0f}% LDS={lp:.0f}% → {cls}'
    elif any('geak' in f for f in files):
        profiler = 'geak-script'
        with open(os.path.join(full, 'geak_output.txt')) as f:
            data_summary = f'{sum(1 for _ in f)} lines'
    print(f'{subdir:<40} {profiler:<15} {data_summary}')
" 2>/dev/null

echo ""
echo "Profile data: $PROFDIR"
echo "Done: $(date)"
