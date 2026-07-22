#!/bin/bash
#SBATCH --job-name=rocprof
#SBATCH --partition=mi300x
#SBATCH --nodelist=rad-mi300x-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/rocprof_direct_%j.log

# Direct rocprof hardware counter collection (bypassing broken rocprof-compute)
set +e
BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROFDIR="$BASE/results/rocprof_direct_${TIMESTAMP}"
mkdir -p "$PROFDIR"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done

echo "Node: $(hostname)  Date: $(date)"
BINARY="$BASE/build-gpu-hsa/run_gpu"
[ -x "$BINARY" ] || BINARY="$BASE/build-gpu-node/run_gpu"
echo "Binary: $BINARY"

SHOTS=500000

# Counter sets for GEAK-style classification
# Set A: instruction mix (VALU vs VMEM vs LDS → compute vs memory vs LDS bound)
# Set B: cache hierarchy (L2 hit rate → memory efficiency)
# Set C: wavefront efficiency (occupancy, stalls)
PSET_A="$PROFDIR/set_a.txt"
PSET_B="$PROFDIR/set_b.txt"
PSET_C="$PROFDIR/set_c.txt"
echo "pmc: SQ_WAVES SQ_INSTS_VALU SQ_INSTS_VMEM SQ_INSTS_LDS SQ_BUSY_CYCLES SQ_WAVE_CYCLES" > "$PSET_A"
echo "pmc: TCC_HIT_sum TCC_MISS_sum TCC_EA_RDREQ_32B_sum TCC_EA_WRREQ_sum" > "$PSET_B"
echo "pmc: SQ_LDS_BANK_CONFLICT GRBM_COUNT" > "$PSET_C"

profile_one() {
    local MODE=$1 FLAG=$2 CIRCUIT=$3 LABEL=$4
    [ -f "$CIRCUIT" ] || return
    local OUTDIR="$PROFDIR/${LABEL}_${MODE}"
    mkdir -p "$OUTDIR"
    echo ""
    echo "--- $MODE $LABEL ---"

    local CMD="$BINARY --circuit $CIRCUIT --shots $SHOTS $FLAG"
    # Warmup
    $CMD >/dev/null 2>&1 || true

    for PSET in "$PSET_A" "$PSET_B" "$PSET_C"; do
        local PNAME=$(basename "$PSET" .txt)
        echo "  Counter set: $PNAME"
        rocprof --stats -i "$PSET" -o "$OUTDIR/${PNAME}.csv" $CMD >/dev/null 2>&1
        RC=$?
        if [ $RC -ne 0 ]; then
            echo "    rocprof failed (rc=$RC) for $PNAME"
            # Try rocprofv3 as fallback
            rocprofv3 --kernel-trace --stats -o "$OUTDIR/${PNAME}_v3" $CMD >/dev/null 2>&1 || echo "    rocprofv3 also failed"
        else
            echo "    OK: $(wc -l < "$OUTDIR/${PNAME}.csv") rows"
        fi
    done
}

# Profile representative circuits
T0="$BASE/tests/fixtures/target_qec.stim"
T1="$BASE/tests/fixtures/cultivation_d5.stim"
T2="$BASE/tests/fixtures/large/surface_d7_t10.stim"

profile_one "svm"      ""        "$T0" "tier0_target_qec"
profile_one "compiled" "--hybrid" "$T0" "tier0_target_qec"
profile_one "svm"      ""        "$T1" "tier1_cultivation_d5"
profile_one "compiled" "--hybrid" "$T1" "tier1_cultivation_d5"
profile_one "svm"      ""        "$T2" "tier2_surface_d7_t10"
profile_one "compiled" "--hybrid" "$T2" "tier2_surface_d7_t10"

# Analyze with GEAK classification
echo ""
echo "=== GEAK Bottleneck Classification from Hardware Counters ==="
python3 << 'PYEOF'
import os, csv, glob, sys

profdir = os.environ.get('PROFDIR', sys.argv[1] if len(sys.argv) > 1 else '.')

print(f"{'Config':<45} {'VALU%':>7} {'VMEM%':>7} {'LDS%':>7} {'L2hit%':>7} {'LDS_conf':>9} {'CLASS'}")
print("-" * 100)

for subdir in sorted(os.listdir(profdir)):
    full = os.path.join(profdir, subdir)
    if not os.path.isdir(full): continue
    if subdir.startswith('set_'): continue

    counters = {}
    for pset in ['set_a', 'set_b', 'set_c']:
        cf = os.path.join(full, f'{pset}.csv')
        if not os.path.exists(cf): continue
        try:
            with open(cf) as f:
                for row in csv.DictReader(f):
                    for k, v in row.items():
                        k = k.strip()
                        try: counters[k] = counters.get(k, 0) + float(v or 0)
                        except ValueError: pass
        except Exception: pass

    valu = counters.get('SQ_INSTS_VALU', 0)
    vmem = counters.get('SQ_INSTS_VMEM', 0)
    lds = counters.get('SQ_INSTS_LDS', 0)
    total = valu + vmem + lds
    l2hit = counters.get('TCC_HIT_sum', 0)
    l2miss = counters.get('TCC_MISS_sum', 0)
    l2total = l2hit + l2miss
    ldsconf = counters.get('SQ_LDS_BANK_CONFLICT', 0)

    if total == 0:
        print(f"{subdir:<45} {'no data':>7}")
        continue

    vp = 100 * valu / total
    mp = 100 * vmem / total
    lp = 100 * lds / total
    l2h = 100 * l2hit / l2total if l2total > 0 else 0
    ldc = 100 * ldsconf / lds if lds > 0 else 0

    # GEAK classification
    if vp > 60 and mp < 40: cls = "COMPUTE-BOUND"
    elif mp > 60 and vp < 40: cls = "MEMORY-BOUND"
    elif lp > 50: cls = "LDS-BOUND"
    elif vp < 40 and mp < 40: cls = "LATENCY-BOUND"
    else: cls = "BALANCED"

    print(f"{subdir:<45} {vp:>7.1f} {mp:>7.1f} {lp:>7.1f} {l2h:>7.1f} {ldc:>9.1f} {cls}")
PYEOF

echo ""
echo "Data: $PROFDIR"
echo "Done: $(date)"
