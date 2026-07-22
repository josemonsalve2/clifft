#!/bin/bash
#SBATCH --job-name=perf-rocprof
#SBATCH --partition=mi300x
#SBATCH --nodelist=rad-mi300x-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=01:45:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/perf_rocprof_%j.log
#SBATCH --exclusive

# Hardware counter collection with rocprof — MI300X
# Targets one representative circuit per tier for each mode.
# Counter sets:
#   Set A: Wavefront occupancy, VALU/VMEM/LDS instruction mix
#   Set B: L2 (TCC) hit rate, HBM traffic
#   Set C: LDS bank conflicts, SIMDs busy

set -euo pipefail
BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROFDIR="$BASE/results/rocprof_${TIMESTAMP}"
mkdir -p "$PROFDIR"

# Set up ROCm environment
for ROCM_CANDIDATE in /opt/rocm-7.2.3 /opt/rocm-7.0.0 /opt/rocm /opt/rocm-*; do
    if [ -d "$ROCM_CANDIDATE/lib" ]; then
        export ROCM_PATH="$ROCM_CANDIDATE"
        export PATH="$ROCM_CANDIDATE/bin:$PATH"
        export LD_LIBRARY_PATH="$ROCM_CANDIDATE/lib:${LD_LIBRARY_PATH:-}"
        echo "Using ROCm: $ROCM_CANDIDATE"
        break
    fi
done

echo "Node: $(hostname)  Job: ${SLURM_JOB_ID:-n/a}  Date: $(date)"
rocm-smi --showid --showproductname 2>/dev/null | head -5

# Build on this node
BINARY="$BASE/build-gpu-node/run_gpu"
if [ ! -x "$BINARY" ]; then
    echo "Building run_gpu on $(hostname)..."
    mkdir -p "$BASE/build-gpu-node"
    cd "$BASE/build-gpu-node"
    cmake "$BASE" -DCLIFFT_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 \
        -DCMAKE_BUILD_TYPE=Release -DCLIFFT_BUILD_PROFILER=ON \
        -DCMAKE_BUILD_PARALLEL_LEVEL=$(nproc) 2>&1
    make -j$(nproc) run_gpu 2>&1
    cd "$BASE"
fi
echo "Binary: $BINARY"

if ! command -v rocprof >/dev/null 2>&1; then
    echo "ERROR: rocprof not found. Is ROCm in PATH?"
    exit 1
fi

# Write counter input files
PSET_A="$PROFDIR/counters_a.txt"
PSET_B="$PROFDIR/counters_b.txt"
PSET_C="$PROFDIR/counters_c.txt"

cat > "$PSET_A" << 'EOF'
pmc: SQ_WAVES SQ_BUSY_CYCLES SQ_WAVE_CYCLES SQ_INSTS_VALU SQ_INSTS_VMEM SQ_INSTS_LDS
EOF
cat > "$PSET_B" << 'EOF'
pmc: TCC_HIT_sum TCC_MISS_sum TCC_EA_RDREQ_32B_sum TCC_EA_WRREQ_sum TCP_TOTAL_CACHE_ACCESSES_sum
EOF
cat > "$PSET_C" << 'EOF'
pmc: SQ_LDS_BANK_CONFLICT SQ_LDS_MEM_VIOLATIONS SQ_INSTS_BRANCH GRBM_COUNT CP_STAT
EOF

SHOTS=500000   # fewer shots for faster profiling runs

profile_one() {
    local MODE=$1 FLAG=$2 CIRCUIT=$3 LABEL=$4 TIER=$5
    [ -f "$CIRCUIT" ] || { echo "  SKIP $CIRCUIT"; return; }

    local OUTDIR="$PROFDIR/t${TIER}_${LABEL}_${MODE}"
    mkdir -p "$OUTDIR"
    echo "  profiling: $MODE $LABEL (tier $TIER)"

    # Warmup
    numactl --cpunodebind=1 --membind=1 \
        "$BINARY" --circuit "$CIRCUIT" --shots 100000 $FLAG >/dev/null 2>&1 || true

    # Profile each counter set
    for PSET_FILE in "$PSET_A" "$PSET_B" "$PSET_C"; do
        local PNAME=$(basename "$PSET_FILE" .txt)
        numactl --cpunodebind=1 --membind=1 \
            rocprof --stats -i "$PSET_FILE" \
                    -o "$OUTDIR/${PNAME}.csv" \
                    "$BINARY" --circuit "$CIRCUIT" --shots $SHOTS $FLAG \
                    > "$OUTDIR/${PNAME}.log" 2>&1 || {
            echo "    WARNING: rocprof failed for $PNAME"
        }
    done
}

# One representative per tier
T0="$BASE/tests/fixtures/target_qec.stim"
T1="$BASE/tests/fixtures/cultivation_d5.stim"
T2="$BASE/tests/fixtures/large/surface_d7_t10.stim"
T3="$BASE/tests/fixtures/large/rank11_q50_d5_tier3.stim"

echo "=== Tier 0 (frame-only) ==="
profile_one "svm"      ""          "$T0" "target_qec"  "0"
profile_one "compiled" "--hybrid"  "$T0" "target_qec"  "0"

echo "=== Tier 1 (register) ==="
profile_one "svm"      ""          "$T1" "cultivation_d5" "1"
profile_one "compiled" "--hybrid"  "$T1" "cultivation_d5" "1"

echo "=== Tier 2 (LDS/coop) ==="
profile_one "svm"      ""          "$T2" "surface_d7_t10" "2"
profile_one "compiled" "--hybrid"  "$T2" "surface_d7_t10" "2"

echo "=== Tier 3 (global/HBM) ==="
profile_one "svm"      ""          "$T3" "rank11_q50" "3"
profile_one "compiled" "--hybrid"  "$T3" "rank11_q50" "3"

echo ""
echo "=== Hardware Counter Summary ==="
python3 - "$PROFDIR" << 'PYEOF'
import os, csv, glob, sys

profdir = sys.argv[1]
MI300X_HBM_BW_GBPS = 5300  # GB/s peak

# Approx clock from GPU: 1.7 GHz base, 2.1 GHz boost
GPU_CLK_GHZ = 1.9  # midpoint estimate

print(f"{'Config':<35} {'VALU%':>7} {'VMEM%':>7} {'LDS%':>7} {'L2hit%':>7} {'LDSconf%':>9}")
print("-" * 80)

for subdir in sorted(os.listdir(profdir)):
    full = os.path.join(profdir, subdir)
    if not os.path.isdir(full): continue
    if not (subdir.startswith('t') and '_' in subdir): continue

    # Read counter sets
    counters = {}
    for pset in ['counters_a', 'counters_b', 'counters_c']:
        cf = os.path.join(full, f'{pset}.csv')
        if not os.path.exists(cf): continue
        try:
            with open(cf) as f:
                for row in csv.DictReader(f):
                    for k, v in row.items():
                        try: counters[k] = counters.get(k, 0) + float(v or 0)
                        except ValueError: pass
        except Exception: pass

    valu = counters.get('SQ_INSTS_VALU', 0)
    vmem = counters.get('SQ_INSTS_VMEM', 0)
    lds  = counters.get('SQ_INSTS_LDS', 0)
    total_insn = valu + vmem + lds
    l2hit = counters.get('TCC_HIT_sum', 0)
    l2miss = counters.get('TCC_MISS_sum', 0)
    l2total = l2hit + l2miss
    ldsconf = counters.get('SQ_LDS_BANK_CONFLICT', 0)

    valu_pct = 100 * valu / total_insn if total_insn > 0 else 0
    vmem_pct = 100 * vmem / total_insn if total_insn > 0 else 0
    lds_pct  = 100 * lds  / total_insn if total_insn > 0 else 0
    l2_hit_pct = 100 * l2hit / l2total if l2total > 0 else 0
    lds_conf_pct = 100 * ldsconf / lds if lds > 0 else 0

    print(f"{subdir:<35} {valu_pct:>7.1f} {vmem_pct:>7.1f} {lds_pct:>7.1f} "
          f"{l2_hit_pct:>7.1f} {lds_conf_pct:>9.1f}")

print("\nNote: LDSconf% = LDS bank conflicts / LDS instructions (0% = no conflicts)")
PYEOF

echo ""
echo "rocprof data: $PROFDIR"
echo "Done: $(date)"
