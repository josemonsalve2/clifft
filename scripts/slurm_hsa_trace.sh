#!/bin/bash
#SBATCH --job-name=hsa-trace
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/hsa_trace_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BIN="$BASE/build-gpu-mlir-mi355x/run_gpu"
STIM="$BASE/tests/fixtures/incremental/01_frame_only/frame_h.stim"
OUTDIR="$BASE/results/hsa_trace_$$"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do
    [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }
done

rm -rf "$HOME/.clifft/kernel_cache/mlir" 2>/dev/null
# Warm up
$BIN --circuit "$STIM" --shots 100 --seed 42 >/dev/null 2>&1
$BIN --circuit "$STIM" --shots 100 --seed 42 --mlir >/dev/null 2>&1

mkdir -p "$OUTDIR"

echo "=== SVM with sys-trace (1000 shots for quick trace) ==="
rocprofv3 --sys-trace -o "$OUTDIR/svm" -- $BIN --circuit "$STIM" --shots 1000 --seed 42 2>&1 | tail -3
echo ""
echo "=== MLIR with sys-trace (1000 shots for quick trace) ==="
rocprofv3 --sys-trace -o "$OUTDIR/mlir" -- $BIN --circuit "$STIM" --shots 1000 --seed 42 --mlir 2>&1 | tail -3

echo ""
echo "=== Extracting HSA dispatch durations ==="
for db in "$OUTDIR"/*.db; do
    [ -f "$db" ] || continue
    echo "--- $db ---"
    python3 -c "
import sqlite3
db = sqlite3.connect('$db')
c = db.cursor()
# Find kernel dispatch table
c.execute(\"SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'rocpd_kernel_dispatch%'\")
tables = c.fetchall()
for (tbl,) in tables:
    uid = tbl.replace('rocpd_kernel_dispatch', '')
    c.execute(f'SELECT COUNT(*) FROM {tbl}')
    cnt = c.fetchone()[0]
    print(f'  {cnt} kernel dispatches in {tbl}')
    c.execute(f'''SELECT ks.string, kd.end - kd.start, kd.grid_size_x, kd.workgroup_size_x
        FROM {tbl} kd JOIN rocpd_string{uid} ks ON kd.kernel_id = ks.id
        ORDER BY (kd.end - kd.start) DESC LIMIT 10''')
    for r in c.fetchall():
        print(f'    {r[0][:50]:50s} {r[1]/1000:.1f} us  grid={r[2]} wg={r[3]}')
    c.execute(f'SELECT SUM(kd.end - kd.start) FROM {tbl}')
    total = c.fetchone()[0] or 0
    print(f'  Total kernel GPU time: {total/1e6:.3f} ms')
db.close()
" 2>&1
done

echo ""
echo "Done: $(date)"
