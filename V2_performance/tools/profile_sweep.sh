#!/bin/bash
#SBATCH --job-name=v2profsweep
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:40:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/V2_performance/raw/profsweep_%j.log
# GPU characterization: V2 (coop+global) AND GPU-SVM across all tiers.
# V2_NO_CPU=1 skips the slow f64 CPU reference so profiler wall time is GPU-only.
# Collects per engine: kernel-trace (time + VGPR/LDS/scratch), HSA dispatch
# overhead, 3 counter passes. Raw CSVs -> V2_performance/raw/<circuit>/<engine>/.
set -u
export V2_NO_CPU=1
BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"; cd "$BASE"
for ROCM in /opt/rocm-7.2.3 /opt/rocm; do [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }; done
export LLVM_PREFIX=/shared/jmonsalv/software/modules/llvm/upstream_05082025
# rebuild run_v2 to pick up V2_NO_CPU support
cmake --build build-v2-nohip -j$(nproc) --target run_v2 2>&1 | grep -iE "error:|Built target run_v2" | head
RAW="$BASE/V2_performance/raw"
V2=build-v2-nohip/run_v2
SVM=build-gpu-mlir-mi355x/run_gpu

# FETCH_SIZE/WRITE_SIZE are NOT collectable on this gfx950 config (rocprofv3
# aborts "Request exceeds the capabilities of the hardware"). Use TCC L2 traffic
# + SQ busy/wait cycles instead. All sets below verified single-pass-collectable.
PMC_A="SQ_WAVES,SQ_INSTS_VALU,SQ_INSTS_MFMA,SQ_INSTS_SALU,SQ_INSTS_LDS"
PMC_B="TCC_HIT_sum,TCC_MISS_sum"
PMC_C="SQ_BUSY_CYCLES,GRBM_GUI_ACTIVE,SQ_WAIT_INST_LDS,SQ_WAVE_CYCLES"

CIRCS=(
"tests/fixtures/incremental/01_frame_only/frame_h.stim 20000"
"tests/fixtures/large/circuit_d3_p0.001.stim 20000"
"tests/fixtures/qv10.stim 20000"
"tests/fixtures/large/surface_d7_t15.stim 10000"
"tests/fixtures/large/surface_d9_t10.stim 10000"
"tests/fixtures/large/surface_d7_t19.stim 5000"
"tests/fixtures/large/surface_d9_t19.stim 5000"
"tests/fixtures/large/surface_d11_t15.stim 5000"
)

prof() {  # $1=outdir  rest=cmd
  local outdir="$1"; shift; local cmd="$*"
  mkdir -p "$outdir"
  eval "$cmd" >/dev/null 2>&1   # warm
  timeout 60 rocprofv3 --kernel-trace --stats --output-format csv -d "$outdir/kt"   -- $cmd >/dev/null 2>&1
  timeout 60 rocprofv3 --hsa-core-trace --stats --output-format csv -d "$outdir/hsa" -- $cmd >/dev/null 2>&1
  timeout 60 rocprofv3 --pmc $PMC_A --output-format csv -d "$outdir/pmcA" -- $cmd >/dev/null 2>&1
  timeout 60 rocprofv3 --pmc $PMC_B --output-format csv -d "$outdir/pmcB" -- $cmd >/dev/null 2>&1
  timeout 60 rocprofv3 --pmc $PMC_C --output-format csv -d "$outdir/pmcC" -- $cmd >/dev/null 2>&1
}

for entry in "${CIRCS[@]}"; do
  c=$(echo "$entry" | awk '{print $1}'); s=$(echo "$entry" | awk '{print $2}')
  name=$(basename "$c" .stim)
  echo "=== $name (shots=$s) ==="
  prof "$RAW/$name/v2"  "$V2 --circuit $c --shots $s --seed 1"
  echo "  v2 done"
  prof "$RAW/$name/svm" "$SVM --circuit $c --shots $s --seed 1 --no-postselection"
  echo "  svm done"
done
echo "SWEEP_DONE"
