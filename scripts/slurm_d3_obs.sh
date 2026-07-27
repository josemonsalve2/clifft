#!/bin/bash
#SBATCH --job-name=d3obs
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:12:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/d3_obs_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BIN="$BASE/build-gpu-mlir-mi355x/run_gpu"
for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }; done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }; done
rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null

D3="$BASE/tests/fixtures/large/circuit_d3_p0.001.stim"
dump(){ python3 -c "import json,sys;d=json.load(sys.stdin);print('passed=',d['passed_shots'],'obs_ones=',d.get('observable_ones'),'logical=',d.get('logical_errors'))" 2>/dev/null; }

echo "=== circuit_d3 --no-postselection: compare observable_ones + logical_errors ==="
echo -n "SVM(gpu):  "; "$BIN" --circuit "$D3" --shots 2000 --seed 42 --no-postselection 2>/dev/null | dump
echo -n "CPU-ref:   "; "$BIN" --circuit "$D3" --shots 2000 --seed 42 --no-postselection --cpu-reference 2>/dev/null | dump
"$BIN" --circuit "$D3" --shots 10 --seed 42 --mlir --no-postselection >/dev/null 2>&1
echo -n "MLIR:      "; "$BIN" --circuit "$D3" --shots 2000 --seed 42 --mlir --no-postselection 2>/dev/null | dump
echo -n "Hybrid:    "; "$BIN" --circuit "$D3" --shots 2000 --seed 42 --hybrid --no-postselection 2>/dev/null | dump
echo ""
echo "Done: $(date)"
