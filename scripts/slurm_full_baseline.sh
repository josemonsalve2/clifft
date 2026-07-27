#!/bin/bash
#SBATCH --job-name=fullbase
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/full_baseline_%j.log

# NOTE: does NOT rebuild (Codex may be editing source). Uses the current binary.
BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BIN="$BASE/build-gpu-mlir-mi355x/run_gpu"
SHOTS=10000

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }; done
for LLVM in /shared/jmonsalv/software/modules/llvm/upstream_05082025 /opt/llvm; do [ -d "$LLVM/bin" ] && { export LLVM_PREFIX="$LLVM"; export PATH="$LLVM/bin:$PATH"; break; }; done

echo "=== FULL BASELINE (BENCHMARKS.md) — binary: $(ls -la $BIN | awk '{print $6,$7,$8}') ==="
echo "Node: $(hostname)  Shots: $SHOTS  Date: $(date)"
echo "Correctness gold standard: SVM (gpu_sample_survivors, per-shot seed). Kernel times = kernel_seconds."
echo ""

rm -rf ~/.clifft/kernel_cache/mlir 2>/dev/null

# name|path|mlir_compile_timeout_s
CIRCUITS=(
  # Register (rank 0-4)
  "frame_h|tests/fixtures/incremental/01_frame_only/frame_h.stim|90"
  "four_t|tests/fixtures/incremental/02_single_expand/four_t.stim|90"
  "circuit_d3_p0.001|tests/fixtures/large/circuit_d3_p0.001.stim|180"
  "surface_d7_t5|tests/fixtures/large/surface_d7_t5.stim|700"
  # Coop (rank 5-10)
  "qv10|tests/fixtures/qv10.stim|180"
  "cultivation_d5|tests/fixtures/cultivation_d5.stim|700"
  "circuit_d5_p0.001|tests/fixtures/large/circuit_d5_p0.001.stim|700"
  "surface_d7_t10|tests/fixtures/large/surface_d7_t10.stim|700"
  "surface_d7_t15|tests/fixtures/large/surface_d7_t15.stim|700"
  "surface_d9_t10|tests/fixtures/large/surface_d9_t10.stim|900"
  # Global (rank 11-19)
  "surface_d7_t19|tests/fixtures/large/surface_d7_t19.stim|700"
  "surface_d9_t15|tests/fixtures/large/surface_d9_t15.stim|900"
  "surface_d9_t19|tests/fixtures/large/surface_d9_t19.stim|900"
  "surface_d11_t15|tests/fixtures/large/surface_d11_t15.stim|1400"
  "surface_d11_t19|tests/fixtures/large/surface_d11_t19.stim|1400"
)

getp(){ python3 -c "import json,sys;print(json.load(sys.stdin)['passed_shots'])" 2>/dev/null; }
getk(){ grep kernel_seconds "$1" | grep -oP '[0-9.]+' | head -1; }

printf "%-20s %4s %5s %8s %8s | %9s %9s %9s | %s\n" CIRCUIT RANK TIER SVM_p MLIR_p SVM_ms MLIR_ms HYB_ms STATUS
printf "%-20s %4s %5s %8s %8s | %9s %9s %9s | %s\n" ------- ---- ---- ----- ------ ------ ------- ------ ------

for entry in "${CIRCUITS[@]}"; do
    IFS='|' read -r name stim timeo <<< "$entry"
    full="$BASE/$stim"
    [ -f "$full" ] || { printf "%-20s MISSING\n" "$name"; continue; }

    rank=$("$BIN" --circuit "$full" --cpu-reference --shots 1 --seed 42 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin)['peak_rank'])" 2>/dev/null)
    rank=${rank:-?}
    if [ "$rank" = "?" ]; then tier=?; elif [ "$rank" -le 4 ]; then tier=REG; elif [ "$rank" -le 10 ]; then tier=COOP; else tier=GLOB; fi

    # SVM
    sf=$(mktemp); sj=$("$BIN" --circuit "$full" --shots $SHOTS --seed 42 2>"$sf"); svm_p=$(echo "$sj"|getp); svm_k=$(getk "$sf"); rm -f "$sf"

    # MLIR: warm compile then timed
    cf=$(mktemp); timeout "$timeo" "$BIN" --circuit "$full" --shots 10 --seed 42 --mlir >/dev/null 2>"$cf"; rc=$?
    cerr=$(grep -iE "error|dominate|FATAL|does not" "$cf" | head -1); rm -f "$cf"
    if [ $rc -eq 124 ]; then mlir_p=TIMEOUT; mlir_k=-
    elif [ -n "$cerr" ]; then mlir_p=CERR; mlir_k=-
    elif [ $rc -ne 0 ]; then mlir_p=CRASH; mlir_k=-
    else
        mf=$(mktemp); mj=$("$BIN" --circuit "$full" --shots $SHOTS --seed 42 --mlir 2>"$mf"); mlir_p=$(echo "$mj"|getp); mlir_k=$(getk "$mf"); rm -f "$mf"
        mlir_p=${mlir_p:-PARSEFAIL}
    fi

    # Hybrid
    hf=$(mktemp); timeout 200 "$BIN" --circuit "$full" --shots 10 --seed 42 --hybrid >/dev/null 2>"$hf"; hrc=$?
    if [ $hrc -eq 0 ]; then hj=$("$BIN" --circuit "$full" --shots $SHOTS --seed 42 --hybrid 2>"$hf"); hyb_k=$(getk "$hf"); else hyb_k=-; fi
    rm -f "$hf"

    st=MISMATCH; [ "$mlir_p" = "$svm_p" ] && st=OK
    case "$mlir_p" in TIMEOUT|CERR|CRASH|PARSEFAIL) st="$mlir_p";; esac

    printf "%-20s %4s %5s %8s %8s | %9s %9s %9s | %s\n" "$name" "$rank" "$tier" "${svm_p:-?}" "${mlir_p:-?}" "${svm_k:-?}" "${mlir_k:-?}" "${hyb_k:-?}" "$st"
done

echo ""
echo "Done: $(date)"
