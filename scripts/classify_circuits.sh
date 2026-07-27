#!/bin/bash
#SBATCH --job-name=classify
#SBATCH --partition=mi350x-es
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/shared/jmonsalv/quantum/clifft_rl/clifft/results/classify_circuits_%j.log

BASE="/shared/jmonsalv/quantum/clifft_rl/clifft"
BIN="$BASE/build-gpu-mlir-mi355x/run_gpu"

for ROCM in /opt/rocm-7.2.3 /opt/rocm /opt/rocm-*; do
    [ -d "$ROCM/lib" ] && { export ROCM_PATH="$ROCM"; export PATH="$ROCM/bin:$PATH"; export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"; break; }
done

echo "=== Circuit Classification by Compiled peak_rank ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

# Header
printf "%-60s %6s %5s %5s %5s %s\n" "CIRCUIT" "RANK" "Q" "DET" "INSTR" "TIER"
printf "%-60s %6s %5s %5s %5s %s\n" "-------" "----" "---" "---" "-----" "----"

REG=0; COOP=0; GLOBAL=0; FAIL=0

for stim in $(find "$BASE/tests/fixtures" "$BASE/docs/guide/circuits" "$BASE/tools/bench/fixtures" -name "*.stim" 2>/dev/null | sort); do
    name=$(echo "$stim" | sed "s|$BASE/||")

    json=$(timeout 30 "$BIN" --circuit "$stim" --cpu-reference --shots 1 --seed 42 2>/dev/null)
    rc=$?

    if [ $rc -ne 0 ] || [ -z "$json" ]; then
        printf "%-60s %6s %5s %5s %5s %s\n" "$name" "FAIL" "-" "-" "-" "ERROR"
        ((FAIL++))
        continue
    fi

    rank=$(echo "$json" | python3 -c "import json,sys; print(json.load(sys.stdin)['peak_rank'])" 2>/dev/null)
    nq=$(echo "$json" | python3 -c "import json,sys; print(json.load(sys.stdin).get('measurements',0))" 2>/dev/null)
    det=$(echo "$json" | python3 -c "import json,sys; print(json.load(sys.stdin).get('detectors',0))" 2>/dev/null)
    instr=$(echo "$json" | python3 -c "import json,sys; print(json.load(sys.stdin).get('num_instructions',0))" 2>/dev/null)

    if [ -z "$rank" ]; then
        printf "%-60s %6s %5s %5s %5s %s\n" "$name" "PARSE" "-" "-" "-" "ERROR"
        ((FAIL++))
        continue
    fi

    if [ "$rank" -le 4 ]; then
        tier="REGISTER"
        ((REG++))
    elif [ "$rank" -le 10 ]; then
        tier="COOP"
        ((COOP++))
    else
        tier="GLOBAL"
        ((GLOBAL++))
    fi

    printf "%-60s %6d %5s %5s %5s %s\n" "$name" "$rank" "$nq" "$det" "$instr" "$tier"
done

echo ""
echo "=== SUMMARY ==="
echo "Register (rank 0-4):  $REG circuits"
echo "Coop     (rank 5-10): $COOP circuits"
echo "Global   (rank 11+):  $GLOBAL circuits"
echo "Failed:               $FAIL circuits"
echo "Total:                $((REG + COOP + GLOBAL + FAIL)) circuits"
echo ""
echo "Done: $(date)"
