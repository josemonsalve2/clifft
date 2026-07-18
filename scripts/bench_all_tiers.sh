#!/bin/bash
# Benchmark all 3 kernel tiers: per-thread, shared-coop, global-coop
# Runs OLD (clifft-amd) and NEW (clifft-gpu) back-to-back on same node
# Uses circuits both implementations can handle (no EXP_VAL)

set -euo pipefail

OLD=/shared/jmonsalv/quantum/clifft_rl/clifft-amd/build-core-hip/run_msc_hip
NEW=./build-gpu/run_gpu

# Tier 1: hook-injection noisy (pure Clifford, peak_rank=0 → per-thread kernel)
CIRCUIT_T1="/shared/jmonsalv/quantum/clifft_rl/clifft-hookinjection/data/circuits/r=3,d=3,p=0.001,noise=SI1000,b=hook_inject_Y_magic_verify,post_r=2,post_d=3,q=17,gates=cz,post_q=17.stim"
SHOTS_T1=10000000

# Tier 2: d5 cultivation (peak_rank=10 → shared-coop kernel with LDS)
CIRCUIT_T2=/shared/jmonsalv/quantum/clifft_rl/clifft-amd/circuit_d5_p=0.001.stim
SHOTS_T2=10000000

# Tier 3: d7 (peak_rank=19 → global-coop kernel with HBM)
CIRCUIT_T3=/shared/jmonsalv/quantum/clifft_rl/clifft-amd/circuit_d7_p0.0005.stim
SHOTS_T3=100000

for TIER in 1 2 3; do
    eval CIRCUIT=\$CIRCUIT_T${TIER}
    eval SHOTS=\$SHOTS_T${TIER}
    echo "================================================================"
    echo "TIER $TIER: $([ $TIER = 1 ] && echo 'Per-thread (registers)' || ([ $TIER = 2 ] && echo 'Shared-coop (LDS)' || echo 'Global-coop (HBM)'))"
    echo "Shots: $SHOTS"
    echo "================================================================"
    for RUN in 1 2; do
        echo -n "OLD run$RUN: "
        $OLD --circuit "$CIRCUIT" --shots $SHOTS --seed 42 2>/dev/null | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(f'rank={d[\"peak_rank\"]} passed={d[\"passed_shots\"]} shots/s={d[\"shots_per_second_sampling_only\"]:.0f}')"
        echo -n "NEW run$RUN: "
        $NEW --circuit "$CIRCUIT" --shots $SHOTS --seed 42 2>/dev/null | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(f'rank={d[\"peak_rank\"]} passed={d[\"passed_shots\"]} shots/s={d[\"shots_per_second_sampling_only\"]:.0f}')"
    done
    echo ""
done
