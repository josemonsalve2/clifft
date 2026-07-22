#!/usr/bin/env python3
"""Generate minimal single-gate circuits for incremental performance evaluation.

Each circuit isolates exactly one HIR operand so that per-gate performance
can be measured and failures are immediately attributable.

Usage:
    python3 scripts/generate_single_gate_circuits.py --outdir tests/fixtures/incremental

Output hierarchy:
    01_frame_only/      — Single frame-tracking gates (rank=0, no amplitudes)
    02_single_expand/   — One T gate → expand to rank=1
    03_single_array/    — One array op at rank=1-4
    04_measure/         — Single measurement (dormant and active)
    05_combinations/    — Two-gate combos (H+T, CNOT+H, etc.)
    06_small_circuits/  — 5-10 gate circuits approaching real QEC
"""
import argparse
import os


def write_circuit(outdir: str, name: str, lines: list[str]):
    path = os.path.join(outdir, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for line in lines:
            f.write(line + '\n')
    print(f"  {name}")


def generate(outdir: str):
    os.makedirs(outdir, exist_ok=True)

    # ================================================================
    # LEVEL 1: Frame-only gates (rank=0, no amplitude array)
    # These test pure Pauli frame tracking — no expand, no array ops
    # ================================================================
    print("Level 1: Frame-only gates")

    # H gate on qubit 0 (frame_h)
    write_circuit(outdir, "01_frame_only/frame_h.stim", [
        "QUBIT_COORDS(0, 0) 0",
        "QUBIT_COORDS(1, 0) 1",
        "H 0",
        "M 0 1",
        "DETECTOR rec[-2]",
        "OBSERVABLE_INCLUDE(0) rec[-1]",
    ])

    # S gate (frame_s)
    write_circuit(outdir, "01_frame_only/frame_s.stim", [
        "QUBIT_COORDS(0, 0) 0",
        "QUBIT_COORDS(1, 0) 1",
        "S 0",
        "M 0 1",
        "DETECTOR rec[-2]",
        "OBSERVABLE_INCLUDE(0) rec[-1]",
    ])

    # CNOT gate (frame_cnot)
    write_circuit(outdir, "01_frame_only/frame_cnot.stim", [
        "QUBIT_COORDS(0, 0) 0",
        "QUBIT_COORDS(1, 0) 1",
        "H 0",
        "CNOT 0 1",
        "M 0 1",
        "DETECTOR rec[-2]",
        "OBSERVABLE_INCLUDE(0) rec[-1]",
    ])

    # CZ gate (frame_cz)
    write_circuit(outdir, "01_frame_only/frame_cz.stim", [
        "QUBIT_COORDS(0, 0) 0",
        "QUBIT_COORDS(1, 0) 1",
        "H 0",
        "CZ 0 1",
        "M 0 1",
        "DETECTOR rec[-2]",
        "OBSERVABLE_INCLUDE(0) rec[-1]",
    ])

    # SWAP (frame_swap)
    write_circuit(outdir, "01_frame_only/frame_swap.stim", [
        "QUBIT_COORDS(0, 0) 0",
        "QUBIT_COORDS(1, 0) 1",
        "H 0",
        "SWAP 0 1",
        "M 0 1",
        "DETECTOR rec[-2]",
        "OBSERVABLE_INCLUDE(0) rec[-1]",
    ])

    # Multiple H gates (10x) — measures dispatch overhead scaling
    write_circuit(outdir, "01_frame_only/frame_h_x10.stim", [
        "QUBIT_COORDS(0, 0) 0",
        "QUBIT_COORDS(1, 0) 1",
    ] + ["H 0"] * 10 + [
        "M 0 1",
        "DETECTOR rec[-2]",
        "OBSERVABLE_INCLUDE(0) rec[-1]",
    ])

    # ================================================================
    # LEVEL 2: Single expand (T gate → rank=1)
    # T gate causes expand: rank goes 0→1
    # ================================================================
    print("Level 2: Single expand (T gate)")

    # One T gate on qubit 0 → expand_t, peak_rank=1
    write_circuit(outdir, "02_single_expand/one_t.stim", [
        "QUBIT_COORDS(0, 0) 0",
        "QUBIT_COORDS(1, 0) 1",
        "QUBIT_COORDS(2, 0) 2",
        "H 0",
        "T 0",           # expand to rank=1
        "M 0 1 2",       # measurement collapses rank back to 0
        "DETECTOR rec[-3]",
        "OBSERVABLE_INCLUDE(0) rec[-1]",
    ])

    # Two T gates on different qubits → peak_rank=2
    write_circuit(outdir, "02_single_expand/two_t.stim", [
        "QUBIT_COORDS(0, 0) 0",
        "QUBIT_COORDS(1, 0) 1",
        "QUBIT_COORDS(2, 0) 2",
        "H 0",
        "H 1",
        "T 0",           # rank 0→1
        "T 1",           # rank 1→2
        "M 0 1 2",
        "DETECTOR rec[-3]",
        "OBSERVABLE_INCLUDE(0) rec[-1]",
    ])

    # Four T gates → peak_rank=4 (max register tier)
    write_circuit(outdir, "02_single_expand/four_t.stim", [
        "QUBIT_COORDS(0, 0) 0",
        "QUBIT_COORDS(1, 0) 1",
        "QUBIT_COORDS(2, 0) 2",
        "QUBIT_COORDS(3, 0) 3",
        "QUBIT_COORDS(4, 0) 4",
    ] + [f"H {i}" for i in range(4)] + [
        f"T {i}" for i in range(4)  # rank 0→1→2→3→4
    ] + [
        "M 0 1 2 3 4",
        "DETECTOR rec[-5]",
        "OBSERVABLE_INCLUDE(0) rec[-1]",
    ])

    # ================================================================
    # LEVEL 3: Single array ops at rank=1
    # After one T gate (rank=1), apply one array operation
    # ================================================================
    print("Level 3: Single array ops at rank=1")

    # array_h at rank=1
    write_circuit(outdir, "03_single_array/array_h_r1.stim", [
        "QUBIT_COORDS(0, 0) 0",
        "QUBIT_COORDS(1, 0) 1",
        "H 0",
        "T 0",         # rank=1
        "H 0",         # array_h on the active qubit
        "M 0 1",
        "DETECTOR rec[-2]",
        "OBSERVABLE_INCLUDE(0) rec[-1]",
    ])

    # array_cnot at rank=2
    write_circuit(outdir, "03_single_array/array_cnot_r2.stim", [
        "QUBIT_COORDS(0, 0) 0",
        "QUBIT_COORDS(1, 0) 1",
        "QUBIT_COORDS(2, 0) 2",
        "H 0",
        "H 1",
        "T 0",          # rank=1
        "T 1",          # rank=2
        "CNOT 0 1",     # array_cnot at rank=2
        "M 0 1 2",
        "DETECTOR rec[-3]",
        "OBSERVABLE_INCLUDE(0) rec[-1]",
    ])

    # array_s at rank=1
    write_circuit(outdir, "03_single_array/array_s_r1.stim", [
        "QUBIT_COORDS(0, 0) 0",
        "QUBIT_COORDS(1, 0) 1",
        "H 0",
        "T 0",
        "S 0",          # array_s at rank=1
        "M 0 1",
        "DETECTOR rec[-2]",
        "OBSERVABLE_INCLUDE(0) rec[-1]",
    ])

    # ================================================================
    # LEVEL 4: Measurements
    # ================================================================
    print("Level 4: Measurements")

    # Dormant measurement (static, no amplitude math)
    write_circuit(outdir, "04_measure/dormant_static.stim", [
        "QUBIT_COORDS(0, 0) 0",
        "QUBIT_COORDS(1, 0) 1",
        "M 0 1",
        "DETECTOR rec[-2]",
        "OBSERVABLE_INCLUDE(0) rec[-1]",
    ])

    # Active diagonal measurement (rank=1 → collapse)
    write_circuit(outdir, "04_measure/active_diagonal.stim", [
        "QUBIT_COORDS(0, 0) 0",
        "QUBIT_COORDS(1, 0) 1",
        "H 0",
        "T 0",          # rank=1
        "M 0",          # active measurement (diagonal), rank 1→0
        "M 1",
        "DETECTOR rec[-2]",
        "OBSERVABLE_INCLUDE(0) rec[-1]",
    ])

    # Noise + measurement (tests noise draw + dormant random)
    write_circuit(outdir, "04_measure/noise_measure.stim", [
        "QUBIT_COORDS(0, 0) 0",
        "QUBIT_COORDS(1, 0) 1",
        "H 0",
        "DEPOLARIZE1(0.001) 0 1",
        "M 0 1",
        "DETECTOR rec[-2]",
        "OBSERVABLE_INCLUDE(0) rec[-1]",
    ])

    # ================================================================
    # LEVEL 5: Combinations (two different gate types)
    # ================================================================
    print("Level 5: Two-gate combinations")

    # H + T (most common combination in QEC)
    write_circuit(outdir, "05_combinations/h_then_t.stim", [
        "QUBIT_COORDS(0, 0) 0",
        "QUBIT_COORDS(1, 0) 1",
        "H 0",
        "T 0",
        "H 0",
        "M 0 1",
        "DETECTOR rec[-2]",
        "OBSERVABLE_INCLUDE(0) rec[-1]",
    ])

    # CNOT + H at rank=2
    write_circuit(outdir, "05_combinations/cnot_h_r2.stim", [
        "QUBIT_COORDS(0, 0) 0",
        "QUBIT_COORDS(1, 0) 1",
        "QUBIT_COORDS(2, 0) 2",
        "H 0",
        "H 1",
        "T 0",
        "T 1",
        "CNOT 0 1",
        "H 0",
        "M 0 1 2",
        "DETECTOR rec[-3]",
        "OBSERVABLE_INCLUDE(0) rec[-1]",
    ])

    # T + measure cycle (the fundamental QEC primitive)
    write_circuit(outdir, "05_combinations/t_measure_cycle.stim", [
        "QUBIT_COORDS(0, 0) 0",
        "QUBIT_COORDS(1, 0) 1",
        "QUBIT_COORDS(2, 0) 2",
        "H 0",
        "T 0",       # expand to rank=1
        "M 0",       # collapse back to rank=0
        "H 1",
        "T 1",       # expand again
        "M 1",
        "M 2",
        "DETECTOR rec[-3]",
        "OBSERVABLE_INCLUDE(0) rec[-1]",
    ])

    # ================================================================
    # LEVEL 6: Small circuits (5-10 gates, approaching real QEC)
    # ================================================================
    print("Level 6: Small circuits")

    # Repetition code distance 3 (simplest QEC)
    write_circuit(outdir, "06_small_circuits/rep_d3.stim", [
        "QUBIT_COORDS(0, 0) 0",
        "QUBIT_COORDS(1, 0) 1",
        "QUBIT_COORDS(2, 0) 2",
        "CNOT 0 1",
        "CNOT 1 2",
        "M 0 1 2",
        "DETECTOR rec[-3] rec[-2]",
        "DETECTOR rec[-2] rec[-1]",
        "OBSERVABLE_INCLUDE(0) rec[-1]",
    ])

    # Small T-gate circuit at rank 4 (register tier boundary)
    write_circuit(outdir, "06_small_circuits/rank4_mixed.stim", [
        "QUBIT_COORDS(0, 0) 0",
        "QUBIT_COORDS(1, 0) 1",
        "QUBIT_COORDS(2, 0) 2",
        "QUBIT_COORDS(3, 0) 3",
        "QUBIT_COORDS(4, 0) 4",
        "H 0", "H 1", "H 2", "H 3",
        "T 0", "T 1", "T 2", "T 3",  # rank=4
        "CNOT 0 1",
        "CZ 2 3",
        "H 0",
        "S 1",
        "M 0 1 2 3 4",
        "DETECTOR rec[-5]",
        "OBSERVABLE_INCLUDE(0) rec[-1]",
    ])

    # Generate manifest
    manifest_path = os.path.join(outdir, "manifest.txt")
    with open(manifest_path, 'w') as f:
        f.write("# Incremental single-gate circuits for performance evaluation\n")
        f.write("# level/circuit_file  description\n")
        for root, dirs, files in sorted(os.walk(outdir)):
            for fname in sorted(files):
                if fname.endswith('.stim'):
                    rel = os.path.relpath(os.path.join(root, fname), outdir)
                    f.write(f"{rel}\n")

    print(f"\nGenerated circuits in {outdir}/")
    print(f"Manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="tests/fixtures/incremental")
    args = parser.parse_args()
    generate(args.outdir)


if __name__ == '__main__':
    main()
