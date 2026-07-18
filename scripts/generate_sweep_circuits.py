#!/usr/bin/env python3
"""Generate a sweep of Stim circuits with controlled T-gate count, qubit count, and depth.

Each circuit has a known peak_rank (= number of T gates before first measurement),
qubit count, and circuit depth (instruction count). These are the three degrees of
freedom for the benchmark sweep.

Usage:
    python3 generate_sweep_circuits.py --outdir circuits/sweep

Produces circuits named: sweep_q{qubits}_t{tgates}_d{depth}.stim
"""
import argparse
import os
import math

def generate_circuit(num_qubits: int, num_t_gates: int, depth_rounds: int) -> str:
    """Generate a Stim circuit with controlled parameters.

    Args:
        num_qubits: Number of physical qubits
        num_t_gates: Number of T gates (determines peak_rank for per-thread tier)
        depth_rounds: Number of rounds of Clifford+T+measurement blocks
    """
    lines = []

    # Qubit coordinates
    for q in range(num_qubits):
        lines.append(f"QUBIT_COORDS({q}, 0) {q}")

    det_idx = 0
    meas_idx = 0

    for r in range(depth_rounds):
        # Clifford layer: random-ish CNOT pattern
        for q in range(0, num_qubits - 1, 2):
            lines.append(f"CNOT {q} {q+1}")
        for q in range(1, num_qubits - 1, 2):
            lines.append(f"CNOT {q} {q+1}")

        # Hadamard layer
        for q in range(num_qubits):
            lines.append(f"H {q}")

        # T-gate injection (only first round to control peak_rank precisely)
        if r == 0:
            for t in range(min(num_t_gates, num_qubits)):
                lines.append(f"T {t}")

        # S gate layer
        for q in range(num_qubits):
            lines.append(f"S {q}")

        # Another CNOT layer
        for q in range(0, num_qubits - 1, 2):
            lines.append(f"CZ {q} {q+1}")

        # Noise
        noise_qubits = " ".join(str(q) for q in range(num_qubits))
        lines.append(f"DEPOLARIZE1(0.001) {noise_qubits}")

        # Measurement round
        meas_qubits = " ".join(str(q) for q in range(num_qubits))
        lines.append(f"M {meas_qubits}")

        # Detectors (parity checks on consecutive measurements)
        for q in range(num_qubits):
            if r > 0:
                lines.append(f"DETECTOR rec[{-(num_qubits - q)}] rec[{-(2*num_qubits - q)}]")
            else:
                lines.append(f"DETECTOR rec[{-(num_qubits - q)}]")
            det_idx += 1

        meas_idx += num_qubits

    # Observable
    obs_recs = " ".join(f"rec[{-(num_qubits - q)}]" for q in range(min(2, num_qubits)))
    lines.append(f"OBSERVABLE_INCLUDE(0) {obs_recs}")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark sweep circuits")
    parser.add_argument("--outdir", default="tests/fixtures/sweep", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Sweep parameters:
    # T-gates: 0, 1, 2, 3, 4 (per-thread tier covers 0-4)
    #          5, 6, 8, 10 (shared-coop tier)
    #          12, 15, 19 (global-coop tier — 2^19 = 512K amplitudes)
    t_gate_counts = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15]

    # Qubits: 5, 9, 17, 33, 65 (1-word and 2-word Pauli frames)
    qubit_counts = [5, 9, 17, 33, 65]

    # Depth rounds: 1, 3, 5, 10 (instruction count scales with depth * qubits)
    depth_rounds = [1, 3, 5, 10]

    generated = 0
    for nq in qubit_counts:
        for nt in t_gate_counts:
            if nt > nq:
                continue  # Can't have more T gates than qubits
            for nd in depth_rounds:
                circuit = generate_circuit(nq, nt, nd)
                fname = f"sweep_q{nq}_t{nt}_d{nd}.stim"
                fpath = os.path.join(args.outdir, fname)
                with open(fpath, "w") as f:
                    f.write(circuit)
                generated += 1

    print(f"Generated {generated} circuits in {args.outdir}/")

    # Also generate a manifest file listing all circuits with their parameters
    manifest_path = os.path.join(args.outdir, "manifest.txt")
    with open(manifest_path, "w") as f:
        f.write("# circuit_file qubits t_gates depth_rounds\n")
        for nq in qubit_counts:
            for nt in t_gate_counts:
                if nt > nq:
                    continue
                for nd in depth_rounds:
                    fname = f"sweep_q{nq}_t{nt}_d{nd}.stim"
                    f.write(f"{fname} {nq} {nt} {nd}\n")

    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
