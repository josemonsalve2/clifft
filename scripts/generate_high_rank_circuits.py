#!/usr/bin/env python3
"""Generate circuits with controlled peak_rank from 0 to 15+.

Peak_rank = number of active qubits at any point = number of T-gates
before the first measurement that reduces active_k. To get high peak_rank,
we inject multiple T-gates on different qubits BEFORE any measurements.

Usage:
    python3 generate_high_rank_circuits.py --outdir tests/fixtures/rank_sweep
"""
import argparse
import os


def generate_circuit(num_qubits: int, target_rank: int, depth: int) -> str:
    """Generate a circuit that reaches peak_rank = target_rank.

    Strategy: inject target_rank T-gates on different qubits (each one
    bumps active_k by 1 via expand), then do measurements to bring it
    back down, then repeat for depth rounds.
    """
    lines = []

    for q in range(num_qubits):
        lines.append(f"QUBIT_COORDS({q}, 0) {q}")

    meas_count = 0

    for r in range(depth):
        # Clifford layer
        for q in range(0, num_qubits - 1, 2):
            lines.append(f"CNOT {q} {q+1}")
        for q in range(num_qubits):
            lines.append(f"H {q}")

        # T-gate injection: target_rank T-gates on different qubits
        # This creates peak_rank = target_rank (each T-gate expands active_k by 1)
        for t in range(min(target_rank, num_qubits)):
            lines.append(f"T {t}")

        # More Clifford gates (these are array ops if active_k > 0)
        for q in range(0, min(target_rank, num_qubits) - 1, 2):
            lines.append(f"CZ {q} {q+1}")

        # Noise
        noise_qubits = " ".join(str(q) for q in range(num_qubits))
        lines.append(f"DEPOLARIZE1(0.001) {noise_qubits}")

        # Measurements (bring active_k back to 0)
        meas_qubits = " ".join(str(q) for q in range(num_qubits))
        lines.append(f"M {meas_qubits}")

        # Detectors
        for q in range(num_qubits):
            if r > 0:
                lines.append(f"DETECTOR rec[{-(num_qubits - q)}] rec[{-(2*num_qubits - q)}]")
            else:
                lines.append(f"DETECTOR rec[{-(num_qubits - q)}]")

        meas_count += num_qubits

    # Observable
    lines.append(f"OBSERVABLE_INCLUDE(0) rec[-1]")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="tests/fixtures/rank_sweep")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Controlled peak_rank sweep:
    # rank 0-4: per-thread tier (VGPRs)
    # rank 5-10: shared-coop tier (LDS)
    # rank 11-15: global-coop tier (HBM)
    target_ranks = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15]
    qubit_counts = [17, 33]
    depths = [1, 3]

    generated = 0
    for nq in qubit_counts:
        for rank in target_ranks:
            if rank > nq:
                continue
            for d in depths:
                circuit = generate_circuit(nq, rank, d)
                fname = f"rank_q{nq}_r{rank}_d{d}.stim"
                with open(os.path.join(args.outdir, fname), "w") as f:
                    f.write(circuit)
                generated += 1

    print(f"Generated {generated} circuits in {args.outdir}/")

    # Manifest
    with open(os.path.join(args.outdir, "manifest.txt"), "w") as f:
        f.write("# circuit_file qubits target_rank depth\n")
        for nq in qubit_counts:
            for rank in target_ranks:
                if rank > nq:
                    continue
                for d in depths:
                    f.write(f"rank_q{nq}_r{rank}_d{d}.stim {nq} {rank} {d}\n")


if __name__ == "__main__":
    main()
