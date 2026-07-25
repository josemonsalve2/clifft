#!/usr/bin/env python3
"""Generate Quantum-Volume-style circuits at a chosen peak_rank.

Why this generator exists
-------------------------
The T-injection generators (generate_large_circuits.py,
generate_high_rank_circuits.py) nominally target a peak_rank, but the
StatevectorSqueezePass sees straight through most of them: T gates on distinct
qubits with only local Clifford glue between them get reordered so that only a
couple of amplitudes are ever live at once. `rank19_q100_d10_wide.stim`, for
instance, compiles to peak_rank=1.

QV circuits do not squeeze. Every layer applies a random SU(4) to a random
pairing of all qubits, so by the end of layer 2 every qubit is entangled with
every other and the statevector genuinely needs 2^N amplitudes. This is exactly
why tools/bench/fixtures/qv20_seed42.stim compiles to peak_rank=20 on the nose.

Structure (matches the existing qv20_seed42 fixture's op counts):
  for each of `layers` layers:
    random pairing of the N qubits
    for each pair (a, b): U3 a; U3 b; (CX a b; U3 a; U3 b) x 3
  M 0 1 ... N-1
=> 3 * (N/2) * layers CX gates and 8 * (N/2) * layers U3 gates.

Usage:
    python3 scripts/generate_qv_circuits.py --qubits 22 --layers 8 \
        --seed 42 --out tests/fixtures/large/qv22_seed42.stim
    python3 scripts/generate_qv_circuits.py --sweep --outdir tests/fixtures/large
"""
import argparse
import os
import random


def _u3(rng: random.Random) -> str:
    """A random U3 with angles in clifft's normalized convention.

    The existing qv20_seed42 fixture's angles live in theta in [0,1],
    phi/lambda in [-1,1]; we match that range so the new fixtures are
    numerically comparable to it.
    """
    return "U3({:.16f},{:.16f},{:.16f})".format(
        rng.uniform(0.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)
    )


def generate_qv(num_qubits: int, layers: int, seed: int) -> str:
    if num_qubits < 2:
        raise ValueError("QV needs at least 2 qubits")
    rng = random.Random(seed)
    lines = []
    for _ in range(layers):
        perm = list(range(num_qubits))
        rng.shuffle(perm)
        # Odd qubit counts leave one qubit idle in a layer; it gets paired in
        # the next one, so peak_rank still reaches num_qubits.
        for i in range(0, num_qubits - 1, 2):
            a, b = perm[i], perm[i + 1]
            lines.append(f"{_u3(rng)} {a}")
            lines.append(f"{_u3(rng)} {b}")
            for _ in range(3):
                lines.append(f"CX {a} {b}")
                lines.append(f"{_u3(rng)} {a}")
                lines.append(f"{_u3(rng)} {b}")
    lines.append("M " + " ".join(str(q) for q in range(num_qubits)))
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qubits", type=int, default=20)
    ap.add_argument("--layers", type=int, default=None,
                    help="default: num_qubits (the standard QV depth)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--sweep", action="store_true",
                    help="generate the rank 20-24 benchmark family")
    ap.add_argument("--outdir", type=str, default="tests/fixtures/large")
    args = ap.parse_args()

    if args.sweep:
        os.makedirs(args.outdir, exist_ok=True)
        # Layer counts shrink as the rank climbs: kernel work per shot is
        # O(layers * 2^rank), so a full-depth QV at rank 24 would be ~100x the
        # cost of the rank-20 one. These keep the family runnable in one sweep
        # while still exercising the global tier at each rank.
        for nq, layers in [(20, 8), (21, 8), (22, 6), (23, 5), (24, 4)]:
            text = generate_qv(nq, layers, args.seed)
            path = os.path.join(args.outdir, f"qv{nq}_L{layers}_seed{args.seed}.stim")
            with open(path, "w") as f:
                f.write(text)
            print(f"{path}  ({nq} qubits, {layers} layers, "
                  f"{3 * (nq // 2) * layers} CX)")
        return

    layers = args.layers if args.layers is not None else args.qubits
    text = generate_qv(args.qubits, layers, args.seed)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text)
        print(f"{args.out}  ({args.qubits} qubits, {layers} layers)")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
