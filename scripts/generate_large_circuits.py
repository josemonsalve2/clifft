#!/usr/bin/env python3
"""Generate large Stim QEC circuits for GPU stress-testing.

Produces circuits spanning peak_rank 0-19 and instruction counts up to 20K+,
targeting the three GPU dispatch tiers:
  - Tier 1 (per-thread):   peak_rank 0-4   -> 1-16 amplitudes/shot
  - Tier 2 (shared-coop):  peak_rank 5-10  -> 32-1024 amplitudes/shot
  - Tier 3 (global-coop):  peak_rank 11-19 -> 2048-524288 amplitudes/shot

Sources:
  - Stim built-in generators (surface_code, color_code) for Clifford circuits
  - SOFT repo (haoliri0/SOFT) cultivation circuits at d3, d5
  - Synthetic cultivation-like circuits at d7, d9 (scaled from d5 pattern)
  - Controlled T-gate injection circuits for precise peak_rank targeting

Usage:
    python3 scripts/generate_large_circuits.py --outdir tests/fixtures/large
"""
import argparse
import math
import os
import sys
import textwrap

try:
    import stim
except ImportError:
    print("ERROR: stim not installed. Run: pip install stim", file=sys.stderr)
    sys.exit(1)


# ─── Stim built-in circuits (Clifford-only, peak_rank=0) ───────────────────

def generate_surface_code(distance: int, rounds: int, noise: float = 0.001) -> str:
    """Generate rotated surface code memory circuit via Stim."""
    c = stim.Circuit.generated(
        code_task="surface_code:rotated_memory_x",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=noise,
    )
    return str(c)


def generate_color_code(distance: int, rounds: int, noise: float = 0.001) -> str:
    """Generate color code memory_xyz circuit via Stim."""
    c = stim.Circuit.generated(
        code_task="color_code:memory_xyz",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=noise,
    )
    return str(c)


# ─── Synthetic cultivation circuits (high peak_rank) ───────────────────────

def _qubit_grid(rows: int, cols: int):
    """Return list of (row, col, index) for a 2D qubit grid."""
    result = []
    idx = 0
    for r in range(rows):
        for c in range(cols):
            result.append((r, c, idx))
            idx += 1
    return result


def generate_cultivation_d7(noise: float = 0.001) -> str:
    """Generate a synthetic d=7 cultivation-like circuit.

    Scaling pattern from SOFT circuits:
      d=3: 15 qubits, 7 T-qubit targets, 29 total T ops, ~198 lines
      d=5: 42 qubits, 19 T-qubit targets, 72 total T ops, ~490 lines
      d=7 (projected): ~80 qubits, ~37 T-qubit targets, ~140 total T ops

    This circuit mimics the structure of the cultivation protocol:
    1. Initialize data qubits and ancilla qubits
    2. Stabilizer measurement rounds (CX patterns)
    3. T-gate injection on data qubits (non-Clifford)
    4. More stabilizer rounds
    5. Final MPP measurements

    The peak_rank will be ~15-19 depending on measurement ordering.
    """
    # For d=7, a color code patch has ~7*7 - some = ~80 qubits
    # Data qubits: ~37, Ancilla qubits: ~43
    nq = 80
    n_data = 37  # qubits receiving T gates
    n_ancilla = nq - n_data

    lines = []
    lines.append(f"# Synthetic d=7 cultivation circuit (scaled from d=5 pattern)")
    lines.append(f"# Qubits: {nq}, T-qubit targets: {n_data}, noise: {noise}")

    # Qubit coordinates on a grid
    grid_side = int(math.ceil(math.sqrt(nq)))
    for q in range(nq):
        r, c = divmod(q, grid_side)
        lines.append(f"QUBIT_COORDS({r}, {c}) {q}")

    data_qubits = list(range(0, n_data))
    ancilla_qubits = list(range(n_data, nq))

    def emit_stabilizer_round(round_num: int, with_noise: bool = True):
        """Emit one round of stabilizer measurements (CX pattern)."""
        lines.append("TICK")
        # Reset ancillas
        anc_str = " ".join(str(q) for q in ancilla_qubits)
        lines.append(f"R {anc_str}")
        if with_noise:
            lines.append(f"X_ERROR({noise}) {anc_str}")

        lines.append("TICK")
        # CX rounds: 4 sub-rounds typical for surface/color code
        for sub in range(4):
            for i, anc in enumerate(ancilla_qubits):
                # Each ancilla connects to ~4 data qubits
                data_idx = (i * 4 + sub) % n_data
                lines.append(f"CX {data_qubits[data_idx]} {anc}")
            if with_noise:
                all_str = " ".join(str(q) for q in range(nq))
                lines.append(f"DEPOLARIZE2({noise}) " +
                           " ".join(f"{data_qubits[(i*4+sub)%n_data]} {anc}"
                                   for i, anc in enumerate(ancilla_qubits)))
            lines.append("TICK")

        # Measure ancillas
        lines.append(f"M({noise}) {anc_str}")

    def emit_t_injection_block(targets: list, gate: str = "T"):
        """Emit T-gate injection on specified targets."""
        target_str = " ".join(str(q) for q in targets)
        lines.append(f"{gate} {target_str}")

    # Phase 1: Initialize
    all_str = " ".join(str(q) for q in range(nq))
    data_str = " ".join(str(q) for q in data_qubits)
    lines.append(f"R {all_str}")
    lines.append("TICK")

    # Phase 2: Initial Hadamard on some data qubits
    h_qubits = data_qubits[:n_data//2]
    lines.append(f"H {' '.join(str(q) for q in h_qubits)}")
    lines.append("TICK")

    # Phase 3: First stabilizer round
    emit_stabilizer_round(0)

    # Phase 4: T-gate injection - Block 1 (small, like d5 has T_DAG on 1 qubit first)
    lines.append("TICK")
    emit_t_injection_block([data_qubits[3]], "T_DAG")

    # Phase 5: More Cliffords
    for q in range(0, n_data - 1, 2):
        lines.append(f"CX {data_qubits[q]} {data_qubits[q+1]}")
    lines.append("TICK")

    # Phase 6: T-gate injection - Block 2 (medium, ~13 targets like d5)
    block2 = data_qubits[:13]
    emit_t_injection_block(block2, "T_DAG")
    lines.append("TICK")

    for q in range(0, len(block2) - 1, 2):
        lines.append(f"CX {block2[q]} {block2[q+1]}")
    lines.append("TICK")

    emit_t_injection_block(block2, "T")
    lines.append("TICK")

    # Phase 7: Stabilizer round 2
    emit_stabilizer_round(1)

    # Phase 8: T-gate injection - Block 3 (large, ~19 targets = peak_rank region)
    block3_a = [data_qubits[i] for i in range(0, n_data, 2)][:10]
    block3_b = [data_qubits[i] for i in range(1, n_data, 2)][:9]
    emit_t_injection_block(block3_a, "T")
    lines.append("TICK")

    for q in range(0, len(block3_a) - 1, 2):
        lines.append(f"CZ {block3_a[q]} {block3_a[q+1]}")
    lines.append("TICK")

    # Largest T block: all n_data targets => peak_rank up to n_data
    emit_t_injection_block(data_qubits[:19], "T_DAG")
    lines.append("TICK")

    for q in range(0, 18, 2):
        lines.append(f"CX {data_qubits[q]} {data_qubits[q+1]}")
    lines.append("TICK")

    emit_t_injection_block(data_qubits[:19], "T")
    lines.append("TICK")

    # Phase 9: Stabilizer rounds 3-5 (to make circuit longer)
    for r in range(3):
        emit_stabilizer_round(r + 2)
        lines.append("SHIFT_COORDS(0, 0, 1)")

    # Phase 10: More T-gate blocks (cultivation iterates)
    emit_t_injection_block(block3_a + block3_b, "T_DAG")
    lines.append("TICK")
    for q in range(0, len(block3_a) - 1, 2):
        lines.append(f"CZ {block3_a[q]} {block3_a[q+1]}")
    emit_t_injection_block(block3_a + block3_b, "T")
    lines.append("TICK")

    # Phase 11: Final stabilizer rounds
    for r in range(3):
        emit_stabilizer_round(r + 5)

    # Phase 12: Final measurement of all qubits
    lines.append("TICK")
    lines.append(f"M({noise}) " + " ".join(str(q) for q in range(nq)))

    # Phase 13: Detectors and observable
    meas_offset = len(ancilla_qubits) * 8 + nq  # approximate
    for q in range(min(20, n_ancilla)):
        lines.append(f"DETECTOR rec[{-(q+1)}]")

    # Observable
    lines.append(f"OBSERVABLE_INCLUDE(0) rec[-1]")

    return "\n".join(lines) + "\n"


def generate_cultivation_d9(noise: float = 0.001) -> str:
    """Generate a synthetic d=9 cultivation-like circuit.

    Scaling:
      d=7 (projected): ~80 qubits, ~37 T-qubit targets
      d=9 (projected): ~130 qubits, ~61 T-qubit targets, peak_rank ~19
    """
    nq = 130
    n_data = 61
    n_ancilla = nq - n_data

    lines = []
    lines.append(f"# Synthetic d=9 cultivation circuit (scaled from d5/d7 pattern)")
    lines.append(f"# Qubits: {nq}, T-qubit targets: {n_data}, noise: {noise}")

    grid_side = int(math.ceil(math.sqrt(nq)))
    for q in range(nq):
        r, c = divmod(q, grid_side)
        lines.append(f"QUBIT_COORDS({r}, {c}) {q}")

    data_qubits = list(range(0, n_data))
    ancilla_qubits = list(range(n_data, nq))

    def emit_stab_round(rnd):
        lines.append("TICK")
        anc_str = " ".join(str(q) for q in ancilla_qubits)
        lines.append(f"R {anc_str}")
        lines.append(f"X_ERROR({noise}) {anc_str}")
        lines.append("TICK")
        for sub in range(4):
            for i, anc in enumerate(ancilla_qubits):
                data_idx = (i * 4 + sub) % n_data
                lines.append(f"CX {data_qubits[data_idx]} {anc}")
            lines.append(f"DEPOLARIZE1({noise}) " +
                        " ".join(str(q) for q in range(nq)))
            lines.append("TICK")
        lines.append(f"M({noise}) {anc_str}")

    all_str = " ".join(str(q) for q in range(nq))
    data_str = " ".join(str(q) for q in data_qubits)
    lines.append(f"R {all_str}")
    lines.append("TICK")
    lines.append(f"H {' '.join(str(q) for q in data_qubits[:n_data//2])}")
    lines.append("TICK")

    # Stabilizer round 0
    emit_stab_round(0)

    # T-gate block 1: small
    lines.append("TICK")
    lines.append(f"T_DAG {data_qubits[3]}")

    # Clifford interleave
    for q in range(0, n_data - 1, 2):
        lines.append(f"CX {data_qubits[q]} {data_qubits[q+1]}")
    lines.append("TICK")

    # T-gate block 2: medium (19 targets, like d5 peak)
    block2 = data_qubits[:19]
    lines.append(f"T_DAG {' '.join(str(q) for q in block2)}")
    lines.append("TICK")
    for q in range(0, len(block2) - 1, 2):
        lines.append(f"CX {block2[q]} {block2[q+1]}")
    lines.append("TICK")
    lines.append(f"T {' '.join(str(q) for q in block2)}")
    lines.append("TICK")

    # Stabilizer round 1
    emit_stab_round(1)

    # T-gate block 3: LARGE (37 targets => peak_rank ~19 after active tracking)
    # In practice, measurements reduce active_k, so we need more T gates
    # between measurements to keep active_k high
    block3 = data_qubits[:37]
    lines.append("TICK")
    lines.append(f"T {' '.join(str(q) for q in block3[:19])}")
    lines.append("TICK")
    for q in range(0, 18, 2):
        lines.append(f"CZ {block3[q]} {block3[q+1]}")
    lines.append("TICK")

    # Add remaining T gates (keeps some active while adding more)
    lines.append(f"T_DAG {' '.join(str(q) for q in block3[19:])}")
    lines.append("TICK")
    for q in range(19, len(block3) - 1, 2):
        lines.append(f"CX {block3[q]} {block3[q+1]}")
    lines.append("TICK")

    lines.append(f"T {' '.join(str(q) for q in block3)}")
    lines.append("TICK")

    # Stabilizer rounds 2-7 (long instruction sequence)
    for r in range(6):
        emit_stab_round(r + 2)
        lines.append("SHIFT_COORDS(0, 0, 1)")

    # More T-gate blocks (cultivation iterate)
    for iteration in range(3):
        block = data_qubits[:19]
        lines.append("TICK")
        lines.append(f"T_DAG {' '.join(str(q) for q in block)}")
        lines.append("TICK")
        for q in range(0, len(block) - 1, 2):
            lines.append(f"CZ {block[q]} {block[q+1]}")
        lines.append("TICK")
        lines.append(f"T {' '.join(str(q) for q in block)}")
        lines.append("TICK")
        emit_stab_round(8 + iteration)

    # Final measurement
    lines.append("TICK")
    lines.append(f"M({noise}) " + " ".join(str(q) for q in range(nq)))

    for q in range(min(30, n_ancilla)):
        lines.append(f"DETECTOR rec[{-(q+1)}]")
    lines.append(f"OBSERVABLE_INCLUDE(0) rec[-1]")

    return "\n".join(lines) + "\n"


# ─── Controlled peak_rank circuits ────────────────────────────────────────

def generate_peak_rank_circuit(
    target_rank: int,
    num_qubits: int,
    depth_rounds: int,
    noise: float = 0.001,
) -> str:
    """Generate a circuit with precisely controlled peak_rank.

    peak_rank = number of T-gate target qubits simultaneously active
    (between expand and contract operations). To achieve peak_rank = K,
    we inject K T-gates on distinct qubits before any measurement reduces
    active_k.

    Args:
        target_rank: Desired peak active_k value (0-19)
        num_qubits: Total physical qubits
        depth_rounds: Number of stabilizer+T+measurement rounds
        noise: Depolarizing noise strength
    """
    assert target_rank <= num_qubits, f"rank {target_rank} > qubits {num_qubits}"

    lines = []
    lines.append(f"# Controlled peak_rank={target_rank} circuit")
    lines.append(f"# Qubits: {num_qubits}, depth: {depth_rounds}, noise: {noise}")

    # Qubit coordinates
    grid_side = int(math.ceil(math.sqrt(num_qubits)))
    for q in range(num_qubits):
        r, c = divmod(q, grid_side)
        lines.append(f"QUBIT_COORDS({r}, {c}) {q}")

    for rnd in range(depth_rounds):
        # Clifford layer: CNOT lattice
        for q in range(0, num_qubits - 1, 2):
            lines.append(f"CX {q} {q+1}")
        lines.append("TICK")

        for q in range(1, num_qubits - 1, 2):
            lines.append(f"CX {q} {q+1}")
        lines.append("TICK")

        # Hadamard layer
        all_str = " ".join(str(q) for q in range(num_qubits))
        lines.append(f"H {all_str}")
        lines.append("TICK")

        # T-gate injection: target_rank T-gates on distinct qubits
        # This is the peak_rank determining section
        if target_rank > 0:
            t_targets = list(range(min(target_rank, num_qubits)))
            t_str = " ".join(str(q) for q in t_targets)
            lines.append(f"T {t_str}")
            lines.append("TICK")

            # Clifford ops within the T-active region (entangle T qubits)
            for q in range(0, len(t_targets) - 1, 2):
                lines.append(f"CZ {t_targets[q]} {t_targets[q+1]}")
            lines.append("TICK")

            # S gates (Clifford, but exercise array kernels when active_k > 0)
            lines.append(f"S {t_str}")
            lines.append("TICK")

            # Conjugate T block
            lines.append(f"T_DAG {t_str}")
            lines.append("TICK")

            # More entangling within active region
            for q in range(0, len(t_targets) - 1, 2):
                lines.append(f"CX {t_targets[q]} {t_targets[q+1]}")
            lines.append("TICK")

            # Second T injection (keeps active_k at target_rank)
            lines.append(f"T {t_str}")
            lines.append("TICK")

        # Noise
        lines.append(f"DEPOLARIZE1({noise}) {all_str}")
        lines.append("TICK")

        # Measurement (contracts active_k back to 0)
        lines.append(f"M({noise}) {all_str}")

        # Detectors
        for q in range(num_qubits):
            if rnd > 0:
                lines.append(f"DETECTOR rec[{-(num_qubits - q)}] rec[{-(2*num_qubits - q)}]")
            else:
                lines.append(f"DETECTOR rec[{-(num_qubits - q)}]")

    # Observable
    lines.append(f"OBSERVABLE_INCLUDE(0) rec[-1]")

    return "\n".join(lines) + "\n"


# ─── Surface code with T-gate injection (realistic QEC + non-Clifford) ────

def generate_surface_code_with_t_gates(
    distance: int,
    rounds: int,
    n_t_injections: int,
    noise: float = 0.001,
) -> str:
    """Generate a surface code circuit with T-gate injection rounds.

    This creates a realistic QEC workload: surface code stabilizer rounds
    interspersed with T-gate injections that raise peak_rank.

    Args:
        distance: Code distance (3, 5, 7, 9, 11)
        rounds: Number of stabilizer rounds
        n_t_injections: Number of T gates to inject between rounds
        noise: Depolarizing noise strength
    """
    # Generate base surface code circuit
    base = stim.Circuit.generated(
        code_task="surface_code:rotated_memory_x",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=noise,
    )
    base_str = str(base)
    base_lines = base_str.split("\n")

    # Find the first TICK after initial setup (injection point)
    tick_indices = [i for i, l in enumerate(base_lines) if l.strip() == "TICK"]

    if len(tick_indices) < 3 or n_t_injections == 0:
        return base_str

    # Insert T gates after the 2nd TICK (after first stabilizer round)
    insert_idx = tick_indices[2] + 1
    n_qubits = base.num_qubits
    t_targets = list(range(min(n_t_injections, n_qubits)))
    t_str = " ".join(str(q) for q in t_targets)

    injection_lines = [
        f"# T-gate injection: {n_t_injections} targets (peak_rank contribution)",
        f"T {t_str}",
        "TICK",
    ]

    # Also add a conjugate T block midway
    if len(tick_indices) > 6:
        mid_idx = tick_indices[len(tick_indices)//2] + 1
        injection_lines_mid = [
            f"T_DAG {t_str}",
            "TICK",
            f"T {t_str}",
            "TICK",
        ]
        # Insert mid first (so indices don't shift)
        result_lines = (
            base_lines[:insert_idx]
            + injection_lines
            + base_lines[insert_idx:mid_idx]
            + injection_lines_mid
            + base_lines[mid_idx:]
        )
    else:
        result_lines = (
            base_lines[:insert_idx]
            + injection_lines
            + base_lines[insert_idx:]
        )

    header = [
        f"# Surface code d={distance} with {n_t_injections} T-gate injections",
        f"# Base: rotated_memory_x, rounds={rounds}, noise={noise}",
    ]

    return "\n".join(header + result_lines) + "\n"


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate large Stim QEC circuits for GPU stress-testing"
    )
    parser.add_argument(
        "--outdir",
        default="tests/fixtures/large",
        help="Output directory for circuit files",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    manifest = []
    generated = 0

    def save(fname: str, content: str, desc: str, expected_peak_rank: int):
        nonlocal generated
        path = os.path.join(args.outdir, fname)
        with open(path, "w") as f:
            f.write(content)
        lines_count = len(content.strip().split("\n"))
        # Count qubits
        nq = len([l for l in content.split("\n") if l.strip().startswith("QUBIT_COORDS")])
        # Count T gates
        t_lines = [l for l in content.split("\n")
                   if l.strip().startswith("T ") or l.strip().startswith("T_DAG ")]
        n_t = sum(len(l.strip().split()) - 1 for l in t_lines)
        manifest.append({
            "file": fname,
            "qubits": nq,
            "t_ops": n_t,
            "lines": lines_count,
            "peak_rank": expected_peak_rank,
            "desc": desc,
        })
        generated += 1
        print(f"  [{generated:3d}] {fname}: {nq} qubits, {n_t} T ops, "
              f"{lines_count} lines, peak_rank={expected_peak_rank}")

    # ── Category 1: Stim built-in surface code (Clifford, peak_rank=0) ──
    print("\n=== Category 1: Surface codes (Clifford, peak_rank=0) ===")
    for d in [7, 9, 11, 13, 15]:
        for r in [d, d * 2, d * 3]:
            content = generate_surface_code(d, r)
            fname = f"surface_d{d}_r{r}.stim"
            desc = f"Surface code d={d} r={r} rotated_memory_x"
            save(fname, content, desc, 0)

    # ── Category 2: Stim built-in color code (Clifford, peak_rank=0) ──
    print("\n=== Category 2: Color codes (Clifford, peak_rank=0) ===")
    for d in [5, 7, 9, 11]:
        for r in [d, d * 2]:
            content = generate_color_code(d, r)
            fname = f"color_d{d}_r{r}.stim"
            desc = f"Color code d={d} r={r} memory_xyz"
            save(fname, content, desc, 0)

    # ── Category 3: Surface code with T-gate injection ──
    print("\n=== Category 3: Surface code + T injection ===")
    for d in [7, 9, 11]:
        for n_t in [5, 10, 15, 19]:
            content = generate_surface_code_with_t_gates(d, d * 2, n_t)
            fname = f"surface_d{d}_t{n_t}.stim"
            desc = f"Surface code d={d} with {n_t} T injections"
            save(fname, content, desc, n_t)

    # ── Category 4: Synthetic cultivation circuits ──
    print("\n=== Category 4: Synthetic cultivation circuits ===")
    for noise in [0.001, 0.005]:
        noise_str = str(noise).replace(".", "")
        content = generate_cultivation_d7(noise)
        fname = f"cultivation_d7_p{noise_str}.stim"
        save(fname, content, f"Synthetic d=7 cultivation p={noise}", 19)

        content = generate_cultivation_d9(noise)
        fname = f"cultivation_d9_p{noise_str}.stim"
        save(fname, content, f"Synthetic d=9 cultivation p={noise}", 19)

    # ── Category 5: Controlled peak_rank circuits ──
    print("\n=== Category 5: Controlled peak_rank circuits ===")
    # Target all three tiers
    rank_configs = [
        # (rank, qubits, depth, desc_tag)
        # Tier 3: global-coop (rank 11-19)
        (11, 50, 5, "tier3"),
        (12, 50, 5, "tier3"),
        (13, 50, 8, "tier3"),
        (14, 50, 8, "tier3"),
        (15, 50, 10, "tier3"),
        (16, 50, 10, "tier3"),
        (17, 50, 10, "tier3"),
        (18, 50, 12, "tier3"),
        (19, 50, 12, "tier3"),
        # High rank with many qubits (stress test)
        (15, 100, 5, "wide"),
        (17, 100, 8, "wide"),
        (19, 100, 10, "wide"),
        # Deep circuits at moderate rank
        (13, 50, 20, "deep"),
        (15, 50, 30, "deep"),
        (17, 50, 40, "deep"),
        (19, 50, 50, "deep"),
    ]

    for rank, nq, depth, tag in rank_configs:
        content = generate_peak_rank_circuit(rank, nq, depth)
        fname = f"rank{rank}_q{nq}_d{depth}_{tag}.stim"
        desc = f"Controlled rank={rank} q={nq} depth={depth} ({tag})"
        save(fname, content, desc, rank)

    # ── Category 6: Extreme circuits (push GPU limits) ──
    print("\n=== Category 6: Extreme circuits (GPU stress) ===")
    # Maximum rank with maximum qubits
    for rank in [15, 17, 19]:
        content = generate_peak_rank_circuit(rank, 200, 15)
        fname = f"extreme_r{rank}_q200_d15.stim"
        desc = f"Extreme rank={rank} q=200 depth=15"
        save(fname, content, desc, rank)

    # Very deep at high rank
    content = generate_peak_rank_circuit(19, 50, 100)
    fname = "extreme_r19_q50_d100.stim"
    desc = "Extreme rank=19 q=50 depth=100 (stress test)"
    save(fname, content, desc, 19)

    # Large surface code at high distance (long Clifford chains)
    for d in [17, 21, 25]:
        content = generate_surface_code(d, d * 3)
        fname = f"surface_d{d}_r{d*3}.stim"
        desc = f"Large surface code d={d} r={d*3}"
        save(fname, content, desc, 0)

    # ── Write manifest ──
    print(f"\n=== Writing manifest ({len(manifest)} circuits) ===")
    manifest_path = os.path.join(args.outdir, "manifest.txt")
    with open(manifest_path, "w") as f:
        f.write("# Large circuit test fixtures for GPU stress testing\n")
        f.write("# Generated by scripts/generate_large_circuits.py\n")
        f.write("#\n")
        f.write("# Columns: file | qubits | t_ops | lines | peak_rank | description\n")
        f.write("#\n")
        f.write(f"# {'file':<45} {'qubits':>6} {'t_ops':>6} {'lines':>6} "
                f"{'peak_rank':>9} description\n")
        f.write("#" + "-" * 120 + "\n")

        # Sort by peak_rank then qubits
        for entry in sorted(manifest, key=lambda x: (x["peak_rank"], x["qubits"])):
            f.write(f"{entry['file']:<45} {entry['qubits']:>6} {entry['t_ops']:>6} "
                    f"{entry['lines']:>6} {entry['peak_rank']:>9} {entry['desc']}\n")

    print(f"\nTotal: {generated} circuits generated in {args.outdir}/")
    print(f"Manifest: {manifest_path}")

    # Summary by tier
    tier1 = [e for e in manifest if e["peak_rank"] <= 4]
    tier2 = [e for e in manifest if 5 <= e["peak_rank"] <= 10]
    tier3 = [e for e in manifest if e["peak_rank"] >= 11]
    print(f"\nTier breakdown:")
    print(f"  Tier 1 (per-thread,  rank 0-4):   {len(tier1)} circuits")
    print(f"  Tier 2 (shared-coop, rank 5-10):   {len(tier2)} circuits")
    print(f"  Tier 3 (global-coop, rank 11-19):  {len(tier3)} circuits")

    max_lines = max(e["lines"] for e in manifest)
    max_qubits = max(e["qubits"] for e in manifest)
    max_rank = max(e["peak_rank"] for e in manifest)
    print(f"\nExtremes:")
    print(f"  Max lines:     {max_lines}")
    print(f"  Max qubits:    {max_qubits}")
    print(f"  Max peak_rank: {max_rank}")


if __name__ == "__main__":
    main()
