#!/usr/bin/env python3
"""Generate multi-GPU comparison plots from sweep CSV data.

Reads CSV files from results/multi_gpu/ and produces PNG plots in
results/multi_gpu/plots/.  Each sweep type (T-gate, qubit, depth) gets
one plot per GPU with one line per approach, plus one combined plot
overlaying all GPUs with all approaches.

Falls back to a text table if matplotlib is unavailable.

Usage:
    python3 scripts/plot_results.py [--datadir DIR] [--plotdir DIR]
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sweep_csv(path):
    """Load a sweep CSV and return list of row dicts."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalise numeric fields
            try:
                row["value"] = float(row["value"])
            except (ValueError, KeyError):
                pass
            try:
                row["peak_rank"] = int(row["peak_rank"])
            except (ValueError, KeyError):
                pass
            try:
                row["shots_per_sec"] = float(row["shots_per_sec"])
            except (ValueError, KeyError):
                row["shots_per_sec"] = 0.0
            rows.append(row)
    return rows


def load_all_sweeps(datadir):
    """Load all merged sweep CSVs.  Returns dict: sweep_name -> [rows]."""
    sweeps = {}
    for sweep_name in ("tgate_sweep", "qubit_sweep", "depth_sweep"):
        merged = os.path.join(datadir, f"all_{sweep_name}.csv")
        if os.path.exists(merged):
            sweeps[sweep_name] = load_sweep_csv(merged)
        else:
            # Fall back: collect individual per-partition files
            rows = []
            for fname in sorted(os.listdir(datadir)):
                if fname.endswith(f"_{sweep_name}.csv"):
                    rows.extend(load_sweep_csv(os.path.join(datadir, fname)))
            if rows:
                sweeps[sweep_name] = rows
    return sweeps


# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------

SWEEP_META = {
    "tgate_sweep": {
        "title": "T-gate Sweep (q=17, d=3)",
        "xlabel": "T-gate count",
        "variable": "t_gates",
    },
    "qubit_sweep": {
        "title": "Qubit Sweep (t=2, d=3)",
        "xlabel": "Qubits",
        "variable": "qubits",
    },
    "depth_sweep": {
        "title": "Depth Sweep (q=17, t=2)",
        "xlabel": "Depth rounds",
        "variable": "depth",
    },
}

# Distinct colours and markers for GPUs
GPU_STYLES = {
    "mi300x":    {"color": "#1f77b4", "marker": "o"},
    "mi300x-es": {"color": "#ff7f0e", "marker": "s"},
    "mi325x":    {"color": "#2ca02c", "marker": "^"},
    "mi350x-es": {"color": "#d62728", "marker": "D"},
}

# Line-styles for approaches
APPROACH_LINESTYLES = {
    "svm":        "-",
    "compiled":   "--",
    "per-op":     "-.",
    "graph":      ":",
    "split":      (0, (3, 1, 1, 1)),
    "persistent": (0, (5, 2)),
    "svm-opt":    (0, (1, 1)),
}


def group_by(rows, key):
    """Group rows by a key, returning dict[key_value] -> [rows]."""
    groups = defaultdict(list)
    for r in rows:
        groups[r.get(key, "unknown")].append(r)
    return dict(groups)


# ---------------------------------------------------------------------------
# Matplotlib plotting
# ---------------------------------------------------------------------------

def plot_per_gpu(sweep_name, rows, plotdir):
    """One plot per GPU: x=sweep value, y=shots/s, one line per approach."""
    import matplotlib.pyplot as plt

    meta = SWEEP_META[sweep_name]
    by_gpu = group_by(rows, "gpu")

    for gpu, gpu_rows in sorted(by_gpu.items()):
        fig, ax = plt.subplots(figsize=(8, 5))
        by_approach = group_by(gpu_rows, "approach")

        for appr, appr_rows in sorted(by_approach.items()):
            xs = sorted(set(r["value"] for r in appr_rows))
            val_to_sps = {r["value"]: r["shots_per_sec"] for r in appr_rows}
            ys = [val_to_sps.get(x, 0) for x in xs]

            ls = APPROACH_LINESTYLES.get(appr, "-")
            ax.plot(xs, ys, label=appr, linestyle=ls, linewidth=2,
                    marker="o", markersize=5)

        ax.set_title(f"{meta['title']} -- {gpu}")
        ax.set_xlabel(meta["xlabel"])
        ax.set_ylabel("Shots / second")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style="sci", axis="y", scilimits=(6, 6))

        fname = f"{gpu}_{sweep_name}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(plotdir, fname), dpi=150)
        plt.close(fig)
        print(f"  Saved {fname}")


def plot_combined(sweep_name, rows, plotdir):
    """One combined plot: all GPUs overlaid, with approach as line style."""
    import matplotlib.pyplot as plt

    meta = SWEEP_META[sweep_name]
    by_gpu = group_by(rows, "gpu")

    fig, ax = plt.subplots(figsize=(10, 6))

    for gpu, gpu_rows in sorted(by_gpu.items()):
        style = GPU_STYLES.get(gpu, {"color": "gray", "marker": "x"})
        by_approach = group_by(gpu_rows, "approach")

        for appr, appr_rows in sorted(by_approach.items()):
            xs = sorted(set(r["value"] for r in appr_rows))
            val_to_sps = {r["value"]: r["shots_per_sec"] for r in appr_rows}
            ys = [val_to_sps.get(x, 0) for x in xs]

            ls = APPROACH_LINESTYLES.get(appr, "-")
            ax.plot(xs, ys,
                    label=f"{gpu} / {appr}",
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=ls,
                    linewidth=1.8,
                    markersize=5)

    ax.set_title(f"{meta['title']} -- All GPUs")
    ax.set_xlabel(meta["xlabel"])
    ax.set_ylabel("Shots / second")
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style="sci", axis="y", scilimits=(6, 6))

    fname = f"combined_{sweep_name}.png"
    fig.tight_layout()
    fig.savefig(os.path.join(plotdir, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


# ---------------------------------------------------------------------------
# Text-table fallback
# ---------------------------------------------------------------------------

def text_table(sweep_name, rows):
    """Print a text table when matplotlib is unavailable."""
    meta = SWEEP_META[sweep_name]
    by_gpu = group_by(rows, "gpu")

    print(f"\n{'=' * 70}")
    print(f"  {meta['title']}")
    print(f"{'=' * 70}")

    for gpu, gpu_rows in sorted(by_gpu.items()):
        by_approach = group_by(gpu_rows, "approach")
        approaches = sorted(by_approach.keys())

        # Collect all x values
        all_xs = sorted(set(r["value"] for r in gpu_rows))

        # Header
        header = f"  {meta['xlabel']:>10}"
        for a in approaches:
            header += f"  {a:>14}"
        print(f"\n  GPU: {gpu}")
        print(header)
        print("  " + "-" * (len(header) - 2))

        for x in all_xs:
            line = f"  {x:>10.0f}"
            for a in approaches:
                val_to_sps = {r["value"]: r["shots_per_sec"]
                              for r in by_approach.get(a, [])}
                sps = val_to_sps.get(x, 0)
                line += f"  {sps:>14.0f}"
            print(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot multi-GPU sweep results")
    parser.add_argument("--datadir",
                        default="results/multi_gpu",
                        help="Directory containing sweep CSVs")
    parser.add_argument("--plotdir",
                        default="results/multi_gpu/plots",
                        help="Directory for output plots")
    args = parser.parse_args()

    sweeps = load_all_sweeps(args.datadir)
    if not sweeps:
        print("No sweep data found in", args.datadir)
        sys.exit(1)

    # Try matplotlib
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
        HAS_MPL = True
    except ImportError:
        HAS_MPL = False
        print("WARNING: matplotlib not available, falling back to text tables")

    os.makedirs(args.plotdir, exist_ok=True)

    for sweep_name, rows in sorted(sweeps.items()):
        print(f"\n--- {SWEEP_META[sweep_name]['title']} ({len(rows)} data points) ---")

        if HAS_MPL:
            plot_per_gpu(sweep_name, rows, args.plotdir)
            plot_combined(sweep_name, rows, args.plotdir)
        else:
            text_table(sweep_name, rows)

    if HAS_MPL:
        print(f"\nAll plots saved to {args.plotdir}/")
    print("Done.")


if __name__ == "__main__":
    main()
