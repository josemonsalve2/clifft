#!/usr/bin/env python3
"""Aggregate a bench_all.sh run's rocprofv3 CSVs into gpu/*.json + summary.{json,md}.

Companion to V2_performance/tools/bench_all.sh. Reads only what rocprofv3 wrote;
derives nothing that is not in the CSVs.

    python3 summarize_bench.py V2_performance/runs/<run_id>

Per circuit and engine it emits:
  kernel_name, LDS_Block_Size, Scratch_Size, VGPR_Count, Accum_VGPR_Count,
  SGPR_Count, Workgroup_Size_X, Grid_Size_X   -- from kt/*_kernel_trace.csv
  n_dispatches, total_kernel_ns, per_dispatch_ns_median
                                              -- from kt/*_kernel_stats.csv
  domain_total_ns, domain_calls               -- from kt/*_domain_stats.csv
  hsa_total_ns                                -- from hsa/*_domain_stats.csv
  counters{}                                  -- from pmc{A,B,C}/*_counter_collection.csv
  v2_vs_svm_kernel_ratio = v2.total_kernel_ns / svm.total_kernel_ns

Ratio convention: LOWER IS BETTER FOR V2. 0.50 means V2 took half the kernel
time. Reported on kernel time, never host wall time (feedback_numa_bind).

Runtime-internal kernels are excluded everywhere -- trace, stats, AND counters.
rocclr's copyBuffer/fillBufferAligned blits are allocation artifacts, not
simulation work, and on the coop-tier circuits a fillBufferAligned dispatch
precedes the real kernel: matching on it silently reports a 256-thread memset's
VGPR/LDS/grid as the simulation kernel's. Matched by PREFIX, since the runtime
appends suffixes (`Aligned`, size variants) that an exact list will miss.

A missing counter is NOT a zero counter. If a pmc pass produced no rows for a
cell, `counters` comes back empty and the derived notes say so
("counters=UNCOLLECTED") rather than asserting MFMA=0. The 20260726 run has
exactly one such cell -- qv24_L4_seed42/svm -- and the earlier version of this
script defaulted the lookup to 0.0, which reported an unmeasured cell as a
measured zero.
"""
import csv
import glob
import json
import os
import statistics
import sys

RUNTIME_KERNEL_PREFIX = "__amd_rocclr_"


def _is_runtime_kernel(name):
    return (name or "").startswith(RUNTIME_KERNEL_PREFIX)


def _rows(pattern):
    """All rows of the single CSV matching `pattern`, as dicts. [] if absent."""
    hits = glob.glob(pattern)
    if not hits:
        return []
    with open(hits[0], newline="") as f:
        return list(csv.DictReader(f))


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def collect_engine(edir):
    """One engine directory (raw/<circuit>/{v2,svm}) -> the per-engine dict."""
    out = {}

    # --- kernel identity + launch geometry, from the trace ------------------
    trace = [r for r in _rows(f"{edir}/kt/*/*_kernel_trace.csv")
             if not _is_runtime_kernel(r.get("Kernel_Name"))]
    if trace:
        r = trace[0]
        out["kernel_name"] = r["Kernel_Name"]
        for k in ("LDS_Block_Size", "Scratch_Size", "VGPR_Count",
                  "Accum_VGPR_Count", "SGPR_Count", "Workgroup_Size_X",
                  "Grid_Size_X"):
            out[k] = r.get(k, "")
        # Per-dispatch durations come from the trace, not the stats file: the
        # stats file reports a mean, and a mean over a 1-element set hides
        # whether there was 1 dispatch or 1000.
        durs = sorted(int(r["End_Timestamp"]) - int(r["Start_Timestamp"])
                      for r in trace)
        out["n_dispatches"] = len(durs)
        out["per_dispatch_ns_median"] = int(statistics.median(durs))

    # --- total kernel time, from the stats file rocprofv3 computed ----------
    for r in _rows(f"{edir}/kt/*/*_kernel_stats.csv"):
        if _is_runtime_kernel(r["Name"]):
            continue
        out["total_kernel_ns"] = int(_num(r["TotalDurationNs"]))
        out.setdefault("kernel_name", r["Name"])
        break

    # --- dispatch domain: how many launches the runtime actually issued -----
    # V2 issues ONE for the whole run (the shot loop is inside the kernel);
    # SVM's count is higher because it also issues its own copies/setup.
    for r in _rows(f"{edir}/kt/*/*_domain_stats.csv"):
        if r["Name"] == "KERNEL_DISPATCH":
            out["domain_total_ns"] = int(_num(r["TotalDurationNs"]))
            out["domain_calls"] = int(_num(r["Calls"]))
            break

    # --- HSA API time: dominated by one-time queue creation, NOT per-shot ---
    for r in _rows(f"{edir}/hsa/*/*_domain_stats.csv"):
        if r["Name"] == "HSA_API":
            out["hsa_total_ns"] = int(_num(r["TotalDurationNs"]))
            break

    # --- hardware counters, summed across dispatches of the real kernel -----
    counters = {}
    for pmc in ("pmcA", "pmcB", "pmcC"):
        for r in _rows(f"{edir}/{pmc}/*/*_counter_collection.csv"):
            if _is_runtime_kernel(r.get("Kernel_Name")):
                continue
            name = r["Counter_Name"]
            counters[name] = counters.get(name, 0.0) + _num(r["Counter_Value"])
    # Always emit the key, even empty: an absent `counters` is ambiguous
    # between "no pmc pass ran" and "the pass ran and found nothing", and a
    # downstream table that silently skips the column reads as complete.
    out["counters"] = dict(sorted(counters.items()))

    # --- derived notes ------------------------------------------------------
    notes = []
    c = out.get("counters", {})
    if not c:
        notes.append("counters=UNCOLLECTED (pmc passes produced no rows)")
    if "SQ_INSTS_MFMA" in c:
        notes.append("MFMA=0 (butterfly, no GEMM)" if c["SQ_INSTS_MFMA"] == 0.0
                     else f"MFMA={c['SQ_INSTS_MFMA']:.0f} (UNEXPECTED)")
    hit, miss = c.get("TCC_HIT_sum"), c.get("TCC_MISS_sum")
    if hit is not None and miss is not None and (hit + miss) > 0:
        notes.append(f"L2_hit%={100.0 * hit / (hit + miss):.1f}")
    waves, wcyc = c.get("SQ_WAVES"), c.get("SQ_WAVE_CYCLES")
    if waves:
        notes.append(f"wave_cycles/wave={wcyc / waves:.0f}")
    if notes:
        out["bottleneck_notes"] = notes
    return out


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <run_dir>")
    run_dir = os.path.abspath(sys.argv[1])
    raw = os.path.join(run_dir, "raw")
    if not os.path.isdir(raw):
        sys.exit(f"no raw/ under {run_dir}")
    gpu_dir = os.path.join(run_dir, "gpu")
    os.makedirs(gpu_dir, exist_ok=True)

    summary = []
    for circuit in sorted(os.listdir(raw)):
        cdir = os.path.join(raw, circuit)
        if not os.path.isdir(cdir):
            continue
        rec = {"circuit": circuit}
        for engine in ("v2", "svm"):
            edir = os.path.join(cdir, engine)
            if os.path.isdir(edir):
                rec[engine] = collect_engine(edir)
        v2k = rec.get("v2", {}).get("total_kernel_ns")
        svmk = rec.get("svm", {}).get("total_kernel_ns")
        if v2k and svmk:
            rec["v2_vs_svm_kernel_ratio"] = round(v2k / svmk, 3)
        with open(os.path.join(gpu_dir, f"{circuit}.json"), "w") as f:
            json.dump(rec, f, indent=1)
            f.write("\n")
        summary.append(rec)

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
        f.write("\n")

    # Header identity comes from manifest.json, not the directory name, so a
    # run copied or re-summarized elsewhere still names the run it came from.
    man = {}
    man_path = os.path.join(run_dir, "manifest.json")
    if os.path.exists(man_path):
        with open(man_path) as f:
            man = json.load(f)
    run_id = man.get("run_id", os.path.basename(run_dir))
    commit = man.get("commit", "?") + ("-dirty" if man.get("dirty") else "")

    ratios = [r["v2_vs_svm_kernel_ratio"] for r in summary
              if "v2_vs_svm_kernel_ratio" in r]
    lines = [f"# Benchmark run {run_id}",
             f"label: {man.get('label', '?')} · commit: {commit} "
             f"· branch: {man.get('branch', '?')}",
             "",
             "| circuit | V2 kernel(µs) | SVM kernel(µs) | V2/SVM | V2 LDS "
             "| V2 VGPR | V2 VALU | MFMA |",
             "|---|---|---|---|---|---|---|---|"]

    def _us(ns):
        return f"{ns / 1000.0:.1f}" if ns else ""

    for r in summary:
        v2, svm = r.get("v2", {}), r.get("svm", {})
        vc = v2.get("counters", {})
        valu = vc.get("SQ_INSTS_VALU")
        mfma = vc.get("SQ_INSTS_MFMA")
        lines.append(
            f"| {r['circuit']} | {_us(v2.get('total_kernel_ns'))} "
            f"| {_us(svm.get('total_kernel_ns'))} "
            f"| {r.get('v2_vs_svm_kernel_ratio', '')} "
            f"| {v2.get('LDS_Block_Size', '')} | {v2.get('VGPR_Count', '')} "
            f"| {'' if valu is None else f'{valu:.2e}'} "
            f"| {'' if mfma is None else mfma} |")
    with open(os.path.join(run_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"wrote {len(summary)} circuits to {run_dir}")
    if ratios:
        print(f"mean {sum(ratios) / len(ratios):.3f}  "
              f"median {statistics.median(ratios):.3f}  "
              f"wins {sum(1 for x in ratios if x < 1.0)}/{len(ratios)}")


if __name__ == "__main__":
    main()
