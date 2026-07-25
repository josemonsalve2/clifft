#!/usr/bin/env python3
"""Digest rocprofv3 raw CSVs (V2_performance/raw/<circuit>/<engine>/) into a
per-circuit + summary characterization under V2_performance/gpu/ and analysis/.
Pure stdlib (no pandas — compute node lacks it). Run on login node after sweep."""
import csv, glob, json, os, sys, statistics

RAW = "/shared/jmonsalv/quantum/clifft_rl/clifft/V2_performance/raw"
GPU = "/shared/jmonsalv/quantum/clifft_rl/clifft/V2_performance/gpu"
ANA = "/shared/jmonsalv/quantum/clifft_rl/clifft/V2_performance/analysis"

# MI355X / CDNA4 specs (Hyperloom HW_SPECS)
HBM_BW_GBPS = 8000.0
ACHIEVABLE_BF16_TFLOPS = 1686.0

def find_csv(d, pat):
    hits = glob.glob(os.path.join(d, "**", pat), recursive=True)
    return hits[0] if hits else None

def read_rows(path):
    if not path or not os.path.exists(path): return []
    with open(path) as f:
        return list(csv.DictReader(f))

def kernel_trace(engine_dir):
    """Kernel time + register/LDS from kt/*kernel_trace.csv and domain_stats."""
    kt = find_csv(os.path.join(engine_dir, "kt"), "*kernel_trace.csv")
    ds = find_csv(os.path.join(engine_dir, "kt"), "*domain_stats.csv")
    out = {}
    rows = read_rows(kt)
    # aggregate our target kernel dispatches (there may be many for coop=1wg/shot)
    durs = []
    for r in rows:
        name = r.get("Kernel_Name", "")
        if "clifft" in name or "sample_kernel" in name or "compiled" in name:
            try:
                durs.append(int(r["End_Timestamp"]) - int(r["Start_Timestamp"]))
            except Exception: pass
            out.setdefault("kernel_name", name)
            for k in ("LDS_Block_Size","Scratch_Size","VGPR_Count","Accum_VGPR_Count",
                      "SGPR_Count","Workgroup_Size_X","Grid_Size_X"):
                if k in r and r[k] != "": out[k] = r[k]
    if durs:
        out["n_dispatches"] = len(durs)
        out["total_kernel_ns"] = sum(durs)
        out["per_dispatch_ns_median"] = int(statistics.median(durs))
    # domain_stats gives an aggregate too
    for r in read_rows(ds):
        if r.get("Name") == "KERNEL_DISPATCH":
            out["domain_total_ns"] = int(float(r["TotalDurationNs"]))
            out["domain_calls"] = int(r["Calls"])
    return out

def hsa_overhead(engine_dir):
    ds = find_csv(os.path.join(engine_dir, "hsa"), "*domain_stats.csv")
    out = {}
    for r in read_rows(ds):
        nm = r.get("Name","")
        if "HSA" in nm.upper():
            out.setdefault("hsa_total_ns", 0)
            out["hsa_total_ns"] += int(float(r.get("TotalDurationNs",0) or 0))
    return out

def counters(engine_dir, sub):
    cc = find_csv(os.path.join(engine_dir, sub), "*counter_collection.csv")
    agg = {}
    for r in read_rows(cc):
        nm = r.get("Kernel_Name","")
        if not ("clifft" in nm or "sample_kernel" in nm or "compiled" in nm): continue
        cn = r.get("Counter_Name");
        try: cv = float(r.get("Counter_Value","0"))
        except Exception: cv = 0.0
        agg[cn] = agg.get(cn, 0.0) + cv
    return agg

def classify(c):
    """Rough bottleneck class from available counters."""
    valu = c.get("SQ_INSTS_VALU",0); lds = c.get("SQ_INSTS_LDS",0)
    waves = c.get("SQ_WAVES",0); mfma = c.get("SQ_INSTS_MFMA",0)
    fetch = c.get("FETCH_SIZE",0); write = c.get("WRITE_SIZE",0)
    hit = c.get("TCC_HIT_sum",0); miss = c.get("TCC_MISS_sum",0)
    notes = []
    if mfma == 0: notes.append("MFMA=0 (no GEMM, butterfly compute as designed)")
    if waves and lds: notes.append(f"LDS_insts/wave={lds/max(waves,1):.0f}")
    if hit+miss: notes.append(f"L2_hit%={100*hit/(hit+miss):.1f}")
    return notes

def main():
    circuits = sorted(d for d in os.listdir(RAW)
                      if os.path.isdir(os.path.join(RAW,d)) and not d.startswith("profsweep"))
    summary = []
    for circ in circuits:
        entry = {"circuit": circ}
        for eng in ("v2","svm"):
            ed = os.path.join(RAW, circ, eng)
            if not os.path.isdir(ed): continue
            e = {}
            e.update(kernel_trace(ed))
            e.update(hsa_overhead(ed))
            cA = counters(ed,"pmcA"); cB = counters(ed,"pmcB"); cC = counters(ed,"pmcC")
            allc = {**cA,**cB,**cC}
            e["counters"] = allc
            e["bottleneck_notes"] = classify(allc)
            entry[eng] = e
        # V2 vs SVM kernel-time ratio (clean kernel-vs-kernel!)
        try:
            v2t = entry["v2"]["total_kernel_ns"]; svt = entry["svm"]["total_kernel_ns"]
            entry["v2_vs_svm_kernel_ratio"] = round(v2t/svt, 3)
        except Exception: pass
        summary.append(entry)
        with open(os.path.join(GPU, f"{circ}.json"), "w") as f:
            json.dump(entry, f, indent=2)
    with open(os.path.join(ANA, "gpu_profile_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    # human table
    lines = ["# GPU profile summary (rocprofv3, clean kernel time)\n",
             "| circuit | V2 kernel(us) | SVM kernel(us) | V2/SVM | V2 LDS | V2 VGPR | V2 VALU | MFMA |",
             "|---|---|---|---|---|---|---|---|"]
    for e in summary:
        v2=e.get("v2",{}); svm=e.get("svm",{})
        def us(x):
            return f"{x/1000:.1f}" if isinstance(x,(int,float)) else "?"
        lines.append("| {} | {} | {} | {} | {} | {} | {:.2e} | {} |".format(
            e["circuit"],
            us(v2.get("total_kernel_ns")), us(svm.get("total_kernel_ns")),
            e.get("v2_vs_svm_kernel_ratio","?"),
            v2.get("LDS_Block_Size","?"), v2.get("VGPR_Count","?"),
            v2.get("counters",{}).get("SQ_INSTS_VALU",0),
            v2.get("counters",{}).get("SQ_INSTS_MFMA","?")))
    with open(os.path.join(ANA, "gpu_profile_summary.md"), "w") as f:
        f.write("\n".join(lines)+"\n")
    print("\n".join(lines))
    print(f"\nWrote {len(summary)} per-circuit JSONs to {GPU}/ and summary to {ANA}/")

if __name__ == "__main__":
    main()
