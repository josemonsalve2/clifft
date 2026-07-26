#!/usr/bin/env python3
"""verify_corr.py — three-way correctness: V2 vs GPU-SVM vs CPU reference.

The three engines do NOT all have the same contract, and conflating them is how
a real bug hid for a while:

  V2 vs SVM   HARD contract. Both are f32 GPU engines and must agree EXACTLY on
              passed_shots and observable_ones. Any difference is a bug.
  V2 vs CPU   SOFT. The CPU reference is f64, so it legitimately differs by a
              handful of shots. A difference here only means something if SVM
              does NOT show the same difference -- so we compare the DEVIATIONS.
              If V2-vs-CPU and SVM-vs-CPU deviate identically, V2 is behaving
              exactly like the reference GPU engine and the gap is pure f32/f64.

All three legs run at the same (shots, seed) per circuit so the comparison is
apples-to-apples. CPU cost grows as 2^peak_rank, so shots taper with rank --
at rank 24 a single shot carries 16M amplitudes in f64.

Runs on a compute node; radha/login has no GPU.
"""
import json
import os
import subprocess
import sys

CLIFFT = os.environ.get("CLIFFT", os.getcwd())
V2 = os.path.join(CLIFFT, "build-v2-nohip/run_v2")
SVM = os.path.join(CLIFFT, "build-gpu-mlir-mi355x/run_gpu")
TIMEOUT = int(os.environ.get("VERIFY_TIMEOUT", "2400"))
SKIP_CPU = os.environ.get("VERIFY_NO_CPU", "") == "1"


def shots_for(rank):
    """Shot count by compiled peak_rank. Bounded by the CPU f64 leg, which is
    the slow one -- the GPU legs would happily take far more."""
    if rank is None or rank <= 14:
        return int(os.environ.get("SHOTS", "2000"))
    if rank <= 21:
        return 200
    return 50


def run_json(cmd):
    """Run a binary and pull the JSON object off stdout. Returns (obj, err)."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT after %ds" % TIMEOUT
    out = p.stdout
    start, end = out.find("{"), out.rfind("}")
    if start < 0 or end < 0:
        tail = (p.stderr or out).strip().splitlines()[-4:]
        return None, "rc=%d no-json: %s" % (p.returncode, " | ".join(tail))
    try:
        return json.loads(out[start:end + 1]), None
    except json.JSONDecodeError as e:
        return None, "bad json: %s" % e


def probe_rank(circuit):
    """Cheap 1-shot GPU run just to learn the compiled peak_rank, so the real
    runs can be sized correctly."""
    env_backup = os.environ.get("V2_NO_CPU")
    os.environ["V2_NO_CPU"] = "1"
    obj, _ = run_json([V2, "--circuit", circuit, "--shots", "1", "--seed", "1"])
    if env_backup is None:
        os.environ.pop("V2_NO_CPU", None)
    else:
        os.environ["V2_NO_CPU"] = env_backup
    return obj.get("peak_rank") if obj else None


def main(circuits):
    print("node=%s cpu_leg=%s" % (os.uname().nodename, "off" if SKIP_CPU else "on"))
    hdr = "%-24s %4s %6s %10s %10s %10s  %-9s %s"
    print(hdr % ("CIRCUIT", "RANK", "SHOTS", "V2", "SVM", "CPU(f64)",
                 "V2==SVM", "V2-vs-CPU vs SVM-vs-CPU"))
    hard_failures, soft_notes = 0, 0
    for c in circuits:
        name = os.path.basename(c).replace(".stim", "")
        rank = probe_rank(c)
        shots = str(shots_for(rank))

        os.environ["V2_NO_CPU"] = "1"
        v2, verr = run_json([V2, "--circuit", c, "--shots", shots, "--seed", "1"])
        sv, serr = run_json([SVM, "--circuit", c, "--shots", shots, "--seed", "1",
                             "--no-postselection"])
        cpu, cerr = (None, "skipped")
        if not SKIP_CPU:
            cpu, cerr = run_json([SVM, "--circuit", c, "--shots", shots, "--seed", "1",
                                  "--no-postselection", "--cpu-reference"])

        if v2 is None or sv is None:
            hard_failures += 1
            print(hdr % (name, rank or "?", shots, "ERR" if v2 is None else "ok",
                         "ERR" if sv is None else "ok", "-", "ERROR",
                         (verr or "") + " ;; " + (serr or "")))
            continue

        vp, vb = v2["v2_passed"], v2["v2_observable_ones"]
        sp, sb = sv["passed_shots"], sv["observable_ones"]
        exact = (vp == sp) and (vb == sb)
        if not exact:
            hard_failures += 1

        # Soft leg: do V2 and SVM deviate from the f64 reference IDENTICALLY?
        if cpu is None:
            cpu_cell, soft = "-", ("skip" if SKIP_CPU else "ERR:" + (cerr or ""))
        else:
            cp, cb = cpu["passed_shots"], cpu["observable_ones"]
            cpu_cell = "%d;%s" % (cp, cb)
            dv = (vp - cp, [a - b for a, b in zip(vb, cb)] if len(vb) == len(cb) else None)
            ds = (sp - cp, [a - b for a, b in zip(sb, cb)] if len(sb) == len(cb) else None)
            if dv == ds:
                soft = "same (f32/f64)" if dv != (0, [0] * len(vb)) else "identical"
            else:
                soft = "DIFFER v2%s svm%s" % (dv, ds)
                soft_notes += 1

        print(hdr % (name, rank or "?", shots, "%d;%s" % (vp, vb), "%d;%s" % (sp, sb),
                     cpu_cell, "EXACT" if exact else "MISMATCH", soft))

    print("VERIFY_DONE hard_failures=%d cpu_deviation_differs=%d"
          % (hard_failures, soft_notes))
    return 1 if (hard_failures or soft_notes) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
