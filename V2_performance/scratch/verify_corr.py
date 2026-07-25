#!/usr/bin/env python3
"""verify_corr.py — byte-exact V2-vs-SVM correctness check.

Both run_v2 and run_gpu emit a JSON object on stdout, so parse that rather than
grepping. The contract V2 must hold against GPU-SVM (f32) is exact agreement on
passed_shots and observable_ones; anything else is a bug, not a tolerance.

Runs on a compute node -- radha/login has no GPU.
"""
import json
import os
import subprocess
import sys

CLIFFT = os.environ.get("CLIFFT", os.getcwd())
V2 = os.path.join(CLIFFT, "build-v2-nohip/run_v2")
SVM = os.path.join(CLIFFT, "build-gpu-mlir-mi355x/run_gpu")
SHOTS = os.environ.get("SHOTS", "2000")
TIMEOUT = int(os.environ.get("VERIFY_TIMEOUT", "1800"))


def run_json(cmd):
    """Run a binary and pull the JSON object off stdout. Returns (obj, err)."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT after %ds" % TIMEOUT
    out = p.stdout
    start = out.find("{")
    end = out.rfind("}")
    if start < 0 or end < 0:
        tail = (p.stderr or out).strip().splitlines()[-4:]
        return None, "rc=%d no-json: %s" % (p.returncode, " | ".join(tail))
    try:
        return json.loads(out[start:end + 1]), None
    except json.JSONDecodeError as e:
        return None, "bad json: %s" % e


def main(circuits):
    env = dict(os.environ, V2_NO_CPU="1")
    os.environ.update(env)
    print("node=%s shots=%s" % (os.uname().nodename, SHOTS))
    print("%-26s %5s %8s %8s %-9s %s" % ("CIRCUIT", "RANK", "V2_PASS", "SVM_PASS", "VERDICT", "DETAIL"))
    failures = 0
    for c in circuits:
        name = os.path.basename(c).replace(".stim", "")
        v2, verr = run_json([V2, "--circuit", c, "--shots", SHOTS, "--seed", "1"])
        sv, serr = run_json([SVM, "--circuit", c, "--shots", SHOTS, "--seed", "1",
                             "--no-postselection"])
        if v2 is None or sv is None:
            failures += 1
            print("%-26s %5s %8s %8s %-9s %s" % (
                name, "?", "ERR" if v2 is None else "ok", "ERR" if sv is None else "ok",
                "ERROR", (verr or "") + " ;; " + (serr or "")))
            continue
        vp, sp = v2["v2_passed"], sv["passed_shots"]
        vb, sb = v2["v2_observable_ones"], sv["observable_ones"]
        rank = v2.get("peak_rank", "?")
        ok = (vp == sp) and (vb == sb)
        if not ok:
            failures += 1
        detail = "obs v2=%s svm=%s" % (vb, sb) if not ok else "obs=%s" % (vb,)
        print("%-26s %5s %8d %8d %-9s %s" % (
            name, rank, vp, sp, "EXACT" if ok else "MISMATCH", detail))
    print("VERIFY_DONE failures=%d" % failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
