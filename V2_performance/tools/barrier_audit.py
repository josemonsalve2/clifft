#!/usr/bin/env python3
"""Audit s_barrier fencing in amdgcn binaries or assembly.

Why this exists: commit 150d09f reported "1439/1509 (95.4%) unfenced barriers"
for the pre-fix specialized kernel, but the code that produced that count was
never committed and the number does not reproduce -- the archived binary gives
1400/1509 (92.8%) under every reasonable definition. This script IS the
definition, so the next reader can re-derive the number instead of quoting it.

Two distinct measurements, both reported, because they answer different
questions:

  unfenced-with-ds-in-flight
      Walk back from each s_barrier until hitting a `s_waitcnt ... lgkmcnt(0)`
      (safe), another s_barrier (nothing in flight from this region), or a `ds_*`
      op (UNSAFE -- an LDS access can still be outstanding when the barrier
      retires). This is the correctness-relevant count: it should be 0.

  fenced-within-N
      Barriers with a lgkmcnt(0) wait in the preceding N instructions. This is a
      proxy for "the compiler emitted the fence for us", and is the number that
      moves from ~3% to ~93% across the fix. It is window-sensitive, which is
      exactly why the first measurement is the one to trust.

Usage:
    barrier_audit.py <file.hsaco|file.s> [...]
    barrier_audit.py --ab                     # rebuild coop_interpreter.c both
                                              # ways and diff (needs no GPU)
"""
import os
import re
import subprocess
import sys

LLVM = os.environ.get("LLVM_PREFIX",
                      "/shared/jmonsalv/software/modules/llvm/upstream_05082025")
CLIFFT = os.environ.get("CLIFFT", "/shared/jmonsalv/quantum/clifft_rl/clifft")


def instructions(path):
    """Normalized instruction list from a .hsaco (disassembled) or a .s file."""
    if path.endswith(".s"):
        out = open(path).read()
        keep = []
        for line in out.splitlines():
            s = line.split(";")[0].strip()
            if not s or s.startswith(".") or s.endswith(":") or s.startswith("//"):
                continue
            keep.append(s)
        return keep
    out = subprocess.run([os.path.join(LLVM, "bin", "llvm-objdump"), "-d", path],
                         capture_output=True, text=True).stdout
    # llvm-objdump puts the address in a trailing `// <addr>: <encoding>` comment
    # and prefixes every instruction line with a tab. Splitting on '//' first is
    # load-bearing: the encoding bytes would otherwise match instruction regexes.
    return [l.split("//")[0].strip() for l in out.splitlines() if l.startswith("\t")]


def audit(instrs, window=3):
    barriers = [i for i, l in enumerate(instrs) if re.match(r"^s_barrier\b", l)]
    fenced = sum(1 for i in barriers
                 if any("lgkmcnt(0)" in x for x in instrs[max(0, i - window):i]))
    unfenced = 0
    for i in barriers:
        j = i - 1
        while j >= 0:
            x = instrs[j]
            if re.search(r"s_waitcnt.*lgkmcnt\(0\)", x):
                break                      # safe: wait covers this barrier
            if re.match(r"^s_barrier", x):
                break                      # prior barrier already drained it
            if re.match(r"^ds_", x):
                unfenced += 1              # LDS op may still be in flight
                break
            j -= 1
    return len(instrs), len(barriers), fenced, unfenced


def report(label, path, window=3):
    n, b, f, u = audit(instructions(path), window)
    pf = 100 * f / b if b else 0.0
    pu = 100 * u / b if b else 0.0
    print(f"{label:24s} instrs={n:6d} barriers={b:5d} "
          f"fenced<={window}={f:5d} ({pf:5.1f}%) ds-in-flight-unfenced={u:5d} ({pu:5.1f}%)")
    return n, b, f, u


def build(tag, defines, outdir):
    """C -> amdgcn asm. No ocml link: device-library code is not V2's, and
    linking it perturbs the barrier count by one. No GPU required."""
    src = os.path.join(CLIFFT, "src/clifft/gpu/mlir/v2/coop_interpreter.c")
    ll = os.path.join(outdir, tag + ".ll")
    asm = os.path.join(outdir, tag + ".s")
    subprocess.run([os.path.join(LLVM, "bin", "clang"),
                    "--target=amdgcn-amd-amdhsa", "-mcpu=gfx950", "-ffreestanding",
                    "-nostdlib", "-nogpulib", "-std=c23", "-O2", "-ffp-contract=off",
                    *defines, "-I" + os.path.join(CLIFFT, "src"),
                    "-emit-llvm", "-S", "-o", ll, src], check=True)
    subprocess.run([os.path.join(LLVM, "bin", "llc"),
                    "-mtriple=amdgcn-amd-amdhsa", "-mcpu=gfx950",
                    "-mattr=+wavefrontsize64", "-filetype=asm", "-O2",
                    "-o", asm, ll], check=True)
    return asm


def ab():
    """Rebuild the interpreter with and without the fence and diff the audit."""
    hdr = os.path.join(CLIFFT, "src/clifft/gpu/mlir/v2/v2_ops.h")
    outdir = os.path.join(CLIFFT, "V2_performance/.scratch/barrier_ab")
    os.makedirs(outdir, exist_ok=True)
    fixed = open(hdr).read()
    fence = ('    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");\n'
             "    __builtin_amdgcn_s_barrier();\n"
             '    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");')
    if fence not in fixed:
        sys.exit("v2_barrier() no longer matches the expected fenced form; "
                 "update the pattern in this script.")
    try:
        for tag, defs in (("post_coop", []), ("post_reg", ["-DV2_REGISTER"])):
            report(tag, build(tag, defs, outdir))
        open(hdr, "w").write(fixed.replace(fence, "    __builtin_amdgcn_s_barrier();"))
        for tag, defs in (("pre_coop", []), ("pre_reg", ["-DV2_REGISTER"])):
            report(tag, build(tag, defs, outdir))
    finally:
        open(hdr, "w").write(fixed)      # always restore, even on failure
    a = audit(instructions(os.path.join(outdir, "pre_coop.s")))[0]
    b = audit(instructions(os.path.join(outdir, "post_coop.s")))[0]
    print(f"\nfence cost (coop interpreter): {a} -> {b} = {100 * (b / a - 1):+.2f}% static instructions")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "--ab":
        ab()
    else:
        for p in args:
            report(os.path.basename(p), p)
