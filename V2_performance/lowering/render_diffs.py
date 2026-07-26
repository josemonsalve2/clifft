#!/usr/bin/env python3
"""Render progressive-lowering diffs between adjacent pipeline stages.

Produces two artifacts per stage pair:
  <out>/<pipeline>.<circuit>.<a>__<b>.diff  - unified diff, greppable, for the report
  <out>/<pipeline>.<circuit>.<a>__<b>.html  - side-by-side colorized, for the deck

Why not just `diff -u`: the stage files are up to 20k lines and the interesting
change is almost never at the top. `--window` extracts the N largest contiguous
change hunks so the report can quote a hunk that actually shows the transform,
rather than the first one alphabetically.

Stage files are NOT line-aligned across a lowering boundary (MLIR SSA names get
renumbered by every pass), so a raw line diff between .mlir and .ll is noise.
That is intentional: the diff is evidence of *how much* the representation
changed, and the hunk excerpts show *what kind* of change. For pairs where the
representations differ in kind, read the stats table in stage_stats.csv
alongside the diff.
"""
import argparse
import difflib
import html
import os
import re
import sys

# Token classes for the side-by-side HTML. Kept deliberately small: this is
# evidence, not a syntax highlighter, and over-coloring hides the +/- signal.
PATTERNS = [
    ("kw", re.compile(r"\b(llvm\.func|llvm\.mlir\.constant|define|declare|"
                      r"module|attributes|call|br|ret|switch|phi)\b")),
    ("op", re.compile(r"\b(llvm\.[a-z_.]+|v_[a-z0-9_]+|s_[a-z0-9_]+|"
                      r"ds_[a-z0-9_]+|global_[a-z0-9_]+|flat_[a-z0-9_]+|"
                      r"buffer_[a-z0-9_]+)\b")),
    ("ssa", re.compile(r"%[A-Za-z0-9_.]+")),
    ("num", re.compile(r"\b-?\d+\.?\d*(e[-+]?\d+)?\b")),
]

CSS = """
body{background:#12141a;color:#d8dee9;font:12px/1.45 'JetBrains Mono',
 'DejaVu Sans Mono',monospace;margin:0;padding:16px}
h1{font-size:15px;color:#88c0d0;margin:0 0 4px}
h2{font-size:12px;color:#7a8494;font-weight:400;margin:0 0 14px}
table{border-collapse:collapse;width:100%;table-layout:fixed}
td{vertical-align:top;padding:0 6px;white-space:pre-wrap;word-break:break-all}
td.ln{width:44px;text-align:right;color:#4b5262;user-select:none;
 border-right:1px solid #232833}
td.code{width:calc(50% - 50px)}
tr.del td.code{background:#3a1d22}
tr.add td.code{background:#16301f}
tr.hdr td{background:#1c2530;color:#81a1c1;padding:5px 8px;font-weight:600;
 border-top:1px solid #2e3440}
.kw{color:#81a1c1}.op{color:#88c0d0}.ssa{color:#b48ead}.num{color:#d08770}
.gone{opacity:.28}
.legend{margin:10px 0 16px;font-size:11px;color:#7a8494}
.legend span{padding:2px 7px;margin-right:8px;border-radius:3px}
.legend .d{background:#3a1d22;color:#d8dee9}.legend .a{background:#16301f;color:#d8dee9}
"""


def colorize(line):
    """Escape then apply non-overlapping token spans, longest-match-first."""
    spans = []
    for cls, pat in PATTERNS:
        for m in pat.finditer(line):
            if not any(s < m.end() and m.start() < e for s, e, _ in spans):
                spans.append((m.start(), m.end(), cls))
    out, prev = [], 0
    for s, e, cls in sorted(spans):
        out.append(html.escape(line[prev:s]))
        out.append(f'<span class="{cls}">{html.escape(line[s:e])}</span>')
        prev = e
    out.append(html.escape(line[prev:]))
    return "".join(out)


def read(p):
    with open(p, errors="replace") as f:
        return f.read().splitlines()


def top_hunks(a, b, n_hunks, ctx, cap):
    """Return the n_hunks largest change regions as (a_lo,a_hi,b_lo,b_hi).

    Each hunk is truncated to `cap` lines per side. Across a lowering boundary
    the SSA namespace is renumbered wholesale, so the "largest change hunk" is
    frequently the entire file; an uncapped render is a megabyte of scrolling
    that shows the same transform the first 60 lines already showed.
    """
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    ops = [o for o in sm.get_opcodes() if o[0] != "equal"]
    ops.sort(key=lambda o: (o[2] - o[1]) + (o[4] - o[3]), reverse=True)
    hunks = []
    for _, i1, i2, j1, j2 in ops[:n_hunks]:
        a1, b1 = max(0, i1 - ctx), max(0, j1 - ctx)
        hunks.append((a1, min(len(a), min(i2 + ctx, a1 + cap)),
                      b1, min(len(b), min(j2 + ctx, b1 + cap))))
    return sorted(hunks)


def emit_html(path, title, sub, a, b, hunks):
    rows = []
    for hi, (a1, a2, b1, b2) in enumerate(hunks):
        rows.append(f'<tr class="hdr"><td colspan="4">hunk {hi+1}/{len(hunks)} '
                    f'&mdash; left {a1+1}..{a2}, right {b1+1}..{b2}</td></tr>')
        sub_a, sub_b = a[a1:a2], b[b1:b2]
        sm = difflib.SequenceMatcher(None, sub_a, sub_b, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for k in range(i2 - i1):
                    la, lb = sub_a[i1 + k], sub_b[j1 + k]
                    rows.append(
                        f'<tr><td class="ln">{a1+i1+k+1}</td>'
                        f'<td class="code">{colorize(la)}</td>'
                        f'<td class="ln">{b1+j1+k+1}</td>'
                        f'<td class="code">{colorize(lb)}</td></tr>')
                continue
            # Pair replaced lines up so the eye can compare them directly;
            # pad the shorter side with dimmed blanks.
            L, R = sub_a[i1:i2], sub_b[j1:j2]
            for k in range(max(len(L), len(R))):
                lt = (f'<td class="ln">{a1+i1+k+1}</td>'
                      f'<td class="code">{colorize(L[k])}</td>') if k < len(L) \
                    else '<td class="ln gone"></td><td class="code gone"></td>'
                rt = (f'<td class="ln">{b1+j1+k+1}</td>'
                      f'<td class="code">{colorize(R[k])}</td>') if k < len(R) \
                    else '<td class="ln gone"></td><td class="code gone"></td>'
                cls = "del" if k >= len(R) else ("add" if k >= len(L) else "del add")
                rows.append(f'<tr class="{cls}">{lt}{rt}</tr>')
    with open(path, "w") as f:
        f.write(f'<!doctype html><meta charset="utf-8"><title>{html.escape(title)}'
                f'</title><style>{CSS}</style>'
                f'<h1>{html.escape(title)}</h1><h2>{html.escape(sub)}</h2>'
                f'<div class="legend"><span class="d">removed / left</span>'
                f'<span class="a">added / right</span></div>'
                f'<table>{"".join(rows)}</table>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="dir holding stage files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pipeline", required=True)
    ap.add_argument("--circuit", help="label for mlir-print-ir-tree-dir layouts, "
                                      "where the dir is the circuit")
    ap.add_argument("--hunks", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=6)
    ap.add_argument("--cap", type=int, default=80,
                    help="max lines rendered per hunk side")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    # Two accepted layouts:
    #   "<circuit>.<n>_<stage>.<ext>"  - our own build_matrix.sh output
    #   "<n>_<pass-name>.mlir"         - mlir-opt --mlir-print-ir-tree-dir output,
    #                                    where the circuit is the directory itself
    stages = {}
    for fn in sorted(os.listdir(args.dir)):
        m = re.match(r"(.+?)\.(\d+)_([A-Za-z0-9]+)\.(\w+)$", fn)
        if m:
            stages.setdefault(m.group(1), []).append(
                (int(m.group(2)), m.group(3), fn))
            continue
        m = re.match(r"(\d+)_([A-Za-z0-9_.-]+)\.(\w+)$", fn)
        if m:
            stages.setdefault(args.circuit or "pass", []).append(
                (int(m.group(1)), m.group(2), fn))

    for circuit, sl in sorted(stages.items()):
        sl.sort()
        for (na, sa, fa), (nb, sb, fb) in zip(sl, sl[1:]):
            a, b = read(os.path.join(args.dir, fa)), read(os.path.join(args.dir, fb))
            base = f"{args.pipeline}.{circuit}.{na}_{sa}__{nb}_{sb}"
            hunks = top_hunks(a, b, args.hunks, args.ctx, args.cap)
            # Hunk-scoped, not whole-file: a full unified diff of two 15k-line
            # stage files is ~2 MB of mostly-renumbered SSA names. The stage
            # files are committed, so the full diff is one command away; what
            # the report and deck actually quote is the largest few hunks.
            with open(os.path.join(args.out, base + ".diff"), "w") as f:
                f.write(f"# {args.pipeline} / {circuit}: {sa} -> {sb}\n"
                        f"# {fa} ({len(a)} lines) -> {fb} ({len(b)} lines)\n"
                        f"# {len(hunks)} largest change hunks, {args.ctx} lines "
                        f"context, capped at {args.cap} lines/side. "
                        f"Full diff: diff -u {fa} {fb}\n")
                for hi, (a1, a2, b1, b2) in enumerate(hunks):
                    f.write(f"\n@@ hunk {hi+1}/{len(hunks)} "
                            f"left {a1+1}..{a2} right {b1+1}..{b2} @@\n")
                    # splitlines() stripped the terminators, so re-add them:
                    # unified_diff copies its input verbatim and would otherwise
                    # emit the whole hunk as one run-on line.
                    f.writelines(difflib.unified_diff(
                        [l + "\n" for l in a[a1:a2]],
                        [l + "\n" for l in b[b1:b2]],
                        f"{fa}:{a1+1}", f"{fb}:{b1+1}", n=args.ctx))
            emit_html(os.path.join(args.out, base + ".html"),
                      f"{args.pipeline} / {circuit}: {sa} → {sb}",
                      f"{fa} ({len(a)} lines) → {fb} ({len(b)} lines) "
                      f"— {len(hunks)} largest change hunks, "
                      f"{args.ctx} lines context, capped {args.cap}/side",
                      a, b, hunks)
            print(f"{base}: {len(a)} -> {len(b)} lines, {len(hunks)} hunks",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
