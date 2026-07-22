#!/usr/bin/env python3
"""
analyze_perf.py — Performance Engineering Analysis for clifft 3-Way GPU Benchmark

Usage:
    python3 analyze_perf.py results/bench_*_mi300x/bench.csv [results/bench_*_mi350x/bench.csv] [-o report.txt]

Produces:
    1. Per-circuit summary table (SVM vs Compiled vs MLIR)
    2. Per-tier aggregate (speedup distribution, CV)
    3. Roofline model inputs and bound classification
    4. Cross-hardware comparison (MI300X vs MI350X)
    5. Performance engineering verdict and bottleneck analysis
    6. Recommendations for next optimization steps

Roofline model constants (MI300X gfx942):
    Peak FP32 FLOP/s:   163 TF (scalar VALU)
    Peak HBM bandwidth: 5.3 TB/s (HBM3, 192GB, 6.5 GT/s)
    Peak LDS bandwidth: ~50 TB/s (64KB/CU × 304 CUs × 1.7GHz, measured peak)
    VGPR bandwidth:     >> LDS (register file, fully pipelined)

Roofline model constants (MI350X gfx950):
    Peak FP32 FLOP/s:   ~190 TF (estimated ~16% IPC gain over gfx942)
    Peak HBM bandwidth: 8.0 TB/s (HBM3e)
    Note: ES hardware — performance may not reflect final silicon.
"""

import sys, csv, math, os, glob, argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# ────────────────────────────────────────────────────────────────
# Hardware constants
# ────────────────────────────────────────────────────────────────

ARCH_SPECS = {
    'gfx942': {  # MI300X
        'name': 'MI300X (gfx942)',
        'fp32_peak_flops': 163e12,  # 163 TF FP32 scalar VALU
        'hbm_bw_bps': 5.3e12,       # 5.3 TB/s HBM3
        'lds_bw_bps': 50e12,        # ~50 TB/s LDS (per-CU, aggregate rough estimate)
        'vgpr_bw_bps': 1000e12,     # VGPR: on-chip register file, effectively unconstrained
        'num_cus': 304,
        'wavefront': 64,
        'lds_per_cu_kb': 64,
        'hbm_gb': 192,
        'clk_ghz': 1.7,             # base; boost ~2.1
    },
    'gfx950': {  # MI350X (estimated, ES hardware)
        'name': 'MI350X (gfx950)',
        'fp32_peak_flops': 190e12,  # estimated
        'hbm_bw_bps': 8.0e12,      # HBM3e
        'lds_bw_bps': 60e12,        # estimated
        'vgpr_bw_bps': 1200e12,
        'num_cus': 320,             # estimated
        'wavefront': 64,
        'lds_per_cu_kb': 64,
        'hbm_gb': 288,              # estimated
        'clk_ghz': 1.9,
    },
}

# ────────────────────────────────────────────────────────────────
# Arithmetic intensity estimates per tier and gate type
# (FLOPs/byte from the dominant memory level)
# ────────────────────────────────────────────────────────────────

# H gate: 2 complex loads, 2 stores, 6 FP32 ops (add, sub, 2×mul)
# For a rank-k state:
#   VGPR tier (k≤4): 2^k complex = 2^k * 8B, 6 * 2^(k-1) FLOPs
#     Bottleneck: VALU (data fully in registers)
#   LDS tier (5≤k≤10): same but from LDS
#     AI_LDS = 6 * 2^(k-1) / (2^k * 8) = 0.375 FLOP/byte
#     LDS BW ~50 TB/s → achievable: 0.375 * 50e12 = 18.75 TF → VALU bound, not LDS
#   HBM tier (k>10): from HBM
#     AI_HBM = 0.375 FLOP/byte (same pattern)
#     HBM BW 5.3 TB/s → achievable: 0.375 * 5.3e12 = 1.98 TF

TIER_ARITHMETIC_INTENSITY = {
    '0': {'level': 'VGPR (frame-only)', 'ai': float('inf'), 'bottleneck': 'VALU throughput / branch overhead'},
    '1': {'level': 'VGPR (register)', 'ai': 100.0,   # effectively VGPR-resident
           'bottleneck': 'VALU pipeline, register pressure, warp-shuffle'},
    '2': {'level': 'LDS', 'ai': 0.375, 'bottleneck': 'LDS bank conflicts, __syncthreads overhead'},
    '3': {'level': 'HBM', 'ai': 0.375, 'bottleneck': 'HBM random-access BW, L2 hit rate'},
}

# ────────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────────

def load_csv(path: str) -> List[dict]:
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows

def stats(vals: List[float]) -> Tuple[float, float, float, float, float]:
    """Returns (mean, std, cv%, min, max)"""
    if not vals: return (0.0, 0.0, 0.0, 0.0, 0.0)
    n = len(vals)
    m = sum(vals) / n
    var = sum((x - m)**2 for x in vals) / max(1, n - 1)
    sd = math.sqrt(var)
    cv = 100 * sd / m if m > 0 else 0.0
    return m, sd, cv, min(vals), max(vals)

def welch_t(a: List[float], b: List[float]) -> Tuple[float, bool]:
    """Welch's t-test. Returns (t-stat, significant at p<0.05)."""
    if len(a) < 2 or len(b) < 2: return (0.0, False)
    ma, mb = sum(a)/len(a), sum(b)/len(b)
    sa2 = sum((x-ma)**2 for x in a) / (len(a)-1)
    sb2 = sum((x-mb)**2 for x in b) / (len(b)-1)
    se = math.sqrt(sa2/len(a) + sb2/len(b))
    if se == 0: return (0.0, False)
    t = (mb - ma) / se
    # df approximation (Welch-Satterthwaite), threshold t > 2.09 for df≈38, p=0.05
    return t, abs(t) > 2.09

def fmt_delta(base: float, other: float) -> str:
    if base <= 0 or other <= 0: return 'N/A'
    return f'{(other - base)/base * 100:+.1f}%'

def fmt_speedup(base: float, other: float) -> str:
    if base <= 0 or other <= 0: return 'N/A'
    return f'{other/base:.2f}x'

# ────────────────────────────────────────────────────────────────
# Roofline bound classification
# ────────────────────────────────────────────────────────────────

def roofline_bound(tier: str, arch: str) -> str:
    spec = ARCH_SPECS.get(arch, ARCH_SPECS['gfx942'])
    info = TIER_ARITHMETIC_INTENSITY.get(tier, {'level': '?', 'ai': 0.375, 'bottleneck': '?'})
    ai = info['ai']
    ridge = spec['fp32_peak_flops'] / spec['hbm_bw_bps']  # FLOP/byte

    if tier in ('0', '1'):
        return f"VALU-bound (VGPR-resident; no HBM traffic)"
    elif ai >= ridge:
        return f"Compute-bound (AI={ai:.2f} > ridge={ridge:.1f} FLOP/byte)"
    else:
        mem_level = 'HBM' if tier == '3' else 'LDS'
        achievable = ai * (spec['hbm_bw_bps'] if tier == '3' else spec['lds_bw_bps'])
        return f"{mem_level}-BW-bound (AI={ai:.3f} < ridge={ridge:.1f}; achievable {achievable/1e12:.1f} TF)"

# ────────────────────────────────────────────────────────────────
# Main analysis
# ────────────────────────────────────────────────────────────────

def analyze(csv_paths: List[str], output: Optional[str] = None):
    # Load all CSVs, tagged by GPU arch
    all_rows = []
    for path in csv_paths:
        rows = load_csv(path)
        # Infer arch from path or from data
        for row in rows:
            row['_source'] = path
        all_rows.extend(rows)

    # Group by (label, mode, gpu_arch)
    by_label_mode_arch: Dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    meta: Dict = {}  # label → (tier, peak_rank, qubits)

    for row in all_rows:
        sps = float(row.get('shots_per_sec', 0) or 0)
        label = row.get('label', 'unknown')
        mode = row.get('mode', 'unknown')
        arch = row.get('gpu_arch', 'gfx942')
        tier = row.get('tier', '?')
        rank = row.get('peak_rank', '?')
        q = row.get('qubits', '?')
        meta[label] = (tier, rank, q)
        if sps > 0:
            by_label_mode_arch[label][mode][arch].append(sps)

    lines = []
    W = 130

    def section(title):
        lines.append('')
        lines.append('=' * W)
        lines.append(f'  {title}')
        lines.append('=' * W)

    def subsection(title):
        lines.append('')
        lines.append(f'  --- {title} ---')

    # ────────────────────────────────────────────
    # Section 1: Per-circuit table
    # ────────────────────────────────────────────
    section('3-WAY GPU KERNEL PERFORMANCE EVALUATION')
    lines.append(f'  Circuits: {len(meta)}  |  Modes: SVM / Compiled HIP Megakernel / MLIR→LLVM-IR')
    lines.append(f'  Metric: shots/sec (higher = better; passed_shots / sample_seconds)')
    lines.append(f'  Statistical significance: Welch\'s t-test p<0.05 (|t|>2.09 for df≈38)')

    all_archs = sorted({row.get('gpu_arch','gfx942') for row in all_rows})

    for arch in all_archs:
        spec = ARCH_SPECS.get(arch, ARCH_SPECS['gfx942'])
        section(f'RESULTS — {spec["name"]}')

        # Column header
        hdr = (f"{'Circuit':<24} {'T':>2} {'Rk':>3} {'Q':>4}  "
               f"{'SVM Mean':>12} {'CV':>5}  "
               f"{'Compiled':>12} {'CV':>5} {'vs SVM':>7} {'Sig':>4}  "
               f"{'MLIR':>12} {'CV':>5} {'vs SVM':>7} {'Sig':>4}")
        lines.append(hdr)
        lines.append('-' * W)

        sorted_labels = sorted(meta.keys(), key=lambda l: (meta[l][0], meta[l][2], l))

        for label in sorted_labels:
            tier, rank, q = meta[label]
            svm_vals  = by_label_mode_arch[label]['svm'][arch]
            comp_vals = by_label_mode_arch[label]['compiled'][arch]
            mlir_vals = by_label_mode_arch[label]['mlir'][arch]

            if not svm_vals and not comp_vals: continue

            sm, ss, scv, *_ = stats(svm_vals)
            cm, cs, ccv, *_ = stats(comp_vals)
            mm, ms, mcv, *_ = stats(mlir_vals)

            t_comp, sig_comp = welch_t(svm_vals, comp_vals)
            t_mlir, sig_mlir = welch_t(svm_vals, mlir_vals)

            line = (f"{label:<24} {tier:>2} {rank:>3} {q:>4}  "
                    f"{sm/1e6:>11.2f}M {scv:>4.1f}%  "
                    f"{cm/1e6:>11.2f}M {ccv:>4.1f}% {fmt_delta(sm,cm):>7} {'YES' if sig_comp else 'no':>4}  "
                    f"{mm/1e6:>11.2f}M {mcv:>4.1f}% {fmt_delta(sm,mm):>7} {'YES' if sig_mlir else 'no':>4}")
            lines.append(line)

        # ────────────────────────────────────────────
        # Section 2: Per-tier aggregates
        # ────────────────────────────────────────────
        subsection(f'Per-Tier Aggregate — {spec["name"]}')
        tier_speedups_comp: Dict[str, List[float]] = defaultdict(list)
        tier_speedups_mlir: Dict[str, List[float]] = defaultdict(list)
        tier_svm_tp:        Dict[str, List[float]] = defaultdict(list)

        for label in meta:
            tier = meta[label][0]
            svm_v  = by_label_mode_arch[label]['svm'][arch]
            comp_v = by_label_mode_arch[label]['compiled'][arch]
            mlir_v = by_label_mode_arch[label]['mlir'][arch]
            sm = sum(svm_v)/len(svm_v) if svm_v else 0
            cm = sum(comp_v)/len(comp_v) if comp_v else 0
            mm = sum(mlir_v)/len(mlir_v) if mlir_v else 0
            if sm > 0:
                tier_svm_tp[tier].append(sm)
                if cm > 0: tier_speedups_comp[tier].append(cm/sm)
                if mm > 0: tier_speedups_mlir[tier].append(mm/sm)

        tier_desc = {
            '0': 'Tier 0 — Frame-only (rank=0, no amplitude array)',
            '1': 'Tier 1 — Register (rank 1-4, VGPR-resident)',
            '2': 'Tier 2 — LDS/Coop (rank 5-10, __shared__ memory)',
            '3': 'Tier 3 — Global/HBM (rank 11-19, HBM)',
        }
        for t in ['0', '1', '2', '3']:
            if t not in tier_svm_tp: continue
            svm_avg = sum(tier_svm_tp[t]) / len(tier_svm_tp[t])
            sc = tier_speedups_comp.get(t, [])
            sm_l = tier_speedups_mlir.get(t, [])

            comp_str = (f"avg {sum(sc)/len(sc):.2f}x [min {min(sc):.2f}x max {max(sc):.2f}x]"
                        if sc else "no data")
            mlir_str = (f"avg {sum(sm_l)/len(sm_l):.2f}x [min {min(sm_l):.2f}x max {max(sm_l):.2f}x]"
                        if sm_l else "no data")

            lines.append(f"  {tier_desc.get(t, f'Tier {t}')}")
            lines.append(f"    SVM avg:          {svm_avg/1e6:.2f}M shots/s  (n={len(tier_svm_tp[t])} circuits)")
            lines.append(f"    Compiled vs SVM:  {comp_str}")
            lines.append(f"    MLIR vs SVM:      {mlir_str}")
            lines.append(f"    Roofline:         {roofline_bound(t, arch)}")
            lines.append(f"    Bottleneck:       {TIER_ARITHMETIC_INTENSITY.get(t,{}).get('bottleneck', '?')}")
            lines.append('')

    # ────────────────────────────────────────────
    # Section 3: Cross-hardware comparison
    # ────────────────────────────────────────────
    if len(all_archs) > 1:
        section('CROSS-HARDWARE COMPARISON: MI300X vs MI350X')
        lines.append(f"  {'Circuit':<24} {'Tier':>5}  {'MI300X SVM':>12}  {'MI350X SVM':>12} {'MI350X/MI300X':>14}  {'MI300X Comp':>12}  {'MI350X Comp':>12} {'Ratio':>6}")
        lines.append('-' * W)

        for label in sorted(meta.keys(), key=lambda l: (meta[l][0], l)):
            tier = meta[label][0]
            arch0_vals = by_label_mode_arch[label]['svm'].get('gfx942', [])
            arch1_vals = by_label_mode_arch[label]['svm'].get('gfx950', [])
            c0 = by_label_mode_arch[label]['compiled'].get('gfx942', [])
            c1 = by_label_mode_arch[label]['compiled'].get('gfx950', [])
            if not arch0_vals and not arch1_vals: continue
            m0 = sum(arch0_vals)/len(arch0_vals) if arch0_vals else 0
            m1 = sum(arch1_vals)/len(arch1_vals) if arch1_vals else 0
            mc0 = sum(c0)/len(c0) if c0 else 0
            mc1 = sum(c1)/len(c1) if c1 else 0
            lines.append(f"  {label:<24} {tier:>5}  {m0/1e6:>11.2f}M  {m1/1e6:>11.2f}M {fmt_speedup(m0,m1):>14}  {mc0/1e6:>11.2f}M  {mc1/1e6:>11.2f}M {fmt_speedup(mc0,mc1):>6}")

    # ────────────────────────────────────────────
    # Section 4: Roofline model summary
    # ────────────────────────────────────────────
    section('ROOFLINE MODEL ANALYSIS')
    for arch in all_archs:
        spec = ARCH_SPECS.get(arch, ARCH_SPECS['gfx942'])
        ridge = spec['fp32_peak_flops'] / spec['hbm_bw_bps']
        subsection(f"{spec['name']}")
        lines.append(f"  Peak FP32 FLOP/s:  {spec['fp32_peak_flops']/1e12:.0f} TF")
        lines.append(f"  Peak HBM BW:       {spec['hbm_bw_bps']/1e12:.1f} TB/s")
        lines.append(f"  Ridge point:       {ridge:.1f} FLOP/byte")
        lines.append(f"  Peak LDS BW:       {spec['lds_bw_bps']/1e12:.0f} TB/s (est.)")
        lines.append(f"  LDS ridge point:   {spec['fp32_peak_flops']/spec['lds_bw_bps']:.2f} FLOP/byte")
        lines.append('')
        lines.append(f"  H gate arithmetic intensity by tier:")
        lines.append(f"    Tier 1 (rank≤4):  VGPR-resident → effectively ∞ AI → VALU-bound")
        lines.append(f"    Tier 2 (rank≤10): AI≈0.375 FLOP/byte (LDS) << LDS ridge → VALU-bound")
        lines.append(f"                      Achievable: 0.375 × {spec['lds_bw_bps']/1e12:.0f} TB/s = {0.375*spec['lds_bw_bps']/1e12:.0f} TF")
        lines.append(f"    Tier 3 (rank≤19): AI≈0.375 FLOP/byte (HBM) << HBM ridge={ridge:.1f}")
        lines.append(f"                      Achievable: 0.375 × {spec['hbm_bw_bps']/1e12:.1f} TB/s = {0.375*spec['hbm_bw_bps']/1e12:.1f} TF")
        lines.append('')
        lines.append(f"  Roofline verdict:")
        lines.append(f"    Tiers 1-2: COMPUTE-BOUND (VALU-limited).  Target: increase VALU occupancy,")
        lines.append(f"               reduce register pressure, eliminate dispatch overhead.")
        lines.append(f"    Tier 3:    MEMORY-BANDWIDTH-BOUND (HBM).  Target: improve L2 hit rate,")
        lines.append(f"               coalesce butterfly accesses, prefetch with pipeline/double-buffer.")

    # ────────────────────────────────────────────
    # Section 5: Engineering verdict
    # ────────────────────────────────────────────
    section('PERFORMANCE ENGINEERING VERDICT')
    lines.append("""
  KERNEL VARIANT COMPARISON (acting as senior GPU perf engineer):

  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │  Variant        │ Tier 0        │ Tier 1        │ Tier 2       │ Tier 3        │
  │  (frame-only)   │ (register)    │ (LDS)         │ (HBM)        │               │
  ├─────────────────┼───────────────┼───────────────┼──────────────┼───────────────┤
  │  SVM Interpreter│ Baseline      │ Baseline      │ Baseline     │ Baseline      │
  │  + warp-shuffle │               │               │              │               │
  ├─────────────────┼───────────────┼───────────────┼──────────────┼───────────────┤
  │  Compiled HIP   │ ~Same or      │ +X%           │ +Y%          │ ~Same         │
  │  Megakernel     │ slight regress│ (no dispatch) │ (no LUT)     │ (BW-limited)  │
  ├─────────────────┼───────────────┼───────────────┼──────────────┼───────────────┤
  │  MLIR→LLVM-IR   │ N/A           │ +Z%           │ N/A          │ N/A           │
  │  (register only)│               │ (LLVM opt)    │              │               │
  └─────────────────┴───────────────┴───────────────┴──────────────┴───────────────┘

  Key findings (to be filled with actual measured deltas from bench.csv):

  1. TIER 0 (frame-only, rank=0):
     - All variants should produce similar throughput.
     - SVM interpreter dispatch loop has minimal overhead when no amplitude math.
     - Compiled megakernel overhead: constant-pool global reads may slightly penalize.
     - EXPECTED: Compiled ≈ SVM ± 5%.

  2. TIER 1 (register, rank 1-4, VGPR-resident):
     - Compiled HIP advantage: eliminates opcode dispatch (switch-case → straight-line).
     - Warp-shuffle reduction is shared by both → equal reduction overhead.
     - MLIR→LLVM-IR advantage over Compiled HIP:
       * LLVM CSE eliminates redundant scatter_bits calculations
       * Loop unrolling: rank≤4 → ≤8 iterations → fully unroll → no loop overhead
       * SCCP: constant-fold baked gate matrices → fewer VALU operations
     - EXPECTED: MLIR ≥ Compiled ≥ SVM; differential ~10-30% if LLVM optimization fires.

  3. TIER 2 (LDS/coop, rank 5-10):
     - SVM: interpreter dispatch inside cooperative kernel (one block = one shot)
     - Compiled coop: straight-line code, no function pointers, baked constants
     - KEY CONCERN: LDS bank conflicts.
       * GpuComplex = 8 bytes = 2 banks; butterfly stride accesses bank N and N^(1<<axis)
       * For single-qubit gates on axis k: half the LDS accesses are strided by 2^k
       * 2^k = 16 bytes = 4 banks → if not swizzled, 2-way bank conflicts on every access
       * MITIGATION: pad LDS array to N+1 elements to break conflict stride
       * Current code: does NOT use LDS padding — bank conflicts expected
     - EXPECTED: Compiled coop ≥ SVM coop due to no dispatch; LDS conflicts hurt both equally.

  4. TIER 3 (global/HBM, rank 11-19):
     - HBM-bandwidth-bound: both SVM and Compiled are constrained by 5.3 TB/s
     - Butterfly access pattern for rank-k gate: stride = 2^axis * 8B
       * For large rank (k≥15): stride >> L2 cache line → L2 miss dominated
       * L2 (16MB per XCD × 8 XCDs = 128MB total) can hold rank-14 state (2^14 × 8B = 128MB)
       * For rank ≥ 15: working set exceeds L2 → true HBM bandwidth bound
     - Per-XCD work stealing (implemented in both SVM and compiled global):
       * Prevents cross-XCD traffic for state vector accesses
       * Each XCD processes its own shots without remote HBM traffic
     - EXPECTED: Compiled ≈ SVM (both HBM-bound).
       * Double-buffer pipeline would help: prefetch next shot while computing current.

  OPTIMIZATION ROADMAP (priority order):

  Priority 1 — HIGH IMPACT (Tier 1, VGPR-bound):
    A. Verify MLIR optimization fires: check .ll output for unrolled loops
    B. If MLIR does not outperform Compiled HIP: add __launch_bounds__ to restrict
       register usage and increase occupancy
    C. Inline GEAK warp-shuffle: ensure it's not a separate function call in compiled path

  Priority 2 — MEDIUM IMPACT (Tier 2, LDS bank conflicts):
    A. Add LDS padding: change __shared__ GpuComplex v[N] to v[N + N/32]
       (one pad element per 32-element row eliminates 2-way conflicts for power-of-2 strides)
    B. Profile with rocprof SQ_LDS_BANK_CONFLICT to confirm conflict rate
    C. Consider vectorized LDS loads: pack re+im into uint2 for single 64-bit transaction

  Priority 3 — MEDIUM IMPACT (Tier 3, HBM prefetch):
    A. Implement double-buffer pipeline: two alternating global_v buffers
       prefetch next shot's state while computing current shot
    B. Use __builtin_amdgcn_s_sendmsg for async L2 prefetch hints
    C. Profile L2 hit rate per tier (TCC_HIT / (TCC_HIT + TCC_MISS))

  Priority 4 — LOW IMPACT (Tier 0, dispatch overhead):
    A. Profile whether opcode dispatch is the limiting factor for rank=0 circuits
    B. Consider: if peak_rank=0, skip amplitude array entirely (pure frame tracking)
""")

    # ────────────────────────────────────────────
    # Output
    # ────────────────────────────────────────────
    text = '\n'.join(lines)
    if output:
        with open(output, 'w') as f:
            f.write(text)
        print(f"Report written to: {output}")
    else:
        print(text)


def main():
    parser = argparse.ArgumentParser(description='Analyze 3-way GPU kernel benchmark results')
    parser.add_argument('csvfiles', nargs='+', help='Benchmark CSV files (bench.csv)')
    parser.add_argument('-o', '--output', help='Output report path')
    args = parser.parse_args()

    # Verify files exist
    missing = [f for f in args.csvfiles if not os.path.exists(f)]
    if missing:
        print(f"ERROR: Missing files: {missing}", file=sys.stderr)
        sys.exit(1)

    analyze(args.csvfiles, args.output)

if __name__ == '__main__':
    main()
