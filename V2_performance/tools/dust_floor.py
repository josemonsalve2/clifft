#!/usr/bin/env python3
"""Where fp32 rounding dust lands: the sizing measurement behind V2_DUST_EPS.

Background. V2_DUST_EPS was copied from the SVM's kDustEpsilon = 1e-18, which is
sized for std::complex<double>. V2 stores CV2Complex = {float re; float im;}, so
analytically-zero interference lands several decades HIGHER and the threshold
could never fire. That is a correctness bug, not dead code: sample_branch()
returns WITHOUT drawing a PRNG value when it clamps, so a branch clamped on one
side and rolled on the other shifts every subsequent draw in that shot and the
two streams never resynchronize. See the V2_DUST_EPS comment in
src/clifft/gpu/mlir/v2/v2_ops.h and section 12 of the report.

This script is the generator for the sizing table in that comment, which was
previously recorded only in prose. It implements BOTH estimators, because they
disagree on the absolute floor by ~1.5 decades while agreeing on everything the
threshold choice actually rests on.

  --model residual (default, reproduces the committed table)
      A statistical model. Each analytically-zero output is assumed to carry a
      relative error of scale eps: resid_i = a_i * eps * z_i with z_i a
      unit-variance complex Gaussian. Then

          p1/total = eps^2 * sum(|a_i|^2 |z_i|^2) / sum(|a_i|^2)

      i.e. eps^2 times a weighted mean of Exp(1) variables. This is an upper
      envelope: it assumes every term rounds at full eps and that the errors
      never cancel against each other.

  --model butterfly
      Direct simulation of the arithmetic the kernel performs:
      out1_i = fl(fl(u*a_i) + fl(v*b_i)) over fp32-stored amplitudes whose exact
      fp64 counterparts cancel identically. No assumed error scale.

Both give the same three structural facts, which are what V2_DUST_EPS is chosen
on:

  1. The floor is RANK-INDEPENDENT. The rounding error is relative to each
     amplitude, so summing more terms averages the residuals rather than
     accumulating them. One constant covers rank 1..26.
  2. The spread TIGHTENS with rank, so the widest tail is at the SMALLEST
     circuits -- a threshold sized on the largest would be too tight.
  3. The floor is insensitive to circuit depth: accumulated error from a long
     gate sequence is itself relative and divides out of the ratio. Verified
     with --depth up to 256 gates (medians move <5%).

They differ on where the floor sits: the residual model concentrates on
eps^2 = 1.42e-14 with a 1.2e-13 low-rank tail, while direct butterfly simulation
puts it ~36x lower, near 4e-16 with a ~5e-15 tail. The residual model is the
conservative one and is what V2_DUST_EPS = 1e-11 was sized against, leaving
~2 decades of margin above its tail (and ~4 above the simulated one). Both sit
~5 decades below the smallest probability fp32 storage can carry meaningfully,
so the choice is not sensitive to which estimator is preferred.

No GPU is required: this is pure fp32 arithmetic.

Usage:  dust_floor.py [--model residual|butterfly] [--trials N] [--cap N]
                      [--depth N]
"""
import argparse
import numpy as np

FP32_EPS = float(np.finfo(np.float32).eps)          # 1.1920929e-07
F32, C64, C128 = np.float32, np.complex64, np.complex128


def residual_trial(half, rng):
    """Statistical model: full-eps relative error on every term, no cancellation."""
    a = rng.standard_normal(half) ** 2 + rng.standard_normal(half) ** 2   # |a_i|^2
    z = (rng.standard_normal(half) ** 2 + rng.standard_normal(half) ** 2) / 2  # |z_i|^2
    return FP32_EPS ** 2 * (a * z).sum() / a.sum()


def butterfly_trial(half, rng, depth):
    """Direct simulation of fl(fl(u*a) + fl(v*b)) on fp32-stored amplitudes."""
    a64 = rng.standard_normal(half) + 1j * rng.standard_normal(half)
    a64 /= np.linalg.norm(a64)
    b64 = rng.standard_normal(half) + 1j * rng.standard_normal(half)
    b64 /= np.linalg.norm(b64)
    a32, b32 = a64.astype(C64), b64.astype(C64)

    # Optional: age the state through `depth` butterflies so it carries realistic
    # accumulated error, tracking the exact fp64 state alongside.
    for _ in range(depth):
        th = rng.uniform(0.1, np.pi / 2 - 0.1)
        u, v = F32(np.cos(th)), F32(np.sin(th))
        a32, b32 = ((u * a32).astype(C64) + (v * b32).astype(C64),
                    (v * a32).astype(C64) - (u * b32).astype(C64))
        a64, b64 = (np.float64(u) * a64 + np.float64(v) * b64,
                    np.float64(v) * a64 - np.float64(u) * b64)

    # Final butterfly, with the partner amplitude set so branch 1 cancels
    # identically in the EXACT state. The fp32 pair keeps whatever relative
    # error it accumulated; the residual is what survives that cancellation.
    th = rng.uniform(0.1, np.pi / 2 - 0.1)
    u, v = F32(np.cos(th)), F32(np.sin(th))
    rel = np.divide(a32.astype(C128) - a64, a64,
                    out=np.zeros(half, dtype=C128), where=np.abs(a64) > 0)
    b32c = ((-(np.float64(u) / np.float64(v)) * a64) * (1 + rel)).astype(C64)

    out1 = (u * a32).astype(C64) + (v * b32c).astype(C64)
    out0 = (v * a32).astype(C64) - (u * b32c).astype(C64)
    # V2 reduces probabilities in f64 from f32 inputs (report section 12.5), so
    # the reduction contributes nothing -- only the stored values do.
    p1 = np.sum(np.abs(out1.astype(C128)) ** 2)
    p0 = np.sum(np.abs(out0.astype(C128)) ** 2)
    return p1 / (p1 + p0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("residual", "butterfly"), default="residual")
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--cap", type=int, default=1 << 14,
                    help="cap on terms per trial. The estimator concentrates as "
                         "1/sqrt(half), so beyond this the distribution is "
                         "already tighter than the reported precision.")
    ap.add_argument("--depth", type=int, default=0,
                    help="butterfly model only: age the state through N gates "
                         "before measuring, to carry accumulated error.")
    ap.add_argument("--ranks", type=int, nargs="+", default=[1, 4, 12, 26])
    a = ap.parse_args()

    print(f"fp32 eps = {FP32_EPS:.7e}   eps^2 = {FP32_EPS ** 2:.3e}")
    print(f"model={a.model}  trials={a.trials}  cap={a.cap}"
          + (f"  depth={a.depth}" if a.model == "butterfly" else "") + "\n")
    print(f"{'rank':>5} {'half':>10} {'used':>7} {'median':>11} {'max':>11} "
          f"{'med/eps^2':>10}")
    for k in a.ranks:
        half = 1 << (k - 1)
        used = min(half, a.cap)
        rng = np.random.default_rng(1234 + k)       # fixed: reproducible
        if a.model == "residual":
            vals = np.array([residual_trial(used, rng) for _ in range(a.trials)])
        else:
            vals = np.array([butterfly_trial(used, rng, a.depth)
                             for _ in range(a.trials)])
        med = np.median(vals)
        flag = "" if used == half else " *"
        print(f"{k:5d} {half:10d} {used:7d} {med:11.3e} {vals.max():11.3e} "
              f"{med / FP32_EPS ** 2:10.3f}{flag}")

    print("\n* half capped at --cap; the estimator has already concentrated.")
    print("Maxima are order statistics: they grow with --trials and are only "
          "comparable\nacross rows at equal --trials.")


if __name__ == "__main__":
    main()
