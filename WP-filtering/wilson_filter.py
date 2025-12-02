#!/usr/bin/env python3
"""
wilson_filter_hkl.py

Iteratively filters reflections from a SHELXL HKLF4-style .hkl file by improving
Wilson-plot linearity (model-independent), AND writes Wilson plot PNGs.

Reads wavelength + cell parameters from an .ins file line starting with:
    CELL 0.01967 15.35 13.56 16.5 90 90 90

HKL format assumed (whitespace-separated):
    h k l Fo2 sigFo2 [optional extra cols...]
Terminator line (e.g. "0 0 0 0 0 0") is preserved.

Wilson:
- x = s^2 = (sin(theta)/lambda)^2 = 1/(4 d^2)
- y = ln(median(Fo2)) per equal-count bin (Fo2 > 0)

Filtering:
- score each reflection by normalized abs residual in log-space within its bin
- remove top fraction per iteration
- stop on patience/min improvement or min_keep_frac

Usage:
  python wilson_filter_hkl.py --ins input.ins --hkl input.hkl --out filtered.hkl --plot wilson.png

Optional:
  --plot_each_dir wilson_iter_plots    # save plot every iteration as wilson_iter_0001.png, ...
"""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# matplotlib is only imported when plotting is requested


@dataclass
class Cell:
    lam: float
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float


@dataclass
class Refl:
    h: int
    k: int
    l: int
    fo2: float
    sig: float
    extra: List[str]
    raw_line: str
    is_terminator: bool = False


def parse_cell_from_ins(ins_path: str) -> Cell:
    with open(ins_path, "r") as f:
        for line in f:
            s = line.strip()
            if s.upper().startswith("CELL"):
                parts = s.split()
                if len(parts) < 8:
                    raise ValueError(f"Found CELL line but expected 7 numbers: {s}")
                lam = float(parts[1])
                a, b, c = float(parts[2]), float(parts[3]), float(parts[4])
                alpha, beta, gamma = float(parts[5]), float(parts[6]), float(parts[7])
                return Cell(lam=lam, a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
    raise ValueError(f"No CELL line found in {ins_path}")


def read_hkl(hkl_path: str) -> List[Refl]:
    refls: List[Refl] = []
    with open(hkl_path, "r") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            parts = raw.split()
            if len(parts) < 5:
                continue
            try:
                h, k, l = int(parts[0]), int(parts[1]), int(parts[2])
                fo2 = float(parts[3])
                sig = float(parts[4])
            except Exception:
                continue
            extra = parts[5:]
            is_term = (h == 0 and k == 0 and l == 0 and abs(fo2) < 1e-12 and abs(sig) < 1e-12)
            refls.append(Refl(h=h, k=k, l=l, fo2=fo2, sig=sig, extra=extra, raw_line=raw, is_terminator=is_term))
    return refls


def metric_tensor(cell: Cell) -> np.ndarray:
    a, b, c = cell.a, cell.b, cell.c
    ar, br, gr = math.radians(cell.alpha), math.radians(cell.beta), math.radians(cell.gamma)
    ca, cb, cg = math.cos(ar), math.cos(br), math.cos(gr)
    return np.array([
        [a*a, a*b*cg, a*c*cb],
        [a*b*cg, b*b, b*c*ca],
        [a*c*cb, b*c*ca, c*c]
    ], dtype=float)


def reciprocal_metric_tensor(cell: Cell) -> np.ndarray:
    return np.linalg.inv(metric_tensor(cell))


def s2_for_hkl(r: Refl, Gstar: np.ndarray) -> float:
    v = np.array([r.h, r.k, r.l], dtype=float)
    inv_d2 = float(v @ Gstar @ v)
    if inv_d2 <= 0.0:
        return float("nan")
    return inv_d2 / 4.0


def robust_line_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    m, b = float(coef[0]), float(coef[1])

    r = y - (m * x + b)
    mad = np.median(np.abs(r - np.median(r))) + 1e-12
    c = 1.345 * mad
    w = np.ones_like(r)
    mask = np.abs(r) > c
    w[mask] = c / (np.abs(r[mask]) + 1e-12)

    W = np.diag(w)
    coef2, *_ = np.linalg.lstsq(W @ A, W @ y, rcond=None)
    return float(coef2[0]), float(coef2[1])


def make_equal_count_bins(values: np.ndarray, nbins: int) -> np.ndarray:
    qs = np.linspace(0, 1, nbins + 1)
    edges = np.quantile(values, qs)
    eps = 1e-12
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + eps
    return edges


def wilson_binned_points(
    refls: List[Refl],
    keep: np.ndarray,
    Gstar: np.ndarray,
    nbins: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      x_bins: median s^2 per bin
      y_bins: ln(median Fo2) per bin
      y_mad:  MAD of ln(Fo2) within bin
      idx_elig: indices of eligible reflections (kept, Fo2>0, finite s^2)
    """
    n = len(refls)
    s2 = np.full(n, np.nan, dtype=float)
    fo2 = np.full(n, np.nan, dtype=float)
    eligible = np.zeros(n, dtype=bool)

    for i, r in enumerate(refls):
        if r.is_terminator or not keep[i]:
            continue
        s2_i = s2_for_hkl(r, Gstar)
        s2[i] = s2_i
        fo2[i] = r.fo2
        if np.isfinite(s2_i) and (r.fo2 > 0.0):
            eligible[i] = True

    idx = np.where(eligible)[0]
    if idx.size < max(50, nbins * 10):
        raise RuntimeError(f"Not enough eligible reflections for Wilson analysis: {idx.size}")

    s2_elig = s2[idx]
    fo2_elig = fo2[idx]

    edges = make_equal_count_bins(s2_elig, nbins)
    bin_id = np.digitize(s2_elig, edges[1:-1], right=False)  # 0..nbins-1

    x_bins = np.full(nbins, np.nan, dtype=float)
    y_bins = np.full(nbins, np.nan, dtype=float)
    y_mad = np.full(nbins, np.nan, dtype=float)

    for b in range(nbins):
        inb = (bin_id == b)
        if not np.any(inb):
            continue
        s2_b = s2_elig[inb]
        fo2_b = fo2_elig[inb]
        med_fo2 = np.median(fo2_b)
        x_bins[b] = float(np.median(s2_b))
        y_bins[b] = math.log(max(float(med_fo2), 1e-300))

        logs = np.log(np.clip(fo2_b, 1e-300, None))
        mad = np.median(np.abs(logs - np.median(logs))) + 1e-12
        y_mad[b] = float(mad)

    return x_bins, y_bins, y_mad, idx


def wilson_fit_and_score(
    x_bins: np.ndarray,
    y_bins: np.ndarray,
    fit_lo_q: float,
    fit_hi_q: float,
) -> Tuple[float, float, float]:
    ok = np.isfinite(x_bins) & np.isfinite(y_bins)
    xb = x_bins[ok]
    yb = y_bins[ok]
    if xb.size < 10:
        raise RuntimeError("Too few bins for Wilson fit.")

    lo = np.quantile(xb, fit_lo_q)
    hi = np.quantile(xb, fit_hi_q)
    fit_mask = (xb >= lo) & (xb <= hi)
    xb_fit = xb[fit_mask]
    yb_fit = yb[fit_mask]
    if xb_fit.size < 10:
        xb_fit, yb_fit = xb, yb

    m, c = robust_line_fit(xb_fit, yb_fit)
    resid = yb_fit - (m * xb_fit + c)
    score = float(np.sqrt(np.mean(resid * resid)))
    return m, c, score


def wilson_outlier_scores(
    refls: List[Refl],
    keep: np.ndarray,
    Gstar: np.ndarray,
    nbins: int,
    fit_lo_q: float,
    fit_hi_q: float,
    seed: int,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Returns:
      score: Wilson fit RMSE on fit region bins
      z: per-reflection outlier score (NaN if not eligible)
      s2: per-reflection s^2
      yhat: per-reflection predicted ln(I)
      x_bins, y_bins: binned points for plotting
      m, c: line fit
    """
    rng = random.Random(seed)

    x_bins, y_bins, y_mad, idx = wilson_binned_points(refls, keep, Gstar, nbins)
    m, c, score = wilson_fit_and_score(x_bins, y_bins, fit_lo_q, fit_hi_q)

    n = len(refls)
    z = np.full(n, np.nan, dtype=float)
    s2 = np.full(n, np.nan, dtype=float)
    yhat = np.full(n, np.nan, dtype=float)

    # Build bin assignment for eligible idx
    s2_elig = []
    fo2_elig = []
    for i in idx:
        s2_elig.append(s2_for_hkl(refls[i], Gstar))
        fo2_elig.append(refls[i].fo2)
    s2_elig = np.array(s2_elig, dtype=float)

    edges = make_equal_count_bins(s2_elig, nbins)
    bin_id = np.digitize(s2_elig, edges[1:-1], right=False)

    for j, i in enumerate(idx):
        r = refls[i]
        s2_i = float(s2_for_hkl(r, Gstar))
        s2[i] = s2_i
        pred = m * s2_i + c
        yhat[i] = pred
        b = int(bin_id[j])
        mad = float(y_mad[b]) if np.isfinite(y_mad[b]) else 1.0
        logi = math.log(max(r.fo2, 1e-300))
        z_i = abs(logi - pred) / (mad + 1e-12)
        z[i] = z_i + (rng.random() * 1e-9)  # deterministic tie-break

    return score, z, s2, yhat, x_bins, y_bins, m, c


def plot_wilson(x_bins, y_bins, m, c, out_png: str, title: str) -> None:
    import matplotlib.pyplot as plt

    ok = np.isfinite(x_bins) & np.isfinite(y_bins)
    xb = x_bins[ok]
    yb = y_bins[ok]
    if xb.size == 0:
        return

    xline = np.linspace(float(np.min(xb)), float(np.max(xb)), 200)
    yline = m * xline + c

    plt.figure()
    plt.scatter(xb, yb, s=18)
    plt.plot(xline, yline, linewidth=2)
    plt.xlabel(r"$s^2 = (\sin\theta/\lambda)^2$ (1/$\AA^2$)")
    plt.ylabel(r"$\ln(\mathrm{median}(F_o^2))$ per bin")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def write_hkl(refls: List[Refl], keep: np.ndarray, out_path: str) -> None:
    with open(out_path, "w") as f:
        for i, r in enumerate(refls):
            if r.is_terminator:
                f.write(r.raw_line + "\n")
                continue
            if keep[i]:
                base = f"{r.h:4d}{r.k:4d}{r.l:4d}{r.fo2:12.2f}{r.sig:12.2f}"
                if r.extra:
                    base += " " + " ".join(r.extra)
                f.write(base + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ins", required=True)
    ap.add_argument("--hkl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plot", default=None, help="Write final Wilson plot PNG to this path")
    ap.add_argument("--plot_each_dir", default=None, help="If set, save Wilson plot each iteration into this directory")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--remove_frac", type=float, default=0.002)
    ap.add_argument("--min_keep_frac", type=float, default=0.85)
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--fit_lo_q", type=float, default=0.05)
    ap.add_argument("--fit_hi_q", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--min_improve", type=float, default=1e-4)
    args = ap.parse_args()

    cell = parse_cell_from_ins(args.ins)
    Gstar = reciprocal_metric_tensor(cell)
    refls = read_hkl(args.hkl)

    nonterm = np.array([not r.is_terminator for r in refls], dtype=bool)
    n_total = int(np.sum(nonterm))
    if n_total == 0:
        raise RuntimeError("No reflections read from HKL.")

    keep = np.ones(len(refls), dtype=bool)
    keep[~nonterm] = True  # keep terminator lines always

    init_kept = int(np.sum(keep & nonterm))

    print(f"# Parsed CELL: lambda={cell.lam} Å, a={cell.a}, b={cell.b}, c={cell.c}, "
          f"alpha={cell.alpha}, beta={cell.beta}, gamma={cell.gamma}")
    print(f"# Starting reflections (non-terminator): {n_total}")
    print(f"# Settings: bins={args.bins}, remove_frac={args.remove_frac}, min_keep_frac={args.min_keep_frac}, "
          f"iters={args.iters}, fit_lo_q={args.fit_lo_q}, fit_hi_q={args.fit_hi_q}")
    print("# Columns: iter  kept  removed_this_iter  wilson_score")

    if args.plot_each_dir:
        os.makedirs(args.plot_each_dir, exist_ok=True)

    best_score = float("inf")
    no_improve = 0

    for it in range(1, args.iters + 1):
        score, z, s2, yhat, x_bins, y_bins, m, c = wilson_outlier_scores(
            refls=refls,
            keep=keep,
            Gstar=Gstar,
            nbins=args.bins,
            fit_lo_q=args.fit_lo_q,
            fit_hi_q=args.fit_hi_q,
            seed=args.seed + it,
        )

        eligible = np.isfinite(z) & keep & nonterm
        eligible_idx = np.where(eligible)[0]
        n_elig = int(eligible_idx.size)
        if n_elig == 0:
            print(f"{it:4d}  {int(np.sum(keep & nonterm)):6d}  {0:16d}  {score:.6f}  # no eligible reflections")
            break

        n_remove = max(1, int(math.floor(args.remove_frac * n_elig)))

        order = np.argsort(z[eligible_idx])[::-1]
        to_remove = eligible_idx[order[:n_remove]]
        keep[to_remove] = False

        kept_now = int(np.sum(keep & nonterm))
        print(f"{it:4d}  {kept_now:6d}  {n_remove:16d}  {score:.6f}")

        if args.plot_each_dir:
            out_png = os.path.join(args.plot_each_dir, f"wilson_iter_{it:04d}.png")
            plot_wilson(x_bins, y_bins, m, c, out_png, title=f"Wilson plot (iter {it}, score={score:.4g})")

        frac_kept = kept_now / max(1, init_kept)
        if frac_kept < args.min_keep_frac:
            print(f"# Stop: kept fraction {frac_kept:.3f} < min_keep_frac {args.min_keep_frac}")
            break

        if best_score - score >= args.min_improve:
            best_score = score
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"# Stop: no significant improvement for {args.patience} iterations "
                      f"(min_improve={args.min_improve})")
                break

    # Final plot on final kept set
    if args.plot:
        score, z, s2, yhat, x_bins, y_bins, m, c = wilson_outlier_scores(
            refls=refls,
            keep=keep,
            Gstar=Gstar,
            nbins=args.bins,
            fit_lo_q=args.fit_lo_q,
            fit_hi_q=args.fit_hi_q,
            seed=args.seed + 9999,
        )
        plot_wilson(x_bins, y_bins, m, c, args.plot, title=f"Wilson plot (final, score={score:.4g})")

    write_hkl(refls, keep, args.out)
    kept_final = int(np.sum(keep & nonterm))
    print(f"# Wrote: {args.out}")
    print(f"# Final kept reflections (non-terminator): {kept_final} / {n_total}  ({kept_final/n_total:.3f})")
    if args.plot:
        print(f"# Wrote: {args.plot}")
    if args.plot_each_dir:
        print(f"# Wrote per-iteration plots to: {args.plot_each_dir}")


if __name__ == "__main__":
    main()

# python wilson_filter.py --ins /home/bubl3932/files/FeAcAc/FeAcAc_11.ins --hkl /home/bubl3932/files/FeAcAc/FeAcAc_11.hkl --out /home/bubl3932/files/FeAcAc/FeAcAc_11_wp_filtered.hkl --plot /home/bubl3932/files/FeAcAc/FeAcAc_11_wp_filtered.png --plot_each_dir /home/bubl3932/files/FeAcAc/wilson_iters --remove_frac 0.001 --patience 8
