#!/usr/bin/env python3
"""
wilson_plot_and_residual.py

Truly unbinned Wilson plot + residual plot (side-by-side).

- Uses ALL eligible reflections directly (no binning) for fitting:
    x = s^2 = 1/(4 d^2)
    y = ln(Fo^2)   (only Fo^2 > 0)

- Ordinary Least Squares (OLS) linear fit.

Outputs a single PNG with:
  (left) unbinned scatter + fitted line
  (right) residuals vs s^2

Usage:
  python wilson_plot_and_residual.py --ins input.ins --hkl input.hkl --out out.png

Optional:
  --fit_lo_q 0.0 --fit_hi_q 1.0     # fit only this central quantile range in x (s^2)
  --max_points 40000                # only affects plotting (fit still uses all points)
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


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
            raw = line.strip()
            if not raw:
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
            is_term = (h == 0 and k == 0 and l == 0 and abs(fo2) < 1e-12 and abs(sig) < 1e-12)
            refls.append(Refl(h=h, k=k, l=l, fo2=fo2, sig=sig, is_terminator=is_term))
    return refls


def metric_tensor(cell: Cell) -> np.ndarray:
    a, b, c = cell.a, cell.b, cell.c
    ar, br, gr = math.radians(cell.alpha), math.radians(cell.beta), math.radians(cell.gamma)
    ca, cb, cg = math.cos(ar), math.cos(br), math.cos(gr)
    return np.array(
        [
            [a * a, a * b * cg, a * c * cb],
            [a * b * cg, b * b, b * c * ca],
            [a * c * cb, b * c * ca, c * c],
        ],
        dtype=float,
    )


def reciprocal_metric_tensor(cell: Cell) -> np.ndarray:
    return np.linalg.inv(metric_tensor(cell))


def s2_for_hkl(r: Refl, Gstar: np.ndarray) -> float:
    v = np.array([r.h, r.k, r.l], dtype=float)
    inv_d2 = float(v @ Gstar @ v)
    if inv_d2 <= 0.0:
        return float("nan")
    return inv_d2 / 4.0


def ols_line_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Ordinary least squares fit y ~ m*x + c."""
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0]), float(coef[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ins", required=True)
    ap.add_argument("--hkl", required=True)
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--fit_lo_q", type=float, default=0.0, help="Fit uses reflections between these x-quantiles")
    ap.add_argument("--fit_hi_q", type=float, default=1.0, help="Fit uses reflections between these x-quantiles")
    ap.add_argument("--max_points", type=int, default=100000, help="Plot at most this many points (random downsample)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not (0.0 <= args.fit_lo_q < args.fit_hi_q <= 1.0):
        raise ValueError("Require 0 <= fit_lo_q < fit_hi_q <= 1")

    cell = parse_cell_from_ins(args.ins)
    Gstar = reciprocal_metric_tensor(cell)
    refls = read_hkl(args.hkl)

    n = len(refls)
    s2 = np.full(n, np.nan, dtype=float)
    fo2 = np.full(n, np.nan, dtype=float)

    for i, r in enumerate(refls):
        if r.is_terminator:
            continue
        s2[i] = s2_for_hkl(r, Gstar)
        fo2[i] = r.fo2

    ok = np.isfinite(s2) & np.isfinite(fo2) & (fo2 > 0.0)
    x_all = s2[ok]
    y_all = np.log(np.clip(fo2[ok], 1e-300, None))

    if x_all.size < 50:
        raise RuntimeError(f"Too few eligible reflections for unbinned Wilson plot: {x_all.size}")

    # Fit range based on reflection-level x quantiles (still unbinned)
    lo = np.quantile(x_all, args.fit_lo_q)
    hi = np.quantile(x_all, args.fit_hi_q)
    fit_mask = (x_all >= lo) & (x_all <= hi)

    x_fit = x_all[fit_mask]
    y_fit = y_all[fit_mask]
    if x_fit.size < 50:
        # fall back to all points if the quantile slice is too narrow
        x_fit, y_fit = x_all, y_all
        lo, hi = float(np.min(x_all)), float(np.max(x_all))

    m, c = ols_line_fit(x_fit, y_fit)

    resid_all = y_all - (m * x_all + c)

    # Downsample for plotting only
    x_plot, y_plot, resid_plot = x_all, y_all, resid_all
    if x_plot.size > args.max_points:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(x_plot.size, size=args.max_points, replace=False)
        x_plot = x_plot[idx]
        y_plot = y_plot[idx]
        resid_plot = resid_plot[idx]

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))

    # Left: unbinned Wilson scatter + fit
    ax1.scatter(x_plot, y_plot, s=6, alpha=0.25)
    xline = np.linspace(float(np.min(x_plot)), float(np.max(x_plot)), 200)
    ax1.plot(xline, m * xline + c, linewidth=2)
    ax1.axvline(lo, linewidth=1)
    ax1.axvline(hi, linewidth=1)
    ax1.set_xlabel(r"$s^2 = (\sin\theta/\lambda)^2$ (1/$\AA^2$)")
    ax1.set_ylabel(r"$\ln(F_o^2)$")
    ax1.set_title("Unbinned Wilson plot (OLS fit)")

    # Right: residual vs s^2
    ax2.scatter(x_plot, resid_plot, s=6, alpha=0.25)
    ax2.axhline(0.0, linewidth=1)
    ax2.axvline(lo, linewidth=1)
    ax2.axvline(hi, linewidth=1)
    ax2.set_xlabel(r"$s^2 = (\sin\theta/\lambda)^2$ (1/$\AA^2$)")
    ax2.set_ylabel(r"$\ln(F_o^2) - (m s^2 + c)$")
    ax2.set_title("Residuals vs $s^2$")

    fig.suptitle(
        f"Unbinned OLS fit: m={m:.6g}, c={c:.6g}  |  fit x-quantiles: [{args.fit_lo_q}, {args.fit_hi_q}]",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"# Wrote: {args.out}")
    print(f"# Eligible reflections: {x_all.size}")
    print(f"# Fit used: {x_fit.size} (x-quantile range [{args.fit_lo_q}, {args.fit_hi_q}])")
    print(f"# Fit: m={m:.8g}, c={c:.8g}")
    print(f"# Points plotted: {x_plot.size} (max_points={args.max_points})")


if __name__ == "__main__":
    main()
# python wilson_plot_and_residual.py --ins /Users/xiaodong/Desktop/LTA1/shelx/t1_no-error-model.ins --hkl /Users/xiaodong/Desktop/LTA1/shelx/t1_no-error-model.hkl --out /Users/xiaodong/Desktop/LTA1/shelx/wilson_unbinned_residual.png
# python wilson_plot_and_residual.py --ins /Users/xiaodong/Desktop/SCXRD-DATA/SCXRDLTA/SCXRD/LTA.ins --hkl /Users/xiaodong/Desktop/SCXRD-DATA/SCXRDLTA/SCXRD/LTA.hkl --out /Users/xiaodong/Desktop/SCXRD-DATA/SCXRDLTA/SCXRD/wilson_unbinned_residual.png