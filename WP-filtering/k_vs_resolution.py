#!/usr/bin/env python3
import argparse, math, re
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
class INSInfo:
    cell: Cell
    fvar: float


@dataclass
class HKLObs:
    h: int
    k: int
    l: int
    fo2: float
    sig: float
    extra: List[str]
    is_terminator: bool = False


@dataclass
class FCFRefl:
    h: int
    k: int
    l: int
    fo2: float
    sig: float
    fc: float
    phi: float


def parse_ins(ins_path: str) -> INSInfo:
    cell = None
    fvar = None
    with open(ins_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            u = s.upper()
            if u.startswith("CELL"):
                parts = s.split()
                if len(parts) < 8:
                    raise ValueError(f"Bad CELL line: {s}")
                cell = Cell(
                    lam=float(parts[1]),
                    a=float(parts[2]), b=float(parts[3]), c=float(parts[4]),
                    alpha=float(parts[5]), beta=float(parts[6]), gamma=float(parts[7]),
                )
            elif u.startswith("FVAR"):
                parts = s.split()
                if len(parts) >= 2:
                    fvar = float(parts[1])
    if cell is None:
        raise ValueError("No CELL line found in .ins")
    if fvar is None:
        raise ValueError("No FVAR line found in .ins")
    return INSInfo(cell=cell, fvar=fvar)


def read_hkl(hkl_path: str) -> List[HKLObs]:
    out = []
    with open(hkl_path, "r") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            parts = raw.split()
            if len(parts) < 5:
                continue
            h, k, l = int(parts[0]), int(parts[1]), int(parts[2])
            fo2 = float(parts[3])
            sig = float(parts[4])
            extra = parts[5:]
            is_term = (h == 0 and k == 0 and l == 0 and abs(fo2) < 1e-12 and abs(sig) < 1e-12)
            out.append(HKLObs(h, k, l, fo2, sig, extra, is_term))
    return out


def read_symops_from_fcf(fcf_path: str) -> List[str]:
    lines = open(fcf_path, "r").read().splitlines()
    ops = []
    for i, line in enumerate(lines):
        if line.strip() == "_space_group_symop_operation_xyz":
            j = i + 1
            while j < len(lines):
                s = lines[j].strip()
                if not s:
                    j += 1
                    continue
                if s.startswith("loop_") or s.startswith("_"):
                    break
                ops.append(s.strip().strip("'").strip('"'))
                j += 1
            break
    if not ops:
        raise ValueError("No symop loop found in .fcf")
    return ops


def read_fcf_reflections(fcf_path: str) -> List[FCFRefl]:
    lines = open(fcf_path, "r").read().splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip() == "_refln_index_h":
            start = i
            break
    if start is None:
        raise ValueError("Could not find reflection header in .fcf")

    data_start = start + 7
    out = []
    for j in range(data_start, len(lines)):
        s = lines[j].strip()
        if not s or s.startswith("_") or s.startswith("loop_"):
            break
        parts = s.split()
        if len(parts) < 7:
            break
        try:
            h, k, l = int(parts[0]), int(parts[1]), int(parts[2])
            fo2 = float(parts[3])
            sig = float(parts[4])
            fc = float(parts[5])
            phi = float(parts[6])
        except Exception:
            break
        out.append(FCFRefl(h, k, l, fo2, sig, fc, phi))
    if not out:
        raise ValueError("No reflections parsed from .fcf")
    return out


# For Pbca (and your case), ops are sign flips only. We parse ±x, ±y, ±z.
def _parse_term(expr: str, axis: str) -> int:
    e = expr.replace(" ", "")
    m = re.match(rf"^([+-]?){axis}([+-].+)?$", e)
    if not m:
        raise ValueError(f"Cannot parse symop term '{expr}' for axis '{axis}'")
    return -1 if m.group(1) == "-" else 1


def parse_symop_signs(op: str) -> Tuple[int, int, int]:
    parts = [p.strip().strip("'").strip('"') for p in op.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Bad symop: {op}")
    sx = _parse_term(parts[0], "x")
    sy = _parse_term(parts[1], "y")
    sz = _parse_term(parts[2], "z")
    return sx, sy, sz


def reciprocal_metric_tensor(cell: Cell) -> np.ndarray:
    a, b, c = cell.a, cell.b, cell.c
    ar, br, gr = map(math.radians, (cell.alpha, cell.beta, cell.gamma))
    ca, cb, cg = math.cos(ar), math.cos(br), math.cos(gr)
    G = np.array(
        [
            [a * a, a * b * cg, a * c * cb],
            [a * b * cg, b * b, b * c * ca],
            [a * c * cb, b * c * ca, c * c],
        ],
        dtype=float,
    )
    return np.linalg.inv(G)


def inv_d2(h: int, k: int, l: int, Gstar: np.ndarray) -> float:
    v = np.array([h, k, l], dtype=float)
    return float(v @ Gstar @ v)


def d_from_invd2(inv_d2_val: float) -> float:
    return 1.0 / math.sqrt(inv_d2_val) if inv_d2_val > 0 else float("nan")


def s2_from_invd2(inv_d2_val: float) -> float:
    return inv_d2_val / 4.0


def s2_to_d(x):
    x = np.array(x, dtype=float)
    return 1.0 / (2.0 * np.sqrt(np.clip(x, 1e-300, None)))


def d_to_s2(dv):
    dv = np.array(dv, dtype=float)
    return 1.0 / (4.0 * np.clip(dv, 1e-300, None) ** 2)


def build_fcf_lookup_expanded(
    fcf: List[FCFRefl], symops: List[str]
) -> Dict[Tuple[int, int, int], Tuple[float, float]]:
    """
    Map (h,k,l) -> (Fo2_fcf, Fc2) expanded to signed equivalents.
    """
    signs = [parse_symop_signs(op) for op in symops]
    lut: Dict[Tuple[int, int, int], Tuple[float, float]] = {}
    for r in fcf:
        fc2 = r.fc * r.fc
        for sx, sy, sz in signs:
            key = (sx * r.h, sy * r.k, sz * r.l)
            if key not in lut:
                lut[key] = (r.fo2, fc2)
    return lut


def add_resolution_top_axis(ax):
    secax = ax.secondary_xaxis("top", functions=(s2_to_d, d_to_s2))
    secax.set_xlabel(r"Resolution $d$ ($\AA$)")
    # Optional: match your SHELXL shell boundaries as ticks
    shells = np.array([0.85, 0.88, 0.92, 0.96, 1.01, 1.07, 1.16, 1.27, 1.45, 1.83])
    xmin, xmax = ax.get_xlim()
    s2_shell = d_to_s2(shells)
    keep = (s2_shell >= xmin) & (s2_shell <= xmax)
    secax.set_xticks(shells[keep])
    return secax


def binned_summary(x, y, bins=60, stat="median", qlo=0.16, qhi=0.84):
    """
    Bin y as a function of x using equal-width bins in x.
    Returns: centers, y_stat, y_lo, y_hi (arrays with NaNs for empty bins)
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size == 0:
        return (np.array([]),) * 4

    edges = np.linspace(np.min(x), np.max(x), bins + 1)
    idx = np.digitize(x, edges) - 1
    idx = np.clip(idx, 0, bins - 1)

    centers = 0.5 * (edges[:-1] + edges[1:])
    y_stat = np.full(bins, np.nan)
    y_lo = np.full(bins, np.nan)
    y_hi = np.full(bins, np.nan)

    for b in range(bins):
        sel = (idx == b)
        if not np.any(sel):
            continue
        yy = y[sel]
        if stat == "mean":
            y_stat[b] = float(np.mean(yy))
        else:
            y_stat[b] = float(np.median(yy))
        y_lo[b] = float(np.quantile(yy, qlo))
        y_hi[b] = float(np.quantile(yy, qhi))

    return centers, y_stat, y_lo, y_hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ins", required=True)
    ap.add_argument("--hkl", required=True)
    ap.add_argument("--fcf", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--min_fc2", type=float, default=1e-4, help="Skip if Fc^2 too small (ratio blowups)")

    ap.add_argument("--bins", type=int, default=10, help="Number of x-bins for binned overlays")
    ap.add_argument("--bin_stat", choices=["median", "mean"], default="median", help="Center statistic per bin")
    ap.add_argument("--qlo", type=float, default=0, help="Lower quantile for bin band")
    ap.add_argument("--qhi", type=float, default=1, help="Upper quantile for bin band")
    ap.add_argument("--no_scatter", action="store_true", help="Do not plot scatter, only binned summary")

    ap.add_argument("--k_ylim_nsigma", type=float, default=0.0,
                    help="For K plot: y-limits = mean(K) ± nsigma*std(K). Use <=0 to disable.")
    args = ap.parse_args()

    out_png = args.out_png
    out_png_K = re.sub(r"\.png$", "_K.png", out_png)
    out_png_lnK = re.sub(r"\.png$", "_lnK.png", out_png)

    ins = parse_ins(args.ins)
    cell = ins.cell
    k_scale = ins.fvar
    Gstar = reciprocal_metric_tensor(cell)

    hkl = read_hkl(args.hkl)
    fcf_refls = read_fcf_reflections(args.fcf)
    symops = read_symops_from_fcf(args.fcf)
    lut = build_fcf_lookup_expanded(fcf_refls, symops)

    rows = []
    n_term = n_nomatch = n_smallfc = 0

    for r in hkl:
        if r.is_terminator:
            n_term += 1
            continue

        key = (r.h, r.k, r.l)
        v = lut.get(key, None)
        if v is None:
            n_nomatch += 1
            continue

        fo2_fcf, fc2 = v
        if fo2_fcf <= 0.0:
            continue
        if fc2 <= args.min_fc2:
            n_smallfc += 1
            continue

        invd2 = inv_d2(r.h, r.k, r.l, Gstar)
        d = d_from_invd2(invd2)
        s2 = s2_from_invd2(invd2)

        # SHELXL variance analysis convention: K = Mean[Fo^2] / Mean[Fc^2]
        K = fo2_fcf / fc2

        rows.append((r.h, r.k, r.l, r.fo2, r.sig, fo2_fcf, fc2, k_scale, K, math.log(K), d, s2))

    if not rows:
        raise RuntimeError("No reflections matched. Check files and symops parsing.")

    with open(args.out_csv, "w") as f:
        f.write("h,k,l,Fo2_hkl,sigFo2_hkl,Fo2_fcf,Fc2,FVAR_scale,K,lnK,d_A,s2\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")

    arr = np.array(rows, dtype=float)
    s2 = arr[:, 11]
    K = arr[:, 8]
    lnK = arr[:, 9]

    # Precompute binned overlays
    cxK, cK, Klo, Khi = binned_summary(s2, K, bins=args.bins, stat=args.bin_stat, qlo=args.qlo, qhi=args.qhi)
    cxL, cL, Llo, Lhi = binned_summary(s2, lnK, bins=args.bins, stat=args.bin_stat, qlo=args.qlo, qhi=args.qhi)

    # -------- K vs s^2 --------
    plt.figure()
    ax = plt.gca()

    if not args.no_scatter:
        ax.scatter(s2, K, s=6, alpha=0.25)

    if cxK.size > 0:
        m = np.isfinite(cK) & np.isfinite(Klo) & np.isfinite(Khi)
        ax.plot(cxK[m], cK[m], linewidth=1.5)
        ax.fill_between(cxK[m], Klo[m], Khi[m], alpha=0.2, linewidth=0)

    ax.set_xlabel(r"$s^2 = (\sin\theta/\lambda)^2$ (1/$\AA^2$)")
    ax.set_ylabel(r"$K = F_{o,\mathrm{fcf}}^2 / F_c^2$")
    ax.set_title(r"Per-reflection $K$ vs $s^2$ (binned overlay)")

    add_resolution_top_axis(ax)

    if args.k_ylim_nsigma and args.k_ylim_nsigma > 0:
        mu = float(np.mean(K))
        sig = float(np.std(K))
        lo = mu - args.k_ylim_nsigma * sig
        hi = mu + args.k_ylim_nsigma * sig
        if hi > 0:
            ax.set_ylim(max(lo, 0.0), hi)

    plt.tight_layout()
    plt.savefig(out_png_K, dpi=250)
    plt.close()

    # -------- ln(K) vs s^2 --------
    plt.figure()
    ax = plt.gca()

    if not args.no_scatter:
        ax.scatter(s2, lnK, s=6, alpha=0.25)

    if cxL.size > 0:
        m = np.isfinite(cL) & np.isfinite(Llo) & np.isfinite(Lhi)
        ax.plot(cxL[m], cL[m], linewidth=1.5)
        ax.fill_between(cxL[m], Llo[m], Lhi[m], alpha=0.2, linewidth=0)

    ax.set_xlabel(r"$s^2 = (\sin\theta/\lambda)^2$ (1/$\AA^2$)")
    ax.set_ylabel(r"$\ln(K)$, where $K = F_{o,\mathrm{fcf}}^2 / F_c^2$")
    ax.set_title(r"Per-reflection $\ln(K)$ vs $s^2$ (binned overlay)")

    add_resolution_top_axis(ax)

    plt.tight_layout()
    plt.savefig(out_png_lnK, dpi=250)
    plt.close()

    print(f"# CELL: lam={cell.lam} Å, a={cell.a}, b={cell.b}, c={cell.c}, alpha={cell.alpha}, beta={cell.beta}, gamma={cell.gamma}")
    print(f"# FVAR(scale) = {k_scale}")
    print(f"# HKL lines: {len(hkl)} (terminators: {n_term})")
    print(f"# FCF ASU refls: {len(fcf_refls)}; symops: {len(symops)}; expanded keys: {len(lut)}")
    print(f"# Rows written: {len(rows)} (no match: {n_nomatch}, Fc2 too small: {n_smallfc})")
    print(f"# Wrote: {args.out_csv}")
    print(f"# Wrote: {out_png_K}")
    print(f"# Wrote: {out_png_lnK}")


if __name__ == "__main__":
    main()
    
