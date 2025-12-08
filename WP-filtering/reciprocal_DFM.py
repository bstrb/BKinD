#!/usr/bin/env python3
"""
3D reciprocal space plot with DFM coloring from SHELX .hkl + .ins/.res

DFM = (Fo^2 - Fc^2) / sqrt( sigma(Fo^2)^2 + (2*u*Fc^2)^2 )

u is chosen so that mean(DFM) = median(DFM)

Optional: resolution-normalize DFM within resolution shells (robust z-score per bin):
  DFM_scaled = (DFM - median_bin) / (1.4826 * MAD_bin)

Requirements:
  - cctbx (your install)
  - numpy
  - plotly  (pip install plotly)

Usage:
  python reciprocal_DFM.py model.ins data.hkl --out dfm_recip.html
  python reciprocal_DFM.py model.ins data.hkl --scale-by-resolution --nbins 20
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from typing import Tuple

import numpy as np

# ---------- parsing: SHELX .ins/.res cell ----------
def parse_shelx_cell(ins_path: str) -> Tuple[float, float, float, float, float, float]:
    """
    Reads CELL line from SHELX .ins/.res:
      CELL wavelength a b c alpha beta gamma
    Returns (a,b,c,alpha,beta,gamma) with angles in degrees.
    """
    cell_re = re.compile(
        r"^\s*CELL\s+([0-9.Ee+-]+)\s+([0-9.Ee+-]+)\s+([0-9.Ee+-]+)\s+([0-9.Ee+-]+)\s+([0-9.Ee+-]+)\s+([0-9.Ee+-]+)\s+([0-9.Ee+-]+)"
    )
    with open(ins_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = cell_re.match(line)
            if m:
                a = float(m.group(2)); b = float(m.group(3)); c = float(m.group(4))
                alpha = float(m.group(5)); beta = float(m.group(6)); gamma = float(m.group(7))
                return a, b, c, alpha, beta, gamma
    raise ValueError(f"No CELL line found in {ins_path}")


# ---------- parsing: SHELX .hkl ----------
@dataclass
class HKLData:
    h: np.ndarray
    k: np.ndarray
    l: np.ndarray
    fo2: np.ndarray
    sig_fo2: np.ndarray

def read_shelx_hkl(hkl_path: str) -> HKLData:
    """
    Reads typical SHELX HKLF 4 format lines:
      h k l Fo^2 sigma(Fo^2) [something]
    Stops at 0 0 0 line.
    """
    hs, ks, ls, fo2s, sigs = [], [], [], [], []
    with open(hkl_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            h, k, l = int(parts[0]), int(parts[1]), int(parts[2])
            if h == 0 and k == 0 and l == 0:
                break
            fo2 = float(parts[3])
            sig = float(parts[4])
            hs.append(h); ks.append(k); ls.append(l)
            fo2s.append(fo2); sigs.append(sig)

    h = np.asarray(hs, dtype=int)
    k = np.asarray(ks, dtype=int)
    l = np.asarray(ls, dtype=int)
    fo2 = np.asarray(fo2s, dtype=float)
    sig_fo2 = np.asarray(sigs, dtype=float)
    return HKLData(h=h, k=k, l=l, fo2=fo2, sig_fo2=sig_fo2)


# ---------- cryst: reciprocal vectors from a,b,c,alpha,beta,gamma ----------
def _cell_to_realspace_matrix(a, b, c, alpha_deg, beta_deg, gamma_deg) -> np.ndarray:
    """
    Returns 3x3 matrix whose columns are real-space basis vectors a,b,c in Cartesian Å.
    Convention: a along x, b in xy-plane, c general.
    """
    deg = math.pi / 180.0
    alpha = alpha_deg * deg
    beta = beta_deg * deg
    gamma = gamma_deg * deg

    ca, cb, cg = math.cos(alpha), math.cos(beta), math.cos(gamma)
    sg = math.sin(gamma)

    ax, ay, az = a, 0.0, 0.0
    bx, by, bz = b * cg, b * sg, 0.0
    cx = c * cb
    cy = c * (ca - cb * cg) / sg
    cz_sq = c * c - cx * cx - cy * cy
    cz = math.sqrt(max(cz_sq, 0.0))

    A = np.array([[ax, bx, cx],
                  [ay, by, cy],
                  [az, bz, cz]], dtype=float)
    return A

def reciprocal_cartesian_vectors_from_hkl(h, k, l, cell) -> np.ndarray:
    """
    Compute reciprocal lattice vectors in Cartesian coordinates (Å^-1, WITHOUT 2π),
    using a*, b*, c* = (A^{-1})^T, where A columns are real basis (Å).
    For each reflection: q = h a* + k b* + l c*
    Returns array shape (N, 3).
    """
    a, b, c, alpha, beta, gamma = cell
    A = _cell_to_realspace_matrix(a, b, c, alpha, beta, gamma)
    B = np.linalg.inv(A).T  # columns are a*, b*, c* in Å^-1 if A in Å
    H = np.stack([h, k, l], axis=0).astype(float)  # (3, N)
    Q = (B @ H).T  # (N, 3)
    return Q

def d_spacing_from_hkl(h, k, l, cell) -> np.ndarray:
    """
    Compute d-spacing (Å) from unit cell for each (h,k,l).
    Uses |q|^2 = 1/d^2 with q in Å^-1 (no 2π).
    """
    a, b, c, alpha, beta, gamma = cell
    A = _cell_to_realspace_matrix(a, b, c, alpha, beta, gamma)
    B = np.linalg.inv(A).T
    H = np.stack([h, k, l], axis=0).astype(float)  # (3, N)
    q = B @ H  # (3, N)
    inv_d2 = np.sum(q * q, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = 1.0 / np.sqrt(inv_d2)
    return d


# ---------- cctbx: Fc^2 for given HKLs ----------
def element_from_label(label: str) -> str:
    m = re.match(r"^([A-Za-z]{1,2})", label.strip())
    if not m:
        raise RuntimeError(f"Cannot infer element from label: {label}")
    el = m.group(1)
    return el[0].upper() + (el[1:].lower() if len(el) > 1 else "")

def compute_fc2_with_cctbx(ins_path: str, h: np.ndarray, k: np.ndarray, l: np.ndarray) -> np.ndarray:
    """
    Build an xray_structure from SHELX .ins/.res text, then compute Fc^2 at provided HKLs using cctbx.

    ED case:
      - SFAC may contain electron scattering params (numeric), so we infer element from atom label.
      - Use electron scattering table.
    """
    from cctbx import crystal, xray
    from cctbx import sgtbx
    from cctbx import miller
    from cctbx.array_family import flex

    def parse_cell_line(line: str):
        p = line.split()
        if len(p) < 8:
            raise ValueError("Bad CELL line")
        a, b, c = map(float, p[2:5])
        al, be, ga = map(float, p[5:8])
        return a, b, c, al, be, ga

    def spacegroup_from_latt(latt_int: int):
        n = abs(int(latt_int))
        cen = {1: "P", 2: "I", 3: "R", 4: "F", 5: "A", 6: "B", 7: "C"}.get(n, "P")
        return sgtbx.space_group_info(symbol=f"{cen} 1").group()

    # --- parse the .ins/.res (only parse atoms after SFAC) ---
    cell = None
    latt = None
    sfac = None
    atoms = []   # (label, element, (x,y,z), occ, uiso)

    def is_int(s: str) -> bool:
        try:
            int(s)
            return True
        except Exception:
            return False

    def is_float(s: str) -> bool:
        try:
            float(s.replace("D", "E"))
            return True
        except Exception:
            return False

    with open(ins_path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            key = parts[0].upper()

            if key == "CELL":
                cell = parse_cell_line(line)
                continue
            if key == "LATT":
                latt = int(parts[1])
                continue
            if key == "SFAC":
                sfac = parts[1:]
                continue
            if key in {"END", "HKLF"}:
                continue

            if sfac is None:
                continue

            # Atom line heuristic:
            # label sfacIndex x y z occ Uiso
            if len(parts) >= 7 and is_int(parts[1]) and all(is_float(x) for x in parts[2:7]):
                label = parts[0]
                sfac_idx = int(parts[1])
                if not (1 <= sfac_idx <= len(sfac)):
                    continue
                x = float(parts[2].replace("D", "E"))
                y = float(parts[3].replace("D", "E"))
                z = float(parts[4].replace("D", "E"))
                occ = float(parts[5].replace("D", "E"))
                uiso = float(parts[6].replace("D", "E"))

                element = element_from_label(label)
                atoms.append((label, element, (x, y, z), occ, uiso))
                continue

    if cell is None:
        raise RuntimeError("No CELL line found in .ins/.res")
    if latt is None:
        latt = 1
    if sfac is None or len(sfac) == 0:
        raise RuntimeError("No SFAC line found in .ins/.res")
    if len(atoms) == 0:
        raise RuntimeError(
            "No atom records parsed after SFAC. If your model uses ANIS or a nonstandard atom format, "
            "paste a few atom lines."
        )

    # --- build crystal symmetry (minimal centering only) ---
    a, b, c, al, be, ga = cell
    sg = spacegroup_from_latt(latt)
    cs = crystal.symmetry(unit_cell=(a, b, c, al, be, ga), space_group=sg)

    # --- build scatterers ---
    scat = []
    for label, element, (x, y, z), occ, uiso in atoms:
        sc = xray.scatterer(label=label, site=(x, y, z), occupancy=occ, scattering_type=element)
        # be explicit for compatibility across builds
        sc.u_iso = uiso
        scat.append(sc)

    xrs = xray.structure(crystal_symmetry=cs)
    xrs.add_scatterers(flex.xray_scatterer(scat))

    # Electron scattering factors
    xrs.scattering_type_registry(table="electron")

    # --- compute Fc^2 at the Miller indices ---
    mi = flex.miller_index(list(zip(h.tolist(), k.tolist(), l.tolist())))
    mset = miller.set(crystal_symmetry=cs, indices=mi, anomalous_flag=False)
    fcalc = mset.structure_factors_from_scatterers(xray_structure=xrs, algorithm="direct").f_calc()

    # fcalc.data() may be flex.complex_double -> use |F|^2
    fc2_flex = flex.norm(fcalc.data())
    fc2 = np.array(fc2_flex, dtype=float)
    return fc2


# ---------- DFM + solve u such that mean(DFM)=median(DFM) ----------
def dfm_values(fo2: np.ndarray, fc2: np.ndarray, sig_fo2: np.ndarray, u: float) -> np.ndarray:
    denom = np.sqrt(sig_fo2**2 + (2.0 * u * fc2)**2)
    denom = np.where(denom == 0.0, np.nan, denom)
    return (fo2 - fc2) / denom

def solve_u_mean_equals_median(fo2, fc2, sig_fo2) -> float:
    """
    Finds u >= 0 such that mean(DFM(u)) - median(DFM(u)) = 0.
    Uses a bracket+bisect strategy if possible; otherwise falls back to minimizing |diff|.
    """
    def diff(u):
        v = dfm_values(fo2, fc2, sig_fo2, u)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return np.nan
        return float(np.mean(v) - np.median(v))

    u_lo = 0.0
    f_lo = diff(u_lo)

    u_hi = 1.0
    f_hi = diff(u_hi)
    expansions = 0
    while (not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_lo * f_hi > 0) and expansions < 25:
        u_hi *= 2.0
        f_hi = diff(u_hi)
        expansions += 1

    if np.isfinite(f_lo) and np.isfinite(f_hi) and f_lo * f_hi <= 0:
        for _ in range(80):
            u_mid = 0.5 * (u_lo + u_hi)
            f_mid = diff(u_mid)
            if not np.isfinite(f_mid):
                u_lo = u_mid
                continue
            if abs(f_mid) < 1e-8:
                return u_mid
            if f_lo * f_mid <= 0:
                u_hi, f_hi = u_mid, f_mid
            else:
                u_lo, f_lo = u_mid, f_mid
        return 0.5 * (u_lo + u_hi)

    grid = np.logspace(-6, 6, 400)
    vals = np.array([abs(diff(u)) if np.isfinite(diff(u)) else np.inf for u in grid])
    best = float(grid[int(np.argmin(vals))])
    if np.isfinite(f_lo) and abs(f_lo) <= np.min(vals):
        best = 0.0
    return best


# ---------- optional: robust resolution scaling ----------
def robust_scale_by_resolution(dfm, d, nbins=20, min_per_bin=50):
    """
    Robustly normalize DFM within resolution bins (by d-spacing).
    Bins are equal-count in 1/d^2 (quantiles).
    Returns dfm_scaled; entries are NaN for bins with too few points.
    """
    dfm = np.asarray(dfm, float)
    d = np.asarray(d, float)

    ok = np.isfinite(dfm) & np.isfinite(d) & (d > 0)
    dfm_scaled = np.full(dfm.shape, np.nan, float)

    inv_d2_ok = 1.0 / (d[ok] ** 2)
    edges = np.quantile(inv_d2_ok, np.linspace(0, 1, nbins + 1))

    inv_d2_full = np.full(dfm.shape, np.nan, float)
    inv_d2_full[ok] = inv_d2_ok

    for i in range(nbins):
        lo, hi = edges[i], edges[i + 1]
        sel = ok & (inv_d2_full >= lo) & (inv_d2_full <= hi if i == nbins - 1 else inv_d2_full < hi)
        vals = dfm[sel]
        if vals.size < min_per_bin:
            continue
        med = np.median(vals)
        mad = np.median(np.abs(vals - med))
        denom = 1.4826 * mad
        if not np.isfinite(denom) or denom == 0:
            denom = np.std(vals)
        if not np.isfinite(denom) or denom == 0:
            continue
        dfm_scaled[sel] = (vals - med) / denom

    return dfm_scaled


# ---------- plot ----------
def make_plotly_3d(Q: np.ndarray,
                   dfm_color: np.ndarray,
                   h: np.ndarray, k: np.ndarray, l: np.ndarray,
                   out_html: str,
                   dfm_raw: np.ndarray | None = None,
                   d: np.ndarray | None = None,
                   color_label: str = "DFM"):
    import plotly.graph_objects as go

    dfm_color = np.asarray(dfm_color, float)
    finite = np.isfinite(dfm_color)
    if finite.sum() == 0:
        raise ValueError("No finite values to plot for coloring.")

    lo, hi = np.percentile(dfm_color[finite], [2, 98])
    dfm_clip = np.clip(dfm_color, lo, hi)

    if dfm_raw is None:
        dfm_raw = dfm_color
    if d is None:
        d = np.full(dfm_color.shape, np.nan, float)

    customdata = np.stack([h, k, l, d, dfm_raw, dfm_color], axis=1)

    fig = go.Figure(
        data=go.Scatter3d(
            x=Q[:, 0], y=Q[:, 1], z=Q[:, 2],
            mode="markers",
            marker=dict(
                size=2,
                color=dfm_clip,
                showscale=True,
                colorbar=dict(title=f"{color_label} (clipped 2–98%)"),
            ),
            customdata=customdata,
            hovertemplate=(
                "hkl=(%{customdata[0]}, %{customdata[1]}, %{customdata[2]})"
                "<br>d=%{customdata[3]:.3f} Å"
                "<br>DFM(raw)=%{customdata[4]:.4f}"
                f"<br>{color_label}=%{{customdata[5]:.4f}}"
                "<br>q=(%{x:.4f}, %{y:.4f}, %{z:.4f})"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=f"Reciprocal space reflections colored by {color_label}",
        scene=dict(
            xaxis_title="q_x (Å⁻¹)",
            yaxis_title="q_y (Å⁻¹)",
            zaxis_title="q_z (Å⁻¹)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    fig.write_html(out_html)
    print(f"Wrote: {out_html}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ins", help="SHELX .ins or .res file (model)")
    ap.add_argument("hkl", help="SHELX .hkl file (Fo^2, sigma)")
    ap.add_argument("--out", default="dfm_recip.html", help="Output HTML file for interactive plot")
    ap.add_argument("--mask-neg-fo2", action="store_true", help="Ignore Fo^2 <= 0 reflections")

    # --- new options ---
    ap.add_argument("--scale-by-resolution", action="store_true",
                    help="Robustly normalize DFM within resolution shells before coloring")
    ap.add_argument("--nbins", type=int, default=20, help="Number of resolution bins (default: 20)")
    ap.add_argument("--min-per-bin", type=int, default=50, help="Min reflections per bin to scale (default: 50)")

    # u options
    ap.add_argument("--u", type=float, default=None,
                    help="Override u (otherwise solve u so mean(DFM)=median(DFM))")

    args = ap.parse_args()

    # Read data
    cell = parse_shelx_cell(args.ins)
    data = read_shelx_hkl(args.hkl)

    # Optional filtering
    sel = np.isfinite(data.fo2) & np.isfinite(data.sig_fo2) & (data.sig_fo2 >= 0)
    if args.mask_neg_fo2:
        sel &= (data.fo2 > 0)

    h = data.h[sel]; k = data.k[sel]; l = data.l[sel]
    fo2 = data.fo2[sel]; sig = data.sig_fo2[sel]

    print(f"Read {data.h.size} HKLs; using {h.size} after filtering")

    # Fc^2
    fc2 = compute_fc2_with_cctbx(args.ins, h, k, l)

    # u
    if args.u is None:
        u = solve_u_mean_equals_median(fo2, fc2, sig)
    else:
        u = float(args.u)

    dfm = dfm_values(fo2, fc2, sig, u)
    dfm_f = dfm[np.isfinite(dfm)]
    print(f"u = {u:.6g}")
    if dfm_f.size:
        print(f"mean(DFM) = {np.mean(dfm_f):.6g}, median(DFM) = {np.median(dfm_f):.6g} (finite only)")

    # Reciprocal points + resolution
    Q = reciprocal_cartesian_vectors_from_hkl(h, k, l, cell)
    d = d_spacing_from_hkl(h, k, l, cell)

    # Choose coloring
    if args.scale_by_resolution:
        dfm_color = robust_scale_by_resolution(dfm, d, nbins=args.nbins, min_per_bin=args.min_per_bin)
        color_label = f"DFM_scaled (bins={args.nbins})"
        n_scaled = np.isfinite(dfm_color).sum()
        print(f"Scaled DFM values computed for {n_scaled}/{dfm.size} reflections (bins with >= {args.min_per_bin} kept).")
    else:
        dfm_color = dfm
        color_label = "DFM"

    # Plot
    make_plotly_3d(Q, dfm_color, h, k, l, args.out, dfm_raw=dfm, d=d, color_label=color_label)


if __name__ == "__main__":
    main()
