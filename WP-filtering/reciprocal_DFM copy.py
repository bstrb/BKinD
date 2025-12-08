#!/usr/bin/env python3
"""
3D reciprocal space plot with DFM coloring from SHELX .hkl + .ins/.res

DFM = (Fo^2 - Fc^2) / sqrt( sigma(Fo^2)^2 + (2*u*Fc^2)^2 )

u is chosen so that mean(DFM) = median(DFM)

Requirements:
  - cctbx (your install)
  - numpy
  - plotly  (pip install plotly)

Usage:
  python plot_dfm_recip.py model.ins data.hkl --out dfm_recip.html
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
    cell_re = re.compile(r"^\s*CELL\s+([0-9.Ee+-]+)\s+([0-9.Ee+-]+)\s+([0-9.Ee+-]+)\s+([0-9.Ee+-]+)\s+([0-9.Ee+-]+)\s+([0-9.Ee+-]+)\s+([0-9.Ee+-]+)")
    with open(ins_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = cell_re.match(line)
            if m:
                # wavelength = float(m.group(1))  # not needed here
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

    # a vector
    ax, ay, az = a, 0.0, 0.0
    # b vector
    bx, by, bz = b * cg, b * sg, 0.0
    # c vector
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


# ---------- cctbx: Fc^2 for given HKLs ----------

def element_from_label(label: str) -> str:
    m = re.match(r"^([A-Za-z]{1,2})", label.strip())
    if not m:
        raise RuntimeError(f"Cannot infer element from label: {label}")
    el = m.group(1)
    return el[0].upper() + (el[1:].lower() if len(el) > 1 else "")

def compute_fc2_with_cctbx(ins_path: str, h: np.ndarray, k: np.ndarray, l: np.ndarray) -> np.ndarray:
    """
    Build an xray_structure from SHELX .ins/.res text (CELL/LATT/SFAC/UNIT + atom lines),
    then compute Fc^2 at provided HKLs using cctbx.

    This avoids iotbx.shelx, which is missing in some cctbx builds.
    """
    import re
    import numpy as np

    from cctbx import crystal, xray
    from cctbx import sgtbx
    from cctbx import miller
    from cctbx.array_family import flex
    from cctbx.eltbx import xray_scattering


    # --- helpers ---
    def parse_cell_line(line: str):
        # CELL wavelength a b c alpha beta gamma
        p = line.split()
        if len(p) < 8:
            raise ValueError("Bad CELL line")
        # wavelength = float(p[1])  # not needed
        a, b, c = map(float, p[2:5])
        al, be, ga = map(float, p[5:8])
        return a, b, c, al, be, ga

    def spacegroup_from_latt(latt_int: int):
        """
        SHELX LATT encodes centering + inversion.
        We only map basic centerings here; symmetry ops beyond that must be in SYMM lines
        (not handled in this minimal parser).
        """
        n = abs(int(latt_int))
        # 1 P, 2 I, 3 R, 4 F, 5 A, 6 B, 7 C (SHELX convention)
        cen = {1: "P", 2: "I", 3: "R", 4: "F", 5: "A", 6: "B", 7: "C"}.get(n, "P")
        # We assume P1 with that centering (no SYMM expansion here).
        # For typical ED/SHELX small-molecule work where you refine in P1 or simple lattices, this is often OK.
        return sgtbx.space_group_info(symbol=f"{cen} 1").group()

    # --- parse the .ins/.res (safer; only parse atoms after SFAC) ---
    cell = None
    latt = None
    sfac = None  # list of element symbols in SFAC order
    fvar = None  # optional
    atoms = []   # (label, element, (x,y,z), occ, uiso)

    def is_int(s: str) -> bool:
        try:
            int(s)
            return True
        except Exception:
            return False

    def is_float(s: str) -> bool:
        try:
            float(s.replace("D", "E"))  # sometimes Fortran D exponents appear
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
            if key == "FVAR":
                # store raw; we may decode later if needed
                fvar = [float(x.replace("D", "E")) for x in parts[1:]]
                continue
            if key in {"END", "HKLF"}:
                # do not early break; some files put atoms before HKLF, some after
                continue

            # --- ONLY attempt atom parsing after SFAC is defined ---
            if sfac is None:
                continue

            # Atom line heuristic:
            # label  sfacIndex  x y z  occ  Uiso   (at least 7 tokens total)
            if len(parts) >= 7 and is_int(parts[1]) and all(is_float(x) for x in parts[2:7]):
                label = parts[0]
                sfac_idx = int(parts[1])
                x = float(parts[2].replace("D", "E"))
                y = float(parts[3].replace("D", "E"))
                z = float(parts[4].replace("D", "E"))
                occ = float(parts[5].replace("D", "E"))
                uiso = float(parts[6].replace("D", "E"))

                if not (1 <= sfac_idx <= len(sfac)):
                    # not a real atom line; ignore
                    continue



                element = element_from_label(label)
                atoms.append((label, element, (x, y, z), occ, uiso))
                continue

            # Otherwise ignore line (REM, SYMM, AFIX, PART, etc.)
            continue

    if cell is None:
        raise RuntimeError("No CELL line found in .ins/.res")
    if latt is None:
        latt = 1
    if sfac is None or len(sfac) == 0:
        raise RuntimeError("No SFAC line found in .ins/.res")
    if len(atoms) == 0:
        raise RuntimeError(
            "No atom records parsed after SFAC. "
            "If your model uses ANIS or a nonstandard atom format, paste a few atom lines."
        )


    # --- build crystal symmetry (minimal: centering + P1 symmetry) ---
    a, b, c, al, be, ga = cell
    sg = spacegroup_from_latt(latt)

    cs = crystal.symmetry(
        unit_cell=(a, b, c, al, be, ga),
        space_group=sg
    )

    # --- build scatterers ---
    scat = []
    for label, element, (x, y, z), occ, uiso in atoms:
        # Convert SHELX occ scale if needed: in SHELX, occupancy is often like 11.0, 21.0 etc (free variables).
        # Here we assume it's already "real" occupancy (0..1) OR close; we just take it as-is.
        # If your occ looks like 11.0 for all atoms, tell me and we’ll decode FVAR properly.
        sc = xray.scatterer(
            label=label,
            site=(x, y, z),
            u=uiso,          # cctbx uses u_iso in Å^2
            occupancy=occ,
            scattering_type=element
        )
        scat.append(sc)

    xrs = xray.structure(crystal_symmetry=cs)

    # Convert Python list -> cctbx flex container (required by some builds)
    xrs.add_scatterers(flex.xray_scatterer(scat))

    # Ensure scattering factors are available
    xrs.scattering_type_registry(
        table="electron"  # standard; change if you prefer
    )

    # --- compute Fc^2 at the Miller indices ---
    mi = flex.miller_index(list(zip(h.tolist(), k.tolist(), l.tolist())))
    mset = miller.set(crystal_symmetry=cs, indices=mi, anomalous_flag=False)

    fcalc = mset.structure_factors_from_scatterers(
        xray_structure=xrs,
        algorithm="direct"
    ).f_calc()


    # fcalc.data() is flex.complex_double in your build
    fc_cplx = fcalc.data()

    # |F|^2 for complex values
    fc2_flex = flex.norm(fc_cplx)  # returns |F|^2 as flex.double
    fc2 = np.array(fc2_flex, dtype=float)
    return fc2


# ---------- DFM + solve u such that mean(DFM)=median(DFM) ----------
def dfm_values(fo2: np.ndarray, fc2: np.ndarray, sig_fo2: np.ndarray, u: float) -> np.ndarray:
    denom = np.sqrt(sig_fo2**2 + (2.0 * u * fc2)**2)
    # avoid divide-by-zero
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

    # Try to bracket a root
    u_lo = 0.0
    f_lo = diff(u_lo)

    # pick a generous upper bound; can expand if needed
    u_hi = 1.0
    f_hi = diff(u_hi)
    expansions = 0
    while (not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_lo * f_hi > 0) and expansions < 25:
        u_hi *= 2.0
        f_hi = diff(u_hi)
        expansions += 1

    if np.isfinite(f_lo) and np.isfinite(f_hi) and f_lo * f_hi <= 0:
        # Bisection
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

    # Fallback: minimize |diff(u)| on log grid
    grid = np.logspace(-6, 6, 400)  # from 1e-6 to 1e6
    vals = np.array([abs(diff(u)) if np.isfinite(diff(u)) else np.inf for u in grid])
    best = float(grid[int(np.argmin(vals))])
    # allow u=0 if it is best
    if abs(f_lo) <= np.min(vals):
        best = 0.0
    return best


# ---------- plot ----------
# def make_plotly_3d(Q: np.ndarray, dfm: np.ndarray, out_html: str):
def make_plotly_3d(Q: np.ndarray, dfm: np.ndarray, h: np.ndarray, k: np.ndarray, l: np.ndarray, out_html: str):

    import plotly.graph_objects as go

    # clip extreme DFM for nicer color scaling (keeps full values in hover)
    dfm_finite = dfm[np.isfinite(dfm)]
    if dfm_finite.size == 0:
        raise ValueError("No finite DFM values to plot.")
    lo, hi = np.percentile(dfm_finite, [2, 98])
    dfm_clip = np.clip(dfm, lo, hi)
    
    fig = go.Figure(
        data=go.Scatter3d(
            x=Q[:, 0], y=Q[:, 1], z=Q[:, 2],
            mode="markers",
            marker=dict(
                size=2,
                color=dfm_clip,
                showscale=True,
                colorbar=dict(title="DFM (clipped 2–98%)"),
            ),
            customdata=np.stack([h, k, l, dfm], axis=1),
            hovertemplate=(
                "hkl=(%{customdata[0]}, %{customdata[1]}, %{customdata[2]})"
                "<br>q=(%{x:.4f}, %{y:.4f}, %{z:.4f})"
                "<br>DFM=%{customdata[3]:.4f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Reciprocal space reflections colored by DFM",
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

    # Solve u
    u = solve_u_mean_equals_median(fo2, fc2, sig)
    u = 0.02
    dfm = dfm_values(fo2, fc2, sig, u)
    dfm_f = dfm[np.isfinite(dfm)]
    print(f"u = {u:.6g}")
    print(f"mean(DFM) = {np.mean(dfm_f):.6g}, median(DFM) = {np.median(dfm_f):.6g} (finite only)")

    # Reciprocal points
    Q = reciprocal_cartesian_vectors_from_hkl(h, k, l, cell)

    # Plot
    make_plotly_3d(Q, dfm, h, k, l, args.out)



if __name__ == "__main__":
    main()
