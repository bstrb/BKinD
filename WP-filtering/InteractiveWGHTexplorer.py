#!/usr/bin/env python3
"""
Interactive WGHT explorer (PyQt6 + Matplotlib Qt canvas) with:
    - FCF mode only (load SHELXL .fcf reflection table)
    - optional SHELXL runner (clone .ins + .hkl into new folder, patch instructions, run, reload .fcf)

WGHT-like model (SHELXL style):
  w = q / [ sigma2(Fo2) + (a*P)^2 + b*P + d + e*sin(theta) ]
  P = f*max(Fo2,0) + (1-f)*Fc2
  q = 1               if c = 0
      exp(c*s^2)       if c > 0
      1 - exp(c*s^2)   if c < 0
where s = sin(theta)/lambda  (Å^-1)

Notes on wavelength:
  - c depends on s^2 -> independent of lambda if s is sin(theta)/lambda
  - e depends on sin(theta) = s*lambda -> lambda matters (often tiny in ED)
"""

from __future__ import annotations

import os
import sys
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QDesktopServices
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
import re


# ==========================
# USER-CONFIG DEFAULTS
# ==========================
DEFAULTS = {
    # Reference resolution points (used only for labels / shell count)
    # Shells are quantiles with count = len(s_ref)
    "s_ref": [0.1, 0.4, 0.8],

    # Wavelength in Angstrom (only affects e*sin(theta))
    "lambda_A": 0.01968,

    # Initial WGHT parameters
    "a": 0.0,
    "b": 0.0,
    "c": 0.0,
    "d": 0.0,
    "e": 0.0,
    "f": 1.0,

    # Binning for FCF mode plots (log Fo2 bins)
    "nbins_logFo2": 100,

    # SHELXL runner defaults
    "shelxl_cmd": "shelxl",     # change if needed
    "runner_force_acta": True,
}


# ==========================
# Data containers
# ==========================
@dataclass
class FcfData:
    path: Path
    Fo2: np.ndarray
    Fc2: np.ndarray
    sigFo2: np.ndarray
    s: np.ndarray        # sin(theta)/lambda in Å^-1
    hkl: np.ndarray      # shape (N,3) ints
    cell: Optional[Dict[str, float]] = None


# ==========================
# CIF/FCF parsing utilities
# ==========================
def _try_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def _read_cell_from_cif(lines: List[str]) -> Dict[str, float]:
    # Minimal parse: look for standard cell tags.
    tags = {
        "_cell_length_a": "a",
        "_cell_length_b": "b",
        "_cell_length_c": "c",
        "_cell_angle_alpha": "alpha",
        "_cell_angle_beta": "beta",
        "_cell_angle_gamma": "gamma",
    }
    cell: Dict[str, float] = {}
    for ln in lines:
        s = ln.strip()
        if not s.startswith("_cell_"):
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        key = parts[0]
        if key in tags:
            val = _try_float(parts[1])
            if val is not None:
                cell[tags[key]] = val
    return cell

def _find_refln_loop(lines: List[str]) -> Tuple[List[str], List[List[str]]]:
    """
    Find the first CIF loop_ that defines _refln_ columns, return (columns, rows_tokens).
    Rows are token lists, each with length >= len(columns) (we slice to exactly len(columns)).
    """
    n = len(lines)
    i = 0
    while i < n:
        if lines[i].strip().lower() != "loop_":
            i += 1
            continue

        # collect column names
        cols: List[str] = []
        j = i + 1
        while j < n:
            t = lines[j].strip()
            if not t:
                j += 1
                continue
            if t.startswith("_"):
                cols.append(t.split()[0])
                j += 1
                continue
            break

        if not cols or not any(c.lower().startswith("_refln_") for c in cols):
            i = j
            continue

        # parse data rows until next loop_/data_/stop_/new tag
        rows: List[List[str]] = []
        k = j
        while k < n:
            raw = lines[k].strip()
            if not raw or raw.startswith("#"):
                k += 1
                continue
            low = raw.lower()
            if low == "loop_" or low.startswith("data_") or low == "stop_" or raw.startswith("_"):
                break

            toks = shlex.split(raw)
            # handle accidental line wraps by pulling next lines until enough tokens
            kk = k
            while len(toks) < len(cols) and kk + 1 < n:
                kk += 1
                nxt = lines[kk].strip()
                if not nxt or nxt.startswith("#"):
                    continue
                nxtlow = nxt.lower()
                if nxtlow == "loop_" or nxtlow.startswith("data_") or nxtlow == "stop_" or nxt.startswith("_"):
                    break
                toks += shlex.split(nxt)

            if len(toks) >= len(cols):
                rows.append(toks[:len(cols)])
            k = kk + 1

        return cols, rows

    raise ValueError("Could not find a CIF loop_ with _refln_ columns in this .fcf")

def _pick_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    lowmap = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        c2 = cand.lower()
        if c2 in lowmap:
            return lowmap[c2]
    return None

def _unit_cell_reciprocal_metric(cell: Dict[str, float]) -> Tuple[float, float, float, float, float, float]:
    """
    Return reciprocal parameters (a*, b*, c*, cos(alpha*), cos(beta*), cos(gamma*)).
    Uses standard crystallography relations.
    """
    a = float(cell["a"]); b = float(cell["b"]); c = float(cell["c"])
    alpha = np.deg2rad(float(cell["alpha"]))
    beta  = np.deg2rad(float(cell["beta"]))
    gamma = np.deg2rad(float(cell["gamma"]))

    ca, cb, cg = np.cos(alpha), np.cos(beta), np.cos(gamma)
    sa, sb, sg = np.sin(alpha), np.sin(beta), np.sin(gamma)

    # Unit cell volume
    V = a * b * c * np.sqrt(max(0.0, 1.0 + 2.0*ca*cb*cg - ca*ca - cb*cb - cg*cg))
    if V <= 0:
        raise ValueError("Invalid unit cell volume computed from cell parameters")

    astar = (b * c * sa) / V
    bstar = (a * c * sb) / V
    cstar = (a * b * sg) / V

    # Reciprocal angles
    cos_alpha_star = (cb*cg - ca) / (sb*sg)
    cos_beta_star  = (ca*cg - cb) / (sa*sg)
    cos_gamma_star = (ca*cb - cg) / (sa*sb)

    # numeric clamp
    cos_alpha_star = float(np.clip(cos_alpha_star, -1.0, 1.0))
    cos_beta_star  = float(np.clip(cos_beta_star, -1.0, 1.0))
    cos_gamma_star = float(np.clip(cos_gamma_star, -1.0, 1.0))

    return astar, bstar, cstar, cos_alpha_star, cos_beta_star, cos_gamma_star

def _s_from_hkl_cell(hkl: np.ndarray, cell: Dict[str, float]) -> np.ndarray:
    """
    Compute s = sin(theta)/lambda = 1/(2d) from hkl and cell.
    """
    astar, bstar, cstar, caS, cbS, cgS = _unit_cell_reciprocal_metric(cell)
    h = hkl[:, 0].astype(float)
    k = hkl[:, 1].astype(float)
    l = hkl[:, 2].astype(float)

    # |g*|^2
    g2 = (
        (h*h) * (astar*astar) +
        (k*k) * (bstar*bstar) +
        (l*l) * (cstar*cstar) +
        2.0*h*k*astar*bstar*cgS +
        2.0*h*l*astar*cstar*cbS +
        2.0*k*l*bstar*cstar*caS
    )
    g2 = np.maximum(g2, 1e-30)
    d = 1.0 / np.sqrt(g2)
    s = 1.0 / (2.0 * d)
    return s

def load_fcf(path: str | Path) -> FcfData:
    p = Path(path).expanduser().resolve()
    txt = p.read_text(errors="replace").splitlines()

    cell = _read_cell_from_cif(txt)
    cols, rows = _find_refln_loop(txt)

    # candidate columns (SHELXL .fcf is CIF-like, but naming can vary)
    col_h = _pick_col(cols, ["_refln_index_h", "_refln_index_h_"])
    col_k = _pick_col(cols, ["_refln_index_k", "_refln_index_k_"])
    col_l = _pick_col(cols, ["_refln_index_l", "_refln_index_l_"])

    col_Fo2 = _pick_col(cols, ["_refln_F_squared_meas", "_refln_F_squared_meas_au", "_refln_F_squared_meas"])
    col_sig = _pick_col(cols, ["_refln_F_squared_sigma", "_refln_F_squared_sigma_au", "_refln_F_squared_sigma"])
    col_Fc2 = _pick_col(cols, ["_refln_F_squared_calc", "_refln_F_squared_calc_au", "_refln_F_squared_calc"])

    col_s = _pick_col(cols, ["_refln_sin_theta_over_lambda", "_refln_sint_over_lamb", "_refln_sin_theta_over_lambda"])
    col_d = _pick_col(cols, ["_refln_d_spacing", "_refln_d_spacing"])

    need = [col_h, col_k, col_l, col_Fo2, col_sig, col_Fc2]
    if any(x is None for x in need):
        raise ValueError(
            "FCF is missing one or more required columns. "
            f"Found columns include: {', '.join(cols[:20])} ..."
        )

    idx = {c: cols.index(c) for c in cols}

    N = len(rows)
    hkl = np.zeros((N, 3), dtype=int)
    Fo2 = np.zeros(N, dtype=float)
    Fc2 = np.zeros(N, dtype=float)
    sig = np.zeros(N, dtype=float)
    s = np.zeros(N, dtype=float)

    for i, r in enumerate(rows):
        hkl[i, 0] = int(float(r[idx[col_h]]))
        hkl[i, 1] = int(float(r[idx[col_k]]))
        hkl[i, 2] = int(float(r[idx[col_l]]))
        Fo2[i] = float(r[idx[col_Fo2]])
        Fc2[i] = float(r[idx[col_Fc2]])
        sig[i] = float(r[idx[col_sig]])

        if col_s is not None:
            s[i] = float(r[idx[col_s]])
        elif col_d is not None:
            dval = float(r[idx[col_d]])
            s[i] = 1.0 / (2.0 * dval) if dval > 0 else 0.0
        else:
            s[i] = np.nan

    # sanitize
    sig = np.maximum(sig, 1e-12)
    Fo2 = np.asarray(Fo2, dtype=float)
    Fc2 = np.asarray(Fc2, dtype=float)

    # compute s if missing
    if np.any(~np.isfinite(s)) or np.all(s <= 0):
        if all(k in cell for k in ("a", "b", "c", "alpha", "beta", "gamma")):
            s = _s_from_hkl_cell(hkl, cell)
        else:
            raise ValueError("No s or d_spacing in FCF and missing cell parameters to compute s from hkl.")

    # filter invalid rows
    ok = np.isfinite(Fo2) & np.isfinite(Fc2) & np.isfinite(sig) & np.isfinite(s) & (sig > 0) & (s > 0)
    Fo2 = Fo2[ok]
    Fc2 = Fc2[ok]
    sig = sig[ok]
    s = s[ok]
    hkl = hkl[ok]

    return FcfData(path=p, Fo2=Fo2, Fc2=Fc2, sigFo2=sig, s=s, hkl=hkl, cell=cell if cell else None)


# ==========================
# Core math for weights/objective
# ==========================
def q_factor_vec(c: float, s: np.ndarray) -> np.ndarray:
    x = s * s
    if c == 0.0:
        return np.ones_like(x)
    if c > 0.0:
        return np.exp(c * x)
    return 1.0 - np.exp(c * x)

def sin_theta_from_s(s: np.ndarray, lambda_A: float) -> np.ndarray:
    return np.clip(s * float(lambda_A), 0.0, 1.0)

def weights_wght(Fo2: np.ndarray, Fc2: np.ndarray, s: np.ndarray, sigma2: np.ndarray,
                 a: float, b: float, c: float, d: float, e: float, f: float, lambda_A: float) -> np.ndarray:
    P = float(f) * np.maximum(Fo2, 0.0) + (1.0 - float(f)) * Fc2
    q = q_factor_vec(float(c), s)
    sin_th = sin_theta_from_s(s, float(lambda_A))
    denom = sigma2 + (float(a) * P) ** 2 + float(b) * P + float(d) + float(e) * sin_th
    return q / np.maximum(denom, 1e-30)

def bin_logx_median(x: np.ndarray, y: np.ndarray, nbins: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x = x[ok]; y = y[ok]
    if x.size == 0:
        return np.array([]), np.array([])
    xmin = np.min(x)
    xmax = np.max(x)
    edges = np.logspace(np.log10(xmin), np.log10(xmax), int(nbins) + 1)
    xc = np.sqrt(edges[:-1] * edges[1:])
    yy = np.full_like(xc, np.nan, dtype=float)
    for i in range(len(xc)):
        m = (x >= edges[i]) & (x < edges[i+1])
        if np.any(m):
            yy[i] = np.nanmedian(y[m])
    ok2 = np.isfinite(yy)
    return xc[ok2], yy[ok2]

def s_shells_quantile(s: np.ndarray, n_shells: int) -> List[Tuple[float, float]]:
    s = np.asarray(s, float)
    s = s[np.isfinite(s) & (s > 0)]
    if s.size == 0:
        return []
    qs = np.linspace(0.0, 1.0, n_shells + 1)
    edges = np.quantile(s, qs)
    shells = []
    for i in range(n_shells):
        lo = float(edges[i])
        hi = float(edges[i+1])
        if hi <= lo:
            continue
        shells.append((lo, hi))
    return shells


def parse_lst_metrics(lst_path: Path) -> Dict[str, object]:
    """Extract OSF (last cycle), split/NPD counts, and K/R1 vs resolution from a SHELXL .lst."""
    text = lst_path.read_text(errors="replace").splitlines()

    def floats_in_line(ln: str) -> List[float]:
        vals: List[float] = []
        for t in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[Ee][-+]?\d+)?", ln):
            try:
                vals.append(float(t))
            except Exception:
                continue
        return vals

    osf_last: Optional[float] = None
    split_count: Optional[int] = None
    npd_count: Optional[int] = None
    res_vals: List[float] = []
    k_vals: List[float] = []
    r1_vals: List[float] = []

    # Track OSF by scanning parameter tables; keep the last occurrence
    for ln in text:
        if "OSF" in ln:
            m = re.search(r"\bOSF\b", ln)
            if not m:
                continue
            parts = ln.strip().split()
            if len(parts) >= 2 and parts[0].isdigit():
                try:
                    osf_last = float(parts[1])
                except Exception:
                    continue

    # Split/NPD counts
    for ln in text:
        m = re.search(r"Warning:\s*(\d+)\s+atoms may be split.*?(\d+)\s+atoms NPD", ln)
        if m:
            split_count = int(m.group(1))
            npd_count = int(m.group(2))
            break

    # Resolution-dependent K and R1
    # Find the last block starting with "Resolution(" then grab following K and R1 lines
    res_block_start = None
    for idx, ln in enumerate(text):
        if ln.strip().startswith("Resolution("):
            res_block_start = idx
    if res_block_start is not None:
        res_line = text[res_block_start].strip()
        # take everything after the first token (e.g., "Resolution(A)")
        res_body = res_line.split(maxsplit=1)[1] if len(res_line.split()) > 1 else res_line
        res_vals = [v for v in floats_in_line(res_body) if np.isfinite(v) and v > 0]

        # subsequent lines: Number in group, GooF, K, R1; grab the first K and first R1 after the block header
        k_line = None
        r1_line = None
        for ln in text[res_block_start+1:]:
            st = ln.strip()
            if not st:
                continue
            if st.startswith("K") and k_line is None:
                k_line = st
                continue
            if st.startswith("R1") and r1_line is None:
                r1_line = st
                break
            if st.startswith("END"):
                break

        def tail_values(line: Optional[str]) -> List[float]:
            if not line:
                return []
            parts = line.split(maxsplit=1)
            tail = parts[1] if len(parts) > 1 else ""
            return floats_in_line(tail)

        k_vals = tail_values(k_line)
        r1_vals = tail_values(r1_line)

    return {
        "osf": osf_last,
        "split": split_count,
        "npd": npd_count,
        "res": res_vals,
        "K": k_vals,
        "R1": r1_vals,
    }


# ==========================
# SHELXL runner utilities
# ==========================
def find_hkl_for_ins(ins_path: Path) -> Optional[Path]:
    base = ins_path.with_suffix("")
    for ext in (".hkl", ".HKL"):
        p = base.with_suffix(ext)
        if p.exists():
            return p
    # sometimes file name differs; no robust way without parsing HKLF usage
    return None

def patch_ins_text(ins_text: str,
                   wght: Tuple[float, float, float, float, float, float],
                   adp_mode: str,            # "keep", "ANIS", "ISOT"
                   set_exti: bool,
                   exti_value: float,
                   force_acta: bool) -> str:
    lines = ins_text.splitlines()

    def cmd_token(line: str) -> str:
        parts = line.strip().split()
        return parts[0].upper() if parts else ""

    targets = {"WGHT", "EXTI", "ANIS", "ISO", "ACTA"}

    # Drop existing targeted commands if we are overwriting them
    out: List[str] = []
    for ln in lines:
        tok = cmd_token(ln)
        if tok == "WGHT":
            continue
        if tok == "EXTI" and set_exti:
            continue
        if tok in {"ANIS", "ISO"} and adp_mode != "keep":
            continue
        if tok == "ACTA" and force_acta:
            continue
        out.append(ln)

    # Find UNIT and the start of the atom list; insert between them
    insert_idx = 0
    unit_idx = None
    for i, ln in enumerate(out):
        if cmd_token(ln) == "UNIT":
            unit_idx = i
            break
    if unit_idx is None:
        unit_idx = 0

    def is_atom_line(ln: str) -> bool:
        parts = ln.strip().split()
        if len(parts) < 2:
            return False
        if parts[0].startswith("!") or parts[0].startswith("#"):
            return False
        # Heuristic: first token is label, second numeric
        try:
            float(parts[1])
        except Exception:
            return False
        return parts[0][0].isalpha()

    insert_idx = unit_idx + 1
    while insert_idx < len(out):
        tok = cmd_token(out[insert_idx])
        if tok in {"HKLF", "END"}:
            break
        if is_atom_line(out[insert_idx]):
            break
        insert_idx += 1

    a, b, c, d, e, f = wght
    new_lines = []
    if force_acta:
        new_lines.append("ACTA")
    if adp_mode.upper() == "ANIS":
        new_lines.append("ANIS")
    elif adp_mode.upper() == "ISOT":
        new_lines.append("ANIS 0")
    if set_exti:
        new_lines.append(f"EXTI {exti_value:.6f}")
    new_lines.append(f"WGHT {a:.6f} {b:.6f} {c:.6f} {d:.6f} {e:.6f} {f:.6f}")

    out = out[:insert_idx] + new_lines + out[insert_idx:]
    return "\n".join(out) + "\n"

def run_shelxl_variant(ins_path: Path,
                       hkl_path: Path,
                       shelxl_cmd: str,
                       run_label: str,
                       wght: Tuple[float, float, float, float, float, float],
                       adp_mode: str,
                       set_exti: bool,
                       exti_value: float,
                       force_acta: bool) -> Tuple[Path, str]:
    """
    Create new folder, copy ins+hkl, patch ins, run shelxl.
    Returns (run_dir, combined_log_text)
    """
    ins_path = ins_path.resolve()
    hkl_path = hkl_path.resolve()
    base = ins_path.stem  # e.g. "sample"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_label.strip())[:40] or "run"
    run_dir = ins_path.parent / f"{base}_wght_{ts}_{label}"
    run_dir.mkdir(parents=True, exist_ok=False)

    # copy inputs
    ins_dst = run_dir / f"{base}.ins"
    hkl_dst = run_dir / hkl_path.name  # keep original name
    shutil.copy2(ins_path, ins_dst)
    shutil.copy2(hkl_path, hkl_dst)

    # patch ins
    patched = patch_ins_text(ins_dst.read_text(errors="replace"), wght, adp_mode, set_exti, exti_value, force_acta)
    ins_dst.write_text(patched)

    # run shelxl on base name (without extension)
    cmd = [shelxl_cmd, base]
    proc = subprocess.run(
        cmd,
        cwd=str(run_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    log = (
        f"$ {' '.join(cmd)}\n\n"
        f"--- STDOUT ---\n{proc.stdout}\n\n"
        f"--- STDERR ---\n{proc.stderr}\n\n"
        f"Exit code: {proc.returncode}\n"
    )
    return run_dir, log


# ==========================
# GUI
# ==========================
class WghtExplorer(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Interactive WGHT Explorer (PyQt6) + FCF + Runner")

        self.cfg = dict(DEFAULTS)

        # Data
        self.fcf_data: Optional[FcfData] = None
        self.lst_metrics: Dict[str, object] = {}

        # Runner state
        self.last_run_dir: Optional[Path] = None

        # Runner state
        self.ins_path: Optional[Path] = None
        self.hkl_path: Optional[Path] = None

        self._build_ui()
        self.update_plots()

    # ---- UI setup ----
    def _build_ui(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)

        # Matplotlib figure and canvas
        self.fig = Figure(figsize=(12.0, 8.0))
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        left = QtWidgets.QVBoxLayout()
        left.addWidget(self.toolbar)
        left.addWidget(self.canvas)

        # Controls in a scroll area
        controls_widget = QtWidgets.QWidget()
        controls_layout = QtWidgets.QFormLayout(controls_widget)
        controls_layout.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)

        # Helpers
        def add_spin(label, default, vmin, vmax, step, decimals=4):
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(vmin, vmax)
            spin.setSingleStep(step)
            spin.setDecimals(decimals)
            spin.setValue(default)
            controls_layout.addRow(label, spin)
            spin.valueChanged.connect(self.update_plots)
            return spin

        def add_combo(label, options, current):
            combo = QtWidgets.QComboBox()
            combo.addItems(options)
            combo.setCurrentText(current)
            controls_layout.addRow(label, combo)
            combo.currentIndexChanged.connect(self.update_plots)
            return combo

        # --- Data section ---
        grp_data = QtWidgets.QGroupBox("Data")
        vdata = QtWidgets.QVBoxLayout(grp_data)

        row = QtWidgets.QHBoxLayout()
        self.btn_load_fcf = QtWidgets.QPushButton("Load .fcf…")
        self.btn_load_fcf.clicked.connect(self.load_fcf_dialog)
        row.addWidget(self.btn_load_fcf)
        self.btn_load_fcf.setToolTip("Load a SHELXL .fcf to use measured Fo^2, Fc^2, sig(Fo^2)")
        vdata.addLayout(row)

        self.line_fcf = QtWidgets.QLineEdit("")
        self.line_fcf.setReadOnly(True)
        self.line_fcf.setPlaceholderText("No .fcf loaded")
        vdata.addWidget(self.line_fcf)
        self.line_fcf.setToolTip("Path of loaded .fcf (read-only)")

        controls_layout.addRow(grp_data)

        # --- WGHT params ---
        c = self.cfg
        self.spin_a = add_spin("a", c["a"], 0.0, 1.0, 0.05, 3)
        self.spin_a.setToolTip("WGHT a: quadratic term (a*P)^2 in denominator; lowers weights at high P")

        self.spin_b = add_spin("b", c["b"], 0.0, 10.0, 0.1, 2)
        self.spin_b.setToolTip("WGHT b: linear term b*P in denominator; reduces weight as P grows")

        self.spin_c = add_spin("c", c["c"], -50.0, 50.0, 0.5, 2)
        self.spin_c.setToolTip("WGHT c: q factor vs s; c>0 boosts high-s (high-res), c<0 damps high-s")

        self.spin_d = add_spin("d", c["d"], 0.0, 10000.0, 100.0, 0)
        self.spin_d.setToolTip("WGHT d: constant noise floor in denominator (raises all weights)")

        self.spin_e = add_spin("e", c["e"], -1000.0, 1000.0, 10.0, 0)
        self.spin_e.setToolTip("WGHT e: sin(theta) term e*sin(theta); adds angle-dependent noise")

        self.spin_f = add_spin("f", c["f"], 0.0, 1.0, 0.05, 2)
        self.spin_f.setToolTip("WGHT f: mix for P = f*Fo^2 + (1-f)*Fc^2; f=1 uses Fo^2 only")

        self.spin_lambda = add_spin("lambda_A", c["lambda_A"], 0.001, 1.0, 0.001, 5)
        self.spin_lambda.setToolTip("Wavelength (Å) used for sin(theta)=s*lambda in the e term")

        # Binning
        self.spin_nbins = QtWidgets.QSpinBox()
        self.spin_nbins.setRange(10, 200)
        self.spin_nbins.setValue(int(c["nbins_logFo2"]))
        self.spin_nbins.valueChanged.connect(self.update_plots)
        controls_layout.addRow("nbins_logFo2", self.spin_nbins)
        self.spin_nbins.setToolTip("Number of log(Fo^2) bins used to compute median curves")

        # --- Runner section ---
        grp_run = QtWidgets.QGroupBox("SHELXL runner (optional)")
        vrun = QtWidgets.QVBoxLayout(grp_run)

        row = QtWidgets.QHBoxLayout()
        self.btn_sel_ins = QtWidgets.QPushButton("Select .ins…")
        self.btn_sel_ins.clicked.connect(self.select_ins_dialog)
        row.addWidget(self.btn_sel_ins)
        self.btn_sel_ins.setToolTip("Pick a .ins file to run shelxl with current WGHT/ADP/EXTI options")

        self.btn_run = QtWidgets.QPushButton("Run variant")
        self.btn_run.clicked.connect(self.run_variant_clicked)
        row.addWidget(self.btn_run)
        self.btn_run.setToolTip("Run shelxl variant; will try to load resulting .fcf for plotting")
        vrun.addLayout(row)

        self.line_ins = QtWidgets.QLineEdit("")
        self.line_ins.setReadOnly(True)
        self.line_ins.setPlaceholderText("No .ins selected")
        vrun.addWidget(self.line_ins)
        self.line_ins.setToolTip("Selected .ins path (read-only)")

        self.line_hkl = QtWidgets.QLineEdit("")
        self.line_hkl.setReadOnly(True)
        self.line_hkl.setPlaceholderText("No .hkl selected / found")
        vrun.addWidget(self.line_hkl)
        self.line_hkl.setToolTip("Selected .hkl path; auto-found by base name when possible")

        self.line_shelxl = QtWidgets.QLineEdit(self.cfg["shelxl_cmd"])
        self.line_shelxl.setPlaceholderText("shelxl executable (in PATH)")
        vrun.addWidget(QtWidgets.QLabel("shelxl command:"))
        vrun.addWidget(self.line_shelxl)
        self.line_shelxl.setToolTip("Command used to invoke shelxl (full path or in PATH)")

        self.line_runlabel = QtWidgets.QLineEdit("test")
        vrun.addWidget(QtWidgets.QLabel("run label:"))
        vrun.addWidget(self.line_runlabel)
        self.line_runlabel.setToolTip("Run label (prefix for shelxl outputs)")

        self.combo_adp = QtWidgets.QComboBox()
        self.combo_adp.addItems(["keep", "ANIS", "ISOT"])
        vrun.addWidget(QtWidgets.QLabel("ADP mode:"))
        vrun.addWidget(self.combo_adp)
        self.combo_adp.setToolTip("ADP override for shelxl: keep existing, force ANIS, or force ISOT")

        row2 = QtWidgets.QHBoxLayout()
        self.chk_exti = QtWidgets.QCheckBox("Set EXTI")
        row2.addWidget(self.chk_exti)
        self.spin_exti = QtWidgets.QDoubleSpinBox()
        self.spin_exti.setRange(-100000.0, 100000.0)
        self.spin_exti.setDecimals(6)
        self.spin_exti.setSingleStep(0.001)
        self.spin_exti.setValue(0.0)
        row2.addWidget(self.spin_exti)
        self.chk_exti.setToolTip("Enable EXTI and set its value for the run")
        self.spin_exti.setToolTip("EXTI value to apply when enabled")
        vrun.addLayout(row2)

        self.chk_acta = QtWidgets.QCheckBox("Force ACTA (ensure .fcf output)")
        self.chk_acta.setChecked(bool(self.cfg["runner_force_acta"]))
        vrun.addWidget(self.chk_acta)
        self.chk_acta.setToolTip("Prepend ACTA to .ins before run to force .fcf output")

        self.txt_runlog = QtWidgets.QPlainTextEdit()
        self.txt_runlog.setReadOnly(True)
        self.txt_runlog.setFixedHeight(140)
        vrun.addWidget(self.txt_runlog)

        self.btn_open_run = QtWidgets.QPushButton("Open run folder")
        self.btn_open_run.clicked.connect(self.open_last_run_folder)
        vrun.addWidget(self.btn_open_run)
        self.btn_open_run.setToolTip("Open the last SHELXL run directory")

        controls_layout.addRow(grp_run)

        # Status text
        self.status_box = QtWidgets.QPlainTextEdit()
        self.status_box.setReadOnly(True)
        self.status_box.setFixedHeight(180)
        controls_layout.addRow("Status", self.status_box)

        # Wrap controls
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls_widget)

        layout.addLayout(left, stretch=3)
        layout.addWidget(scroll, stretch=2)

        # Setup axes
        self._setup_axes()

    def _setup_axes(self) -> None:
        self.fig.clear()
        gs = self.fig.add_gridspec(3, 1, height_ratios=[1.2, 1.0, 0.8])
        self.ax_w = self.fig.add_subplot(gs[0])
        self.ax_c = self.fig.add_subplot(gs[1])
        self.ax_k = self.fig.add_subplot(gs[2])
        self.ax_k_r = self.ax_k.twinx()

        for ax in (self.ax_w, self.ax_c):
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.25)

        self.ax_w.set_xlabel("Fo²")
        self.ax_w.set_ylabel("relative weight = w / (1/σ²)")
        self.ax_w.set_title("Weight behavior vs Fo² (binned medians, shells in s)")

        self.ax_c.set_xlabel("Fo²")
        self.ax_c.set_ylabel("w*(Fo² - Fc²)²  (binned median)")
        self.ax_c.set_title("Weighted contribution vs Fo² (binned medians, shells in s)")

        self.ax_k.set_xscale("log")
        self.ax_k.set_yscale("linear")
        self.ax_k.invert_xaxis()
        self.ax_k.set_xlabel("Resolution (Å)")
        self.ax_k.set_ylabel("K", color="C0")
        self.ax_k.tick_params(axis="y", colors="C0")
        self.ax_k_r.set_ylabel("R1", color="tab:red")
        self.ax_k_r.tick_params(axis="y", colors="tab:red")
        self.ax_k.set_title("K and R1 vs resolution (from .lst)")

        self.lines_w: List = []
        self.lines_c: List = []
        self.lines_k: List = []

    # ---- Data ----
    def load_fcf_dialog(self) -> None:
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select .fcf file", "", "FCF (*.fcf *.FCF);;All (*.*)")
        if not fn:
            return
        try:
            data = load_fcf(fn)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "FCF load error", str(exc))
            return

        self.fcf_data = data
        self.line_fcf.setText(str(data.path))
        self.lst_metrics = {}
        self.update_plots()

    # ---- Runner ----
    def select_ins_dialog(self) -> None:
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select .ins file", "", "INS (*.ins *.INS);;All (*.*)")
        if not fn:
            return
        self.ins_path = Path(fn).resolve()
        self.line_ins.setText(str(self.ins_path))

        hkl = find_hkl_for_ins(self.ins_path)
        if hkl is None:
            self.hkl_path = None
            self.line_hkl.setText("")
            QtWidgets.QMessageBox.information(
                self,
                "HKL not found",
                "Could not find a .hkl with the same base name as the .ins.\n"
                "Select the .hkl manually when you run, if needed."
            )
        else:
            self.hkl_path = hkl
            self.line_hkl.setText(str(hkl))

    def run_variant_clicked(self) -> None:
        if self.ins_path is None:
            QtWidgets.QMessageBox.warning(self, "Runner", "Select a .ins file first.")
            return

        # Ensure we have HKL
        if self.hkl_path is None or not self.hkl_path.exists():
            # prompt user
            fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select .hkl file", str(self.ins_path.parent), "HKL (*.hkl *.HKL);;All (*.*)")
            if not fn:
                return
            self.hkl_path = Path(fn).resolve()
            self.line_hkl.setText(str(self.hkl_path))

        shelxl_cmd = self.line_shelxl.text().strip() or "shelxl"
        run_label = self.line_runlabel.text().strip() or "run"
        adp_mode = self.combo_adp.currentText().strip()
        set_exti = self.chk_exti.isChecked()
        exti_val = float(self.spin_exti.value())
        force_acta = self.chk_acta.isChecked()

        base = self.ins_path.stem

        p = self._read_params()
        wght_tuple = (p["a"], p["b"], p["c"], p["d"], p["e"], p["f"])

        try:
            run_dir, log = run_shelxl_variant(
                ins_path=self.ins_path,
                hkl_path=self.hkl_path,
                shelxl_cmd=shelxl_cmd,
                run_label=run_label,
                wght=wght_tuple,
                adp_mode=adp_mode,
                set_exti=set_exti,
                exti_value=exti_val,
                force_acta=force_acta,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Runner error", str(exc))
            return

        self.txt_runlog.setPlainText(log)
        self.last_run_dir = run_dir

        # parse .lst metrics if available
        lst_path = run_dir / f"{base}.lst"
        if lst_path.exists():
            try:
                self.lst_metrics = parse_lst_metrics(lst_path)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "LST parse warning", f"Ran SHELXL but failed to parse .lst:\n{exc}")
                self.lst_metrics = {}
        else:
            self.lst_metrics = {}

        # try load resulting fcf
        base = self.ins_path.stem
        fcf = run_dir / f"{base}.fcf"
        if not fcf.exists():
            # maybe different case
            fcf2 = run_dir / f"{base}.FCF"
            fcf = fcf2 if fcf2.exists() else fcf

        if fcf.exists():
            try:
                data = load_fcf(fcf)
                self.fcf_data = data
                self.line_fcf.setText(str(data.path))
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "FCF load warning", f"Ran SHELXL but failed to load fcf:\n{exc}")
        else:
            QtWidgets.QMessageBox.information(
                self,
                "No .fcf found",
                "SHELXL run finished, but no .fcf was found in the run folder.\n"
                "If your .ins does not contain ACTA, enable 'Force ACTA' and rerun."
            )

        self.update_plots()

    def open_last_run_folder(self) -> None:
        if self.last_run_dir is None or not self.last_run_dir.exists():
            QtWidgets.QMessageBox.information(self, "Runner", "No run folder available yet.")
            return
        ok = QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.last_run_dir)))
        if not ok:
            QtWidgets.QMessageBox.information(
                self,
                "Runner",
                f"Could not open folder via desktop; path is:\n{self.last_run_dir}"
            )

    # ---- Params ----
    def _read_params(self) -> dict:
        return {
            "a": float(self.spin_a.value()),
            "b": float(self.spin_b.value()),
            "c": float(self.spin_c.value()),
            "d": float(self.spin_d.value()),
            "e": float(self.spin_e.value()),
            "f": float(self.spin_f.value()),
            "lambda_A": float(self.spin_lambda.value()),
        }

    # ---- Plot update ----
    def update_plots(self) -> None:
        p = self._read_params()

        if self.fcf_data is None:
            self.status_box.setPlainText("Load a .fcf or run SHELXL to view weights.")
            # switch to linear before clearing to avoid log warnings when limits are 0/1
            for ax in (self.ax_w, self.ax_c, self.ax_k):
                ax.set_xscale("linear"); ax.set_yscale("linear")
                ax.cla()
                ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.0)
                ax.grid(True, which="both", alpha=0.25)
            self.ax_k_r.cla()
            self.ax_k_r.set_ylim(0.0, 1.0)
            self.canvas.draw_idle()
            return

        Fo2 = self.fcf_data.Fo2
        Fc2 = self.fcf_data.Fc2
        sigFo2 = self.fcf_data.sigFo2
        s = self.fcf_data.s

        sigma2 = np.maximum(sigFo2, 1e-12) ** 2

        # weights + objective contributions
        w = weights_wght(Fo2, Fc2, s, sigma2, p["a"], p["b"], p["c"], p["d"], p["e"], p["f"], p["lambda_A"])
        resid = Fo2 - Fc2
        contrib = w * (resid ** 2)

        # relative weight vs baseline 1/sigma2
        w0 = 1.0 / np.maximum(sigma2, 1e-30)
        relw = w / w0

        # build shells in s (quantiles)
        n_shells = max(1, len(self.cfg["s_ref"]))
        shells = s_shells_quantile(s, n_shells=n_shells)

        # clear lines; set to linear first to avoid log warnings, we'll set log where needed after plotting
        for ax in (self.ax_w, self.ax_c, self.ax_k):
            ax.set_xscale("linear"); ax.set_yscale("linear")
            ax.cla()
        self.ax_k_r.cla()

        self.ax_w.set_xlabel("Fo²")
        self.ax_w.set_ylabel("relative weight = w / (1/σ²)")
        self.ax_w.set_title("Weight behavior vs Fo² (binned medians per resolution shell)")

        self.ax_c.set_xlabel("Fo²")
        self.ax_c.set_ylabel("w*(Fo² - Fc²)² (binned median)")
        self.ax_c.set_title("Weighted objective contribution vs Fo² (binned medians per shell)")

        self.ax_k.set_xlabel("Resolution (Å)")
        self.ax_k.set_ylabel("K", color="C0")
        self.ax_k.tick_params(axis="y", colors="C0")
        self.ax_k_r.set_ylabel("R1", color="tab:red")
        self.ax_k_r.tick_params(axis="y", colors="tab:red")
        self.ax_k.set_title("K and R1 vs resolution (from .lst)")

        nbins = int(self.spin_nbins.value())

        # plot each shell
        labels = []
        for (slo, shi) in shells:
            m = (s >= slo) & (s < shi) & np.isfinite(relw) & np.isfinite(contrib) & (Fo2 > 0)
            if not np.any(m):
                continue

            xw, yw = bin_logx_median(Fo2[m], relw[m], nbins=nbins)
            xc, yc = bin_logx_median(Fo2[m], contrib[m], nbins=nbins)

            # label by d-range
            d_hi = 1.0 / (2.0 * shi)
            d_lo = 1.0 / (2.0 * slo)
            lab = f"d≈{d_hi:.2f}–{d_lo:.2f} Å"
            labels.append(lab)

            if xw.size:
                self.ax_w.plot(xw, np.maximum(yw, 1e-30), label=lab)
            if xc.size:
                self.ax_c.plot(xc, np.maximum(yc, 1e-30), linestyle="--", label=lab)

        if labels:
            self.ax_w.legend(loc="lower left", fontsize=9)
            self.ax_c.legend(loc="upper right", fontsize=9)

        # K/R1 plot from .lst
        k_line = ""
        if self.lst_metrics:
            res_all = np.array(self.lst_metrics.get("res", []), dtype=float)
            k_all = np.array(self.lst_metrics.get("K", []), dtype=float)
            r1_all = np.array(self.lst_metrics.get("R1", []), dtype=float)

            # Plot K
            nk = min(len(res_all), len(k_all))
            if nk > 0:
                res_k = res_all[:nk]
                k_vals = k_all[:nk]
                m = np.isfinite(res_k) & (res_k > 0) & np.isfinite(k_vals)
                if np.any(m):
                    self.ax_k.plot(res_k[m], k_vals[m], marker="o", label="K", color="C0")
                    self.ax_k.legend(loc="upper left", fontsize=9)

            # Plot R1
            nr = min(len(res_all), len(r1_all))
            if nr > 0:
                res_r = res_all[:nr]
                r1_vals = r1_all[:nr]
                m2 = np.isfinite(res_r) & (res_r > 0) & np.isfinite(r1_vals)
                if np.any(m2):
                    self.ax_k_r.plot(res_r[m2], r1_vals[m2], marker="s", color="tab:red", label="R1")
                    self.ax_k_r.legend(loc="upper right", fontsize=9)

            if (nk > 0 or nr > 0):
                k_line = f"K points={nk if nk>0 else 0}, R1 points={nr if nr>0 else 0}"

            # axis limits/ticks using full resolution list; show finite min..max range
            res_valid = res_all[np.isfinite(res_all) & (res_all > 0)]
            if res_valid.size:
                res_min = float(np.min(res_valid))
                res_max = float(np.max(res_valid))
                hi = res_max * 1.05
                lo = res_min / 1.05
                self.ax_k.set_xlim(hi, lo)
                ticks = np.unique(np.round(res_valid, 2))
                ticks = ticks[(ticks > 0)]
                if ticks.size:
                    self.ax_k.set_xticks(ticks)
                    self.ax_k.set_xticklabels([f"{t:.2f}" for t in ticks], rotation=0)

        # apply scales and grids after plotting to avoid log warnings
        for ax in (self.ax_w, self.ax_c):
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.25)

        # set K axis to log-x only after limits are positive
        if self.ax_k.get_xlim()[0] > 0 and self.ax_k.get_xlim()[1] > 0:
            self.ax_k.set_xscale("log")
        else:
            # fallback limits to keep log happy
            self.ax_k.set_xlim(1.0, 0.1)
            self.ax_k.set_xscale("log")
        self.ax_k.set_yscale("linear")
        self.ax_k.invert_xaxis()
        self.ax_k.grid(True, which="both", alpha=0.25)

        # status
        sse = float(np.sum(contrib[np.isfinite(contrib)]))
        wrms = float(np.sqrt(np.mean(contrib[np.isfinite(contrib)])))

        mode_line = f"fcf={self.fcf_data.path.name}  N={Fo2.size}"

        osf_val = self.lst_metrics.get("osf") if isinstance(self.lst_metrics, dict) else None
        split_val = self.lst_metrics.get("split") if isinstance(self.lst_metrics, dict) else None
        npd_val = self.lst_metrics.get("npd") if isinstance(self.lst_metrics, dict) else None
        osf_line = f"OSF(last)={osf_val:.4f}" if isinstance(osf_val, (float, int)) else "OSF(last)=n/a"
        split_line = "split/NPD=n/a"
        if isinstance(split_val, int) and isinstance(npd_val, int):
            split_line = f"split={split_val}  NPD={npd_val}"

        self.status_box.setPlainText(
            f"{mode_line}\n"
            f"WGHT: a={p['a']:.3f} b={p['b']:.3f} c={p['c']:.2f} d={p['d']:.1f} e={p['e']:.1f} f={p['f']:.3f}\n"
            f"lambda_A={p['lambda_A']:.5f}\n"
            f"{osf_line}  {split_line}  {k_line}\n"
            f"SSE=sum[w*r^2]={sse:.3e}  WRMS=sqrt(mean[w*r^2])={wrms:.3e}\n"
            "Notes:\n"
            "  c acts via s^2 (independent of lambda if s=sin(theta)/lambda)\n"
            "  e acts via sin(theta)=s*lambda (lambda matters)\n"
        )

        self.canvas.draw_idle()


def main() -> None:
    if not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"
    app = QtWidgets.QApplication(sys.argv)
    w = WghtExplorer()
    w.resize(1350, 860)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
