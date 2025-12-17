# dfm_scaling.py
#!/usr/bin/env python3
import argparse
import csv
import os
import sys

import pandas as pd
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

from scipy.optimize import minimize

# CCTBX imports
from iotbx.reflection_file_reader import any_reflection_file
from iotbx.shelx.hklf import miller_array_export_as_shelx_hklf as hklf
from scitbx.array_family import flex

HKL = Tuple[int, int, int]
_NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
# Capture: h k l I sigma, and keep exact spans for I and sigma via match.start/end
_INTE_RE = re.compile(
    rf"^(\s*)(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+({_NUM})\s+({_NUM})(.*)$"
)


# -----------------------------
# Helpers
# -----------------------------
def die(msg: str, code: int = 2) -> None:
    raise SystemExit(f"ERROR: {msg}")


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print(f"[RUN] {' '.join(cmd)}   (cwd={cwd})")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def robust_zscore(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        die("MAD=0 in robust z-score (DFM distribution too degenerate).")
    return (x - med) / (1.4826 * mad)


# -----------------------------
# INS patching (reused logic)
# -----------------------------
def modify_ins_file(file_path: Path) -> None:
    """
    Reuses your old logic: between UNIT and FVAR, remove old MERG/FMAPP/ACTA,
    and enforce LIST 4, MERG 0, FMAP 2, ACTA (when encountering LIST).
    """
    modified_lines: List[str] = []
    is_between_unit_fvar = False

    lines = file_path.read_text().splitlines(True)

    for line in lines:
        line_lower = line.lower().strip()

        if "unit" in line_lower:
            is_between_unit_fvar = True
            modified_lines.append(line)
            continue
        elif "fvar" in line_lower:
            is_between_unit_fvar = False
            modified_lines.append(line)
            continue

        if is_between_unit_fvar:
            # remove some conflicting directives (same as old script)
            if "merg" in line_lower and "merge" not in line_lower:
                continue
            if "fmap" in line_lower:
                continue
            if "acta" in line_lower:
                continue
            if line_lower.startswith("list"):
                modified_lines.append("LIST 4\nMERG 0\nFMAP 2\nACTA\n")
                continue

        modified_lines.append(line)

    file_path.write_text("".join(modified_lines))


# -----------------------------
# Parse FVAR + R1 from .res
# -----------------------------
def parse_fvar_and_r1_from_res(res_path: Path) -> Tuple[float, Optional[float], Optional[float]]:
    fvar = None
    r1_4sig = None
    r1_all = None

    fvar_re = re.compile(r"^\s*FVAR\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)", re.IGNORECASE)
    r1_re = re.compile(
        r"R1\s*=\s*([0-9]*\.?[0-9]+)\s*for.*?Fo\s*>\s*4sig.*?and\s*([0-9]*\.?[0-9]+)\s*for\s*all",
        re.IGNORECASE,
    )

    for line in res_path.read_text().splitlines():
        m = fvar_re.match(line)
        if m and fvar is None:
            fvar = float(m.group(1))

        if "R1" in line.upper():
            m2 = r1_re.search(line)
            if m2:
                r1_4sig = float(m2.group(1))
                r1_all = float(m2.group(2))

    if fvar is None:
        die(f"Could not find FVAR in {res_path}")

    return fvar, r1_4sig, r1_all


# -----------------------------
# XDS_ASCII_NEM.HKL generation (sigma recompute)  [reused logic]
# -----------------------------
def format_line(parts: List[str], column_widths: List[int]) -> str:
    return "".join(f"{p:>{w}s}" for p, w in zip(parts, column_widths))


def read_xds_ascii_hkls(xds_ascii_path: Path) -> List[HKL]:
    hkls: List[HKL] = []
    for line in xds_ascii_path.read_text().splitlines():
        if line.startswith("!"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            h, k, l = map(int, parts[:3])
        except ValueError:
            continue
        hkls.append((h, k, l))
    return hkls


def create_xds_ascii_nem_all(run_dir: Path, xds_ascii_path: Path) -> Path:
    """
    Create XDS_ASCII_NEM.HKL in run_dir from:
      - run_dir/INTEGRATE.HKL
      - XDS_ASCII.HKL
    with sigma = (I_xds / I_inte) * sig_inte  (same as old script)
    """
    integrate_path = run_dir / "INTEGRATE.HKL"
    out_path = run_dir / "XDS_ASCII_NEM.HKL"

    if not integrate_path.exists():
        die(f"Missing {integrate_path}")
    if not xds_ascii_path.exists():
        die(f"Missing {xds_ascii_path}")

    # map (h,k,l) -> [ ... floats ... ] from INTEGRATE.HKL split tokens
    integrate_dict: Dict[HKL, List[float]] = {}
    with integrate_path.open("r") as f:
        for line in f:
            if line.startswith("!"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                h, k, l = map(int, parts[:3])
                vals = [float(p) for p in parts]
            except ValueError:
                continue
            integrate_dict[(h, k, l)] = vals

    # keep all HKLs found in XDS_ASCII.HKL
    keep_set = set(read_xds_ascii_hkls(xds_ascii_path))

    column_widths = [6, 6, 6, 11, 11, 8, 8, 9, 10, 4, 4, 8]
    out_lines: List[str] = []

    with xds_ascii_path.open("r") as f:
        for line in f:
            if line.startswith("!"):
                out_lines.append(line)
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            try:
                h, k, l = map(int, parts[:3])
            except ValueError:
                continue

            hkl = (h, k, l)
            if hkl not in keep_set:
                continue

            if hkl in integrate_dict:
                try:
                    xds_I = float(parts[3])
                    inte_vals = integrate_dict[hkl]
                    if inte_vals[3] == 0:
                        die(f"INTEGRATE I=0 for {hkl}, cannot recompute sigma for NEM")
                    calculated_sigma = (xds_I / inte_vals[3]) * inte_vals[4]
                    parts[4] = f"{calculated_sigma:.3E}"
                except Exception as e:
                    die(f"Failed sigma recompute for {hkl}: {e}")

            if len(parts) == len(column_widths):
                out_lines.append(format_line(parts, column_widths) + "\n")
            else:
                out_lines.append(" ".join(parts) + "\n")

    out_path.write_text("".join(out_lines))
    return out_path


def write_xdsconv_inp(run_dir: Path) -> Path:
    """
    XDSCONV.INP:
      XDS_ASCII_NEM.HKL -> SHELX.HKL (HKLF4)
    """
    p = run_dir / "XDSCONV.INP"
    p.write_text(
        "INPUT_FILE=XDS_ASCII_NEM.HKL\n"
        "OUTPUT_FILE=SHELX.hkl SHELX\n"
        "FRIEDEL'S_LAW=FALSE\n"
        "MERGE=FALSE\n"
    )
    return p


# -----------------------------
# Read SHELX.HKL observed data (Fo^2 and sigma) for *all* reflections in that file
# -----------------------------
def read_shelx_hkl_observations(hkl_path: Path) -> Tuple[List[HKL], np.ndarray, np.ndarray]:
    hkls: List[HKL] = []
    fo2: List[float] = []
    sig: List[float] = []

    with hkl_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            if line.lstrip().startswith(("!", "#")):
                continue

            # Prefer strict SHELX fixed columns: 3I4,2F8.2
            try:
                h = int(line[0:4]); k = int(line[4:8]); l = int(line[8:12])
                if (h, k, l) == (0, 0, 0):
                    continue
                I = float(line[12:20]); s = float(line[20:28])
            except Exception:
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    h, k, l = map(int, parts[:3])
                    if (h, k, l) == (0, 0, 0):
                        continue
                    I = float(parts[3]); s = float(parts[4])
                except Exception:
                    continue

            hkls.append((h, k, l))
            fo2.append(I)
            sig.append(s)

    if len(hkls) == 0:
        die(f"No reflections parsed from {hkl_path}")

    return hkls, np.array(fo2, dtype=float), np.array(sig, dtype=float)


# -----------------------------
# CCTBX: compute Fc^2 for arbitrary HKLs from refined .res
# -----------------------------
def compute_fc2_map_from_res(res_path: Path, hkls_unique: List[HKL], anomalous_flag: bool = True) -> Dict[HKL, float]:
    """
    Uses iotbx.shelx.cctbx_xray_structure_from to parse SHELX .res into a cctbx.xray.structure,
    then computes f_calc on the provided HKLs using direct summation, returns Fc^2 map.
    """
    try:
        import iotbx.shelx
        from cctbx import miller
        from scitbx.array_family import flex
        from cctbx.xray.structure_factors.from_scatterers_direct import from_scatterers_direct
    except Exception as e:
        die(f"CCTBX/iotbx imports failed: {e}")

    if not res_path.exists():
        die(f"Missing refined model: {res_path}")

    # NOTE: cctbx_xray_structure_from signature includes an unused first arg ("cls") in iotbx.shelx source.
    xray_structure = iotbx.shelx.cctbx_xray_structure_from(None, filename=str(res_path))
    cs = xray_structure.crystal_symmetry()

    indices_flex = flex.miller_index(hkls_unique)
    mset = miller.set(crystal_symmetry=cs, indices=indices_flex, anomalous_flag=anomalous_flag)

    calc = from_scatterers_direct(xray_structure=xray_structure, miller_set=mset, algorithm="direct")
    f_calc = calc.f_calc()  # miller.array (complex)
    fc2 = flex.abs(f_calc.data()) ** 2

    return {hkl: float(fc2[i]) for i, hkl in enumerate(hkls_unique)}


# -----------------------------
# DFM + optimize u (mean == median)
# -----------------------------

# -----------------------------
# INTEGRATE.HKL reader
# -----------------------------
class Integrate:
    def __init__(self, file_name):
        self.inte = any_reflection_file(str(file_name)).as_miller_arrays(merge_equivalents=False)
        self.inte0 = self.inte[0]
        self.inte1 = self.inte[1]
        self.inte2 = self.inte[2]
        self.df = pd.DataFrame()

    def indices(self):
        return list(self.inte0.indices())

    def data(self):
        return list(self.inte0.data())

    def sigmas(self):
        return list(self.inte0.sigmas())

    def xobs(self):
        return np.array(self.inte1.data())[:, 0]

    def yobs(self):
        return np.array(self.inte1.data())[:, 1]

    def zobs(self):
        return np.array(self.inte1.data())[:, 2]

    def d_spacings(self):
        return list(self.inte0.d_spacings().data())

    def asus(self):
        return list(self.inte0.map_to_asu().indices())

    def as_df(self) -> pd.DataFrame:
        self.df = pd.DataFrame()
        self.df["Miller"] = self.indices()
        self.df["asu"] = self.asus()
        self.df["Intensity"] = self.data()
        self.df["Sigma"] = self.sigmas()
        self.df["I/Sigma"] = self.df["Intensity"] / self.df["Sigma"]
        self.df["Resolution"] = self.d_spacings()
        self.df["xobs"] = self.xobs()
        self.df["yobs"] = self.yobs()
        self.df["zobs"] = self.zobs()
        self.df["Index_INTE"] = np.arange(0, len(self.df.index), 1)
        return self.df

    def print_hklf4(self, outp_name: str):
        stdout_obj = sys.stdout
        sys.stdout = open(outp_name + ".hkl", "w")
        hklf(self.inte0)
        sys.stdout = stdout_obj

def calculate_dfm(u, sample_df):
    return (sample_df["Fo^2"] - sample_df["Fc^2"]) / np.sqrt(
        sample_df["Fo^2_sigma"] ** 2 + (2 * u * sample_df["Fc^2"]) ** 2
    )

def objective_function(u, sample_df):
    dfm_values = calculate_dfm(u, sample_df)
    return abs(np.mean(dfm_values) - np.median(dfm_values))

def read_fcf(file_path):
    miller_indices = []
    Fc2 = []
    Fo2 = []
    sigmas = []
    header_end = False

    with open(file_path, "r") as file:
        for line in file:
            if header_end:
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        h, k, l = map(int, parts[:3])
                        Fc2_value, Fo2_value, sigma_value = map(float, parts[3:6])
                    except ValueError:
                        continue
                    miller_indices.append((h, k, l))
                    Fc2.append(Fc2_value)
                    Fo2.append(Fo2_value)
                    sigmas.append(sigma_value)
            elif line.startswith("loop_"):
                header_end = True

    return pd.DataFrame(
        {"Miller": miller_indices, "Fo^2": Fo2, "Fc^2": Fc2, "Fo^2_sigma": sigmas}
    )


def build_sample_df_with_dfm(iter_dir: str, inte_path: str, initial_u: float) -> pd.DataFrame:
    fcf_path = os.path.join(iter_dir, "SHELX.fcf")
    if not os.path.exists(fcf_path):
        die(f"Missing {fcf_path}")

    sample_fcf_df = read_fcf(fcf_path)
    sample_inte_df = Integrate(inte_path).as_df()

    sample_df = sample_fcf_df.merge(sample_inte_df, on="Miller", how="inner")
    if sample_df.empty:
        die("Merge produced 0 rows. FCF and INTEGRATE.HKL do not match in ED logic.")

    required = ["Miller", "asu", "Fc^2", "Fo^2", "Fo^2_sigma", "Resolution", "xobs", "yobs", "zobs"]
    missing = [c for c in required if c not in sample_df.columns]
    if missing:
        die(f"Missing expected columns after merge: {missing}")

    sample_df = sample_df[required].copy()

    res = minimize(objective_function, initial_u, args=(sample_df,), method="Nelder-Mead")
    optimal_u = float(res.x[0])
    if abs(initial_u - optimal_u) > initial_u + 0.001:
        optimal_u = initial_u

    sample_df["DFM"] = calculate_dfm(optimal_u, sample_df)
    sample_df["absDFM"] = np.abs(sample_df["DFM"])
    sample_df.attrs["optimal_u"] = optimal_u
    return sample_df

def dfm_from_arrays(fo2: np.ndarray, fc2: np.ndarray, sig: np.ndarray, u: float) -> np.ndarray:
    denom = np.sqrt(sig**2 + (2.0 * u * fc2) ** 2)
    if np.any(denom == 0):
        die("Encountered denom==0 in DFM calculation (sigma and Fc2 both zero for some reflection).")
    return (fo2 - fc2) / denom

def median_dfm_per_hkl(hkls: List[HKL], dfm: np.ndarray) -> Dict[HKL, float]:
    buckets: Dict[HKL, List[float]] = {}
    for hkl, v in zip(hkls, dfm):
        buckets.setdefault(hkl, []).append(float(v))
    return {hkl: float(np.median(vals)) for hkl, vals in buckets.items()}

def read_dfm_scale_csv(path: Path) -> Dict[HKL, float]:
    df = pd.read_csv(path)
    return {
        (int(r.h), int(r.k), int(r.l)): float(r.scale)
        for r in df.itertuples(index=False)
    }

# -----------------------------
# INTEGRATE.HKL scaling (preserve original tail columns)
# -----------------------------

def scale_integrate_hkl(
    integrate_in: Path,
    integrate_out: Path,
    hkl_to_scale: Dict[HKL, float],
    scale_sigma: bool,
    require_dfm: bool,
) -> None:
    missing: List[HKL] = []
    n_total = 0
    n_parsed = 0
    n_scaled = 0
    n_passthrough = 0

    with integrate_in.open("r") as fin, integrate_out.open("w") as fout:
        for line in fin:
            n_total += 1

            if line.startswith("!") or (not line.strip()):
                fout.write(line)
                continue

            m = _INTE_RE.match(line)
            if not m:
                # Could be a non-reflection line or a format you don't want to touch
                fout.write(line)
                n_passthrough += 1
                continue

            n_parsed += 1

            try:
                h = int(m.group(2))
                k = int(m.group(3))
                l = int(m.group(4))
                I = float(m.group(5))
                s = float(m.group(6))
            except Exception:
                fout.write(line)
                n_passthrough += 1
                continue

            hkl = (h, k, l)

            if hkl not in hkl_to_scale:
                if require_dfm:
                    missing.append(hkl)
                fout.write(line)
                continue

            scale = float(hkl_to_scale[hkl])
            I2 = I * scale
            s2 = s * scale 

            # Replace I and sigma *in place*, preserving all spacing and tail columns
            i_start, i_end = m.start(5), m.end(5)
            s_start, s_end = m.start(6), m.end(6)

            I_field_w = i_end - i_start
            s_field_w = s_end - s_start

            I_txt = f"{I2:.3E}".rjust(I_field_w)
            s_txt = f"{s2:.3E}".rjust(s_field_w)

            new_line = line[:i_start] + I_txt + line[i_end:s_start] + s_txt + line[s_end:]
            fout.write(new_line)
            n_scaled += 1

    if missing:
        raise SystemExit(f"ERROR: Missing scale for {len(missing)} reflections (first few: {missing[:10]})")

    if n_scaled == 0:
        raise SystemExit(
            "ERROR: Scaled 0 reflection lines. Your INTEGRATE.HKL format did not match the parser, "
            "so INTEGRATE_SCALED.HKL would be identical."
        )

    print(f"[SCALE] total_lines={n_total} parsed_reflections={n_parsed} scaled_reflections={n_scaled} passthrough_lines={n_passthrough}")

# -----------------------------
# Main pipeline per round
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Iterative DFM rescaling: XDS(CORRECT) -> XDS_ASCII_NEM -> XDSCONV -> SHELXL -> CCTBX Fc2 -> DFM -> robust z -> scale INTEGRATE.HKL"
    )
    ap.add_argument("--integrate", default="INTEGRATE.HKL", help="Initial INTEGRATE.HKL (round 0)")
    ap.add_argument("--xds-inp", default="XDS.INP", help="XDS.INP (JOB=CORRECT only)")
    ap.add_argument("--shelx-ins", default="SHELX.INS", help="SHELX.INS template")
    ap.add_argument("--rounds", type=int, required=True, help="Number of rounds to run")
    ap.add_argument("--alpha", type=float, default=0.5, help="alpha in scale=exp(alpha*tanh(z))")
    ap.add_argument("--initial-u", type=float, default=0.01, help="Initial u for optimization")
    ap.add_argument("--scale-sigma", action="store_true", help="Also scale sigma in INTEGRATE.HKL")
    ap.add_argument("--out-dir", default=".", help="Where to create run_### folders (default: .)")

    ap.add_argument("--xds-cmd", default="xds", help="XDS executable (default: xds)")
    ap.add_argument("--xdsconv-cmd", default="xdsconv", help="XDSCONV executable (default: xdsconv)")
    ap.add_argument("--shelxl-cmd", default="shelxl", help="SHELXL executable (default: shelxl)")

    ap.add_argument("--no-modify-ins", action="store_true", help="Do not patch the copied INS file (LIST 4 / MERG 0 etc)")

    args = ap.parse_args()

    integrate0 = Path(args.integrate).resolve()
    xds_inp = Path(args.xds_inp).resolve()
    shelx_ins = Path(args.shelx_ins).resolve()
    out_root = Path(args.out_dir).resolve()

    for p in (integrate0, xds_inp, shelx_ins):
        if not p.exists():
            die(f"Missing input file: {p}")

    out_root.mkdir(parents=True, exist_ok=True)

    # Summary CSV across rounds
    summary_csv = out_root / "round_summary.csv"
    if not summary_csv.exists():
        with summary_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["round", "FVAR", "R1_4sig", "R1_all", "u_opt", "alpha", "scale_sigma"])

    current_integrate = integrate0

    for r in range(args.rounds):
        run_dir = out_root / f"run_{r:03d}"
        run_dir.mkdir(exist_ok=True)

        print(f"\n==================== ROUND {r} ====================")

        # Inputs into run dir
        shutil.copy(current_integrate, run_dir / "INTEGRATE.HKL")
        shutil.copy(xds_inp, run_dir / "XDS.INP")

        # SHELX base name we run is "SHELX"
        # Ensure we have SHELX.ins available
        ins_dst = run_dir / "SHELX.ins"
        shutil.copy(shelx_ins, ins_dst)
        # also keep an uppercase copy if you prefer that convention
        shutil.copy(shelx_ins, run_dir / "SHELX.INS")

        if not args.no_modify_ins:
            modify_ins_file(ins_dst)

        # 1) XDS (CORRECT): XDS.INP must be JOB=CORRECT, produces XDS_ASCII.HKL
        run_cmd([args.xds_cmd], cwd=run_dir)
        xds_ascii = run_dir / "XDS_ASCII.HKL"
        if not xds_ascii.exists():
            die(f"XDS did not produce {xds_ascii}")

        # 2) XDS_ASCII_NEM.HKL (sigma recompute)
        xds_ascii_nem = create_xds_ascii_nem_all(run_dir=run_dir, xds_ascii_path=xds_ascii)
        print(f"[OK] wrote {xds_ascii_nem.name}")

        # 3) XDSCONV: XDS_ASCII_NEM.HKL -> SHELX.HKL
        write_xdsconv_inp(run_dir)
        run_cmd([args.xdsconv_cmd], cwd=run_dir)

        shelx_hkl = run_dir / "SHELX.hkl"
        if not shelx_hkl.exists():
            die("xdsconv did not produce SHELX.hkl")
        # Ensure lower-case .hkl exists for shelxl call
        # shutil.copy(shelx_hkl, run_dir / "SHELX.HKL")

        # 4) SHELXL: SHELX.hkl + SHELX.ins -> SHELX.res + SHELX.fcf
        run_cmd([args.shelxl_cmd, "SHELX"], cwd=run_dir)

        res_path = run_dir / "SHELX.res"
        if not res_path.exists():
            die("shelxl did not produce SHELX.res")

        fcf_candidates = list(run_dir.glob("SHELX.fcf")) + list(run_dir.glob("SHELX.FCF"))
        if not fcf_candidates:
            # also accept any .fcf
            fcf_candidates = list(run_dir.glob("*.fcf")) + list(run_dir.glob("*.FCF"))
        if not fcf_candidates:
            die("shelxl did not produce any .fcf file")
        fcf_path = fcf_candidates[0]
        print(f"[OK] found FCF: {fcf_path.name}")

        # 5) Parse and print FVAR + R1 from .res
        fvar, r1_4sig, r1_all = parse_fvar_and_r1_from_res(res_path)
        print(f"[METRICS] FVAR={fvar:.5f}   R1_4sig={r1_4sig if r1_4sig is not None else 'NA'}   R1_all={r1_all if r1_all is not None else 'NA'}")

        # 6) DFM using:
        #    - Fo^2 and sigma from SHELX.HKL (full reflection list)
        #    - Fc^2 from CCTBX computed from refined .res for that full list

        sample_df = build_sample_df_with_dfm(run_dir, run_dir / "INTEGRATE.HKL", initial_u=args.initial_u)
        opt_u = float(sample_df.attrs.get("optimal_u", np.nan))
        print(f"Optimal u used: {opt_u:.6g}")

        sample_csv = os.path.join(run_dir, "sample_df_with_dfm.csv")
        sample_df.to_csv(sample_csv, index=False)
        print(f"Saved: {sample_csv}")

        z = robust_zscore(sample_df["DFM"].to_numpy(dtype=float))

        keys = list(sample_df["Miller"])
        scales = np.exp(-(args.alpha * np.tanh(z)))
        hkl_to_scale: Dict[HKL, float] = {hkl: float(scales[i]) for i, hkl in enumerate(keys)}

        per_csv = run_dir / "dfm_per_hkl.csv"
        with per_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["h", "k", "l", "DFM", "z_robust", "scale"])
            for i, hkl in enumerate(keys):
                w.writerow([hkl[0], hkl[1], hkl[2], sample_df["DFM"].iloc[i], z[i], hkl_to_scale[hkl]])

        # 7) Apply scaling to INTEGRATE.HKL -> INTEGRATE_SCALED.HKL
        integrate_in = run_dir / "INTEGRATE.HKL"
        integrate_out = run_dir / "INTEGRATE_SCALED.HKL"
        scale_integrate_hkl(
            integrate_in=integrate_in,
            integrate_out=integrate_out,
            hkl_to_scale=hkl_to_scale,
            scale_sigma=args.scale_sigma,
            require_dfm=True,   # per your requirement: if no DFM something is wrong
        )
        print(f"[WRITE] {integrate_out.name}   (alpha={args.alpha}, scale_sigma={args.scale_sigma})")

        # 8) Append summary row
        with summary_csv.open("a", newline="") as f:
            w = csv.writer(f)
            w.writerow([r, fvar, r1_4sig, r1_all, opt_u, args.alpha, args.scale_sigma])

        # 9) Next round uses scaled integrate
        current_integrate = integrate_out

    print("\nDone.")
    print(f"Summary: {summary_csv}")


if __name__ == "__main__":
    main()
# python ./dfm_scaling.py --integrate /Users/xiaodong/Desktop/3DED-DATA/LTA/LTA1/xds/INTEGRATE.HKL --xds-inp /Users/xiaodong/Desktop/3DED-DATA/LTA/LTA1/xds/XDS.INP --shelx-ins /Users/xiaodong/Desktop/3DED-DATA/LTA/LTA1/shelx/t1_no-error-model.ins --rounds 10 --out-dir /Users/xiaodong/Desktop/3DED-DATA/LTA/LTA1