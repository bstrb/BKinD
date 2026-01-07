#!/usr/bin/env python3
import argparse
import os
import sys
import re
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import plotly.express as px

from scipy.optimize import minimize
from pathlib import Path


# CCTBX imports
from iotbx.reflection_file_reader import any_reflection_file
from iotbx.shelx.hklf import miller_array_export_as_shelx_hklf as hklf
# from scitbx.array_family import flex

class Integrate:
    def __init__(self, file_name):
        self.inte = any_reflection_file(file_name).as_miller_arrays(merge_equivalents=False)
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

def calculate_sres(sample_df):
    return (sample_df["Fo^2"] - sample_df["Fc^2"]) / sample_df["Fo^2_sigma"] 

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

def die(msg: str, code: int = 2):
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)

def build_sample_df_with_sres(iter_dir: str, inte_path: str) -> pd.DataFrame:
    fcf_files = list(Path(iter_dir).glob("*.fcf"))
    if len(fcf_files) != 1:
        raise RuntimeError(f"Expected exactly one .fcf in {iter_dir}, found {len(fcf_files)}")

    fcf_path = str(fcf_files[0])
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

    sample_df["SRES"] = calculate_sres(sample_df)
    sample_df["absSRES"] = np.abs(sample_df["SRES"])
    
    return sample_df

def find_single_file(iter_dir: str, pattern: str) -> str:
    files = list(Path(iter_dir).glob(pattern))
    if len(files) != 1:
        raise RuntimeError(f"Expected exactly one '{pattern}' in {iter_dir}, found {len(files)}")
    return str(files[0])

def parse_res_fvar_r1(res_path: str) -> Tuple[Optional[str], Optional[float]]:
    """
    Return (FVAR_line, R1_value).
    - FVAR_line: the full FVAR line as a string (or None if not found)
    - R1_value: float if found anywhere in file like 'R1 = 0.1234' (or None)
    """
    fvar_line = None
    r1_val = None

    # Common patterns seen in SHELXL outputs/logs:
    # "R1 = 0.1234" (sometimes in REM lines, sometimes elsewhere)
    r1_re = re.compile(r"\bR1\b\s*=?\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)

    with open(res_path, "r", errors="replace") as f:
        for line in f:
            s = line.strip()

            if s.upper().startswith("FVAR"):
                # Keep the first FVAR we see (usually what you want)
                if fvar_line is None:
                    fvar_line = s

            # Try to find an R1 in any line
            m = r1_re.search(s)
            if m and r1_val is None:
                try:
                    r1_val = float(m.group(1))
                except ValueError:
                    pass

    return fvar_line, r1_val


def extract_lst_blocks(lst_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract:
      (1) The 'Principal mean square atomic displacements U' section (until the warning line + following blank)
      (2) The 'Analysis of variance for reflections employed in refinement' section (until Recommended weighting scheme)
    Returns (u_block, anova_block) as strings or None if not found.
    """
    with open(lst_path, "r", errors="replace") as f:
        lines = f.readlines()

    u_start = None
    anova_start = None

    for i, line in enumerate(lines):
        if u_start is None and "Principal mean square atomic displacements U" in line:
            u_start = i
        if anova_start is None and "Analysis of variance for reflections employed in refinement" in line:
            anova_start = i

    u_block = None
    if u_start is not None:
        # Heuristic end: stop after we pass a warning line and then hit a blank line,
        # or after a reasonable max span.
        end = min(len(lines), u_start + 250)
        warning_seen = False
        u_end = end

        for j in range(u_start, end):
            if "** Warning:" in lines[j]:
                warning_seen = True
            if warning_seen and lines[j].strip() == "":
                u_end = j
                break

        u_block = "".join(lines[u_start:u_end]).rstrip() + "\n"

    anova_block = None
    if anova_start is not None:
        end = min(len(lines), anova_start + 600)
        anova_end = end
        for j in range(anova_start, end):
            if "Recommended weighting scheme" in lines[j]:
                # include that line + one following line if present
                anova_end = min(len(lines), j + 2)
                break
        anova_block = "".join(lines[anova_start:anova_end]).rstrip() + "\n"

    return u_block, anova_block


def write_summary_report(
    out_path: str,
    df: pd.DataFrame,
    res_path: str,
    lst_path: str,
    top_n: int = 20,
):
    # Top reflections by |SRES|
    cols = ["Miller", "asu", "Resolution", "Fo^2", "Fc^2", "Fo^2_sigma", "SRES", "absSRES", "zobs"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Cannot write summary: missing columns in df: {missing}")

    top = df.sort_values("absSRES", ascending=False).head(top_n)[cols].copy()

    fvar_line, r1_val = parse_res_fvar_r1(res_path)
    u_block, anova_block = extract_lst_blocks(lst_path)

    with open(out_path, "w") as out:
        out.write("SRES summary report\n")
        out.write("=" * 60 + "\n\n")

        out.write(f"Refinement dir: {os.path.dirname(out_path)}\n")

        out.write("From .res\n")
        out.write("-" * 60 + "\n")
        out.write(f"RES file: {res_path}\n")
        out.write(f"FVAR: {fvar_line if fvar_line is not None else 'NOT FOUND'}\n")
        out.write(f"R1: {r1_val if r1_val is not None else 'NOT FOUND'}\n\n")

        out.write("From .lst\n")
        out.write("-" * 60 + "\n")
        out.write(f"LST file: {lst_path}\n\n")

        out.write("Principal mean square atomic displacements U\n")
        out.write("-" * 60 + "\n")
        out.write(u_block if u_block is not None else "NOT FOUND\n")
        out.write("\n")

        out.write("Analysis of variance for reflections employed in refinement\n")
        out.write("-" * 60 + "\n")
        out.write(anova_block if anova_block is not None else "NOT FOUND\n")
        out.write("\n")

        out.write(f"Top {top_n} reflections by |SRES|\n")
        out.write("-" * 60 + "\n")
        # Write as a nice fixed-width table
        out.write(top.to_string(index=False))
        out.write("\n")

def main():
    ap = argparse.ArgumentParser(description="Plot SRES vs zobs for one refinement.")
    ap.add_argument("--refine-dir", required=True, help="Directory containing bkind.fcf")
    ap.add_argument("--integrate-hkl", required=True, help="Path to INTEGRATE.HKL")
    ap.add_argument("--out", default=None, help="Output HTML (default: <refine-dir>/SRES_vs_zobs.html)")
    args = ap.parse_args()

    refine_dir = os.path.abspath(args.refine_dir)
    integrate_hkl = os.path.abspath(args.integrate_hkl)

    out_html = args.out or os.path.join(refine_dir, f"{os.path.basename(refine_dir)}_SRES_vs_zobs.html")

    df = build_sample_df_with_sres(refine_dir, integrate_hkl)
    # Find .res and .lst in the same refinement directory
    res_path = find_single_file(refine_dir, "*.res")
    lst_path = find_single_file(refine_dir, "*.lst")

    summary_path = os.path.join(refine_dir, f"{os.path.basename(refine_dir)}_SRES_summary.txt")
    write_summary_report(
        out_path=summary_path,
        df=df,
        res_path=res_path,
        lst_path=lst_path,
        top_n=100,
    )
    print(f"Wrote: {summary_path}")

    fig = px.scatter(
        df,
        x="zobs",
        # x="Resolution",
        y="SRES",
        title=f"SRES vs zobs{' (Refine dir: ' + os.path.basename(refine_dir) + ')'}",
        labels={"zobs": "Frame (zobs)", "SRES": "SRES"},
        hover_data=["Miller", "asu", "Resolution", "Fo^2", "Fc^2", "Fo^2_sigma"],
    )

    # fig.update_yaxes(range=[-2500, 500])

    fig.write_html(out_html)
    print(f"Wrote: {out_html}")
    print(f"Rows plotted: {len(df)}")

if __name__ == "__main__":
    main()


# python ./plot_sres_vs_zobs.py --refine-dir  /Users/xiaodong/Desktop/3DED-DATA/LTA/LTA4-with-EADP-EXYZ/shelx_anis --integrate-hkl /Users/xiaodong/Desktop/3DED-DATA/LTA/LTA4-with-LTA1-ADPS/xds/INTEGRATE.HKL