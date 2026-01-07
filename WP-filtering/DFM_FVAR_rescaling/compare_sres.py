#!/usr/bin/env python3
import argparse
import os
import sys
import re
from typing import Optional, Tuple, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

# CCTBX imports
from iotbx.reflection_file_reader import any_reflection_file


def die(msg: str, code: int = 2):
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


class Integrate:
    def __init__(self, file_name: str):
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


def calculate_sres(sample_df: pd.DataFrame) -> pd.Series:
    # Standardized residual in Fo^2 space
    return (sample_df["Fo^2"] - sample_df["Fc^2"]) / sample_df["Fo^2_sigma"]


def read_fcf(file_path: str) -> pd.DataFrame:
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


def find_single_file(iter_dir: str, pattern: str) -> str:
    files = list(Path(iter_dir).glob(pattern))
    if len(files) != 1:
        raise RuntimeError(f"Expected exactly one '{pattern}' in {iter_dir}, found {len(files)}")
    return str(files[0])


def build_refinement_df_with_sres(
    refine_dir: str,
    integrate_df: pd.DataFrame,
) -> pd.DataFrame:
    refine_dir = os.path.abspath(refine_dir)

    fcf_files = list(Path(refine_dir).glob("*.fcf"))
    if len(fcf_files) != 1:
        raise RuntimeError(f"Expected exactly one .fcf in {refine_dir}, found {len(fcf_files)}")
    fcf_path = str(fcf_files[0])
    if not os.path.exists(fcf_path):
        die(f"Missing {fcf_path}")

    fcf_df = read_fcf(fcf_path)

    df = fcf_df.merge(integrate_df, on="Miller", how="inner")
    if df.empty:
        die(f"Merge produced 0 rows for {refine_dir}. FCF and INTEGRATE.HKL do not match.")

    required = ["Miller", "asu", "Fc^2", "Fo^2", "Fo^2_sigma", "Resolution", "xobs", "yobs", "zobs"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        die(f"Missing expected columns after merge for {refine_dir}: {missing}")

    df = df[required].copy()
    df["SRES"] = calculate_sres(df)
    df["absSRES"] = np.abs(df["SRES"])

    dup = df["Miller"].duplicated().sum()
    if dup > 0:
        print(f"WARNING: {refine_dir} has {dup} duplicate Miller rows after merge.", file=sys.stderr)

    return df


def build_compare_df(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    a = df_a.copy()
    b = df_b.copy()

    a = a.rename(columns={
        "Fo^2": "Fo^2_A",
        "Fc^2": "Fc^2_A",
        "Fo^2_sigma": "Fo^2_sigma_A",
        "SRES": "SRES_A",
        "absSRES": "absSRES_A",
    })
    b = b.rename(columns={
        "Fo^2": "Fo^2_B",
        "Fc^2": "Fc^2_B",
        "Fo^2_sigma": "Fo^2_sigma_B",
        "SRES": "SRES_B",
        "absSRES": "absSRES_B",
    })

    keep_a = ["Miller", "asu", "Resolution", "xobs", "yobs", "zobs",
              "Fo^2_A", "Fc^2_A", "Fo^2_sigma_A", "SRES_A", "absSRES_A"]
    keep_b = ["Miller",
              "Fo^2_B", "Fc^2_B", "Fo^2_sigma_B", "SRES_B", "absSRES_B"]

    cmp_df = a[keep_a].merge(b[keep_b], on="Miller", how="inner")
    if cmp_df.empty:
        die("No overlapping Miller indices between refinement A and B.")

    cmp_df["dSRES"] = cmp_df["SRES_A"] - cmp_df["SRES_B"]
    cmp_df["abs_dSRES"] = np.abs(cmp_df["dSRES"])
    return cmp_df


def write_compare_report(out_path: str, cmp_df: pd.DataFrame, top_n: int = 100):
    cols = ["Miller", "asu", "Resolution", "zobs", "SRES_A", "SRES_B", "dSRES", "abs_dSRES"]
    missing = [c for c in cols if c not in cmp_df.columns]
    if missing:
        raise RuntimeError(f"Cannot write compare report: missing columns in df: {missing}")

    top = cmp_df.sort_values("abs_dSRES", ascending=False).head(top_n)[cols].copy()

    with open(out_path, "w") as out:
        out.write("SRES comparison report (A vs B)\n")
        out.write("=" * 60 + "\n\n")
        out.write(f"Overlapping reflections: {len(cmp_df)}\n")
        out.write(f"Top {top_n} reflections by |dSRES| = |SRES_A - SRES_B|\n")
        out.write("-" * 60 + "\n")
        out.write(top.to_string(index=False))
        out.write("\n")


def parse_shelx_hkl_line(line: str):
    """
    Parse a standard SHELXL HKL line: h k l I sig
    Returns (h,k,l,I,sig) or None if not parseable.
    """
    s = line.strip()
    if not s:
        return None
    parts = s.split()
    if len(parts) < 5:
        return None
    try:
        h = int(parts[0]); k = int(parts[1]); l = int(parts[2])
        I = float(parts[3]); sig = float(parts[4])
        return h, k, l, I, sig
    except ValueError:
        return None


def format_shelx_hkl_line(h: int, k: int, l: int, I: float, sig: float) -> str:
    # Common fixed-width formatting
    return f"{h:4d}{k:4d}{l:4d}{I:12.2f}{sig:12.2f}\n"


def write_sigma_scaled_hkl(
    input_hkl: str,
    output_hkl: str,
    abs_dSRES_by_miller: Dict[Tuple[int, int, int], float],
    scale_mode: str = "1+abs",
):
    """
    Create a new HKL where sigma is inflated using abs_dSRES for matching Miller indices.

    scale_mode:
      - "1+abs": sigma_new = sigma_old * (1 + abs_dSRES)   [default]
      - "abs":   sigma_new = sigma_old * abs_dSRES
    """
    if scale_mode not in {"1+abs", "abs"}:
        raise ValueError(f"Unknown scale_mode: {scale_mode}")
    
    # Normalize abs_dSRES_by_miller by its max value (so values are in [0, 1])
    if abs_dSRES_by_miller:
        max_abs = max(abs_dSRES_by_miller.values())
        if max_abs > 0.0:
            abs_dSRES_by_miller = {k: (v / max_abs) for k, v in abs_dSRES_by_miller.items()}
        else:
            abs_dSRES_by_miller = {k: 0.0 for k in abs_dSRES_by_miller}

    n_total = 0
    n_scaled = 0
    n_missing = 0

    with open(input_hkl, "r", errors="replace") as fin, open(output_hkl, "w") as fout:
        for line in fin:
            parsed = parse_shelx_hkl_line(line)
            if parsed is None:
                # Preserve non-standard lines as-is
                fout.write(line)
                continue

            h, k, l, I, sig = parsed
            n_total += 1

            # Keep termination line unchanged
            if h == 0 and k == 0 and l == 0:
                fout.write(line)
                continue

            key = (h, k, l)
            if key not in abs_dSRES_by_miller:
                n_missing += 1
                fout.write(format_shelx_hkl_line(h, k, l, I, sig))
                continue

            a = float(abs_dSRES_by_miller[key])

            if scale_mode == "1+abs":
                factor = 1.0 + a
            else:
                factor = a

            sig_new = sig * factor
            n_scaled += 1
            fout.write(format_shelx_hkl_line(h, k, l, I, sig_new))

    print(f"Sigma-scaled HKL written: {output_hkl}")
    print(f"HKL lines parsed (incl. 0 0 0): {n_total}")
    print(f"Scaled reflections: {n_scaled}")
    print(f"No abs_dSRES available (left unchanged): {n_missing}")


def main():
    ap = argparse.ArgumentParser(
        description="Compare per-reflection SRES between two refinements and write an HKL with sigma scaled by abs_dSRES."
    )
    ap.add_argument("--refine-dir-a", required=True, help="Directory containing one .fcf (Refinement A)")
    ap.add_argument("--refine-dir-b", required=True, help="Directory containing one .fcf (Refinement B)")
    ap.add_argument("--integrate-hkl", required=True, help="Path to INTEGRATE.HKL")

    # ap.add_argument(
    #     "--hkl-to-scale",
    #     required=True,
    #     help="Existing SHELXL .hkl to scale (sigma column will be modified).",
    # )

    ap.add_argument(
        "--scale-mode",
        default="1+abs",
        choices=["1+abs", "abs"],
        help="How to scale sigma with abs_dSRES. Default: sigma *= (1 + abs_dSRES).",
    )

    ap.add_argument(
        "--out-prefix",
        default=None,
        help="Prefix for outputs (default: <refine-dir-a>/<A>_vs_<B>)",
    )

    ap.add_argument(
        "--sres-ymin", type=float, default=None,
        help="Optional fixed SRES y-axis minimum for SRES vs zobs plots"
    )
    ap.add_argument(
        "--sres-ymax", type=float, default=None,
        help="Optional fixed SRES y-axis maximum for SRES vs zobs plots"
    )

    args = ap.parse_args()

    refine_dir_a = os.path.abspath(args.refine_dir_a)
    refine_dir_b = os.path.abspath(args.refine_dir_b)
    integrate_hkl = os.path.abspath(args.integrate_hkl)
    # hkl_to_scale = os.path.abspath(args.hkl_to_scale)

    if not os.path.isdir(refine_dir_a):
        die(f"Not a directory: {refine_dir_a}")
    if not os.path.isdir(refine_dir_b):
        die(f"Not a directory: {refine_dir_b}")
    if not os.path.exists(integrate_hkl):
        die(f"Missing INTEGRATE.HKL: {integrate_hkl}")
    # if not os.path.exists(hkl_to_scale):
    #     die(f"Missing HKL to scale: {hkl_to_scale}")

    base_a = os.path.basename(refine_dir_a.rstrip("/"))
    base_b = os.path.basename(refine_dir_b.rstrip("/"))

    out_prefix = args.out_prefix or os.path.join(refine_dir_a, f"{base_a}_vs_{base_b}")

    # Read INTEGRATE.HKL once
    integrate_df = Integrate(integrate_hkl).as_df()

    # Build per-refinement dfs
    df_a = build_refinement_df_with_sres(refine_dir_a, integrate_df)
    df_b = build_refinement_df_with_sres(refine_dir_b, integrate_df)

    # Build compare df
    cmp_df = build_compare_df(df_a, df_b)

    # Outputs (tables)
    compare_txt = f"{out_prefix}_SRES_compare_summary.txt"
    write_compare_report(compare_txt, cmp_df, top_n=100)
    print(f"Wrote: {compare_txt}")

    out_tsv = f"{out_prefix}_SRES_compare_table.tsv"
    cmp_df.to_csv(out_tsv, sep="\t", index=False)
    print(f"Wrote: {out_tsv}")

    # Plots
    fig_a = px.scatter(
        df_a,
        x="zobs",
        y="SRES",
        title=f"SRES vs zobs (A: {base_a})",
        labels={"zobs": "Frame (zobs)", "SRES": "SRES"},
        hover_data=["Miller", "asu", "Resolution", "Fo^2", "Fc^2", "Fo^2_sigma"],
    )
    if args.sres_ymin is not None or args.sres_ymax is not None:
        fig_a.update_yaxes(range=[args.sres_ymin, args.sres_ymax])
    out_a_html = f"{out_prefix}_A_SRES_vs_zobs.html"
    fig_a.write_html(out_a_html)
    print(f"Wrote: {out_a_html} (Rows plotted: {len(df_a)})")

    fig_b = px.scatter(
        df_b,
        x="zobs",
        y="SRES",
        title=f"SRES vs zobs (B: {base_b})",
        labels={"zobs": "Frame (zobs)", "SRES": "SRES"},
        hover_data=["Miller", "asu", "Resolution", "Fo^2", "Fc^2", "Fo^2_sigma"],
    )
    if args.sres_ymin is not None or args.sres_ymax is not None:
        fig_b.update_yaxes(range=[args.sres_ymin, args.sres_ymax])
    out_b_html = f"{out_prefix}_B_SRES_vs_zobs.html"
    fig_b.write_html(out_b_html)
    print(f"Wrote: {out_b_html} (Rows plotted: {len(df_b)})")

    fig_cmp = px.scatter(
        cmp_df,
        x="SRES_A",
        y="SRES_B",
        title=f"SRES per reflection: A ({base_a}) vs B ({base_b})",
        labels={"SRES_A": "SRES (A)", "SRES_B": "SRES (B)"},
        hover_data=["Miller", "asu", "Resolution", "zobs", "dSRES", "abs_dSRES"],
    )
    out_cmp_html = f"{out_prefix}_SRES_A_vs_SRES_B.html"
    fig_cmp.write_html(out_cmp_html)
    print(f"Wrote: {out_cmp_html} (Overlapping reflections: {len(cmp_df)})")

    fig_d = px.scatter(
        cmp_df,
        x="zobs",
        y="dSRES",
        title=f"dSRES vs zobs (A - B): {base_a} minus {base_b}",
        labels={"zobs": "Frame (zobs)", "dSRES": "dSRES = SRES_A - SRES_B"},
        hover_data=["Miller", "asu", "Resolution", "SRES_A", "SRES_B", "abs_dSRES"],
    )
    out_d_html = f"{out_prefix}_dSRES_vs_zobs.html"
    fig_d.write_html(out_d_html)
    print(f"Wrote: {out_d_html}")

    # === Write sigma-scaled HKL ===
    abs_map = {tuple(m): float(a) for m, a in zip(cmp_df["Miller"].tolist(), cmp_df["abs_dSRES"].tolist())}


    hkl_to_scale = list(Path(args.refine_dir_a).glob("*.hkl"))
    if len(hkl_to_scale) != 1:
        raise RuntimeError(f"Expected exactly one .hkl in {args.refine_dir_a}, found {len(hkl_to_scale)}")
    hkl_to_scale = str(hkl_to_scale[0])
    if not os.path.exists(hkl_to_scale):
        die(f"Missing {hkl_to_scale}")

    in_base = os.path.splitext(os.path.basename(hkl_to_scale))[0]
    out_scaled_hkl = os.path.join(os.path.dirname(out_prefix), f"{in_base}_sres_scaled.hkl")
    # If out_prefix includes a directory path, keep outputs alongside plots:
    # plots are written as f"{out_prefix}_....", so directory is dirname(out_prefix)
    out_dir = os.path.dirname(out_prefix)
    out_scaled_hkl = os.path.join(out_dir, f"{in_base}_sres_scaled.hkl")

    write_sigma_scaled_hkl(
        input_hkl=hkl_to_scale,
        output_hkl=out_scaled_hkl,
        abs_dSRES_by_miller=abs_map,
        scale_mode=args.scale_mode,
    )


if __name__ == "__main__":
    main()


# python3 compare_sres.py --refine-dir-a /Users/xiaodong/Desktop/3DED-DATA/LTA/LTA4-with-EADP-EXYZ/shelx_iso --refine-dir-b /Users/xiaodong/Desktop/3DED-DATA/LTA/LTA4-with-EADP-EXYZ/shelx_anis --integrate-hkl /Users/xiaodong/Desktop/3DED-DATA/LTA/LTA4-with-EADP-EXYZ/xds/INTEGRATE.HKL
