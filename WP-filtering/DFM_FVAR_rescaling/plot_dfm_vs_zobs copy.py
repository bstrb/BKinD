#!/usr/bin/env python3
import argparse
import os
import sys
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

def die(msg: str, code: int = 2):
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)

def build_sample_df_with_dfm(iter_dir: str, inte_path: str, initial_u: float) -> pd.DataFrame:
    # fcf_path = os.path.join(iter_dir, "bkind.fcf")
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

    res = minimize(objective_function, initial_u, args=(sample_df,), method="Nelder-Mead")
    optimal_u = float(res.x[0])
    if abs(initial_u - optimal_u) > initial_u + 0.001:
        optimal_u = initial_u

    optimal_u = 0

    sample_df["DFM"] = calculate_dfm(optimal_u, sample_df)
    sample_df["absDFM"] = np.abs(sample_df["DFM"])
    sample_df.attrs["optimal_u"] = optimal_u
    return sample_df


def main():
    ap = argparse.ArgumentParser(description="Plot DFM vs zobs for one refinement.")
    ap.add_argument("--refine-dir", required=True, help="Directory containing bkind.fcf")
    ap.add_argument("--integrate-hkl", required=True, help="Path to INTEGRATE.HKL")
    ap.add_argument("--initial-u", type=float, default=0.01, help="Initial u for DFM optimization (default 0.01)")
    ap.add_argument("--out", default=None, help="Output HTML (default: <refine-dir>/DFM_vs_zobs.html)")
    args = ap.parse_args()

    refine_dir = os.path.abspath(args.refine_dir)
    integrate_hkl = os.path.abspath(args.integrate_hkl)

    out_html = args.out or os.path.join(refine_dir, "DFM_vs_zobs.html")

    df = build_sample_df_with_dfm(refine_dir, integrate_hkl, initial_u=args.initial_u)

    fig = px.scatter(
        df,
        x="zobs",
        # x="Resolution",
        y="DFM",
        title="DFM vs zobs",
        labels={"zobs": "Frame (zobs)", "DFM": "DFM"},
        hover_data=["Miller", "asu", "Resolution", "Fo^2", "Fc^2", "Fo^2_sigma"],
    )

    fig.update_yaxes(range=[-2000, 100])

    fig.write_html(out_html)
    print(f"Wrote: {out_html}")
    print(f"Rows plotted: {len(df)}")
    print(f"Optimal u used: {df.attrs.get('optimal_u', None)}")


if __name__ == "__main__":
    main()
