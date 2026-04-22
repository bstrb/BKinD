#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np
import pandas as pd
from iotbx.reflection_file_reader import any_reflection_file


def die(msg: str, code: int = 2) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


def robust_mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    center = np.median(values)
    return float(np.median(np.abs(values - center)))


def format_hkl(h: int, k: int, l: int) -> str:
    return f"({h}, {k}, {l})"


def raw_hkl_reflection_path(path: str) -> str:
    return path if "=" in os.path.basename(path) else f"{path}=hklf4"


def find_unique_array(arrays, token: str):
    matches = [array for array in arrays if token in array.info().label_string()]
    if len(matches) != 1:
        labels = [array.info().label_string() for array in arrays]
        die(f"Could not find exactly one array containing '{token}'. Arrays: {labels}")
    return matches[0]


def load_raw_hkl(raw_hkl_path: str) -> tuple[pd.DataFrame, object]:
    arrays = any_reflection_file(raw_hkl_reflection_path(raw_hkl_path)).as_miller_arrays(
        merge_equivalents=False
    )
    if not arrays:
        die(f"No reflection arrays parsed from raw HKL: {raw_hkl_path}")

    raw_array = arrays[0]
    df = pd.DataFrame(list(raw_array.indices()), columns=["h", "k", "l"])
    df["Fo^2_raw"] = list(raw_array.data())
    df["Fo^2_sigma_raw"] = list(raw_array.sigmas())
    df = df[(df["h"] != 0) | (df["k"] != 0) | (df["l"] != 0)].reset_index(drop=True)
    return df, raw_array


def load_integrate_df(integrate_hkl_path: str) -> pd.DataFrame:
    arrays = any_reflection_file(integrate_hkl_path).as_miller_arrays(merge_equivalents=False)
    if len(arrays) < 3:
        die(f"Expected at least three arrays in INTEGRATE.HKL, got {len(arrays)}")

    intensity_array = arrays[0]
    xyzobs_array = find_unique_array(arrays, "xyzobs")
    xyzobs = np.asarray(xyzobs_array.data(), dtype=float)
    df = pd.DataFrame(list(intensity_array.indices()), columns=["h", "k", "l"])
    df["Resolution"] = list(intensity_array.d_spacings().data())
    df["xobs"] = xyzobs[:, 0]
    df["yobs"] = xyzobs[:, 1]
    df["zobs"] = xyzobs[:, 2]
    df["frame"] = np.rint(df["zobs"]).astype(int)
    return df


def load_fcf_lookup(fcf_path: str) -> tuple[pd.DataFrame, object]:
    arrays = any_reflection_file(fcf_path).as_miller_arrays(merge_equivalents=False)
    calc_array = find_unique_array(arrays, "_refln_F_squared_calc")
    obs_array = find_unique_array(arrays, "_refln_F_squared_meas")

    calc_asu = calc_array.map_to_asu()
    obs_asu = obs_array.map_to_asu()

    calc_indices = list(calc_asu.indices())
    obs_indices = list(obs_asu.indices())
    if calc_indices != obs_indices:
        die("FCF calc and observed arrays do not share the same ASU indices after mapping.")

    df = pd.DataFrame(list(calc_indices), columns=["asu_h", "asu_k", "asu_l"])
    df["Fc^2"] = list(calc_asu.data())
    df["Fo^2_fcf"] = list(obs_asu.data())
    df["Fo^2_sigma_fcf"] = list(obs_asu.sigmas())

    grouped = (
        df.groupby(["asu_h", "asu_k", "asu_l"], as_index=False)
        .agg(
            {
                "Fc^2": "median",
                "Fo^2_fcf": "median",
                "Fo^2_sigma_fcf": "median",
            }
        )
        .copy()
    )

    counts = df.groupby(["asu_h", "asu_k", "asu_l"]).size().reset_index(name="fcf_asu_count")
    grouped = grouped.merge(counts, on=["asu_h", "asu_k", "asu_l"], how="left")
    return grouped, calc_array


def build_sres_dataframe(
    raw_hkl_path: str,
    fcf_path: str,
    integrate_hkl_path: str,
    min_sigma: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    raw_df, raw_array = load_raw_hkl(raw_hkl_path)
    integrate_df = load_integrate_df(integrate_hkl_path)
    fcf_lookup_df, fcf_calc_array = load_fcf_lookup(fcf_path)

    raw_dups = int(raw_df.duplicated(["h", "k", "l"]).sum())
    integrate_dups = int(integrate_df.duplicated(["h", "k", "l"]).sum())
    if raw_dups:
        die(f"Raw HKL has {raw_dups} duplicate exact HKLs; cannot assign frame rows uniquely.")
    if integrate_dups:
        die(f"INTEGRATE.HKL has {integrate_dups} duplicate exact HKLs; cannot assign frame rows uniquely.")

    raw_array_with_cs = raw_array.customized_copy(crystal_symmetry=fcf_calc_array.crystal_symmetry())
    raw_asu_indices = list(raw_array_with_cs.map_to_asu().indices())
    raw_df["asu_h"] = [h for h, _, _ in raw_asu_indices]
    raw_df["asu_k"] = [k for _, k, _ in raw_asu_indices]
    raw_df["asu_l"] = [l for _, _, l in raw_asu_indices]

    df = raw_df.merge(integrate_df, on=["h", "k", "l"], how="inner", validate="one_to_one")
    if df.empty:
        die("Raw HKL and INTEGRATE.HKL produced 0 exact HKL matches.")

    df = df.merge(fcf_lookup_df, on=["asu_h", "asu_k", "asu_l"], how="inner", validate="many_to_one")
    if df.empty:
        die("Could not map any raw reflections to Fc^2 values from the FCF.")

    mad_raw = robust_mad(df["Fo^2_raw"].to_numpy())
    mad_fc2 = robust_mad(df["Fc^2"].to_numpy())
    if not np.isfinite(mad_raw) or mad_raw <= 0:
        die(f"Invalid MAD for raw Fo^2: {mad_raw}")
    if not np.isfinite(mad_fc2) or mad_fc2 <= 0:
        die(f"Invalid MAD for Fc^2: {mad_fc2}")

    mad_scale = mad_fc2 / mad_raw
    df["Fo^2_scaled"] = mad_scale * df["Fo^2_raw"]
    df["Fo^2_sigma_scaled"] = mad_scale * df["Fo^2_sigma_raw"]

    sigma_mask = np.isfinite(df["Fo^2_sigma_scaled"]) & (df["Fo^2_sigma_scaled"] > min_sigma)
    dropped_sigma = int((~sigma_mask).sum())
    df = df.loc[sigma_mask].copy()
    if df.empty:
        die("All rows were dropped because the scaled sigma was non-finite or too small.")

    df["SRES"] = (df["Fo^2_scaled"] - df["Fc^2"]) / df["Fo^2_sigma_scaled"]
    df["absSRES"] = np.abs(df["SRES"])
    df["Miller"] = [format_hkl(h, k, l) for h, k, l in df[["h", "k", "l"]].itertuples(index=False, name=None)]
    df["asu"] = [
        format_hkl(h, k, l)
        for h, k, l in df[["asu_h", "asu_k", "asu_l"]].itertuples(index=False, name=None)
    ]
    df["hkl"] = df["Miller"]
    df = df.sort_values("zobs").reset_index(drop=True)

    stats = {
        "raw_rows": float(len(raw_df)),
        "integrate_rows": float(len(integrate_df)),
        "matched_exact_rows": float(len(raw_df.merge(integrate_df, on=["h", "k", "l"], how="inner"))),
        "fcf_asu_rows": float(len(fcf_lookup_df)),
        "final_rows": float(len(df)),
        "dropped_sigma_rows": float(dropped_sigma),
        "mad_raw": float(mad_raw),
        "mad_fc2": float(mad_fc2),
        "mad_scale": float(mad_scale),
        "median_raw": float(np.median(df["Fo^2_raw"])),
        "median_fc2": float(np.median(df["Fc^2"])),
    }
    return df, stats


def write_summary(path: str, stats: dict[str, float], df: pd.DataFrame) -> None:
    top = df.sort_values("absSRES", ascending=False).head(25)[
        [
            "Miller",
            "asu",
            "Resolution",
            "zobs",
            "Fo^2_raw",
            "Fo^2_scaled",
            "Fo^2_sigma_scaled",
            "Fc^2",
            "SRES",
            "absSRES",
        ]
    ].copy()

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("Raw-observation SRES summary\n")
        handle.write("=" * 60 + "\n\n")
        handle.write(f"Raw HKL rows:            {int(stats['raw_rows'])}\n")
        handle.write(f"INTEGRATE rows:          {int(stats['integrate_rows'])}\n")
        handle.write(f"Exact raw/integrate:     {int(stats['matched_exact_rows'])}\n")
        handle.write(f"FCF unique ASU rows:     {int(stats['fcf_asu_rows'])}\n")
        handle.write(f"Final plotted rows:      {int(stats['final_rows'])}\n")
        handle.write(f"Dropped by sigma filter: {int(stats['dropped_sigma_rows'])}\n")
        handle.write("\n")
        handle.write(f"MAD(raw Fo^2):           {stats['mad_raw']:.6g}\n")
        handle.write(f"MAD(Fc^2):               {stats['mad_fc2']:.6g}\n")
        handle.write(f"MAD scale factor:        {stats['mad_scale']:.6g}\n")
        handle.write(f"Median(raw Fo^2):        {stats['median_raw']:.6g}\n")
        handle.write(f"Median(Fc^2):            {stats['median_fc2']:.6g}\n")
        handle.write("\n")
        handle.write("Top 25 reflections by |SRES|\n")
        handle.write("-" * 60 + "\n")
        handle.write(top.to_string(index=False))
        handle.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build a viewer-ready CSV for raw-observation SRES using "
            "INTEGRATE.HKL + SHELX raw HKL + SHELX FCF."
        )
    )
    ap.add_argument("--raw-hkl", required=True, help="SHELX HKLF4 .hkl with raw Fo^2 and sigma(Fo^2)")
    ap.add_argument("--fcf", required=True, help="SHELX .fcf with Fc^2 values")
    ap.add_argument("--integrate-hkl", required=True, help="XDS INTEGRATE.HKL for observed frame positions")
    ap.add_argument("--out-csv", required=True, help="Output CSV path")
    ap.add_argument("--out-summary", default=None, help="Optional summary text path")
    ap.add_argument("--min-sigma", type=float, default=1e-12, help="Drop rows with scaled sigma <= this")
    args = ap.parse_args()

    out_csv = os.path.abspath(args.out_csv)
    out_summary = os.path.abspath(args.out_summary) if args.out_summary else None
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    if out_summary is not None:
        os.makedirs(os.path.dirname(out_summary) or ".", exist_ok=True)

    df, stats = build_sres_dataframe(
        raw_hkl_path=os.path.abspath(args.raw_hkl),
        fcf_path=os.path.abspath(args.fcf),
        integrate_hkl_path=os.path.abspath(args.integrate_hkl),
        min_sigma=args.min_sigma,
    )

    df.to_csv(out_csv, index=False)
    print(f"Wrote CSV: {out_csv}")
    print(f"Rows: {len(df)}")
    print(f"MAD scale factor: {stats['mad_scale']:.6g}")

    if out_summary is not None:
        write_summary(out_summary, stats, df)
        print(f"Wrote summary: {out_summary}")


if __name__ == "__main__":
    main()
