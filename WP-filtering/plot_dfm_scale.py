#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def robust_sigma_mad(x: np.ndarray) -> float:
    """Robust std estimate via MAD: sigma ~= 1.4826 * median(|x - median(x)|)."""
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad

def classical_sigma(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.std(x, ddof=1)) if x.size > 1 else 0.0

def compute_scale(dfm: np.ndarray, sigma_d: float, z0: float, lam: float) -> np.ndarray:
    # Guard against sigma_d = 0
    denom = z0 * sigma_d
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError(f"Invalid denom=z0*sigma_d = {denom}. Check DFM distribution / sigma mode.")
    return np.exp(-lam * np.tanh(dfm / denom))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Input CSV containing a column named 'DFM'")
    ap.add_argument("--dfm-col", default="DFM", help="Name of DFM column (default: DFM)")
    ap.add_argument("--sigma-mode", choices=["mad", "std"], default="mad",
                    help="How to estimate sigma_D from DFM: mad (robust) or std (classical)")
    ap.add_argument("--z0", type=float, default=3.0,
                    help="Transition scale in 'sigmas' (default: 3.0)")
    ap.add_argument("--max-factor", type=float, default=10.0,
                    help="Maximum up/down factor at saturation (default: 10.0 => min=0.1, max=10)")
    ap.add_argument("--out-prefix", default="dfm_scale",
                    help="Output prefix for plots and CSV (default: dfm_scale)")
    ap.add_argument("--sample", type=int, default=0,
                    help="If >0, randomly subsample this many points for scatter plot (helps huge CSVs)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.dfm_col not in df.columns:
        raise SystemExit(f"Missing column '{args.dfm_col}'. Columns: {list(df.columns)}")

    dfm = pd.to_numeric(df[args.dfm_col], errors="coerce").to_numpy()
    dfm = dfm[np.isfinite(dfm)]
    if dfm.size == 0:
        raise SystemExit("No finite DFM values found.")

    sigma_d = robust_sigma_mad(dfm) if args.sigma_mode == "mad" else classical_sigma(dfm)
    if not np.isfinite(sigma_d) or sigma_d <= 0:
        raise SystemExit(f"sigma_D is invalid ({sigma_d}).")

    # Choose lambda so that saturation corresponds to max-factor
    # Because tanh(...) -> +/-1 => s -> exp(-/+lambda)
    lam = float(np.log(args.max_factor))

    # Compute scales for the full dataframe (keep alignment with rows)
    dfm_full = pd.to_numeric(df[args.dfm_col], errors="coerce").to_numpy()
    scales = np.full_like(dfm_full, np.nan, dtype=float)
    mask = np.isfinite(dfm_full)
    scales[mask] = compute_scale(dfm_full[mask], sigma_d=sigma_d, z0=args.z0, lam=lam)
    df["scale_factor"] = scales

    # Save augmented CSV
    out_csv = f"{args.out_prefix}_with_scale.csv"
    df.to_csv(out_csv, index=False)

    print(f"sigma_D ({args.sigma_mode}) = {sigma_d:.6g}")
    print(f"z0 = {args.z0}")
    print(f"max_factor = {args.max_factor}  -> lambda = ln(max_factor) = {lam:.6g}")
    print(f"Wrote: {out_csv}")

    # Prepare data for plotting
    df_plot = df.loc[np.isfinite(df["scale_factor"]) & np.isfinite(df[args.dfm_col]), [args.dfm_col, "scale_factor"]].copy()
    if args.sample and args.sample > 0 and len(df_plot) > args.sample:
        df_plot = df_plot.sample(n=args.sample, random_state=0)

    # 1) Scatter: DFM vs scale
    plt.figure()
    plt.scatter(df_plot[args.dfm_col].values, df_plot["scale_factor"].values, s=8)
    plt.xlabel("DFM")
    plt.ylabel("Scale factor  s = exp(-λ tanh(DFM/(z0 σD)))")
    plt.title(f"DFM→Scale (sigma_D={sigma_d:.3g}, z0={args.z0}, max_factor={args.max_factor})")
    plt.yscale("log")  # helpful when you allow factors like 0.1..10
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    out_png1 = f"{args.out_prefix}_scatter.png"
    plt.savefig(out_png1, dpi=200, bbox_inches="tight")

    # 2) Histogram of scale factors
    plt.figure()
    sf = df_plot["scale_factor"].values
    plt.hist(sf[np.isfinite(sf)], bins=80)
    plt.xlabel("Scale factor")
    plt.ylabel("Count")
    plt.title("Scale factor distribution")
    plt.grid(True, linestyle="--", linewidth=0.5)
    out_png2 = f"{args.out_prefix}_hist.png"
    plt.savefig(out_png2, dpi=200, bbox_inches="tight")

    print(f"Wrote: {out_png1}")
    print(f"Wrote: {out_png2}")

if __name__ == "__main__":
    main()
