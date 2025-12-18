#!/usr/bin/env python3
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description="Plot exp(z-score(DFM)) vs DFM from a CSV.")
    ap.add_argument("csv", help="Input CSV path")
    ap.add_argument("--dfm-col", default=None,
                    help="DFM column name (default: 'DFM' if present else last column)")
    ap.add_argument("--out-prefix", default=None,
                    help="Output prefix (default: input filename without extension)")
    ap.add_argument("--alpha", type=float, default=0.2, help="Scatter alpha (default: 0.2)")
    ap.add_argument("--s", type=float, default=6.0, help="Marker size (default: 6)")
    ap.add_argument("--no-show", action="store_true", help="Do not display plot (still saves PNG)")
    args = ap.parse_args()

    in_path = args.csv
    if not os.path.isfile(in_path):
        print(f"ERROR: file not found: {in_path}", file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(in_path)

    # Determine DFM column
    if args.dfm_col is not None:
        dfm_col = args.dfm_col
        if dfm_col not in df.columns:
            print(f"ERROR: --dfm-col '{dfm_col}' not found. Columns: {list(df.columns)}", file=sys.stderr)
            sys.exit(2)
    else:
        dfm_col = "DFM" if "DFM" in df.columns else df.columns[-1]

    dfm = pd.to_numeric(df[dfm_col], errors="coerce").to_numpy()
    dfm = dfm[np.isfinite(dfm)]

    if dfm.size < 2:
        print(f"ERROR: Not enough numeric DFM values in column '{dfm_col}'.", file=sys.stderr)
        sys.exit(2)

    # Compute z-scores standard way
    # mu = float(np.mean(dfm))
    # sigma = float(np.std(dfm, ddof=1))
    # if not math.isfinite(sigma) or sigma == 0.0:
    #     print(f"ERROR: DFM std is zero or non-finite (std={sigma}). Can't z-score normalize.", file=sys.stderr)
    #     sys.exit(2)

    # z = (dfm - mu) / sigma
    # y = np.exp(-z/5)

    # Compute z-scores robust way (via MAD)
    mu = np.median(dfm)
    sigma = 1.4826 * np.median(np.abs(dfm - mu))
    z = (dfm - mu) / sigma
    scale = np.exp(-(np.log(args.alpha) * np.tanh(z/np.max(np.abs(z)))))
    scale = np.exp(-(np.log(args.alpha) * np.tanh(z/3)))
    # y = np.log(scale)
    y = scale

    out_prefix = args.out_prefix
    if out_prefix is None:
        out_prefix = os.path.splitext(os.path.basename(in_path))[0]

    plt.figure()
    plt.scatter(dfm, y, s=args.s)
    plt.xlabel(dfm_col)
    plt.ylabel(f"exp(z-score({dfm_col}))")
    plt.title(f"exp(z-score({dfm_col})) vs {dfm_col}  (mean={mu:.4g}, std={sigma:.4g})")
    plt.tight_layout()

    

    out_png = f"{out_prefix}_exp_z_vs_dfm_{args.alpha}.png"
    plt.savefig(out_png, dpi=250)
    print(f"DFM column: {dfm_col}")
    print(f"n={dfm.size}  mean={mu:.6g}  std={sigma:.6g}")
    print(f"Saved: {out_png}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
