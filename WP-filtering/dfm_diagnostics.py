#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd

def mad_centered_sigma(x):
    # 1.4826 * median(|x - median(x)|)
    m = np.median(x)
    return 1.4826 * np.median(np.abs(x - m))

def mad0_sigma(x):
    # 1.4826 * median(|x|)
    return 1.4826 * np.median(np.abs(x))

def pct_abs_sigma(x, p):
    return np.percentile(np.abs(x), p)

def scale_factor(dfm, sigma_d, z0, lam):
    denom = z0 * sigma_d
    return np.exp(-lam * np.tanh(dfm / denom))

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 dfm_diagnostics.py sample.csv [DFMcol]")
        sys.exit(1)

    csv = sys.argv[1]
    dfm_col = sys.argv[2] if len(sys.argv) > 2 else "DFM"

    df = pd.read_csv(csv)
    x = pd.to_numeric(df[dfm_col], errors="coerce").to_numpy()
    x = x[np.isfinite(x)]
    if x.size == 0:
        print("No finite DFM values.")
        sys.exit(1)

    pos = x[x > 0]
    neg = x[x < 0]

    # Core stats
    print(f"N = {x.size}")
    print(f"mean = {np.mean(x):.6g}   median = {np.median(x):.6g}   std = {np.std(x, ddof=1):.6g}")
    print(f"min = {np.min(x):.6g}   max = {np.max(x):.6g}")
    print("percentiles (DFM):", np.percentile(x, [0.5, 1, 5, 50, 95, 99, 99.5]).round(6).tolist())
    print("percentiles (|DFM|):", np.percentile(np.abs(x), [50, 75, 90, 95, 99, 99.5]).round(6).tolist())

    # Side-by-side tails
    print(f"\npos count = {pos.size}   neg count = {neg.size}")
    if pos.size:
        print("pos percentiles:", np.percentile(pos, [50, 90, 95, 99]).round(6).tolist())
    if neg.size:
        # note: neg are negative; show magnitude too
        print("neg percentiles:", np.percentile(neg, [50, 10, 5, 1]).round(6).tolist())
        print("neg |.| percentiles:", np.percentile(np.abs(neg), [50, 90, 95, 99]).round(6).tolist())

    # Different "sigma" definitions
    sig_std = np.std(x, ddof=1)
    sig_mad_center = mad_centered_sigma(x)
    sig_mad0 = mad0_sigma(x)
    sig_abs90 = pct_abs_sigma(x, 90)
    sig_abs95 = pct_abs_sigma(x, 95)

    print("\nSigma candidates:")
    print(f"std(DFM)                 = {sig_std:.6g}")
    print(f"1.4826*median(|DFM-med|) = {sig_mad_center:.6g}")
    print(f"1.4826*median(|DFM|)     = {sig_mad0:.6g}")
    print(f"percentile(|DFM|,90)     = {sig_abs90:.6g}")
    print(f"percentile(|DFM|,95)     = {sig_abs95:.6g}")

    # Show mapping at ±k*sigma for a couple of sigmas
    z0 = 3.0
    max_factor = 10.0
    lam = np.log(max_factor)

    def report(sig, label):
        print(f"\nScale mapping using {label} (sigma={sig:.6g}), z0={z0}, max_factor={max_factor}:")
        for k in [0.5, 1, 2, 3]:
            d = k * sig
            sp = scale_factor(+d, sig, z0, lam)
            sn = scale_factor(-d, sig, z0, lam)
            print(f"  k={k:>3}:  s(+kσ)={sp:.6g}   s(-kσ)={sn:.6g}   product={sp*sn:.6g}")

    report(sig_std, "std(DFM)")
    report(sig_mad_center, "MAD around median")
    report(sig_mad0, "MAD from 0 via |DFM|")
    report(sig_abs90, "p90(|DFM|)")
    report(sig_abs95, "p95(|DFM|)")

if __name__ == "__main__":
    main()
