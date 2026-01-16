#!/usr/bin/env python3
import argparse
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

pio.renderers.default = "browser"


# =========================
# Sigma model (frozen once)
# =========================

def sigma_from_iobs(iobs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Toy measurement uncertainty for Fo^2, generated ONCE per dataset:
      sigma = frac * max(Iobs, 0), frac ~ U(0.08, 0.12)
    """
    i = np.maximum(iobs, 0.0)
    frac = rng.uniform(0.08, 0.12, size=i.shape)
    sigma = frac * i
    return np.maximum(sigma, 1e-6)


# ==========================================
# Resolution-dependent Ic model (ED-inspired)
# ==========================================

def icalc_resolution_dependent(iobs: np.ndarray, s: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Resolution-dependent toy ED model error:
      - Low resolution (s small): Ic biased high + larger ± noise ("smearing")
      - High resolution (s large): small bias + small noise (more trustworthy)
      - Includes additive noise floor that increases at low resolution
      - Includes multiplicative noise that scales with intensity
    """
    i = np.maximum(iobs, 0.0)
    low = 1.0 - s  # 1 at low res, 0 at high res

    # Robust intensity scale for additive floor
    i_scale = np.percentile(i, 95) + 1e-12

    # Systematic dynamical bias (Ic > Io on average), strongest at low res but kept small
    bias_frac_min = 0.005
    bias_frac_max = 0.08
    bias_frac = bias_frac_min + low * (bias_frac_max - bias_frac_min)
    bias = bias_frac * i

    # Multiplicative smearing noise (±), stronger at low res, further reduced to limit separation
    noise_frac_min = 0.008
    noise_frac_max = 0.12
    noise_frac = noise_frac_min + low * (noise_frac_max - noise_frac_min)
    mult_sigma = noise_frac * i

    # Additive floor (±), stronger at low res but small to avoid big lift
    floor_frac_hi = 0.001   # ~0.1% of i_scale at high res
    floor_frac_lo = 0.008   # ~0.8% of i_scale at low res
    floor_frac = floor_frac_hi + low * (floor_frac_lo - floor_frac_hi)
    floor_sigma = floor_frac * i_scale

    noise_sigma = np.sqrt(mult_sigma**2 + floor_sigma**2)

    icalc = iobs + bias + rng.normal(0.0, noise_sigma, size=i.shape)

    # Optional: keep Ic non-negative (uncomment if you want)
    # icalc = np.maximum(icalc, 0.0)

    return icalc


# =========================
# SHELXL WGHT-like weight
# =========================

def shelxl_weight(
    fo2: np.ndarray,
    fc2: np.ndarray,
    sig: np.ndarray,
    s: np.ndarray,
    a: float, b: float, c: float, d: float, e: float, f_mix: float
) -> np.ndarray:
    """
    If c >= 0:
      w = exp(c*s^2) / (sigma(Fo^2)^2 + (aP)^2 + bP + d + e*s)
    If c < 0:
      w = (1 - exp(c*s^2)) / (sigma(Fo^2)^2 + (aP)^2 + bP + d + e*s)

    P(f) = f*max(0, Fo^2) + (1-f)*Fc^2
    """
    P = f_mix * np.maximum(fo2, 0.0) + (1.0 - f_mix) * fc2

    denom = (sig ** 2) + (a * P) ** 2 + b * P + d + e * s
    denom = np.maximum(denom, 1e-300)

    if c >= 0.0:
        numer = np.exp(c * (s ** 2))
    else:
        numer = 1.0 - np.exp(c * (s ** 2))
        numer = np.maximum(numer, 0.0)

    return numer / denom


def main():
    ap = argparse.ArgumentParser()

    # Resolution range in Angstrom (d-spacing)
    ap.add_argument("--dmin", type=float, default=0.8, help="High-resolution limit (small Å)")
    ap.add_argument("--dmax", type=float, default=3.0, help="Low-resolution limit (large Å)")
    ap.add_argument("--n", type=int, default=1001)

    # Intensity range used to construct Io(d)
    ap.add_argument("--iobs-min", type=float, default=10.0)
    ap.add_argument("--iobs-max", type=float, default=1000.0)

    # WGHT a–f only
    ap.add_argument("-a", type=float, required=True)
    ap.add_argument("-b", type=float, required=True)
    ap.add_argument("-c", type=float, required=True)
    ap.add_argument("-d", type=float, required=True)
    ap.add_argument("-e", type=float, required=True)
    ap.add_argument("-f", dest="f_mix", type=float, required=True)

    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if args.dmin <= 0 or args.dmax <= 0:
        raise ValueError("dmin and dmax must be > 0 Å.")
    if args.dmin >= args.dmax:
        raise ValueError("dmin must be smaller than dmax (dmin=high-res, dmax=low-res).")

    rng = np.random.default_rng(args.seed)

    # d-axis in Å (low-res large Å -> high-res small Å)
    dA = np.linspace(args.dmax, args.dmin, args.n)  # decreasing with index

    # Internal normalized resolution proxy s in [0, 1]:
    # s=0 at low res (d=dmax), s=1 at high res (d=dmin)
    s = (args.dmax - dA) / (args.dmax - args.dmin)

    # Construct Io(d): stronger at low resolution (large Å), weaker at high resolution (small Å)
    gamma = 2.0
    iobs = args.iobs_max - (args.iobs_max - args.iobs_min) * (s ** gamma)

    # Freeze sigma once
    sig = sigma_from_iobs(iobs, rng=rng)

    # Simulate Ic with low-res bias + smearing
    icalc = icalc_resolution_dependent(iobs, s=s, rng=rng)

    # Compute w and term
    w = shelxl_weight(
        fo2=iobs, fc2=icalc, sig=sig, s=s,
        a=args.a, b=args.b, c=args.c, d=args.d, e=args.e, f_mix=args.f_mix
    )
    term = w * (iobs - icalc) ** 2

    # Plot: left axis w and term; right axis Io and Ic
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(
        title="Resolution-dependent plot vs Å: Io, Ic, w, and w*(Io-Ic)^2",
        xaxis_title="d (Å)  [low res → high res]"
    )

    fig.add_trace(go.Scatter(x=dA, y=term, mode="lines", name="w*(Io-Ic)^2"), secondary_y=False)
    fig.add_trace(go.Scatter(x=dA, y=w, mode="lines", name="w", line=dict(dash="dot")), secondary_y=False)

    fig.add_trace(go.Scatter(x=dA, y=iobs, mode="lines", name="Io"), secondary_y=True)
    fig.add_trace(go.Scatter(x=dA, y=icalc, mode="lines", name="Ic", line=dict(dash="dash")), secondary_y=True)

    fig.update_yaxes(title_text="w and w*(Io-Ic)^2", secondary_y=False)
    fig.update_yaxes(title_text="Io and Ic", secondary_y=True)

    # Make the axis read naturally: left = low res (large Å), right = high res (small Å)
    fig.update_xaxes(autorange="reversed")

    fig.show()


if __name__ == "__main__":
    main()


# Example:
# python plot_wght_dual.py --dmin 0.5 --dmax 10.0 -a 0.1 -b 0.0 -c 0.0 -d 0.0 -e 0.0 -f 0.33
