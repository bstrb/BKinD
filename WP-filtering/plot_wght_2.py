#!/usr/bin/env python3
"""
Plot SHELXL-style WGHT behavior for multiple parameter sets (a,b,c,d,e,f).

WGHT definition (SHELXL help-style):
  w = q / [ sigma^2(Fo^2) + (a*P)^2 + b*P + d + e*sin(theta) ]
  P = f*max(Fo^2,0) + (1-f)*Fc^2
  q = 1                         if c = 0
      exp[c*(sin(theta)/lambda)^2]      if c > 0
      1 - exp[c*(sin(theta)/lambda)^2]  if c < 0

We treat P as an "intensity-like" scale variable (you can think ~Fo^2 level)
and choose a simple sigma^2(Fo^2) model for illustration.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# User inputs
# -----------------------------
OUTDIR = "wght_plots_full"
os.makedirs(OUTDIR, exist_ok=True)

# Wavelength (Å). Only needed to convert s=sinθ/λ into sinθ.
LAMBDA_A = 0.71073  # Mo Kα; change if you want (e.g., 1.54184 for Cu Kα)

# P range (intensity-like scale)
P_MIN, P_MAX, NPTS = 1e-1, 1e6, 2500
P = np.logspace(np.log10(P_MIN), np.log10(P_MAX), NPTS)

# Choose a sigma^2(Fo^2) model (illustrative)
SIGMA2_MODEL = "poisson"   # "poisson" or "constant_rel" or "constant_abs"
REL_SIGMA_I = 0.20         # used if constant_rel: sigma(Fo^2)=REL_SIGMA_I*Fo^2
ABS_SIGMA_I = 100.0        # used if constant_abs: sigma(Fo^2)=ABS_SIGMA_I

# Resolution points to show (s = sinθ/λ, in Å^-1)
# Pick 2–4 values; these control the c and e effects.
S_VALUES = [0.1, 0.4, 0.8]   # low, mid, high angle-ish

# WGHT parameter sets you want to compare
# Edit / add as many as you like:
WGHT_SETS = [
    dict(a=0.01, b=0.0, c=0.0,  d=0.0,  e=0.0,  f=1/3, label="default-ish: a=0.01, f=1/3"),
    dict(a=0.01, b=0.5, c=0.0,  d=0.0,  e=0.0,  f=1/3, label="add b term: b=0.5"),
    dict(a=0.01, b=0.0, c=+10,  d=0.0,  e=0.0,  f=1/3, label="c=+10 (exp upweight high angle)"),
    dict(a=0.01, b=0.0, c=-10,  d=0.0,  e=0.0,  f=1/3, label="c=-10 (saturating upweight high angle)"),
    dict(a=0.01, b=0.0, c=0.0,  d=50.0, e=0.0,  f=1/3, label="add d (constant denom offset)"),
    dict(a=0.01, b=0.0, c=0.0,  d=0.0,  e=200., f=1/3, label="add e*sin(theta) (angle-dependent denom)"),
]


# -----------------------------
# Core functions
# -----------------------------
def sigma2_fo2(P: np.ndarray) -> np.ndarray:
    """Illustrative sigma^2(Fo^2) model vs P."""
    m = SIGMA2_MODEL.lower().strip()
    if m == "poisson":
        # Counting-statistics dominated: Var(I) ~ I
        return P.copy()
    if m == "constant_rel":
        # Constant relative on Fo^2: sigma(I)=rel*I -> sigma^2=(rel*I)^2
        return (REL_SIGMA_I * P) ** 2
    if m == "constant_abs":
        return np.full_like(P, ABS_SIGMA_I**2)
    raise ValueError(f"Unknown SIGMA2_MODEL: {SIGMA2_MODEL}")


def q_factor(c: float, s: np.ndarray) -> np.ndarray:
    """q as a function of c and s=(sinθ/λ)."""
    x = (s ** 2)  # (sinθ/λ)^2
    if c == 0.0:
        return np.ones_like(x)
    if c > 0.0:
        return np.exp(c * x)
    # c < 0
    return 1.0 - np.exp(c * x)


def wght_weight(P: np.ndarray, s: float, params: dict, sigma2: np.ndarray) -> np.ndarray:
    """
    Compute w(P, s) for one WGHT parameter set.
    We approximate:
      Fc^2 ~ P  (for plotting/intuition)
      max(Fo^2,0) ~ P (for plotting/intuition)
    so P_mix ends up equal to P regardless of f.
    (f still matters in real refinement; this is a didactic plot.)
    """
    a = float(params["a"])
    b = float(params["b"])
    c = float(params["c"])
    d = float(params["d"])
    e = float(params["e"])

    # Convert s=sinθ/λ to sinθ using lambda, clamp to [0,1]
    sin_theta = np.clip(s * LAMBDA_A, 0.0, 1.0)

    q = q_factor(c, np.array([s]))[0]
    denom = sigma2 + (a * P) ** 2 + b * P + d + e * sin_theta
    return q / denom


# -----------------------------
# Plots
# -----------------------------
def plot_weights_vs_P():
    sigma2 = sigma2_fo2(P)
    w0 = 1.0 / sigma2

    # For each s, make a plot showing baseline + all WGHT sets
    for s in S_VALUES:
        fig, ax = plt.subplots(figsize=(8.4, 5.2))
        ax.plot(P, w0, label=f"Baseline: w=1/σ² (σ² model={SIGMA2_MODEL})", linestyle=":")

        for ps in WGHT_SETS:
            w = wght_weight(P, s, ps, sigma2)
            ax.plot(P, w, label=ps["label"])

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("P (intensity-like scale; ~Fo² level)")
        ax.set_ylabel("Weight w")
        ax.set_title(f"Weights vs P at s=sinθ/λ={s} Å⁻¹  (λ={LAMBDA_A} Å)")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, f"weights_vs_P_s_{s:.2f}.png"), dpi=220)
        plt.close(fig)


def plot_relative_factor_vs_P():
    sigma2 = sigma2_fo2(P)
    w0 = 1.0 / sigma2

    # For each s, plot relative factor for each WGHT set
    for s in S_VALUES:
        fig, ax = plt.subplots(figsize=(8.4, 5.2))
        for ps in WGHT_SETS:
            w = wght_weight(P, s, ps, sigma2)
            rel = w / w0
            ax.plot(P, rel, label=ps["label"])

        ax.set_xscale("log")
        ax.set_xlabel("P (intensity-like scale)")
        ax.set_ylabel("Relative factor  w_withWGHT / w_baseline")
        ax.set_title(f"Relative down-weighting vs P at s={s} Å⁻¹  (λ={LAMBDA_A} Å)")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, f"relative_factor_vs_P_s_{s:.2f}.png"), dpi=220)
        plt.close(fig)


def main():
    plot_weights_vs_P()
    plot_relative_factor_vs_P()
    print(f"Saved PNGs in: {OUTDIR}/")
    print("Tip: edit WGHT_SETS and S_VALUES to match the story you want to tell.")


if __name__ == "__main__":
    main()
