#!/usr/bin/env python3
import argparse
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Force opening in your default browser (reliable in VS Code on macOS)
pio.renderers.default = "browser"


# =========================
# Synthetic ED-like generator
# =========================

def sample_resolution_shells(n: int, rng: np.random.Generator, s_min: float, s_max: float) -> np.ndarray:
    """Sample s = 1/d with density ~ s^2 (uniform in reciprocal volume)."""
    u = rng.uniform(0.0, 1.0, size=n)
    s_cubed = u * (s_max ** 3 - s_min ** 3) + s_min ** 3
    return np.cbrt(s_cubed)


def make_true_intensity(s: np.ndarray, i0: float, b_wilson: float) -> np.ndarray:
    """Wilson-like falloff: I_true = I0 * exp(-B s^2)."""
    return i0 * np.exp(-b_wilson * (s ** 2))


def apply_dynamical_underestimation(i_true: np.ndarray, s: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mimic low-s underestimation of strong reflections and detector compression."""
    # Compression of intense spots (further relaxed to bring Io closer to Ic)
    i_sat = 3.0e4
    comp = i_true / (1.0 + (i_true / i_sat))

    # Additional low-resolution damping (strong dynamical attenuation)
    alpha = 0.12
    s_low = 0.25
    dyn = 1.0 - alpha * np.exp(-(s / s_low) ** 2)

    base = comp * dyn

    # Smearing at low resolution (zero-mean noise, larger at low s)
    low = np.exp(-(s / s_low) ** 2)
    noise_frac_min = 0.02
    noise_frac_max = 0.35
    noise_frac = noise_frac_min + low * (noise_frac_max - noise_frac_min)
    noise_sigma = noise_frac * np.maximum(base, 0.0)

    return base + rng.normal(0.0, noise_sigma, size=i_true.shape)


def add_observation_noise(i_mean: np.ndarray, s: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Add Poisson-like plus background/read noise and return Fo^2 and sigma(Fo^2)."""
    gain = 1.0
    sigma_bg = 25.0
    sigma_read = 5.0
    sigma_res = 0.005  # gentler resolution inflation to keep high-res closer

    var = gain * np.maximum(i_mean, 0.0) + sigma_bg ** 2 + sigma_read ** 2
    var *= 1.0 + sigma_res * (s ** 2)
    sigma = np.sqrt(np.maximum(var, 1e-12))

    i_obs = i_mean + rng.normal(0.0, sigma, size=i_mean.shape)
    return i_obs, sigma


def make_calc_intensity(i_true: np.ndarray, s: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Construct Ic^2 with scale/B bias and mild random model error."""
    # Slight positive bias that fades with resolution; noise allows over/under estimates
    bias_frac_max = 0.02
    bias = bias_frac_max * np.exp(-(s / 0.28) ** 2)  # mostly low-res

    model_noise_frac = 0.06

    ic = i_true * (1.0 + bias)
    ic *= 1.0 + rng.normal(0.0, model_noise_frac, size=i_true.shape)
    return np.maximum(ic, 0.0)


# =========================
# Weight function
# =========================

def sigma_from_iobs(iobs: np.ndarray, s: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Deterministic sigma consistent with the noise model (no extra randomness)
    gain = 1.0
    sigma_bg = 25.0
    sigma_read = 5.0
    sigma_res = 0.005

    var = gain * np.maximum(iobs, 0.0) + sigma_bg ** 2 + sigma_read ** 2
    var *= 1.0 + sigma_res * (s ** 2)
    return np.sqrt(np.maximum(var, 1e-12))


def shelxl_weight(
    fo2: np.ndarray,
    fc2: np.ndarray,
    s: np.ndarray,
    a: float, b: float, c: float, d: float, e: float, f_mix: float,
    rng: np.random.Generator,
) -> np.ndarray:

    """
    WGHT scheme:

      If c >= 0:
        w = exp(c*s^2) / (sigma(Fo^2)^2 + (aP)^2 + bP + d + e*s)
      If c < 0:
        w = (1 - exp(c*s^2)) / (sigma(Fo^2)^2 + (aP)^2 + bP + d + e*s)

    P(f) = f*max(0, Fo^2) + (1-f)*Fc^2
    """
    sig = sigma_from_iobs(fo2, s=s, rng=rng)
    P = f_mix * np.maximum(fo2, 0.0) + (1.0 - f_mix) * fc2

    denom = (sig ** 2) + (a * P) ** 2 + b * P + d + e * s
    denom = np.maximum(denom, 1e-300)

    if c >= 0.0:
        numer = np.exp(c * (s ** 2))
    else:
        numer = 1.0 - np.exp(c * (s ** 2))
        numer = np.maximum(numer, 0.0)

    return numer / denom


# =========================
# Plot helpers
# =========================

def binned_stats(x: np.ndarray, y: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centers = 0.5 * (bins[1:] + bins[:-1])
    med = np.zeros_like(centers)
    for i in range(len(bins) - 1):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        if np.any(mask):
            med[i] = np.median(y[mask])
        else:
            med[i] = np.nan
    return centers, med


def main():
    ap = argparse.ArgumentParser()

    # Keep CLI minimal: n for sampling density, WGHT a–f, seed
    ap.add_argument("--n", type=int, default=4000)

    ap.add_argument("-a", type=float, default=0.0)
    ap.add_argument("-b", type=float, default=0.0)
    ap.add_argument("-c", type=float, default=0.0)
    ap.add_argument("-d", type=float, default=0.0)
    ap.add_argument("-e", type=float, default=0.0)
    ap.add_argument("-f", dest="f_mix", type=float, default=0.3333)

    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Resolution range (s = 1/d)
    s_min = 0.2   # ~5 Å
    s_max = 2    # ~0.5 Å

    s = sample_resolution_shells(args.n, rng, s_min, s_max)
    i_true = make_true_intensity(s, i0=3.0e4, b_wilson=1.2)
    i_mean_obs = apply_dynamical_underestimation(i_true, s, rng)
    iobs, sigma_fo2 = add_observation_noise(i_mean_obs, s, rng)
    iclc = make_calc_intensity(i_true, s, rng)

    w = shelxl_weight(
        fo2=iobs, fc2=iclc, s=s,
        a=args.a, b=args.b, c=args.c, d=args.d, e=args.e, f_mix=args.f_mix,
        rng=rng,
    )
    term = w * (iobs - iclc) ** 2

    # Prepare resolution in Å
    d = 1.0 / np.maximum(s, 1e-6)

    # Binning for smoother trends
    d_bins = np.linspace(d.min(), d.max(), 36)

    d_centers, w_med_d = binned_stats(d, w, d_bins)
    _, term_med_d = binned_stats(d, term, d_bins)

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("w and term vs d", "Io and Ic vs d", "Io vs Ic"),
        specs=[[{}, {}, {}]],
    )

    # Panel 1: weights and term vs resolution
    fig.add_trace(go.Scatter(x=d, y=w, mode="markers", name="w (scatter)",
                             marker=dict(size=4, opacity=0.35, color="steelblue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=d, y=term, mode="markers", name="w*(Io-Ic)^2 (scatter)",
                             marker=dict(size=4, opacity=0.35, color="indianred")), row=1, col=1)
    fig.add_trace(go.Scatter(x=d_centers, y=w_med_d, mode="lines", name="median w",
                             line=dict(color="navy", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=d_centers, y=term_med_d, mode="lines", name="median w*(Io-Ic)^2",
                             line=dict(color="darkred", width=2, dash="dash")), row=1, col=1)

    fig.update_xaxes(title_text="d (Å)", autorange="reversed", row=1, col=1)
    fig.update_yaxes(title_text="w, w*(Io-Ic)^2", type="log", row=1, col=1)

    # Panel 2: Io and Ic vs resolution (intensities)
    fig.add_trace(go.Scatter(x=d, y=iobs, mode="markers", name="Io vs d",
                             marker=dict(size=3, opacity=0.25, color="gray")), row=1, col=2)
    fig.add_trace(go.Scatter(x=d, y=iclc, mode="markers", name="Ic vs d",
                             marker=dict(size=3, opacity=0.35, color="darkorange")), row=1, col=2)

    fig.update_xaxes(title_text="d (Å)", autorange="reversed", row=1, col=2)
    fig.update_yaxes(title_text="Io, Ic", row=1, col=2)

    # Panel 3: Io vs Ic scatter with identity line (intensities)
    fig.add_trace(go.Scatter(x=iclc, y=iobs, mode="markers", name="Io vs Ic",
                             marker=dict(size=3, opacity=0.3, color="purple")), row=1, col=3)
    diag_min = float(min(iobs.min(), iclc.min()))
    diag_max = float(max(iobs.max(), iclc.max()))
    fig.add_trace(go.Scatter(x=[diag_min, diag_max], y=[diag_min, diag_max], mode="lines",
                             name="Io=Ic", line=dict(color="black", dash="dash")), row=1, col=3)

    fig.update_xaxes(title_text="Ic", row=1, col=3)
    fig.update_yaxes(title_text="Io", row=1, col=3)

    fig.update_layout(title="WGHT behavior vs resolution (synthetic ED-like, Io/Ic shown)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0.0))

    fig.show()


if __name__ == "__main__":
    main()


# Example: python plot_wght.py -a 0.0 -b 0.0 -c 1.0 -d 0.0 -e 0.0 -f 0.3333