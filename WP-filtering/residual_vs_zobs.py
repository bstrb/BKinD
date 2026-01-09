#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px


def parse_hklf4_raw(hkl_path: str) -> pd.DataFrame:
    """
    Parse HKLF4-style .hkl:
      h k l Fo^2 sigma(Fo^2) [extra cols...]
    Example line:
      0  -1   0 191450. 3906.09   1
    We ignore any trailing columns beyond sigma.
    We omit the terminator line: 0 0 0 0 0 0 (or any 0 0 0 with zeros).
    """
    rows = []
    with open(hkl_path, "r", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 5:
                continue

            try:
                h = int(parts[0]); k = int(parts[1]); l = int(parts[2])
                fo2 = float(parts[3]); sig = float(parts[4])
            except Exception:
                continue

            # Omit terminal / invalid row
            if h == 0 and k == 0 and l == 0:
                # you said the terminator looks like: 0 0 0 0 0 0
                # treat any (0,0,0) as terminator here
                continue

            rows.append({"h": h, "k": k, "l": l, "fo2": fo2, "sig_fo2": sig})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Parsed 0 reflections from raw HKLF4 .hkl. Check file format.")
    return df


def parse_integrate_hkl_for_zobs(integrate_path: str) -> pd.DataFrame:
    """
    Parse XDS INTEGRATE.HKL and extract h,k,l and ZOBS.
    Assumes 21 items per record (standard XDS layout).
    ZOBS is column 15 in the documented list (0-based index 14).
    """
    rows = []
    with open(integrate_path, "r", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            if line.startswith("!"):
                continue
            parts = line.split()
            if len(parts) < 15:
                continue

            try:
                h = int(parts[0]); k = int(parts[1]); l = int(parts[2])
                zobs = float(parts[14])
            except Exception:
                continue

            rows.append({"h": h, "k": k, "l": l, "zobs": zobs})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Parsed 0 records from INTEGRATE.HKL. Check file format.")
    return df


def parse_fcf_fcalc2(fcf_path: str) -> pd.DataFrame:
    """
    Parse CIF-like .fcf and extract h,k,l and _refln_F_squared_calc (Fc^2).
    Does NOT filter on observed status.
    """
    lines = Path(fcf_path).read_text(errors="replace").splitlines()

    wanted = [
        "_refln_index_h",
        "_refln_index_k",
        "_refln_index_l",
        "_refln_F_squared_calc",
    ]

    def is_loop_start(s: str) -> bool:
        return s.strip().lower() == "loop_"

    def is_tag_line(s: str) -> bool:
        return s.strip().startswith("_")

    def is_new_block(s: str) -> bool:
        st = s.strip()
        return st.lower().startswith("data_") or is_loop_start(st) or is_tag_line(st)

    i = 0
    data_rows = []
    while i < len(lines):
        if is_loop_start(lines[i]):
            i += 1
            tags = []
            while i < len(lines) and is_tag_line(lines[i]):
                tags.append(lines[i].strip().split()[0])
                i += 1

            if not all(t in tags for t in wanted):
                # skip loop data
                while i < len(lines) and lines[i].strip() != "" and not is_new_block(lines[i]):
                    i += 1
                continue

            tag_to_idx = {t: idx for idx, t in enumerate(tags)}

            while i < len(lines):
                row = lines[i].strip()
                if row == "" or is_new_block(row):
                    break
                parts = row.split()
                if len(parts) < len(tags):
                    i += 1
                    continue

                try:
                    h = int(parts[tag_to_idx["_refln_index_h"]])
                    k = int(parts[tag_to_idx["_refln_index_k"]])
                    l = int(parts[tag_to_idx["_refln_index_l"]])
                    fcalc2 = float(parts[tag_to_idx["_refln_F_squared_calc"]])
                except Exception:
                    i += 1
                    continue

                data_rows.append({"h": h, "k": k, "l": l, "fcalc2": fcalc2})
                i += 1

            break  # stop after the first matching reflection loop
        i += 1

    df = pd.DataFrame(data_rows)
    if df.empty:
        raise ValueError("Could not find reflection loop with _refln_F_squared_calc in the .fcf.")
    return df


def main():
    ap = argparse.ArgumentParser(
        description="Plot standardized residual (s*Fo^2 - Fc^2)/(s*sigma(Fo^2)) vs ZOBS using raw HKLF4 .hkl + .fcf + INTEGRATE.HKL"
    )
    ap.add_argument("--raw_hkl", required=True, help="Raw HKLF4 .hkl containing Fo^2 and sigma(Fo^2)")
    ap.add_argument("--fcf", required=True, help=".fcf containing Fc^2 (_refln_F_squared_calc)")
    ap.add_argument("--integrate", required=True, help="INTEGRATE.HKL (used only to extract ZOBS)")
    ap.add_argument("--scale", required=True, type=float, help="Overall scale factor s to apply to Fo^2 and sigma(Fo^2)")
    ap.add_argument("--min_sigma", type=float, default=1e-12, help="Drop rows with (s*sigma) <= this")
    args = ap.parse_args()

    s = args.scale
    if not np.isfinite(s) or s <= 0:
        raise ValueError("--scale must be a positive finite number.")

    raw = parse_hklf4_raw(args.raw_hkl)
    fcf = parse_fcf_fcalc2(args.fcf)
    zdf = parse_integrate_hkl_for_zobs(args.integrate)

    # Merge strictly on exact h,k,l
    df = raw.merge(fcf, on=["h", "k", "l"], how="inner", validate="many_to_one")
    df = df.merge(zdf, on=["h", "k", "l"], how="inner", validate="many_to_one")

    # Apply scaling to Fo^2 and sigma(Fo^2)
    df["fo2_scaled"] = s * df["fo2"]
    df["sig_scaled"] = s * df["sig_fo2"]

    # Filter invalid sigma
    df = df[np.isfinite(df["sig_scaled"])].copy()
    df = df[df["sig_scaled"] > args.min_sigma].copy()

    # Standardized residual (NO sqrt) as requested
    df["resid_std"] = (df["fo2_scaled"] - df["fcalc2"]) / df["sig_scaled"]

    # Helpful hover label
    df["hkl"] = df["h"].astype(str) + " " + df["k"].astype(str) + " " + df["l"].astype(str)

    # Sort by ZOBS for nicer viewing
    df = df.sort_values("zobs").reset_index(drop=True)

    print(f"Raw HKLF4 reflections:       {len(raw)}")
    print(f"FCF reflections:            {len(fcf)}")
    print(f"INTEGRATE ZOBS records:     {len(zdf)}")
    print(f"Matched & plotted points:   {len(df)}")

    fig = px.scatter(
        df,
        x="zobs",
        y="resid_std",
        title=f"Standardized residual vs ZOBS (scale s={s:g})",
        labels={"zobs": "ZOBS", "resid_std": "(s*Fo^2 - Fc^2)/(s*sigma(Fo^2))"},
        hover_data={
            "hkl": True,
            "zobs": ":.3f",
            "fo2": ":.6g",
            "sig_fo2": ":.6g",
            "fo2_scaled": ":.6g",
            "sig_scaled": ":.6g",
            "fcalc2": ":.6g",
            "h": False, "k": False, "l": False,
        },
    )

    # Make hover look clean
    fig.update_traces(
        marker=dict(size=6),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "ZOBS=%{x:.3f}<br>"
            "resid=%{y:.4f}<br>"
            "Fo^2=%{customdata[1]:.6g}, sigma=%{customdata[2]:.6g}<br>"
            "s*Fo^2=%{customdata[3]:.6g}, s*sigma=%{customdata[4]:.6g}<br>"
            "Fc^2=%{customdata[5]:.6g}"
            "<extra></extra>"
        )
    )
    out_dir = os.path.dirname(args.raw_hkl)
    html_path = os.path.join(out_dir, "residual_vs_zobs.html")
    fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"Wrote interactive plot: {html_path}")

if __name__ == "__main__":
    main()

# python residual_vs_zobs.py --raw_hkl /home/bubl3932/files/3DED-DATA/LTA/LTA4/unlocked/t4_no-error-model.hkl --fcf /home/bubl3932/files/3DED-DATA/LTA/LTA4/unlocked/t4_no-error-model.fcf --integrate /home/bubl3932/files/3DED-DATA/LTA/LTA4/xds/INTEGRATE.HKL --scale 1600