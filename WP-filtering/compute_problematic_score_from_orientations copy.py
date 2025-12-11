#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd

# python compute_problematic_score_from_orientations.py --orientations_csv LTA1_orientations.csv --problematic_csv CORRECT_problematic_axes_scored.csv --out_csv LTA1_orientations_scored.csv


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Return angle in radians between vectors v1 and v2."""
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0.0 or n2 == 0.0:
        return np.nan
    v1 /= n1
    v2 /= n2
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return float(np.arccos(dot))


def parse_problematic_csv(problematic_csv: str):
    """
    Read problematic axes from a CSV with columns: u,v,w,score
    Supports comment lines starting with '#'.
    Returns list of (u,v,w,score).
    """
    df = pd.read_csv(problematic_csv, comment="#")
    required = {"u", "v", "w", "Score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{problematic_csv} is missing columns: {sorted(missing)}")

    axes = []
    for _, r in df.iterrows():
        u, v, w = int(r["u"]), int(r["v"]), int(r["w"])
        sc = float(r["Score"])
        axes.append((u, v, w, sc))
    return axes


def main():
    ap = argparse.ArgumentParser(
        description="Compute problematic score for each orientation (like stream script) using predicted problematic axes."
    )
    ap.add_argument(
        "--orientations_csv",
        required=True,
        help="Path to orientations.csv (must contain a_x..a_z, b_x..b_z, c_x..c_z).",
    )
    ap.add_argument(
        "--problematic_csv",
        required=True,
        help="Path to problematic axes CSV (columns u,v,w,score; comments allowed with '#').",
    )
    ap.add_argument(
        "--out_csv",
        required=True,
        help="Output CSV path.",
    )
    ap.add_argument(
        "--beam",
        default="0,0,1",
        help="Beam direction in lab frame as 'x,y,z'. Default: 0,0,1",
    )
    ap.add_argument(
        "--min_angle_deg",
        type=float,
        default=0.0,
        help="If >0, require best angle <= this threshold; otherwise set scores to NaN for that row.",
    )
    args = ap.parse_args()

    beam_xyz = np.array([float(x) for x in args.beam.split(",")], dtype=float)
    if np.linalg.norm(beam_xyz) == 0.0:
        raise ValueError("--beam must be non-zero")

    ori = pd.read_csv(args.orientations_csv)
    needed = ["a_x", "a_y", "a_z", "b_x", "b_y", "b_z", "c_x", "c_y", "c_z"]
    missing = [c for c in needed if c not in ori.columns]
    if missing:
        raise ValueError(f"{args.orientations_csv} missing columns: {missing}")

    axes = parse_problematic_csv(args.problematic_csv)
    if not axes:
        raise ValueError("No axes found in problematic CSV.")

    # Pre-store axis list as arrays for speed
    axes_uvw = [(u, v, w) for (u, v, w, _) in axes]
    axes_sc = [sc for (_, _, _, sc) in axes]

    out_best_u = []
    out_best_v = []
    out_best_w = []
    out_best_sc = []
    out_best_angle_deg = []
    out_best_angle_norm = []
    out_ang_over_score = []
    out_best_score = []

    for _, r in ori.iterrows():
        astar = np.array([r["a_x"], r["a_y"], r["a_z"]], dtype=float)
        bstar = np.array([r["b_x"], r["b_y"], r["b_z"]], dtype=float)
        cstar = np.array([r["c_x"], r["c_y"], r["c_z"]], dtype=float)

        best_ang = np.inf
        best_idx = None

        # Find closest axis in angle to beam
        for i, (u, v, w) in enumerate(axes_uvw):
            axis_vec = u * astar + v * bstar + w * cstar
            ang = angle_between(axis_vec, beam_xyz)
            if np.isnan(ang):
                continue
            if ang < best_ang:
                best_ang = ang
                best_idx = i

        if best_idx is None or not np.isfinite(best_ang):
            out_best_u.append(np.nan)
            out_best_v.append(np.nan)
            out_best_w.append(np.nan)
            out_best_sc.append(np.nan)
            out_best_angle_deg.append(np.nan)
            out_best_angle_norm.append(np.nan)
            out_ang_over_score.append(np.nan)
            out_best_score.append(np.nan)
            continue

        best_angle_deg = np.degrees(best_ang)
        if args.min_angle_deg > 0 and best_angle_deg > args.min_angle_deg:
            # outside threshold: blank the scoring outputs
            out_best_u.append(np.nan)
            out_best_v.append(np.nan)
            out_best_w.append(np.nan)
            out_best_sc.append(np.nan)
            out_best_angle_deg.append(best_angle_deg)
            out_best_angle_norm.append(np.nan)
            out_ang_over_score.append(np.nan)
            out_best_score.append(np.nan)
            continue

        sc = float(axes_sc[best_idx])
        angle_norm = best_ang / (0.5 * np.pi)  # same normalization as stream script
        best_score = (1.0 - angle_norm) * sc   # same as stream script
        ang_over_score = angle_norm * (2.0 - sc)  # same as stream script's "problematic_score"

        u, v, w = axes_uvw[best_idx]
        out_best_u.append(u)
        out_best_v.append(v)
        out_best_w.append(w)
        out_best_sc.append(sc)
        out_best_angle_deg.append(best_angle_deg)
        out_best_angle_norm.append(angle_norm)
        out_ang_over_score.append(ang_over_score)
        out_best_score.append(best_score)

    out = ori.copy()
    out["best_u"] = out_best_u
    out["best_v"] = out_best_v
    out["best_w"] = out_best_w
    out["best_axis_score"] = out_best_sc
    out["best_angle_deg"] = out_best_angle_deg
    out["best_angle_norm"] = out_best_angle_norm
    # Matches the stream script's naming/meaning:
    out["problematic_score"] = out_ang_over_score
    out["best_score"] = out_best_score

    out.to_csv(args.out_csv, index=False)
    print(f"Wrote: {args.out_csv}")


if __name__ == "__main__":
    main()
