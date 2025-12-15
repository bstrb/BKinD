#!/usr/bin/env python3
import argparse
import os
import sys
import shutil
import subprocess
import math
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import minimize

# CCTBX imports
from iotbx.reflection_file_reader import any_reflection_file
from iotbx.shelx.hklf import miller_array_export_as_shelx_hklf as hklf
from scitbx.array_family import flex


# -----------------------------
# Helpers / Utilities
# -----------------------------
def die(msg: str, code: int = 2):
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def unique_run_dir(out_dir: str, run_label: str | None = None) -> str:
    ensure_dir(out_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"run_{stamp}" if not run_label else f"run_{stamp}_{run_label}"
    run_dir = os.path.join(out_dir, base)
    i = 1
    while os.path.exists(run_dir):
        run_dir = os.path.join(out_dir, f"{base}_{i:02d}")
        i += 1
    ensure_dir(run_dir)
    return run_dir

def manage_files(action, source_directory, target_directory, filename=None, new_filename=None, extension=None):
    source_file_path = None

    if filename:
        source_file_path = os.path.join(source_directory, filename)
    elif extension:
        for f in os.listdir(source_directory):
            if f.endswith(extension):
                source_file_path = os.path.join(source_directory, f)
                break
        if not source_file_path:
            print(f"No file with extension '{extension}' found in '{source_directory}'.")
            return False
    else:
        print("Either 'filename' or 'extension' must be provided.")
        return False

    if not new_filename:
        new_filename = os.path.basename(source_file_path)

    target_file_path = os.path.join(target_directory, new_filename)
    ensure_dir(target_directory)

    try:
        if action == "move":
            shutil.move(source_file_path, target_file_path)
        elif action == "copy":
            shutil.copy(source_file_path, target_file_path)
        else:
            print(f"Invalid action '{action}'. Use 'move' or 'copy'.")
            return False
        return True
    except FileNotFoundError:
        print(f"Error: '{source_file_path}' not found.")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def modify_ins_file(file_path):
    modified_lines = []
    is_between_unit_fvar = False

    try:
        with open(file_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            line_lower = line.lower().strip()
            if "unit" in line_lower:
                is_between_unit_fvar = True
                modified_lines.append(line)
                continue
            elif "fvar" in line_lower:
                is_between_unit_fvar = False
                modified_lines.append(line)
                continue

            if is_between_unit_fvar:
                if "merg" in line_lower and "merge" not in line_lower:
                    continue
                if "fmap" in line_lower:
                    continue
                if "acta" in line_lower:
                    continue
                if line_lower.startswith("list"):
                    modified_lines.append("LIST 4\nMERG 0\nFMAP 2\nACTA\n")
                    continue

            modified_lines.append(line)

        with open(file_path, "w") as f:
            f.writelines(modified_lines)

    except Exception as e:
        print(f"An error occurred: {e}")

def parse_fvar_from_res(res_path: str) -> float:
    """
    Parses FVAR from a SHELX .res file line like:
      FVAR      14.66029
    Uses the first float after FVAR.
    """
    with open(res_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.upper().startswith("FVAR"):
                parts = s.split()
                for tok in parts[1:]:
                    try:
                        return float(tok)
                    except ValueError:
                        continue
                die(f"Found FVAR line but could not parse float: {s}")
    die(f"No FVAR line found in {res_path}")

# def parse_fvar_from_lst(lst_path: str) -> float:
#     """
#     Parses single-line:
#         FVAR      14.66029
#     Uses the first float after FVAR.
#     """
#     with open(lst_path, "r") as f:
#         for line in f:
#             s = line.strip()
#             if not s:
#                 continue
#             if s.upper().startswith("FVAR"):
#                 parts = s.split()
#                 floats = []
#                 for tok in parts[1:]:
#                     try:
#                         floats.append(float(tok))
#                     except ValueError:
#                         pass
#                 if not floats:
#                     die(f"Found FVAR line but could not parse float(s): {s}")
#                 return float(floats[0])
#     die(f"No FVAR line found in {lst_path}")

def format_line(parts, column_widths):
    formatted_parts = []
    for part, width in zip(parts, column_widths):
        formatted_parts.append(f"{part:>{width}}")
    return "".join(formatted_parts)

def read_xds_ascii_hkls(xds_ascii_path: str) -> list[tuple[int, int, int]]:
    hkls = []
    with open(xds_ascii_path, "r") as f:
        for line in f:
            if line.startswith("!"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                h, k, l = map(int, parts[:3])
            except ValueError:
                continue
            hkls.append((h, k, l))
    return hkls


# -----------------------------
# INTEGRATE.HKL reader
# -----------------------------
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


# -----------------------------
# NEM XDS_ASCII creation (fixed width) + filtering
# -----------------------------
def create_xds_ascii_nem_filtered(xds_dir: str, target_dir: str, keep_set: set[tuple[int, int, int]]):
    xds_path = os.path.join(xds_dir, "XDS_ASCII.HKL")
    integrate_path = os.path.join(xds_dir, "INTEGRATE.HKL")
    output_path = os.path.join(target_dir, "XDS_ASCII_NEM.HKL")

    column_widths = [6, 6, 6, 11, 11, 8, 8, 9, 10, 4, 4, 8]

    if not os.path.exists(xds_path):
        die(f"Missing {xds_path}")
    if not os.path.exists(integrate_path):
        die(f"Missing {integrate_path}")

    integrate_dict: dict[tuple[int, int, int], list[float]] = {}
    with open(integrate_path, "r") as f:
        for line in f:
            if line.startswith("!"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                h, k, l = map(int, parts[:3])
                vals = [float(p) for p in parts]
            except ValueError:
                continue
            integrate_dict[(h, k, l)] = vals

    output_data = []
    with open(xds_path, "r") as f:
        for line in f:
            if line.startswith("!"):
                output_data.append(line)
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            try:
                h, k, l = map(int, parts[:3])
            except ValueError:
                continue

            hkl = (h, k, l)
            if hkl not in keep_set:
                continue

            if hkl in integrate_dict and len(parts) > 4:
                try:
                    xds_I = float(parts[3])
                    inte_vals = integrate_dict[hkl]
                    calculated_sigma = (xds_I / inte_vals[3]) * inte_vals[4]
                    parts[4] = f"{calculated_sigma:.3E}"
                except (IndexError, ZeroDivisionError, ValueError):
                    pass

            # Keep fixed-width *only if* token count matches widths.
            if len(parts) != len(column_widths):
                output_data.append(line if line.endswith("\n") else (line + "\n"))
            else:
                out_line = format_line(parts, column_widths)
                output_data.append(out_line + "\n")

    with open(output_path, "w") as f:
        f.writelines(output_data)

    return output_path

def create_xdsconv_inp(target_directory: str):
    xdsconv_file_path = os.path.join(target_directory, "XDSCONV.INP")
    with open(xdsconv_file_path, "w") as f:
        f.write("INPUT_FILE=XDS_ASCII_NEM.HKL\n")
        f.write("OUTPUT_FILE=bkind.hkl SHELX ! or CCP4_I or CCP4_F or SHELX or CNS\n")
        f.write("FRIEDEL'S_LAW=FALSE ! store anomalous signal in output file even if weak\n")
        f.write("MERGE=FALSE\n")
    return xdsconv_file_path


# -----------------------------
# DFM logic
# -----------------------------
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

def build_sample_df_with_dfm(iter_dir: str, inte_path: str, initial_u: float) -> pd.DataFrame:
    fcf_path = os.path.join(iter_dir, "bkind.fcf")
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

    sample_df["DFM"] = calculate_dfm(optimal_u, sample_df)
    sample_df["absDFM"] = np.abs(sample_df["DFM"])
    sample_df.attrs["optimal_u"] = optimal_u
    return sample_df


# -----------------------------
# SHELX HKL scaling
# -----------------------------
def scale_shelx_hkl_file(in_path: str, out_path: str,
                         hkl_to_factor: dict[tuple[int,int,int], float],
                         default_factor: float):

    n_scaled = 0
    n_lines = 0

    with open(in_path, "r") as fin, open(out_path, "w") as fout:
        for line in fin:
            n_lines += 1

            # Keep comments/blank
            if not line.strip() or line.lstrip().startswith(("!", "#")):
                fout.write(line)
                continue

            # Try strict SHELX fixed columns: 3I4,2F8.2 plus tail
            try:
                h = int(line[0:4]); k = int(line[4:8]); l = int(line[8:12])
                I = float(line[12:20]); sig = float(line[20:28])
                tail = line[28:].rstrip("\n")
            except Exception:
                # Fallback: whitespace parse, but still write fixed widths
                parts = line.split()
                if len(parts) < 5:
                    fout.write(line)
                    continue
                try:
                    h, k, l = map(int, parts[:3])
                    I = float(parts[3]); sig = float(parts[4])
                    tail = ""  # don't guess tail in fallback
                except Exception:
                    fout.write(line)
                    continue

            if (h, k, l) == (0, 0, 0):
                fout.write(line)
                continue

            factor = hkl_to_factor.get((h, k, l), default_factor)
            I_new = I * factor
            sig_new = sig * factor

            fout.write(f"{h:4d}{k:4d}{l:4d}{I_new:8.2f}{sig_new:8.2f}{tail}\n")
            n_scaled += 1

    return n_scaled, n_lines


# -----------------------------
# Iteration mechanics
# -----------------------------
def prepare_run_ins(run_dir: str, shelx_dir: str):
    ok = manage_files("copy", shelx_dir, run_dir, extension=".ins", new_filename="bkind.ins")
    if not ok:
        die(f"Could not copy .ins from {shelx_dir}")
    modify_ins_file(os.path.join(run_dir, "bkind.ins"))

def run_xdsconv_and_shelxl(iter_dir: str, xds_dir: str, keep_set: set[tuple[int,int,int]]):
    ensure_dir(iter_dir)

    create_xds_ascii_nem_filtered(xds_dir, iter_dir, keep_set)
    create_xdsconv_inp(iter_dir)

    p = subprocess.run(["xdsconv"], cwd=iter_dir)
    if p.returncode != 0:
        die(f"xdsconv failed in {iter_dir} with exit code {p.returncode}")

    if not os.path.exists(os.path.join(iter_dir, "bkind.hkl")):
        die("xdsconv did not produce bkind.hkl")

    p = subprocess.run(["shelxl", "bkind"], cwd=iter_dir)
    if p.returncode != 0:
        die(f"shelxl failed in {iter_dir} with exit code {p.returncode}")

    if not os.path.exists(os.path.join(iter_dir, "bkind.fcf")):
        die("shelxl did not produce bkind.fcf")
    if not os.path.exists(os.path.join(iter_dir, "bkind.lst")):
        die("shelxl did not produce bkind.lst")


# -----------------------------
# Main pipeline
# -----------------------------
def iterative_dfm_filter_rescale_and_final_validate(
    shelx_dir: str,
    xds_dir: str,
    out_dir: str,
    remove_fraction: float,
    fvar_tol: float,
    max_iters: int,
    initial_u: float,
    min_remaining: int,
    run_label: str | None,
    make_plots: bool,
):
    shelx_dir = os.path.abspath(shelx_dir)
    xds_dir = os.path.abspath(xds_dir)
    out_dir = os.path.abspath(out_dir)

    if not (0.0 < remove_fraction < 1.0):
        die("--remove-fraction must be between 0 and 1 (exclusive)")

    run_dir = unique_run_dir(out_dir, run_label=run_label)
    print(f"Run directory: {run_dir}")

    # Copy INTEGRATE.HKL into run_dir (never touch originals)
    ok = manage_files("copy", xds_dir, run_dir, filename="INTEGRATE.HKL", new_filename="INTEGRATE.HKL")
    if not ok:
        die(f"Could not copy INTEGRATE.HKL from {xds_dir}")
    inte_path = os.path.join(run_dir, "INTEGRATE.HKL")

    # Prepare fixed bkind.ins once
    prepare_run_ins(run_dir, shelx_dir)

    # Initial keep set from XDS_ASCII.HKL
    xds_ascii_path = os.path.join(xds_dir, "XDS_ASCII.HKL")
    if not os.path.exists(xds_ascii_path):
        die(f"Missing {xds_ascii_path}")

    all_hkls = read_xds_ascii_hkls(xds_ascii_path)
    keep_set: set[tuple[int, int, int]] = set(all_hkls)
    if len(keep_set) == 0:
        die("No HKLs found in XDS_ASCII.HKL")

    removal_rows = []
    fvar_hist = []

    prev_fvar = None
    final_fvar = None

    for it in range(max_iters):
        if len(keep_set) <= min_remaining:
            print(f"Stopping: keep_set size {len(keep_set)} <= min_remaining {min_remaining}")
            break

        iter_dir = os.path.join(run_dir, f"iter_{it:03d}")
        ensure_dir(iter_dir)

        # Copy fixed ins into iteration dir
        shutil.copy(os.path.join(run_dir, "bkind.ins"), os.path.join(iter_dir, "bkind.ins"))

        print(f"\n=== Iteration {it} ===")
        print(f"Keeping {len(keep_set)} reflections")

        run_xdsconv_and_shelxl(iter_dir, xds_dir, keep_set)

        # fvar = parse_fvar_from_lst(os.path.join(iter_dir, "bkind.lst"))
        fvar = parse_fvar_from_res(os.path.join(iter_dir, "bkind.res"))
        final_fvar = fvar
        fvar_hist.append({"iteration": it, "FVAR": fvar, "n_keep": len(keep_set)})
        print(f"FVAR: {fvar:.6g}")

        sample_df = build_sample_df_with_dfm(iter_dir, inte_path, initial_u=initial_u)
        opt_u = float(sample_df.attrs.get("optimal_u", np.nan))
        print(f"Optimal u used: {opt_u:.6g}")

        sample_csv = os.path.join(iter_dir, "sample_df_with_dfm.csv")
        sample_df.to_csv(sample_csv, index=False)
        print(f"Saved: {sample_csv}")

        if make_plots:
            fig = px.scatter(
                sample_df,
                x="zobs",
                y="DFM",
                title=f"DFM vs Frame (iter {it:03d})",
                labels={"zobs": "Frame (zobs)", "DFM": "DFM"},
            )
            out_html = os.path.join(iter_dir, f"DFM_vs_frame_iter_{it:03d}.html")
            fig.write_html(out_html)
            print(f"Saved plot: {out_html}")

        if prev_fvar is not None:
            delta = abs(fvar - prev_fvar)
            print(f"|ΔFVAR| vs previous: {delta:.6g} (tol={fvar_tol})")
            if delta <= fvar_tol:
                print("Converged: FVAR change within tolerance. Stopping iterations.")
                break

        # Remove top fraction by abs(DFM)
        n_candidates = len(sample_df)
        n_remove = int(math.ceil(remove_fraction * n_candidates))
        n_remove = max(1, n_remove)

        if len(keep_set) - n_remove < min_remaining:
            n_remove = max(0, len(keep_set) - min_remaining)
            print(f"Adjusted n_remove to {n_remove} to respect min_remaining={min_remaining}")

        if n_remove <= 0:
            print("No removals performed (would violate min_remaining). Stopping.")
            break

        worst = sample_df.sort_values("absDFM", ascending=False).head(n_remove)
        remove_hkls = list(worst["Miller"])

        for hkl, dfm, absdfm in zip(worst["Miller"], worst["DFM"], worst["absDFM"]):
            removal_rows.append({
                "h": int(hkl[0]),
                "k": int(hkl[1]),
                "l": int(hkl[2]),
                "iteration_removed": it,
                "DFM_at_removal": float(dfm),
                "absDFM_at_removal": float(absdfm),
                "FVAR_at_removal": float(fvar),
            })

        before = len(keep_set)
        for hkl in remove_hkls:
            keep_set.discard(tuple(map(int, hkl)))
        after = len(keep_set)
        print(f"Removed {before - after} reflections (target {n_remove})")

        prev_fvar = fvar

    if final_fvar is None:
        die("No iterations completed; final FVAR is undefined.")

    # Save logs
    removal_df = pd.DataFrame(removal_rows)
    fvar_df = pd.DataFrame(fvar_hist)
    removal_csv = os.path.join(run_dir, "removed_hkls_with_fvar.csv")
    fvar_csv = os.path.join(run_dir, "fvar_history.csv")
    removal_df.to_csv(removal_csv, index=False)
    fvar_df.to_csv(fvar_csv, index=False)
    print(f"\nSaved removal log: {removal_csv}")
    print(f"Saved FVAR history: {fvar_csv}")

    remaining_df = pd.DataFrame([{"h": h, "k": k, "l": l} for (h, k, l) in sorted(keep_set)])
    remaining_csv = os.path.join(run_dir, "remaining_hkls.csv")
    remaining_df.to_csv(remaining_csv, index=False)
    print(f"Saved remaining HKLs: {remaining_csv}")
    print(f"Final FVAR used for remaining reflections: {final_fvar:.6g}")

    # Build HKL -> FVAR_at_removal mapping for removed reflections
    removed_map: dict[tuple[int, int, int], float] = {}
    for row in removal_rows:
        removed_map[(row["h"], row["k"], row["l"])] = float(row["FVAR_at_removal"])

    # Normalize scale factors to avoid blowing up I/SIG:
    # scale(hkl) = FVAR_at_removal / FVAR_max   (removed)
    # scale(default) = final_fvar / FVAR_max    (remaining)
    fvar_max = max([final_fvar] + [float(r["FVAR_at_removal"]) for r in removal_rows]) if removal_rows else final_fvar
    removed_map_norm = {hkl: (fvar / fvar_max) for hkl, fvar in removed_map.items()}
    default_factor_norm = final_fvar / fvar_max
    print(f"Using normalized scaling with FVAR_max={fvar_max:.6g} (default scale={default_factor_norm:.6g})")

    # -----------------------------
    # 2) FINAL validation run:
    #    redo xdsconv + shelxl + DFM after applying per-HKL scaling to bkind.hkl
    # -----------------------------
    final_dir = os.path.join(run_dir, "final_after_rescaling")
    ensure_dir(final_dir)

    # Use ALL reflections for the final run (including previously removed),
    # because they are now scaled (down/up) according to saved FVAR_at_removal / final FVAR.
    all_set = set(all_hkls)

    # Copy fixed .ins
    shutil.copy(os.path.join(run_dir, "bkind.ins"), os.path.join(final_dir, "bkind.ins"))

    print(f"\n=== Final validation run (after rescaling) ===")
    print(f"Using all reflections: {len(all_set)}")

    # Run xdsconv + shelxl ONCE to create baseline bkind.hkl, then replace with scaled version
    create_xds_ascii_nem_filtered(xds_dir, final_dir, all_set)
    create_xdsconv_inp(final_dir)

    p = subprocess.run(["xdsconv"], cwd=final_dir)
    if p.returncode != 0:
        die(f"xdsconv failed in {final_dir} with exit code {p.returncode}")

    bkind_hkl = os.path.join(final_dir, "bkind.hkl")
    if not os.path.exists(bkind_hkl):
        die("xdsconv did not produce bkind.hkl in final validation run")

    # Preserve the unscaled bkind.hkl inside final_dir
    unscaled = os.path.join(final_dir, "bkind_unscaled.hkl")
    shutil.move(bkind_hkl, unscaled)

    # Write scaled bkind.hkl for SHELXL using your rule:
    # removed HKLs: I_new = I * FVAR_at_removal
    # remaining HKLs: I_new = I * FVAR_final
    # n_scaled, n_lines = scale_shelx_hkl_file(unscaled, bkind_hkl, removed_map_norm, default_factor=default_factor_norm)
    n_scaled, n_lines = scale_shelx_hkl_file(unscaled, bkind_hkl, removed_map_norm, default_factor=default_factor_norm)
    print(f"Scaled final bkind.hkl: scaled {n_scaled} reflections / {n_lines} lines")
    print(f"Preserved unscaled: {unscaled}")

    # Run SHELXL on the scaled bkind.hkl
    p = subprocess.run(["shelxl", "bkind"], cwd=final_dir)
    if p.returncode != 0:
        die(f"shelxl failed in {final_dir} with exit code {p.returncode}")

    if not os.path.exists(os.path.join(final_dir, "bkind.fcf")):
        die("shelxl did not produce bkind.fcf in final validation run")
    if not os.path.exists(os.path.join(final_dir, "bkind.lst")):
        die("shelxl did not produce bkind.lst in final validation run")

    # fvar_final_run = parse_fvar_from_lst(os.path.join(final_dir, "bkind.lst"))
    fvar_final_run = parse_fvar_from_res(os.path.join(final_dir, "bkind.res"))
    print(f"Final-run FVAR (from .lst after rescaling): {fvar_final_run:.6g}")

    sample_df_final = build_sample_df_with_dfm(final_dir, inte_path, initial_u=initial_u)
    opt_u_final = float(sample_df_final.attrs.get("optimal_u", np.nan))
    print(f"Final-run optimal u used: {opt_u_final:.6g}")

    final_csv = os.path.join(final_dir, "sample_df_with_dfm.csv")
    sample_df_final.to_csv(final_csv, index=False)
    print(f"Saved final sample_df: {final_csv}")

    if make_plots:
        fig = px.scatter(
            sample_df_final,
            x="zobs",
            y="DFM",
            title="DFM vs Frame (final run after rescaling)",
            labels={"zobs": "Frame (zobs)", "DFM": "DFM"},
        )
        out_html = os.path.join(final_dir, "DFM_vs_frame_final_after_rescaling.html")
        fig.write_html(out_html)
        print(f"Saved final plot: {out_html}")

    print(f"\nFinal validation directory: {final_dir}")


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Iterative |DFM| removal until FVAR converges, rescale raw .hkl by per-HKL FVAR_at_removal/final FVAR, then final xdsconv+shelxl+DFM validation."
    )
    ap.add_argument("--shelx-dir", required=True, help="Directory containing template .ins (copied to bkind.ins)")
    ap.add_argument("--xds-dir", required=True, help="Directory containing XDS_ASCII.HKL and INTEGRATE.HKL")
    ap.add_argument("--out-dir", required=True, help="Output directory; a unique run_* subfolder will be created")

    ap.add_argument("--remove-fraction", type=float, default=0.01, help="Fraction removed per iteration by |DFM| (default 0.01)")
    ap.add_argument("--fvar-tol", type=float, default=0.1, help="Absolute tolerance for FVAR convergence (default 0.1)")
    ap.add_argument("--max-iters", type=int, default=50, help="Max iterations (default 50)")
    ap.add_argument("--initial-u", type=float, default=0.01, help="Initial u for DFM optimization (default 0.01)")
    ap.add_argument("--min-remaining", type=int, default=0, help="Stop if remaining reflections <= this (default 0)")
    ap.add_argument("--run-label", default=None, help="Optional label appended to run folder name")
    ap.add_argument("--no-plots", action="store_true", help="Disable Plotly HTML plots")

    args = ap.parse_args()

    iterative_dfm_filter_rescale_and_final_validate(
        shelx_dir=args.shelx_dir,
        xds_dir=args.xds_dir,
        out_dir=args.out_dir,
        remove_fraction=args.remove_fraction,
        fvar_tol=args.fvar_tol,
        max_iters=args.max_iters,
        initial_u=args.initial_u,
        min_remaining=args.min_remaining,
        run_label=args.run_label,
        make_plots=not args.no_plots,
    )

if __name__ == "__main__":
    main()
