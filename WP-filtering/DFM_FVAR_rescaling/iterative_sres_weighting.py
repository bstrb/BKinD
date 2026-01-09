#!/usr/bin/env python3
import argparse
import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.express as px

# CCTBX imports
from iotbx.reflection_file_reader import any_reflection_file


def die(msg: str, code: int = 2):
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)

# -----------------------------
# Hardcoded defaults (as asked)
# -----------------------------

ISO_BASENAME = "iso"
ANISO_BASENAME = "aniso"
OUT_ROOTNAME = "iterative_sres"  # output folder created next to the input dir
 

# -----------------------------
# Utilities: running + parsing
# -----------------------------

def run_cmd(cmd, cwd: str):
    print(f"[run] (cwd={cwd}) {' '.join(cmd)}")
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # print(p.stdout)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed (code {p.returncode}): {' '.join(cmd)}")

def detect_npds_in_lst(lst_path: str) -> bool:
    if not os.path.exists(lst_path):
        raise RuntimeError(f"Missing LST: {lst_path}")
    needles = [
        "NON POSITIVE DEFINITE",
    ]
    with open(lst_path, "r", errors="replace") as f:
        text = f.read().upper()
    return any(n in text for n in needles)

def insert_anis_after_ls(ins_in: str, ins_out: str):
    """
    Create an ANISO ins from an ISO .res/.ins by:
      - Leaving L.S. unchanged
      - Inserting ANIS and EXYZ immediately after the first L.S. line
    """
    with open(ins_in, "r", errors="replace") as f:
        lines = f.readlines()

    has_anis = any(l.strip().upper() == "ANIS" for l in lines)
    has_exyz = any(l.strip().upper() == "EXYZ" for l in lines)

    out = []
    inserted = False

    for line in lines:
        out.append(line)

        if (not inserted) and line.strip().upper().startswith("L.S"):
            if not has_anis:
                out.append("ANIS\n")
            if not has_exyz:
                out.append("EXYZ\n")
            inserted = True

    # if not inserted:
    #     out.append("\nANIS\nEXYZ\n")
    #     print(
    #         f"WARNING: No 'L.S.' line found in {ins_in}; appended ANIS/EXYZ at end.",
    #         file=sys.stderr
    #     )

    with open(ins_out, "w") as f:
        f.writelines(out)

# -----------------------------
# INTEGRATE reader (zobs etc.)
# -----------------------------

class Integrate:
    def __init__(self, file_name: str):
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

# -----------------------------
# SRES computation
# -----------------------------

def calculate_sres(sample_df: pd.DataFrame) -> pd.Series:
    return (sample_df["Fo^2"] - sample_df["Fc^2"]) / sample_df["Fo^2_sigma"]

def read_fcf(file_path: str) -> pd.DataFrame:
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

    return pd.DataFrame({"Miller": miller_indices, "Fo^2": Fo2, "Fc^2": Fc2, "Fo^2_sigma": sigmas})

def build_refinement_df_with_sres(refine_dir: str, integrate_df: pd.DataFrame) -> pd.DataFrame:
    fcf_files = list(Path(refine_dir).glob("*.fcf"))
    if len(fcf_files) != 1:
        raise RuntimeError(f"Expected exactly one .fcf in {refine_dir}, found {len(fcf_files)}")
    fcf_path = str(fcf_files[0])

    fcf_df = read_fcf(fcf_path)
    df = fcf_df.merge(integrate_df, on="Miller", how="inner")
    if df.empty:
        die(f"Merge produced 0 rows for {refine_dir}. FCF and INTEGRATE.HKL do not match.")

    required = ["Miller", "asu", "Fc^2", "Fo^2", "Fo^2_sigma", "Resolution", "zobs"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        die(f"Missing expected columns after merge for {refine_dir}: {missing}")

    df = df[required].copy()

    df["SRES"] = calculate_sres(df)
    df["absSRES"] = np.abs(df["SRES"])
    return df


def build_compare_df(df_iso: pd.DataFrame, df_aniso: pd.DataFrame) -> pd.DataFrame:
    a = df_iso.copy().rename(columns={
        "Fo^2": "Fo^2_A", "Fc^2": "Fc^2_A", "Fo^2_sigma": "Fo^2_sigma_A",
        "SRES": "SRES_A", "absSRES": "absSRES_A",
    })
    b = df_aniso.copy().rename(columns={
        "Fo^2": "Fo^2_B", "Fc^2": "Fc^2_B", "Fo^2_sigma": "Fo^2_sigma_B",
        "SRES": "SRES_B", "absSRES": "absSRES_B",
    })

    keep_a = ["Miller", "Resolution", "zobs",
              "Fo^2_A", "Fc^2_A", "Fo^2_sigma_A", "SRES_A"]
    keep_b = ["Miller", "Fo^2_B", "Fc^2_B", "Fo^2_sigma_B", "SRES_B"]

    cmp_df = a[keep_a].merge(b[keep_b], on="Miller", how="inner")
    if cmp_df.empty:
        die("No overlapping Miller indices between ISO and ANISO refinements.")
        
    cmp_df["abs_dSRES"] = np.abs(cmp_df["SRES_A"] - cmp_df["SRES_B"])
    return cmp_df


# -----------------------------
# HKL scaling
# -----------------------------
def parse_shelx_hkl_line(line: str):
    """
    Fixed-column SHELX HKL: 4 4 4 8 8 4
    Returns (h,k,l,I,sig,flag_int).
    """
    if not line.strip():
        return None
    raw = line.rstrip("\n")
    if len(raw) < 28:
        return None

    try:
        h = int(raw[0:4])
        k = int(raw[4:8])
        l = int(raw[8:12])

        I_field = raw[12:20].strip()
        sig_field = raw[20:28].strip()

        I = float(I_field) if I_field else 0.0
        sig = float(sig_field) if sig_field else 0.0

        # flag is a fixed 4-char field; default to 1 if missing/blank
        flag_field = raw[28:32] if len(raw) >= 32 else ""
        flag_field = flag_field.strip()
        flag = int(flag_field) if flag_field else 1

        return h, k, l, I, sig, flag
    except ValueError:
        return None
def _format_8char_number(x: float, prefer_no_decimals: bool = False) -> str:
    if prefer_no_decimals:
        s = f"{int(round(x))}."
        if len(s) <= 8:
            return s.rjust(8)
    for ndp in (2, 1, 0):
        s = f"{x:.{ndp}f}"
        if len(s) <= 8:
            return s.rjust(8)
    for fmt in ("{:.2e}", "{:.1e}", "{:.0e}"):
        s = fmt.format(x)
        if len(s) <= 8:
            return s.rjust(8)
    return s[-8:].rjust(8)

def format_shelx_hkl_line(h: int, k: int, l: int, I: float, sig: float, flag: int) -> str:
    I_str = _format_8char_number(I, prefer_no_decimals=True)
    sig_str = _format_8char_number(sig, prefer_no_decimals=False)
    return f"{h:4d}{k:4d}{l:4d}{I_str}{sig_str}{flag:4d}\n"

def write_sigma_scaled_hkl(
    input_hkl: str,
    output_hkl: str,
    abs_dSRES_by_miller: Dict[Tuple[int, int, int], float],
):
    # Normalize by max -> [0, 1]
    if not abs_dSRES_by_miller:
        raise RuntimeError("abs_dSRES_by_miller is empty (no overlaps?)")

    max_abs = max(abs_dSRES_by_miller.values())
    if max_abs > 0.0:
        abs_dSRES_by_miller = {k: (v / max_abs) for k, v in abs_dSRES_by_miller.items()}
    else:
        abs_dSRES_by_miller = {k: 0.0 for k in abs_dSRES_by_miller}

    # NEW: if all normalized values are 0, then all factors are 1.0 (no scaling)
    if all(v == 0.0 for v in abs_dSRES_by_miller.values()):
        die("All sigma scale factors would be 1.0 (all |ΔSRES| normalized to 0). No HKL scaling would occur.")

    n_total = 0
    n_scaled = 0
    n_missing = 0

    with open(input_hkl, "r", errors="replace") as fin, open(output_hkl, "w") as fout:
        for line in fin:
            parsed = parse_shelx_hkl_line(line)
            if parsed is None:
                fout.write(line)
                continue
            
            n_total += 1

            h, k, l, I, sig, flag = parsed

            if h == 0 and k == 0 and l == 0:
                fout.write(line)
                continue

            key = (h, k, l)
            if key not in abs_dSRES_by_miller:
                n_missing += 1
                fout.write(format_shelx_hkl_line(h, k, l, I, sig, flag))
                continue

            a = float(abs_dSRES_by_miller[key])  # normalized
            sig_new = sig * (1.0 + a**2)
            n_scaled += 1
            fout.write(format_shelx_hkl_line(h, k, l, I, sig_new, flag))


    print(f"Sigma-scaled HKL written: {output_hkl}")
    print(f"HKL lines parsed (incl. 0 0 0): {n_total}")
    print(f"Scaled reflections: {n_scaled}")
    print(f"No abs_dSRES available (left unchanged): {n_missing}")

# -----------------------------
# Plotting
# -----------------------------

def plot_iteration(out_prefix: str, df_iso: pd.DataFrame, df_aniso: pd.DataFrame, cmp_df: pd.DataFrame):
    cmp_df.to_csv(f"{out_prefix}_SRES_compare_table.tsv", sep="\t", index=False)

    def safe_hover(df: pd.DataFrame, cols):
        """Return only hover columns that exist in df (prevents plotly KeyError)."""
        return [c for c in cols if c in df.columns]

    # ISO: SRES vs zobs
    fig_iso = px.scatter(
        df_iso, x="zobs", y="SRES",
        title="SRES vs zobs (ISO)",
        labels={"zobs": "Frame (zobs)", "SRES": "SRES"},
        hover_data=safe_hover(df_iso, ["Miller", "Resolution", "Fo^2", "Fc^2", "Fo^2_sigma", "asu"]),
    )
    fig_iso.write_html(f"{out_prefix}_ISO_SRES_vs_zobs.html")

    # ANISO: SRES vs zobs
    fig_aniso = px.scatter(
        df_aniso, x="zobs", y="SRES",
        title="SRES vs zobs (ANISO)",
        labels={"zobs": "Frame (zobs)", "SRES": "SRES"},
        hover_data=safe_hover(df_aniso, ["Miller", "Resolution", "Fo^2", "Fc^2", "Fo^2_sigma", "asu"]),
    )
    fig_aniso.write_html(f"{out_prefix}_ANISO_SRES_vs_zobs.html")

    # Compare: SRES_A vs SRES_B
    fig_cmp = px.scatter(
        cmp_df, x="SRES_A", y="SRES_B",
        title="SRES per reflection: ISO (x) vs ANISO (y)",
        labels={"SRES_A": "SRES (ISO)", "SRES_B": "SRES (ANISO)"},
        hover_data=safe_hover(
            cmp_df,
            ["Miller", "Resolution", "zobs", "abs_dSRES",
             "Fo^2_A", "Fc^2_A", "Fo^2_sigma_A",
             "Fo^2_B", "Fc^2_B", "Fo^2_sigma_B"]
        ),
    )
    fig_cmp.write_html(f"{out_prefix}_SRES_ISO_vs_SRES_ANISO.html")

    # Since dSRES is no longer computed, plot |ΔSRES| vs zobs instead
    fig_absd = px.scatter(
        cmp_df, x="zobs", y="abs_dSRES",
        title="|ΔSRES| vs zobs (|SRES_ISO - SRES_ANISO|)",
        labels={"zobs": "Frame (zobs)", "abs_dSRES": "|ΔSRES|"},
        hover_data=safe_hover(
            cmp_df,
            ["Miller", "Resolution", "SRES_A", "SRES_B",
             "Fo^2_A", "Fc^2_A", "Fo^2_sigma_A",
             "Fo^2_B", "Fc^2_B", "Fo^2_sigma_B"]
        ),
    )
    fig_absd.write_html(f"{out_prefix}_abs_dSRES_vs_zobs.html")

# -----------------------------
# Main iteration
# -----------------------------
def find_input_iso_pair(iso_dir: str) -> Tuple[str, str]:
    """
    Given a directory, find exactly one .ins and one .hkl to start from.
    """
    ins_files = list(Path(iso_dir).glob("*.ins"))
    hkl_files = list(Path(iso_dir).glob("*.hkl"))
    if len(ins_files) != 1:
        raise RuntimeError(f"Expected exactly one .ins in {iso_dir}, found {len(ins_files)}")
    if len(hkl_files) != 1:
        raise RuntimeError(f"Expected exactly one .hkl in {iso_dir}, found {len(hkl_files)}")
    return str(ins_files[0]), str(hkl_files[0])


def main():
    ap = argparse.ArgumentParser(
        description="Iteratively: ISO + ANISO refinements; if ANISO has NPDs, scale sigmas using 1+norm(|ΔSRES|) and repeat."
    )
    ap.add_argument("--integrate-hkl", required=True, help="Path to INTEGRATE.HKL (for zobs/xobs/yobs/resolution merge)")
    ap.add_argument("--iso-dir", required=True, help="Directory containing the starting ISO .ins and .hkl")
    ap.add_argument("--max-iter", type=int, default=5, help="Max number of iterations (default: 5)")
    args = ap.parse_args()

    integrate_hkl = os.path.abspath(args.integrate_hkl)
    start_dir = os.path.abspath(args.iso_dir)

    if not os.path.exists(integrate_hkl):
        die(f"Missing INTEGRATE.HKL: {integrate_hkl}")
    if not os.path.isdir(start_dir):
        die(f"--iso-ins must be a directory containing one .ins and one .hkl. Got: {start_dir}")

    start_ins, start_hkl = find_input_iso_pair(start_dir)
    print(f"[input] ISO start INS: {start_ins}")
    print(f"[input] ISO start HKL: {start_hkl}")

    # outputs next to input dir
    out_dir = os.path.join(start_dir, OUT_ROOTNAME)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[output] Root: {out_dir}")

    integrate_df = Integrate(integrate_hkl).as_df()

    # This HKL is updated each iteration (points to the most recent scaled HKL)
    hkl_current = os.path.abspath(start_hkl)

    # This INS is updated each iteration: ISO uses last ISO .res as next ISO .ins
    iso_ins_current = os.path.abspath(start_ins)

    stable_base = os.path.splitext(os.path.basename(start_hkl))[0]

    for it in range(1, args.max_iter + 1):
        iter_dir = os.path.join(out_dir, f"iter_{it:02d}")
        iso_dir = os.path.join(iter_dir, "ISO")
        aniso_dir = os.path.join(iter_dir, "ANISO")
        os.makedirs(iso_dir, exist_ok=True)
        os.makedirs(aniso_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print(f"ITERATION {it}")
        print(f"HKL current: {hkl_current}")
        print(f"ISO ins current: {iso_ins_current}")
        print("=" * 80)

        # -----------------
        # ISO refinement
        # -----------------
        # copy ins + hkl into ISO folder, named iso.ins / iso.hkl
        iso_ins_dst = os.path.join(iso_dir, f"{ISO_BASENAME}.ins")
        iso_hkl_dst = os.path.join(iso_dir, f"{ISO_BASENAME}.hkl")
        shutil.copy2(iso_ins_current, iso_ins_dst)
        shutil.copy2(hkl_current, iso_hkl_dst)

        run_cmd(["shelxl", ISO_BASENAME], cwd=iso_dir, )

        iso_res = os.path.join(iso_dir, f"{ISO_BASENAME}.res")
        if not os.path.exists(iso_res):
            die(f"ISO refinement did not produce: {iso_res}")

        # -----------------
        # ANISO refinement
        # -----------------
        # Use ISO .res as the starting point, but insert ANIS right after L.S.
        aniso_ins_dst = os.path.join(aniso_dir, f"{ANISO_BASENAME}.ins")
        aniso_hkl_dst = os.path.join(aniso_dir, f"{ANISO_BASENAME}.hkl")

        insert_anis_after_ls(iso_res, aniso_ins_dst)
        shutil.copy2(hkl_current, aniso_hkl_dst)

        run_cmd(["shelxl", ANISO_BASENAME], cwd=aniso_dir)

        # Update ISO ins for next iteration: use the ISO .res as next ISO .ins (in next iter's ISO folder)
        iso_ins_current = iso_res

        # NPD check on ANISO
        aniso_lst = os.path.join(aniso_dir, f"{ANISO_BASENAME}.lst")
        has_npd = detect_npds_in_lst(aniso_lst)
        print(f"[NPD check] ANISO has NPDs: {has_npd}")

        # -----------------
        # SRES compare + plots
        # -----------------
        df_iso = build_refinement_df_with_sres(iso_dir, integrate_df)
        df_aniso = build_refinement_df_with_sres(aniso_dir, integrate_df)
        cmp_df = build_compare_df(df_iso, df_aniso)

        out_prefix = os.path.join(iter_dir, f"iter_{it:02d}")
        plot_iteration(out_prefix, df_iso, df_aniso, cmp_df)

        # Stop condition: no NPDs in ANISO
        if not has_npd:
            print(f"[stop] No NPDs in ANISO at iteration {it}. Stopping.")
            break

        # -----------------
        # Sigma scale HKL for next iteration
        # -----------------
        abs_map = {tuple(m): float(a) for m, a in zip(cmp_df["Miller"].tolist(), cmp_df["abs_dSRES"].tolist())}

        # stable naming: <startbase>_sres_scaled_iterXX.hkl (no snowball)
        hkl_scaled = os.path.join(iter_dir, f"{stable_base}_sres_scaled_iter{it:02d}.hkl")

        write_sigma_scaled_hkl(
            input_hkl=hkl_current,
            output_hkl=hkl_scaled,
            abs_dSRES_by_miller=abs_map,
        )

        # Next iteration uses the newly scaled HKL
        hkl_current = hkl_scaled

    print("\nDone.")

if __name__ == "__main__":
    main()

# python3 iterative_sres_weighting.py --integrate-hkl /Users/xiaodong/Desktop/3DED-DATA/LTA/LTA4-with-EADP-EXYZ/xds/INTEGRATE.HKL --iso-dir /Users/xiaodong/Desktop/3DED-DATA/LTA/LTA4-with-EADP-EXYZ/shelx_iso 
 