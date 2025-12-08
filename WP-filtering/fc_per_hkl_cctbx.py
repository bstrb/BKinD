#!/usr/bin/env python3
import argparse
import math
import csv

from cctbx import miller
from cctbx import uctbx
from cctbx.array_family import flex
from iotbx import shelx


def read_hkl_hklf4(path):
    """
    Read HKLF4-style .hkl:
    h k l Fo^2 sig(Fo^2) [batch/flag...]
    Terminator: 0 0 0 0 0 0 (often last column 0 too)
    Returns: indices(list of (h,k,l)), fo2(list), sig(list)
    """
    idx = []
    fo2 = []
    sig = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 5:
                continue
            h, k, l = int(parts[0]), int(parts[1]), int(parts[2])
            I = float(parts[3])
            sI = float(parts[4])

            # common terminator
            if h == 0 and k == 0 and l == 0 and abs(I) < 1e-12 and abs(sI) < 1e-12:
                break

            idx.append((h, k, l))
            fo2.append(I)
            sig.append(sI)
    return idx, fo2, sig


def inv_d2_from_cell(cell: uctbx.unit_cell, hkl):
    return cell.d_star_sq(hkl)  # 1/d^2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ins", required=True, help="SHELX .ins or .res containing the (refined) model")
    ap.add_argument("--hkl", required=True, help="HKLF4 .hkl with measured reflections")
    ap.add_argument("--out_csv", required=True, help="Output CSV with Fc per measured HKL")
    ap.add_argument("--min_fc2", type=float, default=1e-12, help="Floor to consider Fc^2 nonzero")
    args = ap.parse_args()

    # --- Read structure from SHELX file ---
    # Works for .ins or .res; uses SFAC/UNIT + atoms + CELL/LATT/SYMM.
    reader = shelx.reader(file_name=args.ins)
    xs = reader.xray_structure()

    cs = xs.crystal_symmetry()
    uc = cs.unit_cell()
    sg = cs.space_group()

    print(f"# Read model from: {args.ins}")
    print(f"# Unit cell: {uc.parameters()}")
    print(f"# Space group: {sg.info()}")

    # --- Read measured reflections ---
    indices, fo2, sig = read_hkl_hklf4(args.hkl)
    print(f"# Read HKL: {args.hkl}")
    print(f"# Measured reflections (non-terminator): {len(indices)}")

    # --- Build miller set for exactly those indices ---
    miller_indices = flex.miller_index(indices)

    # anomalous_flag=True preserves Friedel mates separately if present in file
    ms = miller.set(
        crystal_symmetry=cs,
        indices=miller_indices,
        anomalous_flag=True,
    )

    # --- Compute Fc for those indices ---
    fc_array = xs.structure_factors(miller_set=ms).f_calc()
    fc = fc_array.data()  # flex.complex_double

    # --- Write CSV ---
    with open(args.out_csv, "w", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(["h", "k", "l", "Fo2", "sigFo2", "Fc", "Fc2", "lnK", "s2", "d_A"])

        n_written = 0
        n_skipped_fc = 0

        for i, hkl in enumerate(indices):
            I = fo2[i]
            sI = sig[i]
            fci = fc[i]
            Fc = abs(fci)  # amplitude
            Fc2 = Fc * Fc

            inv_d2 = inv_d2_from_cell(uc, hkl)  # 1/d^2
            d = (1.0 / math.sqrt(inv_d2)) if inv_d2 > 0 else float("nan")
            s2 = inv_d2 / 4.0 if inv_d2 > 0 else float("nan")

            lnK = ""
            if I > 0.0 and Fc2 > args.min_fc2:
                lnK = math.log(I / Fc2)
            elif Fc2 <= args.min_fc2:
                n_skipped_fc += 1

            w.writerow([hkl[0], hkl[1], hkl[2], I, sI, Fc, Fc2, lnK, s2, d])
            n_written += 1

    print(f"# Wrote: {args.out_csv}")
    print(f"# Rows written: {n_written}")
    print(f"# Reflections with Fc^2 <= {args.min_fc2}: {n_skipped_fc}")


if __name__ == "__main__":
    main()
