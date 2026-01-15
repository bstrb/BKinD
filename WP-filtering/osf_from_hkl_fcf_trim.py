#!/usr/bin/env python3
"""
osf_from_hkl_fcf_trim.py

Recompute SHELXL FVAR(1) (overall scale factor, OSF) from:
- HKLF4 .hkl file: h k l Fo^2 sigma(Fo^2)
- .fcf file (CIF-like): must contain h,k,l and either Fc or Fc^2

Objective (WGHT 0 0):
  minimize sum_h ( (Fo^2 - k*Fc^2)^2 / sigma^2 )

Closed-form solution:
  k = sum(w * Fc2 * Fo2) / sum(w * Fc2^2),  w = 1/sigma^2

Robust trimming:
  compute k, compute normalized residuals r = (Fo2 - k*Fc2)/sigma,
  reject a fraction of largest |r|, recompute k on remaining.
  Repeat for a few iterations if requested.

Usage examples:
  python osf_from_hkl_fcf_trim.py data.hkl model.fcf --fractions 0 0.001 0.005 0.01 0.02 --iters 2
  python osf_from_hkl_fcf_trim.py data.hkl model.fcf --fraction 0.01 --iters 3

Notes:
- Matches reflections by exact (h,k,l).
- Keeps Fo^2 negative values (SHELX-style); you can optionally filter them.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class HKLObs:
    fo2: float
    sig: float


def read_hkl_hklf4(path: str) -> Dict[Tuple[int, int, int], HKLObs]:
    out: Dict[Tuple[int, int, int], HKLObs] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                h, k, l = int(parts[0]), int(parts[1]), int(parts[2])
                fo2 = float(parts[3])
                sig = float(parts[4])
            except ValueError:
                continue
            # SHELX HKL terminator often: 0 0 0 0 0
            if h == 0 and k == 0 and l == 0 and fo2 == 0 and sig == 0:
                break
            if sig <= 0:
                sig = 1e-6
            out[(h, k, l)] = HKLObs(fo2=fo2, sig=sig)
    if not out:
        raise ValueError(f"No reflections read from {path}")
    return out


def _is_data_tag(tok: str) -> bool:
    return tok.startswith("_")


def _tokenize_cif_line(line: str) -> List[str]:
    # Minimal CIF tokenization: split on whitespace, ignore comments.
    # (Good enough for typical .fcf loops.)
    if "#" in line:
        line = line.split("#", 1)[0]
    return line.strip().split()


def read_fcf_fc2(path: str) -> Dict[Tuple[int, int, int], float]:
    """
    Parse an .fcf (CIF-like) and return dict (h,k,l) -> Fc2.

    Accepts either:
      _refln_F_squared_calc  (Fc^2)
    or
      _refln_F_calc          (Fc)  -> squares it
    Also commonly present:
      _refln_index_h, _refln_index_k, _refln_index_l

    Implementation: scan for CIF loops and look for a loop that contains h,k,l and Fc or Fc^2.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    i = 0
    n = len(lines)

    best: Dict[Tuple[int, int, int], float] = {}

    while i < n:
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        if line.lower().startswith("loop_"):
            # read tags
            i += 1
            tags: List[str] = []
            while i < n:
                tline = lines[i].strip()
                if not tline:
                    i += 1
                    continue
                toks = _tokenize_cif_line(tline)
                if toks and _is_data_tag(toks[0]):
                    tags.append(toks[0])
                    i += 1
                    continue
                break

            # if no tags, continue
            if not tags:
                continue

            # figure columns
            tag_lc = [t.lower() for t in tags]

            def find_tag(*cands: str) -> Optional[int]:
                for c in cands:
                    c = c.lower()
                    if c in tag_lc:
                        return tag_lc.index(c)
                return None

            ih = find_tag("_refln_index_h", "_diffrn_refln_index_h")
            ik = find_tag("_refln_index_k", "_diffrn_refln_index_k")
            il = find_tag("_refln_index_l", "_diffrn_refln_index_l")

            ifc2 = find_tag("_refln_f_squared_calc", "_refln_f_squared_calc_esd")  # esd unlikely here
            ifc = find_tag("_refln_f_calc")

            has_hkl = (ih is not None and ik is not None and il is not None)
            has_fc = (ifc2 is not None) or (ifc is not None)

            # read data rows until next loop_/tag or stop
            rows: List[List[str]] = []
            while i < n:
                tline = lines[i].strip()
                if not tline:
                    i += 1
                    continue
                low = tline.lower()
                if low.startswith("loop_") or (tline and tline[0] == "_"):
                    break
                toks = _tokenize_cif_line(tline)
                if toks:
                    rows.append(toks)
                i += 1

            if has_hkl and has_fc and rows:
                # try to parse this loop into (h,k,l)->Fc2
                tmp: Dict[Tuple[int, int, int], float] = {}
                for toks in rows:
                    if len(toks) < len(tags):
                        # CIF values can wrap; MVP: skip incomplete lines
                        continue
                    try:
                        h = int(float(toks[ih]))  # some write as "1.0"
                        k = int(float(toks[ik]))
                        l = int(float(toks[il]))
                        if ifc2 is not None:
                            fc2 = float(toks[ifc2])
                        else:
                            fc = float(toks[ifc])  # type: ignore[arg-type]
                            fc2 = fc * fc
                    except Exception:
                        continue
                    tmp[(h, k, l)] = fc2

                # keep the biggest matching loop found so far
                if len(tmp) > len(best):
                    best = tmp

            continue

        i += 1

    if not best:
        raise ValueError(
            f"Could not find a suitable CIF loop in {path} containing h,k,l and Fc or Fc^2.\n"
            "Expected tags like _refln_index_h/_k/_l and _refln_F_squared_calc or _refln_F_calc."
        )
    return best


def compute_k_weighted_ls(fo2: np.ndarray, sig: np.ndarray, fc2: np.ndarray) -> float:
    # WGHT 0 0: w = 1/sig^2
    w = 1.0 / (sig * sig)
    num = np.sum(w * fc2 * fo2)
    den = np.sum(w * fc2 * fc2)
    if den <= 0:
        raise ValueError("Denominator in k computation is non-positive. Check Fc^2 values.")
    return float(num / den)


def trimmed_scale(
    fo2: np.ndarray,
    sig: np.ndarray,
    fc2: np.ndarray,
    reject_fraction: float,
    iters: int,
    keep_positive_fc2: bool = True,
) -> Tuple[float, int]:
    """
    Compute k with iterative trimming.

    reject_fraction: fraction to drop (0..0.9) based on largest |(Fo2 - k*Fc2)/sig|
    iters: number of trim iterations (>=1)

    Returns: (k, n_kept)
    """
    if reject_fraction < 0 or reject_fraction >= 0.95:
        raise ValueError("reject_fraction must be in [0, 0.95).")

    mask = np.ones_like(fo2, dtype=bool)
    if keep_positive_fc2:
        mask &= (fc2 > 0)

    for _ in range(max(iters, 1)):
        idx = np.where(mask)[0]
        if idx.size < 10:
            raise ValueError("Too few reflections left after masking/trimming.")

        k = compute_k_weighted_ls(fo2[idx], sig[idx], fc2[idx])

        if reject_fraction <= 0:
            return k, int(idx.size)

        r = (fo2[idx] - k * fc2[idx]) / sig[idx]  # normalized residual for WGHT 0 0
        absr = np.abs(r)
        n_keep = int(round((1.0 - reject_fraction) * idx.size))
        n_keep = max(n_keep, 10)

        # keep the smallest |r|
        keep_local = np.argpartition(absr, n_keep - 1)[:n_keep]
        new_mask = np.zeros_like(mask)
        new_mask[idx[keep_local]] = True
        mask = new_mask

    # final recompute on last mask
    idx = np.where(mask)[0]
    k = compute_k_weighted_ls(fo2[idx], sig[idx], fc2[idx])
    return k, int(idx.size)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("hkl", help="HKLF4 .hkl file (h k l Fo^2 sigFo^2)")
    ap.add_argument("fcf", help=".fcf file containing Fc or Fc^2 (CIF loop)")
    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument("--fraction", type=float, default=None, help="Single reject fraction, e.g. 0.01 for 1%")
    g.add_argument("--fractions", type=float, nargs="+", default=None,
                   help="List of reject fractions, e.g. 0 0.001 0.005 0.01 0.02")
    ap.add_argument("--iters", type=int, default=2, help="Trim iterations per fraction (default 2)")
    ap.add_argument("--min-sig", type=float, default=1e-6, help="Floor for sigma (default 1e-6)")
    ap.add_argument("--drop-neg-fo2", action="store_true",
                    help="Optionally drop reflections with Fo^2 < 0 before scaling")
    args = ap.parse_args()

    obs = read_hkl_hklf4(args.hkl)
    fc2_map = read_fcf_fc2(args.fcf)

    # match
    keys = sorted(set(obs.keys()) & set(fc2_map.keys()))
    if not keys:
        raise SystemExit("No overlapping (h,k,l) between .hkl and .fcf.")

    fo2 = np.array([obs[k].fo2 for k in keys], dtype=float)
    sig = np.array([max(obs[k].sig, args.min_sig) for k in keys], dtype=float)
    fc2 = np.array([fc2_map[k] for k in keys], dtype=float)

    # optional filter Fo^2 < 0
    base_mask = np.ones_like(fo2, dtype=bool)
    if args.drop_neg_fo2:
        base_mask &= (fo2 >= 0)

    fo2 = fo2[base_mask]
    sig = sig[base_mask]
    fc2 = fc2[base_mask]

    print(f"Matched reflections: {len(keys)}  (after optional Fo^2 filter: {fo2.size})")
    print("WGHT 0 0 assumed: w = 1/sigma^2")
    print(f"Trim iterations per fraction: {args.iters}")

    if args.fractions is not None:
        fracs = args.fractions
    elif args.fraction is not None:
        fracs = [args.fraction]
    else:
        # sensible default sweep
        fracs = [0.0, 0.001, 0.005, 0.01, 0.02]

    # baseline (no trimming)
    k0, n0 = trimmed_scale(fo2, sig, fc2, reject_fraction=0.0, iters=1)
    print(f"\nBaseline: reject=0.0000  kept={n0:6d}  k={k0: .10g}  sqrt(k)={math.sqrt(max(k0, 0.0)):.6f}")

    print("\nRejectFrac   Kept     k                sqrt(k)     delta_k(%)")
    print("---------   ------   ---------------   ---------   ----------")
    for f in fracs:
        k, nk = trimmed_scale(fo2, sig, fc2, reject_fraction=float(f), iters=args.iters)
        dk = 100.0 * (k - k0) / k0 if k0 != 0 else float("nan")
        print(f"{f:9.4f}   {nk:6d}   {k: .10g}   {math.sqrt(max(k, 0.0)):.6f}   {dk: .4f}")


if __name__ == "__main__":
    main()

# python osf_from_hkl_fcf_trim.py /Users/xiaodong/Desktop/3DED-DATA/LTA/LTA1/osf_from_trimmed_hkl/LTA1osf.hkl /Users/xiaodong/Desktop/3DED-DATA/LTA/LTA1/osf_from_trimmed_hkl/LTA1osf.fcf --fractions 0 0.1 0.2 0.3 --iters 1

