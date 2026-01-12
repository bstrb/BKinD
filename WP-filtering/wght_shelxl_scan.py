#!/usr/bin/env python3
"""
wght_shelxl_scan.py

1) Rewrites the WGHT line in <base>.ins inside a target folder
2) Runs: shelxl <base>
3) Parses the “Resolution(A) … / … K … / … GooF … / … R1 …” table from <base>.lst
4) Parses the line like:
      <n> atoms may be split and <m> atoms NPD
   including warning forms like:
      ** Warning:     0  atoms may be split and     2  atoms NPD **
5) Parses OSF from the LAST least-squares cycle block in <base>.lst
6) Saves:
   - a plot of K vs resolution
   - the parsed table as a .txt
Both outputs are named after the WGHT used.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def sanitize_wght_tag(wght_line: str) -> str:
    s = wght_line.strip()
    s = re.sub(r"\s+", "_", s)
    s = s.replace(".", "p")
    s = re.sub(r"[^A-Za-z0-9_+-]", "", s)
    return s


def rewrite_wght_in_ins(ins_path: Path, wght_line: str, backup: bool = True) -> None:
    if not ins_path.exists():
        raise FileNotFoundError(f"Missing .ins file: {ins_path}")

    txt = ins_path.read_text(errors="replace").splitlines(keepends=True)

    if backup:
        bak = ins_path.with_suffix(ins_path.suffix + ".bak")
        shutil.copy2(ins_path, bak)

    out_lines: List[str] = []
    replaced = False

    for line in txt:
        if line.lstrip().upper().startswith("WGHT"):
            m = re.search(r"(\r\n|\n)$", line)
            line_ending = m.group(1) if m else "\n"
            out_lines.append(wght_line.rstrip() + line_ending)
            replaced = True
        else:
            out_lines.append(line)

    if not replaced:
        end_idx = None
        for i, line in enumerate(out_lines):
            if line.strip().upper() == "END":
                end_idx = i
        insert_line = wght_line.rstrip() + "\n"
        if end_idx is None:
            out_lines.append(insert_line)
        else:
            out_lines.insert(end_idx, insert_line)

    ins_path.write_text("".join(out_lines))


def run_shelxl(workdir: Path, base: str) -> None:
    cmd = ["shelxl", base]
    proc = subprocess.run(
        cmd,
        cwd=str(workdir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "SHELXL failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"cwd: {workdir}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}\n"
        )


def _to_float(token: str) -> Optional[float]:
    t = token.strip()
    if not t:
        return None
    if t.lower() == "inf":
        return float("inf")
    try:
        return float(t)
    except ValueError:
        return None


def parse_resolution_table(lst_path: Path) -> Dict[str, List[float]]:
    if not lst_path.exists():
        raise FileNotFoundError(f"Missing .lst file: {lst_path}")

    lines = lst_path.read_text(errors="replace").splitlines()

    header_idx = None
    for i, line in enumerate(lines):
        if "Resolution(A)" in line:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find 'Resolution(A)' header in the .lst file.")

    def parse_row(idx: int) -> Tuple[str, List[float]]:
        line = lines[idx].strip()
        parts = re.split(r"\s+", line)

        first_num = None
        for j, p in enumerate(parts):
            if _to_float(p) is not None:
                first_num = j
                break
        if first_num is None:
            raise ValueError(f"Could not parse numeric row at line {idx+1}: {lines[idx]!r}")

        label = " ".join(parts[:first_num])
        nums: List[float] = []
        for p in parts[first_num:]:
            v = _to_float(p)
            if v is not None:
                nums.append(v)
        return label, nums

    _, res = parse_row(header_idx)

    wanted = {
        "Number in group": "number_in_group",
        "GooF": "goof",
        "K": "k",
        "R1": "r1",
    }

    out: Dict[str, List[float]] = {"resolution": res}

    for i in range(header_idx + 1, min(header_idx + 40, len(lines))):
        line = lines[i].strip()
        if not line:
            continue
        for label, key in wanted.items():
            if line.startswith(label):
                _, nums = parse_row(i)
                out[key] = nums

    missing = [k for k in wanted.values() if k not in out]
    if missing:
        raise ValueError(f"Did not find required rows in .lst table: {missing}")

    return out


def parse_split_npd_line(lst_path: Path) -> str:
    if not lst_path.exists():
        return f"Split/NPD line not found (missing file): {lst_path}"

    text = lst_path.read_text(errors="replace")
    pat = re.compile(
        r"(?i)(\d+)\s+atoms\s+may\s+be\s+split\s+and\s+(\d+)\s+atoms\s+NPD"
    )
    m = pat.search(text)
    if not m:
        return "Split/NPD line not found: '<n> atoms may be split and <m> atoms NPD'"
    n, m2 = m.group(1), m.group(2)
    return f"{n} atoms may be split and {m2} atoms NPD"


def parse_last_cycle_osf(lst_path: Path) -> Optional[float]:
    """
    Find OSF value in the LAST 'Least-squares cycle ...' block.
    In that block, find the line that ends with 'OSF' and parse the numeric 'value' column.

    Example line:
         1    36.60794     1.19550     0.000    OSF
    """
    if not lst_path.exists():
        return None

    lines = lst_path.read_text(errors="replace").splitlines()

    # Find last occurrence of a cycle header
    cycle_idxs = [i for i, line in enumerate(lines) if line.startswith(" Least-squares cycle")]
    if not cycle_idxs:
        return None
    start = cycle_idxs[-1]

    # Define end as next cycle header or EOF
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith(" Least-squares cycle"):
            end = j
            break

    # Within this last block, look for an OSF row
    # Format is typically: N value esd shift/esd parameter
    osf_pat = re.compile(
        r"^\s*\d+\s+([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s+[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?\s+[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?\s+OSF\s*$"
    )
    for i in range(start, end):
        m = osf_pat.match(lines[i])
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None

    return None


def save_table_txt(workdir: Path, tag: str, table: Dict[str, List[float]], lst_path: Path) -> Path:
    out_path = workdir / f"{tag}_resolution_table.txt"

    res = table["resolution"]
    finite_res = [r for r in res if r != float("inf")]

    def align(vals: List[float]) -> List[float]:
        return vals[:len(finite_res)]

    ngrp = align(table["number_in_group"])
    goof = align(table["goof"])
    kvals = align(table["k"])
    r1 = align(table["r1"])

    split_npd = parse_split_npd_line(lst_path)
    osf = parse_last_cycle_osf(lst_path)

    with out_path.open("w") as f:
        f.write(f"WGHT tag: {tag}\n")
        f.write(f"{split_npd}\n")
        f.write(f"OSF (last cycle): {osf if osf is not None else 'not found'}\n")
        f.write("Columns correspond to finite resolution bins (inf excluded).\n\n")
        f.write("Resolution(A):      " + "  ".join(f"{x:>6.2f}" for x in finite_res) + "\n")
        f.write("Number in group:    " + "  ".join(f"{x:>6.0f}" for x in ngrp) + "\n")
        f.write("GooF:               " + "  ".join(f"{x:>6.3f}" for x in goof) + "\n")
        f.write("K:                  " + "  ".join(f"{x:>6.3f}" for x in kvals) + "\n")
        f.write("R1:                 " + "  ".join(f"{x:>6.3f}" for x in r1) + "\n")

    return out_path


def save_k_plot(workdir: Path, tag: str, table: Dict[str, List[float]]) -> Path:
    import matplotlib.pyplot as plt

    res = table["resolution"]
    finite_res = [r for r in res if r != float("inf")]
    k = table["k"][:len(finite_res)]

    out_path = workdir / f"{tag}_K_vs_resolution.png"

    plt.figure()
    plt.plot(finite_res, k, marker="o")
    plt.xlabel("Resolution (Å)")
    plt.ylabel("K")
    plt.title(f"K vs Resolution ({tag})")
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Folder containing <base>.ins/.hkl etc")
    ap.add_argument("--base", required=True, help="Base name, e.g. wght_study (without extension)")
    ap.add_argument("--wght", required=True, help='WGHT line (either "WGHT a b" or just "a b")')
    ap.add_argument("--no-backup", action="store_true", help="Do not create <base>.ins.bak")
    args = ap.parse_args()

    workdir = Path(args.dir).expanduser().resolve()
    base = args.base

    wght_line = args.wght.strip()
    if not wght_line.upper().startswith("WGHT"):
        wght_line = "WGHT " + wght_line

    ins_path = workdir / f"{base}.ins"
    lst_path = workdir / f"{base}.lst"

    rewrite_wght_in_ins(ins_path, wght_line, backup=(not args.no_backup))
    run_shelxl(workdir, base)

    table = parse_resolution_table(lst_path)

    tag = sanitize_wght_tag(wght_line)
    txt_path = save_table_txt(workdir, tag, table, lst_path)
    plot_path = save_k_plot(workdir, tag, table)

    print(f"Done.\n- Table: {txt_path}\n- Plot:  {plot_path}")

if __name__ == "__main__":
    main()

# python wght_shelxl_scan.py --dir /Users/xiaodong/Desktop/3DED-DATA/LTA/LTA4/wght_study --base wght_study --wght "0.2 0.0"
