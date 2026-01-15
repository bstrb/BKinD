#!/usr/bin/env python3
"""
shelxl_wght_run.py

Run SHELXL with a user-specified WGHT (a-f), in an isolated subfolder.

Inputs (ONLY):
  --dir   Path containing exactly one matching pair: <base>.ins and <base>.hkl
  --wghts Six numbers: a b c d e f   (or a full "WGHT a b c d e f" line)

Behavior:
  - Creates a subfolder under --dir named by the WGHT values
  - Copies <base>.ins and <base>.hkl into that subfolder (NO backups)
  - Rewrites/insert WGHT line in the copied .ins
  - Runs: shelxl <base>  (in the subfolder)
  - Parses <base>.lst for:
      * NPD count (atoms NPD)
      * OSF in the last least-squares cycle block
      * Resolution table:
          Resolution(A), Number in group, GooF, K, R1
  - Writes:
      * summary.txt
      * metrics_vs_resolution.png
  - Prints the same summary to terminal
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ------------------------- Helpers -------------------------

def _sanitize_tag(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = s.replace(".", "p")
    s = re.sub(r"[^A-Za-z0-9_+-]", "", s)
    return s


def _parse_wghts_arg(wghts: str) -> Tuple[str, str]:
    """
    Returns (wght_line, folder_tag)
    Accepts either:
      "WGHT a b c d e f"
    or:
      "a b c d e f"
    """
    raw = wghts.strip()
    if raw.upper().startswith("WGHT"):
        tokens = re.split(r"\s+", raw)
        tokens = tokens[1:]
    else:
        tokens = re.split(r"\s+", raw)

    if len(tokens) != 6:
        raise ValueError(
            f"--wghts must provide 6 numbers (a b c d e f). Got {len(tokens)} tokens: {tokens}"
        )

    # Validate numeric
    vals: List[float] = []
    for t in tokens:
        try:
            vals.append(float(t))
        except ValueError as e:
            raise ValueError(f"Non-numeric WGHT token: {t!r}") from e

    # Normalize formatting (keep readable, but stable)
    def fmt(x: float) -> str:
        # avoid scientific unless needed
        if abs(x) >= 1e4 or (abs(x) > 0 and abs(x) < 1e-3):
            return f"{x:.6g}"
        return f"{x:g}"

    norm = [fmt(v) for v in vals]
    wght_line = "WGHT " + " ".join(norm)
    tag = _sanitize_tag("WGHT_" + "_".join(norm))
    return wght_line, tag


def _find_base_pair(workdir: Path) -> Tuple[str, Path, Path]:
    """
    Find a unique <base>.ins with matching <base>.hkl in workdir.

    Rules:
      - If exactly one .ins exists and matching .hkl exists -> use it.
      - Else choose the unique basename that has both .ins and .hkl.
      - Else error.
    """
    ins_files = sorted(workdir.glob("*.ins"))
    hkl_files = {p.stem for p in workdir.glob("*.hkl")}

    if len(ins_files) == 1:
        ins = ins_files[0]
        base = ins.stem
        hkl = workdir / f"{base}.hkl"
        if hkl.exists():
            return base, ins, hkl
        raise FileNotFoundError(
            f"Found one .ins ({ins.name}) but missing matching .hkl ({base}.hkl)."
        )

    candidates = []
    for ins in ins_files:
        if ins.stem in hkl_files:
            candidates.append(ins.stem)

    candidates = sorted(set(candidates))
    if len(candidates) == 1:
        base = candidates[0]
        return base, workdir / f"{base}.ins", workdir / f"{base}.hkl"

    if not ins_files:
        raise FileNotFoundError(f"No .ins files found in: {workdir}")
    if not candidates:
        raise FileNotFoundError(
            f"Found .ins files but no matching .hkl with same basename in: {workdir}"
        )
    raise RuntimeError(
        f"Ambiguous: multiple .ins/.hkl basenames found in {workdir}: {candidates}\n"
        "Keep only one matching pair in the directory to use this script (since inputs are only --dir and --wghts)."
    )


def _rewrite_wght_in_ins(ins_path: Path, wght_line: str) -> None:
    """
    Replace existing WGHT line(s); if none exist, insert before END (or append).
    No backups.
    """
    txt = ins_path.read_text(errors="replace").splitlines(keepends=True)

    out: List[str] = []
    replaced = False
    for line in txt:
        if line.lstrip().upper().startswith("WGHT"):
            # Preserve original line ending style if present
            m = re.search(r"(\r\n|\n)$", line)
            line_ending = m.group(1) if m else "\n"
            out.append(wght_line.rstrip() + line_ending)
            replaced = True
        else:
            out.append(line)

    if not replaced:
        insert_line = wght_line.rstrip() + "\n"
        end_idx = None
        for i, line in enumerate(out):
            if line.strip().upper() == "END":
                end_idx = i
        if end_idx is None:
            out.append(insert_line)
        else:
            out.insert(end_idx, insert_line)

    ins_path.write_text("".join(out))


def _run_shelxl(workdir: Path, base: str) -> None:
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


def _parse_resolution_table(lst_path: Path) -> Dict[str, List[float]]:
    """
    Extracts the table starting at "Resolution(A)" and the rows:
      Number in group, GooF, K, R1
    Returns dict with keys:
      resolution, number_in_group, goof, k, r1
    """
    lines = lst_path.read_text(errors="replace").splitlines()

    header_idx = None
    for i, line in enumerate(lines):
        if "Resolution(A)" in line:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find 'Resolution(A)' header in the .lst file.")

    def parse_row(idx: int) -> Tuple[str, List[float]]:
        parts = re.split(r"\s+", lines[idx].strip())
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

    # scan a window below header
    for i in range(header_idx + 1, min(header_idx + 60, len(lines))):
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


def _parse_npd_count(lst_path: Path) -> Optional[int]:
    """
    Finds the last occurrence of "<n> atoms may be split and <m> atoms NPD"
    Returns m as int.
    """
    text = lst_path.read_text(errors="replace")
    pat = re.compile(r"(?i)(\d+)\s+atoms\s+may\s+be\s+split\s+and\s+(\d+)\s+atoms\s+NPD")
    matches = list(pat.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    return int(m.group(2))


def _parse_last_cycle_osf(lst_path: Path) -> Optional[float]:
    """
    Find OSF value inside the LAST "Least-squares cycle" block.
    Looks for a line containing OSF and parses the first float after the leading index.
    Typical line:
       1    36.60794     1.19550     0.000    OSF
    """
    lines = lst_path.read_text(errors="replace").splitlines()

    cycle_idxs = [i for i, line in enumerate(lines) if line.startswith(" Least-squares cycle")]
    if not cycle_idxs:
        return None
    start = cycle_idxs[-1]

    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith(" Least-squares cycle"):
            end = j
            break

    osf_pat = re.compile(
        r"^\s*\d+\s+([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s+.*\bOSF\b"
    )
    for i in range(start, end):
        m = osf_pat.match(lines[i])
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
    return None


def _save_metrics_plot(outdir: Path, table: Dict[str, List[float]], tag: str) -> Path:
    import matplotlib.pyplot as plt

    res = table["resolution"]
    # Exclude "inf" bin from x-axis; also keep arrays aligned
    finite_idx = [i for i, r in enumerate(res) if r != float("inf")]
    x = [res[i] for i in finite_idx]

    goof = [table["goof"][i] for i in finite_idx]
    k = [table["k"][i] for i in finite_idx]
    r1 = [table["r1"][i] for i in finite_idx]

    out_path = outdir / "metrics_vs_resolution.png"

    fig, ax1 = plt.subplots()

    # GooF often dwarfs K/R1, so use a second axis but keep it ONE figure
    l1, = ax1.plot(x, goof, marker="o", label="GooF")
    ax1.set_xlabel("Resolution (Å)")
    ax1.set_ylabel("GooF")
    ax1.invert_xaxis()

    ax2 = ax1.twinx()
    l2, = ax2.plot(x, k, marker="o", label="K")
    l3, = ax2.plot(x, r1, marker="o", label="R1")
    ax2.set_ylabel("K / R1")

    # Combined legend
    lines = [l1, l2, l3]
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="best")

    fig.suptitle(f"WGHT scan metrics ({tag})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return out_path


def _format_summary(
    base: str,
    wght_line: str,
    npd: Optional[int],
    osf: Optional[float],
    table: Dict[str, List[float]],
) -> str:
    res = table["resolution"]
    finite_idx = [i for i, r in enumerate(res) if r != float("inf")]
    x = [res[i] for i in finite_idx]

    ngrp = [table["number_in_group"][i] for i in finite_idx]
    goof = [table["goof"][i] for i in finite_idx]
    k = [table["k"][i] for i in finite_idx]
    r1 = [table["r1"][i] for i in finite_idx]

    lines: List[str] = []
    lines.append(f"Base: {base}")
    lines.append(f"{wght_line}")
    lines.append(f"NPD atoms (last reported): {npd if npd is not None else 'NOT FOUND'}")
    lines.append(f"OSF (last LS cycle): {osf if osf is not None else 'NOT FOUND'}")
    lines.append("")
    lines.append("Columns correspond to finite resolution bins (inf excluded).")
    lines.append("")
    lines.append("Resolution(A):      " + "  ".join(f"{v:>6.2f}" for v in x))
    lines.append("Number in group:    " + "  ".join(f"{v:>6.0f}" for v in ngrp))
    lines.append("GooF:               " + "  ".join(f"{v:>10.3f}" for v in goof))
    lines.append("K:                  " + "  ".join(f"{v:>10.3f}" for v in k))
    lines.append("R1:                 " + "  ".join(f"{v:>10.3f}" for v in r1))
    lines.append("")
    return "\n".join(lines)


# ------------------------- Main -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Directory containing <base>.ins and <base>.hkl (one unique pair).")
    ap.add_argument("--wghts", required=True, help='Six numbers "a b c d e f" or full "WGHT a b c d e f".')
    args = ap.parse_args()

    root = Path(args.dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"--dir is not a directory: {root}")

    wght_line, tag = _parse_wghts_arg(args.wghts)
    base, ins_src, hkl_src = _find_base_pair(root)

    outdir = root / tag
    outdir.mkdir(parents=True, exist_ok=False)

    # Copy only the requested inputs (no backups, do not modify originals)
    ins_dst = outdir / ins_src.name
    hkl_dst = outdir / hkl_src.name
    shutil.copy2(ins_src, ins_dst)
    shutil.copy2(hkl_src, hkl_dst)

    # Edit WGHT in the copied ins
    _rewrite_wght_in_ins(ins_dst, wght_line)

    # Run SHELXL in the new folder
    _run_shelxl(outdir, base)

    # Parse outputs
    lst_path = outdir / f"{base}.lst"
    if not lst_path.exists():
        raise FileNotFoundError(f"Expected .lst not found after SHELXL run: {lst_path}")

    table = _parse_resolution_table(lst_path)
    npd = _parse_npd_count(lst_path)
    osf = _parse_last_cycle_osf(lst_path)

    # Save plot + summary
    plot_path = _save_metrics_plot(outdir, table, tag)
    summary = _format_summary(base, wght_line, npd, osf, table)
    summary_path = outdir / "summary.txt"
    summary_path.write_text(summary)

    # Print summary to terminal
    print(summary)
    print(f"Saved:\n  - {summary_path}\n  - {plot_path}")
    print(f"Run folder:\n  - {outdir}")


if __name__ == "__main__":
    main()


# python wght_shelxl_scan.py --dir /Users/xiaodong/Desktop/LTA1/wght_study --wghts "0.2 0 0 0 0 0"
