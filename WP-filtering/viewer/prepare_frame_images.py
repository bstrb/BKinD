#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image


try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    RESAMPLE_BILINEAR = Image.BILINEAR


READABLE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".webp"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Prepare frame images for the Dash viewer by converting them to compact PNGs."
    )
    ap.add_argument("--src-dir", required=True, help="Directory containing source images")
    ap.add_argument("--dst-dir", required=True, help="Directory to write PNG images into")
    ap.add_argument(
        "--max-size",
        type=int,
        default=256,
        help="Resize so the longest edge is at most this many pixels (default: 256)",
    )
    ap.add_argument(
        "--percentile-low",
        type=float,
        default=0.5,
        help="Lower percentile used when normalizing grayscale images to 8-bit",
    )
    ap.add_argument(
        "--percentile-high",
        type=float,
        default=99.5,
        help="Upper percentile used when normalizing grayscale images to 8-bit",
    )
    ap.add_argument("--overwrite", action="store_true", help="Rebuild PNGs even if they already exist")
    return ap.parse_args()


def normalize_grayscale(arr: np.ndarray, low_pct: float, high_pct: float) -> Image.Image:
    arr = np.asarray(arr)
    finite = np.isfinite(arr)
    if not finite.any():
        return Image.fromarray(np.zeros(arr.shape[:2], dtype=np.uint8), mode="L")

    values = arr[finite].astype(np.float32)
    lo = np.percentile(values, low_pct)
    hi = np.percentile(values, high_pct)
    if not np.isfinite(lo):
        lo = float(values.min())
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0

    scaled = np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    return Image.fromarray(np.rint(scaled * 255.0).astype(np.uint8), mode="L")


def convert_image(path: Path, low_pct: float, high_pct: float) -> Image.Image:
    with Image.open(path) as img:
        arr = np.asarray(img)
        if arr.ndim == 2:
            out = normalize_grayscale(arr, low_pct=low_pct, high_pct=high_pct)
        elif arr.ndim == 3 and arr.shape[2] in (3, 4):
            out = img.convert("RGB")
        else:
            out = normalize_grayscale(arr[..., 0], low_pct=low_pct, high_pct=high_pct)
        return out.copy()


def resize_if_needed(img: Image.Image, max_size: int) -> Image.Image:
    if max_size <= 0:
        return img

    width, height = img.size
    longest_edge = max(width, height)
    if longest_edge <= max_size:
        return img

    scale = max_size / float(longest_edge)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return img.resize(new_size, RESAMPLE_BILINEAR)


def main() -> None:
    args = parse_args()
    src_dir = Path(args.src_dir).expanduser().resolve()
    dst_dir = Path(args.dst_dir).expanduser().resolve()
    dst_dir.mkdir(parents=True, exist_ok=True)

    if not src_dir.is_dir():
        raise SystemExit(f"Source directory not found: {src_dir}")

    files = sorted(
        path for path in src_dir.iterdir() if path.is_file() and path.suffix.lower() in READABLE_SUFFIXES
    )
    if not files:
        raise SystemExit(f"No supported images found in {src_dir}")

    written = 0
    skipped = 0
    failed = 0

    for src_path in files:
        dst_path = dst_dir / f"{src_path.stem}.png"
        if dst_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            img = convert_image(src_path, low_pct=args.percentile_low, high_pct=args.percentile_high)
            img = resize_if_needed(img, args.max_size)
            img.save(dst_path, format="PNG", optimize=True)
            written += 1
        except Exception as exc:
            failed += 1
            print(f"Could not process {src_path.name}: {exc}")

    print(f"Source images: {len(files)}")
    print(f"Wrote PNGs:    {written}")
    print(f"Skipped:       {skipped}")
    print(f"Failed:        {failed}")
    print(f"Output dir:    {dst_dir}")


if __name__ == "__main__":
    main()
