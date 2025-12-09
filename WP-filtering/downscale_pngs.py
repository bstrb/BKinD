#!/usr/bin/env python3
import os
import argparse
from PIL import Image
from tqdm import tqdm

# python downscale_pngs.py --src-dir /Users/xiaodong/Desktop/3DED-DATA/LTA/LTA1/images/png --dst-dir /Users/xiaodong/Desktop/3DED-DATA/LTA/LTA1/images/png_256 --size 256


def downscale_pngs(src_dir, dst_dir, size):
    """
    Downsize PNGs from src_dir to dst_dir at resolution size × size.
    
    Keeps filenames identical.
    """

    os.makedirs(dst_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(src_dir) if f.lower().endswith(".png")])

    if not files:
        print("No PNGs found in", src_dir)
        return

    print(f"Found {len(files)} PNGs.")
    print(f"Downsizing to {size}×{size}...")

    for f in tqdm(files):
        src_path = os.path.join(src_dir, f)
        dst_path = os.path.join(dst_dir, f)

        try:
            img = Image.open(src_path).convert("RGB")
            img = img.resize((size, size), Image.BILINEAR)
            img.save(dst_path, format="PNG")
        except Exception as e:
            print(f"Could not process {f}: {e}")

    print(f"\nDone! Resized images saved to: {dst_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Downscale PNG images to a fixed resolution."
    )
    parser.add_argument("--src-dir", required=True, help="Directory containing PNG images.")
    parser.add_argument("--dst-dir", required=True, help="Directory to save resized PNGs.")
    parser.add_argument("--size", type=int, required=True, help="Output resolution (e.g. 256).")
    args = parser.parse_args()

    downscale_pngs(args.src_dir, args.dst_dir, args.size)


if __name__ == "__main__":
    main()
