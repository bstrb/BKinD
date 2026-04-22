# Reflection Viewer

This folder now supports two related workflows:

1. Open an existing CSV plus prepared frame images in the Dash viewer.
2. Build a raw-observation `SRES` table from `INTEGRATE.HKL + raw .hkl + .fcf`,
   optionally convert frame TIFFs to PNGs, and launch the viewer in one step.
3. Save the full pipeline as a reusable config file so the same dataset can be
   rebuilt later with one command.

## Generic viewer

Use this when you already have a CSV with columns such as `zobs`, `frame`, `DFM`,
or `SRES`, plus an image folder.

```bash
cd WP-filtering/viewer
chmod +x run_viewer.sh
./run_viewer.sh --csv /path/to/data.csv --image-dir /path/to/images --y-col SRES
```

Notes:
- The viewer now updates the image on hover, with click as a fallback.
- If `frame` is missing, it uses `round(zobs)`.
- Image filenames can be `00001.png`, `000001.png`, `frame_0001.png`, etc.
  The viewer indexes the last integer in each filename.

## Raw-observation SRES workflow

Use this when you want to start from:
- `INTEGRATE.HKL` for observed frame positions
- SHELX raw `*.hkl` for `Fo^2` and `sigma(Fo^2)`
- SHELX `*.fcf` for `Fc^2`
- optional frame images such as TIFFs

```bash
cd WP-filtering/viewer
chmod +x run_sres_viewer.sh
./run_sres_viewer.sh \
  --integrate-hkl /path/to/INTEGRATE.HKL \
  --raw-hkl /path/to/shelx.hkl \
  --fcf /path/to/shelx.fcf \
  --image-src /path/to/frame_tiffs
```

What it does:
- `build_sres_data.py`
  Uses cctbx to map each raw SHELX observation onto an `Fc^2` value from the
  `.fcf`, merges the exact observation with `xyzobs/zobs` from `INTEGRATE.HKL`,
  applies a robust MAD-based scale factor to `Fo^2` and `sigma(Fo^2)`, and
  writes `sres_viewer_data.csv`.
- `prepare_frame_images.py`
  Converts TIFF/PNG/JPG frames into compact PNGs for the viewer.
- `dfm_viewer.py`
  Launches the plot plus frame browser.

Outputs go to `viewer/generated/` by default:
- `sres_viewer_data.csv`
- `sres_viewer_summary.txt`
- `images/*.png`
- `sres_pipeline.env`

## Saved pipeline configs

You can save a dataset pipeline in two ways.

1. Run once from command-line arguments and let the script save a reusable
   config in the output directory:

```bash
./run_sres_viewer.sh \
  --integrate-hkl /path/to/INTEGRATE.HKL \
  --raw-hkl /path/to/shelx.hkl \
  --fcf /path/to/shelx.fcf \
  --image-src /path/to/frame_tiffs \
  --build-only
```

That writes `OUT_DIR/sres_pipeline.env`. Later you can reopen the same dataset:

```bash
./run_sres_viewer.sh --config /path/to/OUT_DIR/sres_pipeline.env
```

2. Start from the template file [sres_pipeline.template.env](/home/bubl3932/projects/BKinD/WP-filtering/viewer/sres_pipeline.template.env:1),
   fill in the paths once, then run:

```bash
./run_sres_viewer.sh --config /path/to/your_dataset.env
```

You can also save a copy of the effective settings anywhere you like:

```bash
./run_sres_viewer.sh ... --save-config /path/to/my_dataset.env --build-only
```

## Conda environments

- `run_viewer.sh` uses `dfmviewer`
- `run_sres_viewer.sh` uses:
  - `cctbx-env` for the cctbx mapping step
  - `dfmviewer` for image prep and Dash

You can override those names with:

```bash
VIEWER_ENV=myviewer CCTBX_ENV=mycctbx ./run_sres_viewer.sh ...
```
