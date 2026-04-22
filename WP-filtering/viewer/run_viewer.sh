#!/bin/bash

set -euo pipefail

# Run from the script's own directory so relative paths work no matter where
# the launcher is invoked from.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Make sure Conda is available
if ! command -v conda &> /dev/null ; then
    echo "Conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

# Enable conda activate in scripts
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create environment if missing
if ! conda env list | grep -q "^dfmviewer"; then
    echo "Creating Conda environment 'dfmviewer'..."
    conda env create -f environment.yml
fi

VIEWER_ARGS=("$@")
HAS_CSV_ARG=0
for arg in "$@"; do
    if [[ "$arg" == "--csv" ]]; then
        HAS_CSV_ARG=1
        break
    fi
done

if [[ "$HAS_CSV_ARG" -eq 0 ]]; then
    GENERATED_CSV="$SCRIPT_DIR/generated/sres_viewer_data.csv"
    GENERATED_IMAGES="$SCRIPT_DIR/generated/images"
    SAMPLE_CSV="$SCRIPT_DIR/sample_df_no_filter.csv"

    if [[ -f "$GENERATED_CSV" ]]; then
        VIEWER_ARGS=(--csv "$GENERATED_CSV" --image-dir "$GENERATED_IMAGES" "${VIEWER_ARGS[@]}")
    elif [[ ! -f "$SAMPLE_CSV" ]]; then
        cat <<EOF
No default CSV was found in:
  $SAMPLE_CSV

To build and open the raw-observation SRES viewer, run:
  ./run_sres_viewer.sh --integrate-hkl /path/to/INTEGRATE.HKL --raw-hkl /path/to/shelx.hkl --fcf /path/to/shelx.fcf --image-src /path/to/frames

Or open an existing CSV directly:
  ./run_viewer.sh --csv /path/to/data.csv --image-dir /path/to/images --y-col SRES
EOF
        exit 1
    fi
fi

echo "Activating environment..."
conda activate dfmviewer

echo "Starting viewer..."
python dfm_viewer.py "${VIEWER_ARGS[@]}"
