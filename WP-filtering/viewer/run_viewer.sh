#!/bin/bash

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

echo "Activating environment..."
conda activate dfmviewer

echo "Starting viewer..."
python dfm_viewer.py
