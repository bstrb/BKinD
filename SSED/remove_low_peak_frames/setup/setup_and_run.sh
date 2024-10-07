#!/bin/bash

# Define the environment name
ENV_NAME="remove_low_peak_env"
HEAD_SCRIPT="gui_remove_frames.py"

# Function to check for tkinter
check_tkinter() {
    python -c "import tkinter" &> /dev/null
    if [ $? -ne 0 ]; then
        echo "tkinter is not installed. Installing it now..."
        sudo apt-get update
        sudo apt-get install python3-tk -y
    fi
}

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Miniconda or Anaconda."
    exit 1
fi

# Ensure the conda functions are available
source $(conda info --base)/etc/profile.d/conda.sh

# Check if the environment exists
if conda env list | awk '{print $1}' | grep -w "^${ENV_NAME}$" &> /dev/null; then
    echo "Activating the existing environment: $ENV_NAME"
    conda activate "$ENV_NAME"
else
    echo "Environment $ENV_NAME does not exist. Creating it now..."
    conda create -n "$ENV_NAME" -c conda-forge python=3.12.2 h5py numpy tqdm -y
    conda activate "$ENV_NAME"
    echo "Environment $ENV_NAME created and activated."
fi

# Check and install tkinter
check_tkinter

# Run the head script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
echo "Running head script from directory: $SCRIPT_DIR"
python "$SCRIPT_DIR/$HEAD_SCRIPT"

# Deactivate the conda environment
conda deactivate
