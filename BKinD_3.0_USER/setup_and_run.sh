#!/bin/bash

# Define the environment name
ENV_NAME="bkind"
HEAD_SCRIPT="bkind.py"

# Function to check for tkinter
check_tkinter() {
    python -c "import tkinter" &> /dev/null
    if [ $? -ne 0 ]; then
        echo "tkinter is not installed. Installing it now..."
        sudo apt-get update
        sudo apt-get install python3-tk -y
#    else
#        echo "tkinter is already installed."
    fi
}

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Miniconda or Anaconda."
    exit 1
fi

# Check if the environment exists
if conda env list | awk '{print $1}' | grep -w "^${ENV_NAME}$" &> /dev/null; then
    echo "Activating the existing environment: $ENV_NAME"
    source activate "$ENV_NAME"
else
    echo "Environment $ENV_NAME does not exist. Creating it now..."
    conda create -n "$ENV_NAME" -c conda-forge cctbx-base python=3.12.2 numpy==1.26.4 pandas==2.2.1 plotly==5.19.0 tqdm==4.66.4 pillow==10.3.0 -y
    source activate "$ENV_NAME"
    echo "Environment $ENV_NAME created and activated."
fi

# Check and install tkinter
check_tkinter

# Run the head script
echo "Running head script..."
python "$HEAD_SCRIPT"

# Deactivate the conda environment
conda deactivate
