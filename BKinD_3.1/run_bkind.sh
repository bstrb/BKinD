#!/bin/bash

# Enable debugging
set -x

# Check if the script is running
echo "Running run_bkind.sh script"

# Source the Conda script to enable the `conda` command
source /opt/anaconda3/etc/profile.d/conda.sh

# Activate the Conda environment
conda activate bkind

# Run the Python script
python bkind.py

# Close the terminal window
osascript -e 'tell application "Terminal" to close first window' & exit