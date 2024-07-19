# #!/bin/bash

# # Define the environment name
# ENV_NAME="bkind"
# HEAD_SCRIPT="bkind.py"

# # Check if conda is installed
# if ! command -v conda &> /dev/null; then
#     echo "Conda is not installed. Please install Miniconda or Anaconda."
#     exit 1
# fi

# # Check if the environment exists
# if conda env list | grep -q "$ENV_NAME"; then
#     echo "Activating the existing environment: $ENV_NAME"
#     source activate "$ENV_NAME"
# else
#     echo "Environment $ENV_NAME does not exist. Creating it now..."
#     conda create -n "$ENV_NAME" -c conda-forge cctbx-base python=3.12.2 numpy==1.26.4 pandas==2.2.1 plotly==5.19.0 tqdm==4.66.4 pillow==10.3.0 -y
#     source activate "$ENV_NAME"
#     echo "Environment $ENV_NAME created and activated."
# fi

# # Run the head script
# echo "Running head script..."
# python "$HEAD_SCRIPT"

# # Deactivate the conda environment
# conda deactivate

#!/bin/bash

# Define the environment name
ENV_NAME="bkind"
HEAD_SCRIPT="bkind.py"

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

# Run the head script
echo "Running head script..."
python "$HEAD_SCRIPT"

# Deactivate the conda environment
conda deactivate
