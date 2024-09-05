# copy_and_reduce_hkl.py
 
# Standard Library Imports
import os

# Third-Party Imports
import pandas as pd

from util.file.find_file import find_file

def copy_and_reduce_hkl(source_directory, target_directory):
    """
    Copies an INTEGRATE.HKL file from the source directory to the target directory, reduces it based on Miller indices specified in 'remaining_data.csv', and validates the reduction.

    Parameters:
    - source_directory: str, the directory containing the 'INTEGRATE.HKL' file.
    - target_directory: str, the directory where 'remaining_data.csv' is located and where the reduced HKL will be saved.

    Returns:
    - None, but saves the reduced 'INTEGRATE.HKL' in the target directory and prints validation results.
    """
    # Define file paths
    source_hkl_path = os.path.join(source_directory, 'INTEGRATE.HKL')
    target_hkl_path = os.path.join(target_directory, 'INTEGRATE.HKL')
    miller_csv_path = find_file(target_directory,'.csv')

    # Load and prepare Miller indices
    miller_df = pd.read_csv(miller_csv_path)
    if 'Miller' not in miller_df.columns:
        raise ValueError("CSV file must contain a 'Miller' column.")
    miller_indices = set(tuple(map(int, m.split(','))) for m in miller_df['Miller'].str.strip('()'))

    # Read and filter HKL file
    reduced_miller_indices = []
    with open(source_hkl_path, 'r') as original_hkl, open(target_hkl_path, 'w') as reduced_hkl:
        is_header = True
        for line in original_hkl:
            if is_header:
                reduced_hkl.write(line)
                if line.strip() == '!END_OF_HEADER':
                    is_header = False
                continue

            parts = line.strip().split()
            if len(parts) < 3:
                continue
            miller = tuple(map(int, parts[:3]))

            if miller in miller_indices:
                reduced_hkl.write(line)
                reduced_miller_indices.append(miller)