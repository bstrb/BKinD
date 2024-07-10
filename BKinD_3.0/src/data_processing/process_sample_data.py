# process_sample_data.py

# Standard Library Imports
import os

# Third-Party Imports
import pandas as pd
from scipy.optimize import minimize

# Utility Read Imports
from util.read.extract_space_group_number_from_cif import extract_space_group_number_from_cif
from util.read.extract_unit_cell_from_fcf import extract_unit_cell_from_fcf
from util.read.extract_parts_from_path import extract_parts_from_path
from util.read.read_fcf import read_fcf

from util.read.extract_space_group_symbol_from_cif import extract_space_group_symbol_from_cif

# Utility File Imports
from util.file.find_file import find_file

# Utility Process Imports
from util.process.calculate_dfm import calculate_dfm
from util.process.objective_function import objective_function

# Class Imports
from ed.integrate import Integrate

# X-ray Specific Imports
from xray.calculate_asu_and_resolution import calculate_asu_and_resolutions

def process_sample_data(user_directory, xray=False):
    """
    Processes sample data from .fcf and INTEGRATE.HKL files located in a user directory, 
    merges them, calculates DFM, and adjusts the instability factor (u) so that the mean and median of DFM become the same.

    Args:
    user_directory (str): Directory containing the .fcf and INTEGRATE.HKL files.

    Returns:
    pd.DataFrame: Processed sample DataFrame with calculated and adjusted DFM values.
    """

    # Finding files
    fcf_path = find_file(user_directory, '.fcf')

    # Extract unit cell parameters and resolution
    unit_cell_params = extract_unit_cell_from_fcf(fcf_path)
    if unit_cell_params is None:
        print("Failed to extract unit cell parameters or resolution.")
        return pd.DataFrame()
    
    spacegroup_no = extract_space_group_number_from_cif(user_directory)

    spacegroup_symbol = extract_space_group_symbol_from_cif(user_directory)

    # Create a DataFrame from .fcf
    sample_fcf_df = read_fcf(fcf_path)

    if xray:
        # Create a DataFrame with ASU and resolutions
        df_asu_resolutions = calculate_asu_and_resolutions(fcf_path, unit_cell_params, spacegroup_no)

        # Merge the ASU and resolutions from df_asu_resolutions to sample_df
        sample_df = sample_fcf_df.merge(df_asu_resolutions[['Miller', 'asu', 'Resolution']], on='Miller', how='left')
        
        sample_df=sample_df.drop_duplicates()

        required_columns = ["Miller", "asu", "Fc^2", "Fo^2", "Fo^2_sigma", "Resolution"]
    else:
        # Create a DataFrame from 'INTEGRATE.HKL'
        integrate_path = find_file(user_directory, 'INTEGRATE.HKL')
        sample_inte_df = Integrate(integrate_path).as_df()

        # Merge and organize data using a common column known to exist in both data sets
        sample_df = sample_fcf_df.merge(sample_inte_df, on="Miller")  # Adjusted to use 'Miller' as the merge key
        
        
        required_columns = ["Miller", "asu", "Fc^2", "Fo^2", "Fo^2_sigma", "Resolution", "xobs", "yobs", "zobs"]
    
    sample_df = sample_df[required_columns]

    # Optimize the instability factor (u) to make the mean and median of DFM the same
    initial_u = 0.01
    result = minimize(objective_function, initial_u, args=(sample_df,), method='Nelder-Mead')

    optimal_u = result.x[0]

    second_part, fourth_part = extract_parts_from_path(user_directory)

    # Write the optimal_u value to filtering_stats.txt
    stats_output_path = os.path.join(user_directory, 'filtering_stats.txt')
    with open(stats_output_path, 'w') as stats_file:
            stats_file.write(f"Filtering Diffraction Data for Crystal {second_part} to {fourth_part} % Completeness\nSpace Group Number: {spacegroup_no} Space Group Symbol:{spacegroup_symbol}\nUnit Cell Parameters: {unit_cell_params}\n")

    # Ensure the optimal instability factor 'u' is a reasonable value
    if abs(initial_u - optimal_u) > initial_u + 0.001:
        optimal_u = initial_u  # Default to initial u if the optimized result too far from initial u
        with open(stats_output_path, 'a') as stats_file:
            stats_file.write(f"Optimal Instability Factor (u): {optimal_u:.3g}. (Default value since optima found at: {result.x[0]})\n")
    else:
        with open(stats_output_path, 'a') as stats_file:
            stats_file.write(f"Optimal Instability Factor (u): {optimal_u:.3g}\n")

    # Calculate DFM with the optimal u
    sample_df['DFM'] = calculate_dfm(optimal_u, sample_df)

    # Save the DataFrame to the user directory
    output_path = os.path.join(user_directory, 'sample_df.csv')
    sample_df.to_csv(output_path, index=False)
