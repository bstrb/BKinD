# filter_function.py

# Standard Library Imports
import os
import warnings

# Third-Party Imports
import numpy as np
import pandas as pd

# Utility File Imports
from util.file.find_file import find_file

# Utility Read Imports
from util.read.extract_unit_cell_from_fcf import extract_unit_cell_from_fcf
from util.read.extract_space_group_number_from_cif import extract_space_group_number_from_cif

# Utility Stats Imports
from util.stats.save_statistics import save_statistics

# Utility Data Imports
from util.data.compute_completeness_from_df_sgn_uc import compute_completeness_from_df_sgn_uc
from util.data.filter_data_step import filter_data_step
from util.data.save_filtered_data import save_filtered_data

def filter_extreme_data(output_folder, target_completeness, extreme_percent_step, max_iterations=10000):
    """
    Filters out data points from a DataFrame based on deviations from the mean DFM value in steps,
    aiming to reduce the number of unique 'asu' to a target percentage.

    Parameters:
    - output_folder (str): Directory path to save the filtered data files and statistics.
    - target_completeness (float): Target percentage of unique 'asu' to retain.
    - extreme_percent_step (float): Step percentage to filter extreme data.
    - max_iterations (int): Maximum number of iterations for filtering.

    Raises:
    - FileNotFoundError: If the sample DataFrame CSV file does not exist.
    """
    
    # Load the DataFrame from the CSV file
    df_path = os.path.join(output_folder, 'sample_df.csv')
    if not os.path.exists(df_path):
        raise FileNotFoundError(f"The file {df_path} does not exist.")
    
    df = pd.read_csv(df_path)
    fcf_path = find_file(output_folder, '.fcf')

    # Extract unit cell parameters and resolution
    uc = extract_unit_cell_from_fcf(fcf_path)
    if uc is None:
        print("Failed to extract unit cell parameters or resolution.")
        return pd.DataFrame()
    
    sgn = extract_space_group_number_from_cif(output_folder)

    original_count = len(df)
    filtered_df = pd.DataFrame(columns=df.columns)
    remaining_df = df.copy()
    initial_completeness = 100 * compute_completeness_from_df_sgn_uc(df, sgn, uc)
    possible_asu = 100 * df['asu'].nunique() / initial_completeness
    target_asu = target_completeness * possible_asu / 100

    for iteration in range(max_iterations):
        # Check for NaNs and infinite values
        remaining_df = remaining_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['DFM'])
        
        if remaining_df.empty or remaining_df['asu'].nunique() <= target_asu:
            break
        
        # Calculate the mean and deviation of the remaining DFM values
        current_mean = remaining_df['DFM'].mean()
        deviation = abs(remaining_df['DFM'] - current_mean)
        cutoff = deviation.quantile(1 - extreme_percent_step / 100)
        # print(cutoff)
        # Filter the data
        remaining_df, current_filtered = filter_data_step(remaining_df, current_mean, cutoff)
        # print(len(remaining_df))

        if not current_filtered.empty:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                filtered_columns = [col for col in current_filtered.columns if not current_filtered[col].isna().all()]
                current_filtered = current_filtered[filtered_columns]
                filtered_df = pd.concat([filtered_df, current_filtered], ignore_index=True)

    target_folder = os.path.join(output_folder, f"filtered_{target_completeness}")
    save_filtered_data(remaining_df, filtered_df, target_folder)
    resulting_completeness = 100 * compute_completeness_from_df_sgn_uc(remaining_df, sgn, uc)
    # print (resulting_completeness) 
    data_filtered_count = len(remaining_df)
    data_filtered_percentage = 100 * (data_filtered_count / original_count)
    filtered_completeness = 100 * compute_completeness_from_df_sgn_uc(filtered_df, sgn, uc)

    stats_filename = os.path.join(output_folder, 'filtering_stats.txt')
    save_statistics(
        stats_filename,
        original_count,
        initial_completeness,
        target_completeness,
        resulting_completeness,
        iteration,
        data_filtered_percentage,
        data_filtered_count,
        remaining_df['asu'].nunique(),
        filtered_completeness
    )
