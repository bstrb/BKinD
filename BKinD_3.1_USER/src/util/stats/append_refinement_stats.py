# append_refinement_stats.py

# Standard Library Imports
import os
import logging
from multiprocessing import Pool

# Third-Party Imports
import pandas as pd

def read_summary_data(summary_csv_path):
    """
    Reads the summary CSV and returns the first row as a formatted string.
    """
    try:
        summary_df = pd.read_csv(summary_csv_path)
        if not summary_df.empty:
            first_row = summary_df.iloc[0]
            stats = (f"R1: {first_row['R1']}, Rint: {first_row['Rint']}, FVAR: {first_row['FVAR']}, "
                    f"Number of NPD's: {first_row['NPD']}\n Highest Diff Peak: {first_row['highest_diff_peak']}, "
                    f"Deepest Hole: {first_row['deepest_hole']}, One Sigma Level: {first_row['one_sigma_level']}\n")
            return stats
        else:
            logging.warning("CSV file is empty: %s", summary_csv_path)
    except Exception as e:
        logging.error("Failed to read or parse CSV file %s: %s", summary_csv_path, e)
    return "No data available\n"

def process_target_asu(args):
    """
    Process a single target ASU.
    """
    main_directory, target_asu = args
    subdir_name = f"filtered_{float(target_asu)}"
    subdir_path = os.path.join(main_directory, subdir_name, "REFINEMENT_STATISTICS")
    summary_csv_path = os.path.join(subdir_path, "summary.csv")

    if os.path.exists(summary_csv_path):
        stats = read_summary_data(summary_csv_path)
        return f"Target Completeness: {target_asu}\n{stats}"
    else:
        logging.warning("Summary CSV not found: %s", summary_csv_path)
        return f"Summary statistics not available for ASU {target_asu}\n"

def append_refinement_stats(main_directory, target_asus):
    """
    Appends refinement statistics from CSV files within subdirectories to a specified text file located in the main directory.
    """
    text_file_path = os.path.join(main_directory, "filtering_stats.txt")
    
    try:
        with Pool() as pool:
            results = pool.map(process_target_asu, [(main_directory, target_asu) for target_asu in target_asus])
        
        with open(text_file_path, 'a') as file:
            for result in results:
                file.write(result)
    except FileNotFoundError:
        logging.error("Text file not found: %s", text_file_path)
    except Exception as e:
        logging.error("An error occurred: %s", e)
