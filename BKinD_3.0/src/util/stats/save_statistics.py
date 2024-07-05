# save_statistics.py

# Standard Library Imports
import os

def check_original_data_written(stats_filename):
    """Check if the original data counts are already written in the file."""
    if not os.path.exists(stats_filename):
        return False

    with open(stats_filename, 'r') as file:
        for line in file:
            if "Original Data Count" in line or "Initial Completeness" in line:
                return True
    return False

def save_statistics(stats_filename, original_count, initial_completeness, target_completeness, iteration, data_filtered_percentage, data_filtered_count, remaining_unique_asus, filtered_completeness):
    """
    Save filtering statistics to a file.

    Parameters:
    - stats_filename (str): Path to the statistics file.
    - original_count (int): Original count of data points.
    - initial_completeness (int): Initial completeness.
    - target_completeness (float): Target completeness.
    - iteration (int): Number of iterations.
    - data_filtered_percentage (float): Percentage of data retained after filtering.
    - remaining_unique_asus (int): Count of unique 'asu' remaining.
    """
    average_multiplicity = data_filtered_count / remaining_unique_asus if remaining_unique_asus != 0 else 0

    try:
        original_data_written = check_original_data_written(stats_filename)

        with open(stats_filename, 'a') as file:
            if not original_data_written:
                file.write(f"Original Data Count: {original_count}\n")
                file.write(f"Initial Completeness: {initial_completeness:.4f} %\n")
            file.write(f"Target Completeness: {target_completeness} %\n")
            file.write(f"Completeness of Data Filtered Away: {filtered_completeness:.1f} %\n")
            file.write(f"Number of Iterations: {iteration}\n")
            file.write(f"Resulting Data Percentage: {data_filtered_percentage:.2f} %\n")
            file.write(f"Resulting Data Count: {data_filtered_count}\n")
            file.write(f"Remaining Unique 'asu' Counts: {remaining_unique_asus}\n")
            file.write(f"Resulting Average Multiplicity: {average_multiplicity:.1f}\n")
    except IOError as e:
        print(f"Error writing to {stats_filename}: {e}")