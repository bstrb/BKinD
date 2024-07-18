# aggregate.py

# Standard Library Imports
import os

# Third-Party Imports
import pandas as pd


def aggregate_filtered(output_folder, target_percentages):
    """
    Aggregates all filtered files into a new subfolder and copies the remaining data from the
    lowest target percentage subfolder.

    Parameters:
    - output_folder: str, the base directory where filtered data subfolders are located.
    - target_percentages: list, a sorted list of target percentages used in filtering.
    """
    # Create a new directory to store the aggregated filtered
    aggregate_folder = os.path.join(output_folder, "aggregated_filtered")
    os.makedirs(aggregate_folder, exist_ok=True)

    # Iterate through each target percentage folder to gather filtered
    for target in target_percentages:
        source_folder = os.path.join(output_folder, f"filtered_{target}")
        # filtered_file_path = os.path.join(source_folder, 'filtered_data.csv')
        filtered_file_path = os.path.join(source_folder, 'removed_data.csv')

        if os.path.exists(filtered_file_path):
            df_filtered = pd.read_csv(filtered_file_path)
            # Save the filtered file with a new name that includes the target percentage
            df_filtered.to_csv(os.path.join(aggregate_folder, f"filtered_{target}.csv"), index=False)
            # df_filtered.to_csv(os.path.join(aggregate_folder, f"removed_data{target}.csv"), index=False)
        else:
            print(f"No filtered file found for {target}%.")

    # Find the remaining data from the subfolder with the lowest target percentage
    highest_target = max(target_percentages)
    highest_folder = os.path.join(output_folder, f"filtered_{highest_target}")
    # highest_folder = os.path.join(output_folder, f"removed_data{highest_target}")
    remaining_data_file_path = os.path.join(highest_folder, 'remaining_data.csv')

    if os.path.exists(remaining_data_file_path):
        df_remaining = pd.read_csv(remaining_data_file_path)
        # Optionally, save the remaining data in the aggregated folder
        df_remaining.to_csv(os.path.join(aggregate_folder, f"remaining_data.csv"), index=False)
    else:
        print(f"No remaining data file found for the lowest target percentage {highest_target}%.")
    return aggregate_folder
