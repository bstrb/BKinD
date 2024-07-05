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
        filtered_file_path = os.path.join(source_folder, 'filtered_data.csv')

        if os.path.exists(filtered_file_path):
            df_filtered = pd.read_csv(filtered_file_path)
            # Save the filtered file with a new name that includes the target percentage
            df_filtered.to_csv(os.path.join(aggregate_folder, f"filtered_{target}.csv"), index=False)
        else:
            print(f"No filtered file found for {target}%.")

    # Find the remaining data from the subfolder with the lowest target percentage
    lowest_target = min(target_percentages)
    lowest_folder = os.path.join(output_folder, f"filtered_{lowest_target}")
    remaining_data_file_path = os.path.join(lowest_folder, 'remaining_data.csv')

    if os.path.exists(remaining_data_file_path):
        df_remaining = pd.read_csv(remaining_data_file_path)
        # Optionally, save the remaining data in the aggregated folder
        df_remaining.to_csv(os.path.join(aggregate_folder, f"remaining_data.csv"), index=False)
    else:
        print(f"No remaining data file found for the lowest target percentage {lowest_target}%.")
    return aggregate_folder

#####################################################################

# def clean_aggregate(folder_path):
#     """
#     Clean files in a folder so that each file only contains unique data not present in files with a higher target percentage.

#     Parameters:
#     - folder_path: str, path to the folder containing the extreme data CSV files.
#     """
#     # Suppress specific pandas FutureWarnings
#     warnings.simplefilter(action='ignore', category=FutureWarning)

#     # Step 1: Collect all CSV files and sort them by decreasing target percentages
#     files = [f for f in os.listdir(folder_path) if f.startswith('filtered_') and f.endswith('.csv')]
#     files.sort(key=lambda x: float(x.split('_')[1].replace('.csv', '')), reverse=True)

#     # Append 'remaining_data.csv' after sorting
#     files.append('remaining_data.csv')

#     # Step 2: Initialize a DataFrame to store all seen filtered
#     all_seen_filtered = pd.DataFrame()

#     for file in files:
#         current_path = os.path.join(folder_path, file)
#         current_filtered = pd.read_csv(current_path)

#         # Filter out any filtered already seen in higher percentage files
#         if not all_seen_filtered.empty:
#             combined = pd.concat([current_filtered, all_seen_filtered]).drop_duplicates(keep=False)
#             unique_to_current = combined.head(len(current_filtered))
#         else:
#             unique_to_current = current_filtered

#         # Update the all_seen_filtered DataFrame
#         all_seen_filtered = pd.concat([all_seen_filtered, unique_to_current], ignore_index=True)

#         # Save the filtered data back to the same file or delete if empty
#         if unique_to_current.empty:
#             os.remove(current_path)
#         else:
#             unique_to_current.to_csv(current_path, index=False)