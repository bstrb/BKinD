# clean_aggregate.py

# Standard Library Imports
import os
import warnings

# Third-Party Imports
import pandas as pd

def clean_aggregate(folder_path):
    """
    Clean files in a folder so that each file only contains unique data not present in files with a higher target percentage.

    Parameters:
    - folder_path: str, path to the folder containing the extreme data CSV files.
    """
    # Suppress specific pandas FutureWarnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Step 1: Collect all CSV files and sort them by decreasing target percentages
    files = [f for f in os.listdir(folder_path) if f.startswith('filtered_') and f.endswith('.csv')]
    files.sort(key=lambda x: float(x.split('_')[1].replace('.csv', '')), reverse=False)

    # Append 'remaining_data.csv' after sorting
    files.append('remaining_data.csv')

    # Step 2: Initialize a DataFrame to store all seen filtered
    all_seen_filtered = pd.DataFrame()

    for file in files:
        current_path = os.path.join(folder_path, file)
        current_filtered = pd.read_csv(current_path)

        # Filter out any filtered already seen in higher percentage files
        if not all_seen_filtered.empty:
            combined = pd.concat([current_filtered, all_seen_filtered]).drop_duplicates(keep=False)
            unique_to_current = combined.head(len(current_filtered))
        else:
            unique_to_current = current_filtered

        # Update the all_seen_filtered DataFrame
        all_seen_filtered = pd.concat([all_seen_filtered, unique_to_current], ignore_index=True)

        # Save the filtered data back to the same file or delete if empty
        if unique_to_current.empty:
            os.remove(current_path)
        else:
            unique_to_current.to_csv(current_path, index=False)