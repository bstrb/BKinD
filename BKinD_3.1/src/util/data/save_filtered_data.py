# save_filtered_data.py

# Standard Library Imports
import os

def save_filtered_data(remaining_df, removed_df, target_folder):
    """
    Save remaining and removed data to CSV files.

    Parameters:
    - remaining_df (pd.DataFrame): DataFrame containing remaining data points.
    - removed_df (pd.DataFrame): DataFrame containing removed data points.
    - target_folder (str): Directory path to save the data files.
    """
    os.makedirs(target_folder, exist_ok=True)
    remaining_data_path = os.path.join(target_folder, 'remaining_data.csv')
    removed_data_path = os.path.join(target_folder, 'removed_data.csv')
    remaining_df.to_csv(remaining_data_path, index=False)
    removed_df.to_csv(removed_data_path, index=False)
