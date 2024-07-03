# save_filtered_data.py

# Standard Library Imports
import os

def save_filtered_data(remaining_df, filtered_df, target_folder):
    """
    Save remaining and filtered data to CSV files.

    Parameters:
    - remaining_df (pd.DataFrame): DataFrame containing remaining data points.
    - filtered_df (pd.DataFrame): DataFrame containing filtered data points.
    - target_folder (str): Directory path to save the data files.
    """
    os.makedirs(target_folder, exist_ok=True)
    remaining_data_path = os.path.join(target_folder, 'remaining_data.csv')
    filtered_data_path = os.path.join(target_folder, 'filtered_data.csv')
    remaining_df.to_csv(remaining_data_path, index=False)
    filtered_df.to_csv(filtered_data_path, index=False)