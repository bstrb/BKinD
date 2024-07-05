# move_filtered_data.py

import os
import shutil

def move_filtered_data(folder_path):
    # Define the source file path and the new directory path
    source_file = os.path.join(folder_path, 'filtered_data.csv')
    new_folder = os.path.join(folder_path, 'solve_filtered')
    
    # Check if the source file exists
    if not os.path.isfile(source_file):
        print(f"No file named 'filtered_data.csv' found in {folder_path}.")
        return
    
    # Create the new folder if it doesn't exist
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    
    # Define the destination file path
    destination_file = os.path.join(new_folder, 'filtered_data.csv')
    
    # Move the file
    shutil.move(source_file, destination_file)
