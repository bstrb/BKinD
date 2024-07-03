# create_subfolder.py

# Standard Library Imports
import os
import shutil

def create_subfolder(base_directory, crys, completeness, xray=False):
    """
    Creates a subfolder named 'bkind_{crys}_to_{completeness}_completeness' in the specified base directory. 
    If the subfolder already exists, it is cleaned of all contents.
    
    Parameters:
    - base_directory (str): The path to the directory where the subfolder will be created.
    - crys (str): The crystal identifier.
    - completeness (int or float): The completeness value.
    - xray (bool): If True, prefix 'XRAY_' to the subfolder name.
    
    Returns:
    - str: The path to the created (and cleaned, if necessary) subfolder.
    """
    # Determine the folder name based on the xray flag
    prefix = "XRAY_" if xray else ""
    subfolder_path = os.path.join(base_directory, f'bkind_{prefix}{crys}_to_{int(completeness)}_completeness')

    # Check if the directory already exists
    if os.path.exists(subfolder_path):
        # Remove all contents of the directory
        for filename in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove file or link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directory and all its contents
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
                return None
         
    # Ensure the directory exists (create if it does not)
    try:
        os.makedirs(subfolder_path, exist_ok=True)
    except Exception as e:
        print(f"An error occurred while creating the subfolder: {e}")
        return None

    return subfolder_path
