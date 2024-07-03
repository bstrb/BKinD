# get_subfolder.py

# Standard Library Imports
import os

def get_subfolder(base_directory, crys, target_completeness, xray=False):
    """
    Creates a subfolder named 'bkind_{crys}_to_{target_completeness}_completeness' in the specified base directory and returns its path.

    Parameters:
    - base_directory (str): The path to the directory where the subfolder will be created.
    - crys (str): The crystal identifier.
    - target_completeness (int or float): The target completeness value.
    - xray (bool): If True, prefix 'XRAY_' to the subfolder name.

    Returns:
    - str: The path to the created subfolder.
    """
    # Determine the folder name based on the xray flag
    prefix = "XRAY_" if xray else ""
    subfolder_name = f'bkind_{prefix}{crys}_to_{int(target_completeness)}_completeness'
    subfolder_path = os.path.join(base_directory, subfolder_name)

    return subfolder_path
