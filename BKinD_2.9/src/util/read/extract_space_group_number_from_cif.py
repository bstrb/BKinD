# extract_space_group_number_from_cif.py

# Utility File Imports
from util.file.find_file import find_file

def extract_space_group_number_from_cif(dir):
    """
    Extracts the space group number from a .cif file in the specified folder.
    
    Args:
    dir (str): Path to the folder containing the .cif file.
    
    Returns:
    int: The space group number, or None if not found.
    """
    space_group_number = None
    
    # Search for the .cif file in the given folder
    cif_file_path = find_file(dir, '.cif')
    
    # Extract the space group number from the .cif file
    with open(cif_file_path, 'r') as file:
        for line in file:
            if line.startswith('_space_group_IT_number'):
                space_group_number = int(line.split()[1])
                break
    
    if space_group_number is None:
        print("Space group number not found in the .cif file.")
    
    return space_group_number
