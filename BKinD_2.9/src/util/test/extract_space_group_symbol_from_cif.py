# extract_space_group_symbol_from_cif.py
# %%
# Utility File Imports
from find_file import find_file

def extract_space_group_symbol_from_cif(dir):
    """
    Extracts the space group symbol from a .cif file in the specified folder.
    
    Args:
    dir (str): Path to the folder containing the .cif file.
    
    Returns:
    str: The space group symbol without spaces, or None if not found.
    """
    space_group_symbol = None
    
    # Search for the .cif file in the given folder
    cif_file_path = find_file(dir, '.cif')
    
    # Extract the space group symbol from the .cif file
    with open(cif_file_path, 'r') as file:
        for line in file:
            if line.startswith('_space_group_name_H-M_alt'):
                space_group_symbol = ' '.join(line.split()[1:]).replace(' ', '')
                break
    
    if space_group_symbol is None:
        print("Space group symbol not found in the .cif file.")
    
    return space_group_symbol

# directory = '/Users/xiaodong/Desktop/bkind_LTA_to_96.5_completeness/filtered_96.5'

# space_group_symbol = extract_space_group_symbol_from_cif(directory)
    
# if space_group_symbol:
#     print(f"Extracted space group symbol: {space_group_symbol}")
# else:
#     print("No space group symbol found.")
# # %%
