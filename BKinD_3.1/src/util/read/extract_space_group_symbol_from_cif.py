# extract_space_group_symbol_from_cif.py

# Utility File Imports
from util.file.find_file import find_file

def format_space_group_symbol(symbol):
    # Check for two consecutive digits and place the second in parentheses
    formatted_symbol = ""
    i = 0
    while i < len(symbol):
        if i < len(symbol) - 1 and symbol[i].isdigit() and symbol[i+1].isdigit():
            formatted_symbol += f"{symbol[i]}({symbol[i+1]})"
            i += 2  # Skip the next character as it's already processed
        else:
            formatted_symbol += symbol[i]
            i += 1
    
    # Remove spaces and replace slashes with underscores
    formatted_symbol = formatted_symbol.replace(" ", "").replace("/", "_")
    
    return formatted_symbol

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
                space_group_symbol = ' '.join(line.split()[1:])
                break
    
    if space_group_symbol is None:
        # print("Space group symbol not found in the .cif file.")
        return None
    
    # Format the space group symbol
    formatted_symbol = format_space_group_symbol(space_group_symbol)
    
    return formatted_symbol
