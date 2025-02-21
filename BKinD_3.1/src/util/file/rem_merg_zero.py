# rem_merg_zero.py

# Standard Library Imports
import os

# File Imports
from util.file.find_file import find_file

def rem_merg_zero(directory):
    """
    Modifies the .ins file in the specified directory by changing "MERG 0" to "REM MERG 0".

    Parameters:
    - directory: str, the directory containing the .ins file to be modified.

    Returns:
    - None, but modifies the .ins file in the specified directory.
    """
    # ins_file_name = os.path.basename(directory) + '.ins'
    # ins_file_path = os.path.join(directory, ins_file_name)

    ins_file_path = find_file(directory, '.ins')

    if not os.path.isfile(ins_file_path):
        raise FileNotFoundError(f"No .ins file found in the specified directory: {ins_file_path}")

    with open(ins_file_path, 'r') as ins_file:
        lines = ins_file.readlines()
    
    with open(ins_file_path, 'w') as ins_file:
        for line in lines:
            if 'MERG 0' in line.upper():
                line = 'REM ' + line
            ins_file.write(line)