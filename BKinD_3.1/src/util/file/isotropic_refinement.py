# isotropic_refinement.py

# Standard Library Imports
import os

# File Imports
from util.file.find_file import find_file

def isotropic_refinement(directory):
    """
    Modifies the .ins file in the specified directory by keeping only the first 5 entries for each atom line.

    Parameters:
    - directory: str, the directory containing the .ins file to be modified.

    Returns:
    - None, but modifies the .ins file in the specified directory.
    """

    # Locate the .ins file
    ins_file_path = find_file(directory, '.ins')

    if not os.path.isfile(ins_file_path):
        raise FileNotFoundError(f"No .ins file found in the specified directory: {ins_file_path}")

    with open(ins_file_path, 'r') as ins_file:
        lines = ins_file.readlines()

    with open(ins_file_path, 'w') as ins_file:
        for line in lines:
            # Identify lines that contain atom information based on the example format
            parts = line.split()
            if len(parts) > 5 and parts[0][0].isalpha() and parts[1].isdigit():
                # Keep only the first 5 entries
                truncated_line = " ".join(parts[:6]) + "\n"
                ins_file.write(truncated_line)
            else:
                # Write the line as it is if it doesn't match the atom line format
                ins_file.write(line)
