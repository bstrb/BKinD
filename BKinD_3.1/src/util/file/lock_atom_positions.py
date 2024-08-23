# lock_atom_positions.py

# Standard Library Imports
import os
import re

# File Imports
from util.file.find_file import find_file

def lock_atom_positions(directory):
    """
    Modifies the .ins file in the specified directory by adding 10 to the x, y, and z coordinates of atom positions,
    while leaving lines starting with ZERR, HKLF, or QXX (where XX is any number) untouched.

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
            # Skip lines starting with ZERR, HKLF, or QXX (where XX is any number)
            if line.strip().startswith("ZERR") or line.strip().startswith("HKLF") or re.match(r'^Q\d{1,2}', line.strip()):
                ins_file.write(line)
                continue

            # Identify lines that contain atom information based on the example format
            parts = line.split()
            if len(parts) > 5 and parts[0][0].isalpha() and parts[1].isdigit():
                # Assuming atom lines have x, y, z as the 3rd, 4th, and 5th elements
                try:
                    parts[2] = f"{float(parts[2]) + 10:.6f}"
                    parts[3] = f"{float(parts[3]) + 10:.6f}"
                    parts[4] = f"{float(parts[4]) + 10:.6f}"
                except ValueError:
                    # If the conversion fails, skip this line (this might be a non-atom line)
                    pass

                # Rebuild the line, ensuring it doesn't exceed 80 characters
                line = " ".join(parts)
                if len(line) > 80:
                    line = line[:80] + "\n"
                else:
                    line += "\n"

            ins_file.write(line)
