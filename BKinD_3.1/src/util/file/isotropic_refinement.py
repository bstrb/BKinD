# isotropic_refinement.py

# Standard Library Imports
import os
import re

# File Imports
from util.file.find_file import find_file

def isotropic_refinement(directory):
    """
    Modifies the .ins file in the specified directory by keeping only the first 6 entries for each atom line.
    If the following line starts with numbers after some spacing, it will be removed.
    Skips processing of lines starting with ZERR, HKLF, or QXX (where XX is any number) but leaves them intact.

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
        skip_next_line = False  # Flag to skip the next line if needed

        for i, line in enumerate(lines):
            # Check if the line should be left intact (ZERR, HKLF, QXX)
            if line.strip().startswith("ZERR") or line.strip().startswith("HKLF") or re.match(r'^Q\d{1,2}', line.strip()):
                ins_file.write(line)
                continue  # Skip further processing and move to the next line

            if skip_next_line:
                skip_next_line = False
                continue

            parts = line.split()

            # Check if the line is an atom line (starts with an atom label and contains numbers)
            if len(parts) > 6 and parts[0][0].isalpha() and parts[1].isdigit():
                # Write only the first 6 entries of the atom line
                truncated_line = " ".join(parts[:7]) + "\n"
                ins_file.write(truncated_line)

                # Check if the next line starts with spaces followed by numbers
                if i + 1 < len(lines) and re.match(r'^\s+\d', lines[i + 1]):
                    skip_next_line = True  # Mark to skip the next line
            else:
                # Write the line as it is if it doesn't match the atom line format
                ins_file.write(line)
