import re

def extract_r1_value(res_file_path):
    """
    Extract the second R1 value from the dynaml.res file.

    Args:
    - res_file_path (str): Path to the dynaml.res file.

    Returns:
    - float: The second R1 value (for all data), or None if not found.
    """
    try:
        with open(res_file_path, 'r') as file:
            for line in file:
                if line.startswith("REM R1"):
                    match = re.search(r'R1\s*=\s*\d+\.\d+\s*for\s*\d+\s*Fo\s*>\s*4sig\(Fo\)\s*and\s*(\d+\.\d+)\s*for\s*all', line)
                    if match:
                        return float(match.group(1))
        return None  # Return None if R1 value is not found
    except FileNotFoundError:
        print(f"File not found: {res_file_path}")
        return None
