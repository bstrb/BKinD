# extract_parts_from_path.py

# Standard Library Imports
import os
import re

def extract_parts_from_path(path):
    """
    Extract the relevant parts of the folder name from the given path.
    
    Args:
        path (str): The full path from which to extract parts.
    
    Returns:
        tuple: A tuple containing the relevant parts of the folder name.
    """
    # Extract the folder name
    folder_name = os.path.basename(os.path.normpath(path))

    # Define both regular expressions
    pattern1 = re.compile(r'bkind_([^_]+)_to_([^_]+)_completeness')
    pattern2 = re.compile(r'bkind_([^_]+)_([^_]+)_to_([^_]+)_completeness')

    # Try matching the first pattern
    match1 = pattern1.search(folder_name)
    if match1:
        second_part = match1.group(1)
        fourth_part = match1.group(2)
        return second_part, fourth_part

    # Try matching the second pattern
    match2 = pattern2.search(folder_name)
    if match2:
        second_part = match2.group(2)
        fourth_part = match2.group(3)
        return second_part, fourth_part

    # Raise an error if no pattern matches
    raise ValueError(f"The path '{path}' does not match the expected patterns.")

