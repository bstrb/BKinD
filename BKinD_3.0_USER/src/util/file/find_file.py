# find_file.py

# Standard Library Imports
import os

def find_file(path, pattern):
    """
    Generic file search to find a specific file pattern in the given directory.

    Args:
    path (str): The directory path where to search for the file.
    pattern (str): The file pattern to search for (e.g., '.fcf').

    Returns:
    str: The path to the file if found, otherwise None.
    """
    for file in os.listdir(path):
        if file.endswith(pattern):
            return os.path.join(path, file)
    # print(f"No {pattern} file found in the source directory.")
    return None

