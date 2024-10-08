# find_stream_files.py

import os
import fnmatch

def find_stream_files(directory):
    # Check if the directory is a valid directory
    if not os.path.isdir(directory):
        raise ValueError("Provided path is not a valid directory")

    # List all files in the directory
    all_files = os.listdir(directory)

    # Filter files that end with 'stream'
    stream_files = fnmatch.filter(all_files, '*.stream')

    full_paths = [os.path.join(directory, file) for file in stream_files]

    return full_paths
