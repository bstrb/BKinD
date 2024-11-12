import os

def find_first_file(directory, extension):
    # Find the first file in the directory with the specified extension
    for file_name in os.listdir(directory):
        if file_name.endswith(extension):
            return os.path.join(directory, file_name)  # Return the full path
    return None