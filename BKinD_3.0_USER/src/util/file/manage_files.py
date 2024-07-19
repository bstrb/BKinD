# manage_files.py

# Standard library imports
import os
import shutil

def manage_files(action, source_directory, target_directory, filename=None, new_filename=None, extension=None):
    """
    Manages files by moving or copying them, with options to rename or find files by extension.

    Parameters:
    - action: The action to perform - 'move' or 'copy'.
    - source_directory: The directory containing the source file.
    - target_directory: The directory where the file will be moved/copied.
    - filename: The name of the file to be moved/copied. If not provided, extension must be specified.
    - new_filename: The new name for the file in the target directory. Optional.
    - extension: The file extension to search for if filename is not provided. Optional.
    """

    source_file_path = None

    if filename:
        source_file_path = os.path.join(source_directory, filename)
    elif extension:
        # Find the first file with the given extension in the source directory
        for file in os.listdir(source_directory):
            if file.endswith(extension):
                source_file_path = os.path.join(source_directory, file)
                break
        if not source_file_path:
            print(f"No file with extension '{extension}' found in '{source_directory}'.")
            return False
    else:
        print("Either 'filename' or 'extension' must be provided.")
        return False

    if not new_filename:
        new_filename = os.path.basename(source_file_path)
        
    target_file_path = os.path.join(target_directory, new_filename)
    
    # Ensure the target directory exists
    os.makedirs(target_directory, exist_ok=True)

    try:
        if action == 'move':
            shutil.move(source_file_path, target_file_path)
            # print(f"File '{os.path.basename(source_file_path)}' moved to '{target_file_path}'.")
        elif action == 'copy':
            shutil.copy(source_file_path, target_file_path)
            # print(f"File '{os.path.basename(source_file_path)}' copied to '{target_file_path}'.")
        else:
            print(f"Invalid action '{action}'. Use 'move' or 'copy'.")
            return False
        return True
    except FileNotFoundError:
        print(f"Error: '{source_file_path}' not found.")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
