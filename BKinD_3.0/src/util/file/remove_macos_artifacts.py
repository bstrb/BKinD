# remove_macos_artifacts.py

# Standard library imports
import os
import subprocess

def remove_macos_artifacts(path):
    """Remove macOS-created metadata files from a directory."""
    if not os.path.exists(path):
        print("Given path does not exist:", path)
        return
    if not os.access(path, os.R_OK):
        print("No read permissions on the directory:", path)
        return
    
    try:
        command = ['find', path, '-name', '._*', '-delete']
        subprocess.run(command, check=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        pass
