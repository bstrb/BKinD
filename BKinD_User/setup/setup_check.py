# setup_check.py

# Standard library imports
import os
import sys
from importlib import metadata, util
import subprocess

# Third-party imports
import tkinter as tk
from tkinter import messagebox
    
def is_conda_environment():
    """Check if running in a conda environment."""
    return 'CONDA_DEFAULT_ENV' in os.environ

def check_python_version():
    """Check if the current Python version meets the requirement."""
    required_version = (3, 12, 2)
    current_version = sys.version_info
    return current_version[:3] == required_version

def show_messagebox_and_exit(title, message, error=True):
    """Display a messagebox and exit the application."""
    root = tk.Tk()
    root.withdraw()
    if error:
        messagebox.showerror(title, message)
    else:
        messagebox.showinfo(title, message) 
    root.destroy()
    sys.exit(message)

def is_package_installed(package_name):
    """Check if a package is installed."""
    return util.find_spec(package_name) is not None

def is_cctbx_installed():
    """Check if CCTBX is installed."""
    try:
        import iotbx
        return True
    except ImportError:
        return False

def check_packages_installed():
    """Check if required packages are installed."""
    required_packages = ['numpy', 'pandas', 'plotly', 'tqdm']
    missing_packages = [pkg for pkg in required_packages if not is_package_installed(pkg)]

    return missing_packages

def check_package_versions():
    """Check if the correct package versions are installed."""
    required_versions = {
        'numpy': '1.26.4',
        'pandas': '2.2.1',
        'plotly': '5.19.0',
        'tqdm': '4.66.4'
    }

    incorrect_versions = {}

    for package, required_version in required_versions.items():
        try:
            installed_version = metadata.version(package)
            if installed_version != required_version:
                incorrect_versions[package] = installed_version
        except metadata.PackageNotFoundError:
            incorrect_versions[package] = 'not installed'

    return incorrect_versions

def is_command_installed(command):
    """Check if a command is installed."""
    try:
        subprocess.run([command], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False

def check_xds_installed():
    """Check if XDS is installed."""
    return is_command_installed("xds")

def check_shelx_installed():
    """Check if SHELX is installed."""
    return is_command_installed("shelxl")

def is_pillow_installed():
    """Check if pillow package is installed."""
    try:
        from PIL import Image
        return True
    except ImportError:
        return False
