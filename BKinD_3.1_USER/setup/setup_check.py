# setup_check.py

# Standard library imports
import os
import sys
import subprocess
from importlib import metadata, util

# Third-party imports
try:
    import tkinter as tk
    from tkinter import messagebox
    
except ImportError:
    print("tkinter module for GUI missing. Install in terminal using 'sudo apt-get install python3-tk'")
    sys.exit("Exiting: Missing tkinter module. Please install manually and restart the application.")

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

def install_conda_package(package_name):
    """Attempt to install a package using Conda."""
    try:
        subprocess.run(["conda", "install", package_name, "-y"], check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def is_package_installed(package_name):
    """Check if a package is installed."""
    return util.find_spec(package_name) is not None

def is_cctbx_installed():
    """Check if CCTBX is installed and offer to install if missing."""
    try:
        import iotbx
        return True
    except ImportError:
        if messagebox.askyesno("Installation Required", "CCTBX is not installed. Install it now?"):
            if install_conda_package("cctbx-base"):
                show_messagebox_and_exit("Installation Successful", "CCTBX (cctbx-base) installed successfully. Please restart the application after installation of CCTBX.", error=False)
            else:
                show_messagebox_and_exit("Installation Failed", "Failed to install cctbx-base. Please install manually and restart the application.")
        else:
            show_messagebox_and_exit("Installation Skipped", "CCTBX installation was skipped. Consult the readme.txt for instructions on how to install CCTBX.")

def check_and_install_packages():
    """Check and install required packages."""
    required_packages = ['numpy', 'pandas', 'plotly', 'tqdm']
    missing_packages = [pkg for pkg in required_packages if not is_package_installed(pkg)]
    
    if missing_packages:
        if messagebox.askyesno("Missing Packages", f"The following packages are missing: {', '.join(missing_packages)}. Install them now?"):
            for package in missing_packages:
                if not install_conda_package(package):
                    show_messagebox_and_exit("Installation Failed", f"Failed to install {package}.")
            messagebox.showinfo("Installation Successful", "All packages were successfully installed.")
        else:
            show_messagebox_and_exit("Installation Aborted", "The application cannot run without all necessary packages.")
    return True

def check_and_install_package_versions():
    """Check if the correct package versions are installed and offer to install them if not."""
    packages = {
        'numpy': '1.26.4',
        'pandas': '2.2.1',
        'plotly': '5.19.0',
        'tqdm': '4.66.4'
    }
    
    try:
        for package, required_version in packages.items():
            installed_version = metadata.version(package)
            if installed_version != required_version:
                if messagebox.askyesno("Package Error", f"{package} {required_version} is required, but {installed_version} is installed. Install the required version?"):
                    if not install_conda_package(f"{package}=={required_version}"):
                        show_messagebox_and_exit("Installation Failed", f"Failed to install {package} {required_version}.")
                    else:
                        messagebox.showinfo("Installation Successful", f"{package} {required_version} installed successfully.")
                else:
                    show_messagebox_and_exit("Installation Aborted", f"{package} version is not correct and was not updated.")
        return True
    except metadata.PackageNotFoundError:
        show_messagebox_and_exit("Package Error", "One or more package versions are missing.")
    except subprocess.CalledProcessError:
        show_messagebox_and_exit("Installation Failed", "Failed to install the required packages.")

def is_command_installed(command):
    """Check if a command is installed."""
    try:
        subprocess.run([command], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False

def check_xds_installed():
    """Check if XDS is installed."""
    if not is_command_installed("xds"):
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("XDS Not Found", "XDS is not installed or not found in PATH. Please install XDS and try again. Consult the readme.txt for instructions on how to install XDS.")
        root.destroy()
        return False
    return True

def check_shelx_installed():
    """Check if SHELX is installed."""
    if not is_command_installed("shelxl"):
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("SHELX Not Found", "SHELX is not installed or not found in PATH. Please install SHELX and try again. Consult the readme.txt for instructions on how to install SHELX.")
        root.destroy()
        return False
    return True 

def check_and_install_pillow():
    """Check if pillow package is installed and prompt the user to install it if not."""
    try:
        from PIL import Image
    except ImportError:
        if messagebox.askyesno("Missing Package", "pillow package is not installed. Install it now?"):
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pillow"])
                messagebox.showinfo("Installation Successful", "pillow was successfully installed.")
            except subprocess.CalledProcessError:
                show_messagebox_and_exit("Installation Failed", "Failed to install pillow package.")
        else:
            show_messagebox_and_exit("Installation Aborted", "The application cannot run without pillow package.")
    return True
