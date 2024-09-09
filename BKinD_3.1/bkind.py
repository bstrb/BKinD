# bkind.py

# Standard library imports
import os
import sys
import warnings
import re
import subprocess

# Third-party imports
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except ImportError:
    print("tkinter module for GUI missing. Install in terminal using 'sudo apt-get install python3-tk'")
    sys.exit("Exiting: Missing tkinter module. Please install manually and restart the application.")

# Custom module imports
from setup.setup_check import (
    is_conda_environment,
    # check_python_version,
    # is_cctbx_installed,
    # check_and_install_packages,
    # check_and_install_package_versions,
    # check_and_install_pillow,
    check_xds_installed,
    check_shelx_installed,
)

# Get the absolute path of the script and directory information
script_path = os.path.abspath(__file__)
script_directory = os.path.dirname(script_path)
folder_name = os.path.basename(script_directory)

# Extract the decimal number from the folder name
folder_number = re.search(r'\d+\.\d+', folder_name)
version = folder_number.group(0)

# Add the src directory to the Python path
sys.path.append(os.path.join(script_directory, 'src'))

# Suppress all warnings globally
warnings.filterwarnings("ignore")


def ensure_conda_environment():
    """Ensure the script is running in a Conda environment."""
    if not is_conda_environment():
        show_error_and_exit("Environment Error", "This application must be run within a Conda environment.")


# def ensure_python_version():
#     """Ensure the correct Python version is being used."""
#     if not check_python_version():
#         root = tk.Tk()
#         root.withdraw()
#         if messagebox.askyesno("Version Error", "This application requires Python 3.12.2. Install it now?"):
#             root.destroy()
#             try:
#                 subprocess.run(["conda", "install", "python==3.12.2", "-y"], check=True)
#                 messagebox.showinfo("Installation Successful", "Python 3.12.2 installed successfully. Please restart the application with 'python bkind.py' using conda python version 3.12.2.")
#             except subprocess.CalledProcessError:
#                 show_error_and_exit("Installation Failed", "Failed to install Python 3.12.2.")
#         else:
#             show_error_and_exit("Version Error", "Exiting: Incorrect Python version.")


def show_error_and_exit(title, message):
    """Display an error message and exit the application."""
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(title, message)
    root.destroy()
    sys.exit(message)


def check_requirements():
    """Check all necessary requirements and installations."""
    # if not is_cctbx_installed():
    #     sys.exit("CCTBX installation required.")
    # if not check_and_install_packages():
    #     sys.exit("Exiting: Application setup incomplete.")
    # if not check_and_install_package_versions():
    #     sys.exit("Exiting: Required package versions are not installed.")
    # if not check_and_install_pillow():
    #     sys.exit("Exiting: pillow package needed for image display.")
    if not check_xds_installed():
        sys.exit("XDS installation required.")
    if not check_shelx_installed():
        sys.exit("SHELX installation required.")


def initialize_app():
    """Initialize and run the application."""
    from src.gui.app_gui import App

    root = tk.Tk()
    apply_theme(root)
    app = App(root, ttk.Style(root), version)
    root.mainloop()


def apply_theme(root):
    """Apply a theme to the Tkinter application."""
    style = ttk.Style(root)
    style.theme_use('clam')
    style.configure("TButton", padding=6, relief="flat", background="#ccc")
    style.configure("TLabel", font=("Arial", 16))


def update_repo():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Run the update_repo.sh script
    update_script = os.path.join(script_dir, '/setup/update_repo.sh')
    
    try:
        result = subprocess.run(['bash', update_script], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Failed to update the repository: {e.stderr}")

def main():
    update_repo()
    ensure_conda_environment()
    # ensure_python_version()
    check_requirements()
    initialize_app()

if __name__ == "__main__":
    main()
