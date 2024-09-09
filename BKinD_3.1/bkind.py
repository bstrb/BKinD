# bkind.py

# %%

# Standard library imports
import os
import sys
import warnings
import re
import subprocess

# Third-party imports
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, simpledialog
except ImportError:
    print("tkinter module for GUI missing. Install in terminal using 'sudo apt-get install python3-tk'")
    sys.exit("Exiting: Missing tkinter module. Please install manually and restart the application.")

# Custom module imports
from setup.setup_check import (
    is_conda_environment,
    check_python_version,
    is_cctbx_installed,
    is_pillow_installed,
    check_packages_installed,
    check_package_versions,
    check_xds_installed,
    check_shelx_installed,
)

# Get the absolute path of the script and directory information
script_path = os.path.abspath(__file__)
script_directory = os.path.dirname(script_path)
folder_name = os.path.basename(script_directory)

# Extract the decimal number from the folder name
folder_number = re.search(r'\d+\.\d+', folder_name)
version = folder_number.group(0) if folder_number else ''

# Add the src directory to the Python path
sys.path.append(os.path.join(script_directory, 'src'))

# Suppress all warnings globally
warnings.filterwarnings("ignore")

def show_error_and_exit(title, message):
    """Display an error message and exit the application."""
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(title, message)
    root.destroy()
    sys.exit(message)

def ensure_conda_environment():
    """Ensure the script is running in a Conda environment."""
    if not is_conda_environment():
        show_error_and_exit("Environment Error", "This application must be run within a Conda environment.")

def check_xds_and_shelx():
    """Check if XDS and SHELX are installed. Exit if they are not."""
    if not check_xds_installed():
        show_error_and_exit("XDS Not Found", "XDS is not installed. Please install XDS and try again.")
    if not check_shelx_installed():
        show_error_and_exit("SHELX Not Found", "SHELX is not installed. Please install SHELX and try again.")

def offer_to_create_env(reason):
    """Offer to create a new Conda environment with required packages if there are missing dependencies."""
    root = tk.Tk()
    root.withdraw()
    if messagebox.askyesno("Environment Setup", f"The current environment is missing some dependencies ({reason}). Would you like to create a new Conda environment?"):
        env_name = simpledialog.askstring("Environment Name", "Enter the name for the new Conda environment:")
        if env_name:
            create_conda_environment(env_name)
        else:
            show_error_and_exit("Environment Creation Cancelled", "No environment name provided. Exiting...")
    else:
        messagebox.showinfo("Manual Setup Required", 
                            "The application cannot continue without the correct environment. "
                            "Please install the required dependencies manually by referring to the README file, "
                            "or use the clickable icons to automatically create a new environment with the proper setup.")
        sys.exit("Exiting: Dependencies are missing.")
    root.destroy()

def create_conda_environment(env_name):
    """Create a new Conda environment with the required packages."""
    command = [
        "conda", "create", "-n", env_name, "-c", "conda-forge", "cctbx-base", "python=3.12.2",
        "numpy==1.26.4", "pandas==2.2.1", "plotly==5.19.0", "tqdm==4.66.4", "pillow==10.3.0", "-y"
    ]
    try:
        subprocess.run(command, check=True)
        messagebox.showinfo("Success", f"Conda environment '{env_name}' created successfully. Please activate it with 'conda activate {env_name}' and restart the application.")
        sys.exit(f"Please activate the environment using: 'conda activate {env_name}'")
    except subprocess.CalledProcessError:
        show_error_and_exit("Environment Creation Failed", f"Failed to create the Conda environment '{env_name}'.")

def check_python_cctbx_pillow():
    """Check Python version and CCTBX installation."""
    if not check_python_version():
        offer_to_create_env("Python version")
    elif not is_cctbx_installed():
        offer_to_create_env("CCTBX")
    elif not is_pillow_installed():
        offer_to_create_env("Pillow")

def check_packages():
    """Check if all required packages and their versions are installed."""
    missing_packages = check_packages_installed()
    if missing_packages:
        offer_to_create_env(f"Missing packages: {', '.join(missing_packages)}")

    incorrect_versions = check_package_versions()
    if incorrect_versions:
        offer_to_create_env(f"Incorrect package versions: {', '.join([f'{pkg} (installed: {ver})' for pkg, ver in incorrect_versions.items()])}")

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
    update_script = os.path.join(script_dir, 'setup/update_repo.sh')
    
    try:
        result = subprocess.run(['bash', update_script], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Failed to update the repository: {e.stderr}")

def main():
    update_repo()
    ensure_conda_environment()  # Check Conda environment first
    check_xds_and_shelx()  # Ensure XDS and SHELX are installed
    check_python_cctbx_pillow()  # Ensure correct Python version and CCTBX
    check_packages()  # Ensure correct package versions
    initialize_app()  # Run the application

if __name__ == "__main__":
    main()
