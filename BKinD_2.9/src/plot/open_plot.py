# open_plot.py

# Standard library imports
import platform
import subprocess

def open_plot(fig, plot_filename):
    os_name = platform.system()
    
    if os_name == 'Darwin':
        fig.show()
    elif os_name == 'Linux' and 'microsoft' in platform.uname().release.lower():
        # Translate the POSIX path to a Windows path
        windows_path = plot_filename.replace('/mnt/c', 'C:')
        windows_path = windows_path.replace('/', '\\')

        # Use Windows command to open the file
        # Make sure the path is enclosed in double quotes
        command = f'cmd.exe /c start "{windows_path}"'
        result = subprocess.run(command, shell=True, stderr=subprocess.DEVNULL)
    else:
        print("Unsupported OS. This script supports only macOS and WSL.")
        return