# # open_plot.py

# # Standard library imports
# import platform
# import subprocess

# def open_plot(fig, plot_filename):
#     os_name = platform.system()
    
#     if os_name == 'Darwin':
#         fig.show()
#     elif os_name == 'Linux' and 'microsoft' in platform.uname().release.lower():
#         # Translate the POSIX path to a Windows path
#         windows_path = plot_filename.replace('/mnt/c', 'C:')
#         windows_path = windows_path.replace('/', '\\')

#         # Use Windows command to open the file
#         # Make sure the path is enclosed in double quotes
#         command = f'cmd.exe /c start "{windows_path}"'
#         result = subprocess.run(command, shell=True, stderr=subprocess.DEVNULL)
#     else:
#         print("Unsupported OS. This script supports only macOS and WSL.")
#         return
    
# open_plot.py

# Standard library imports
import platform
import subprocess

def open_plot(fig, plot_filename):
    os_name = platform.system()
    
    if os_name == 'Darwin':
        # macOS: Just show the figure
        fig.show()
    elif os_name == 'Linux' and 'microsoft' in platform.uname().release.lower():
        # For WSL
        windows_path = plot_filename
        
        # Check if the path is under /mnt (e.g., /mnt/c)
        if windows_path.startswith('/mnt/'):
            # Convert /mnt/c/path/to/file to C:\path\to\file
            drive_letter = windows_path[5].upper()  # Extract the drive letter
            windows_path = f'{drive_letter}:\\' + windows_path[7:].replace('/', '\\')
        elif windows_path.startswith('/home'):
            # Convert /home/user/path to \\wsl.localhost\Ubuntu\home\user\path
            windows_path = r'\\\wsl.localhost\Ubuntu' + windows_path.replace('/', '\\')

        # Use Windows command to open the file
        # Make sure the path is enclosed in double quotes
        command = f'cmd.exe /c start \"{windows_path}\"'
        try:
            subprocess.run(command, shell=True, check=True, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"Failed to open the plot. Error: {e.stderr.decode()}")
    else:
        print("Unsupported OS. This script supports only macOS and WSL.")
