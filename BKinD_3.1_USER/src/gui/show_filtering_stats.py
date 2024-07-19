# show_filtering_stats.py

# Standard library imports
import os
import platform
import subprocess

# Third-party imports
from tkinter import messagebox

def show_filtering_stats(self):
    os_name = platform.system()
    file_path = os.path.join(self.output_folder, 'filtering_stats.txt')
    
    if os_name == 'Darwin':  # macOS
        subprocess.call(['open', file_path])
    elif os_name == 'Linux' and 'microsoft' in platform.uname().release.lower():  # WSL
        # Convert the WSL path to a Windows path
        if self.output_folder.startswith('/mnt/'):
            drive_letter = self.output_folder[5]
            windows_path = f"{drive_letter.upper()}:" + self.output_folder[6:].replace('/', '\\') + '\\filtering_stats.txt'
        else:
            result = subprocess.run(['wslpath', '-w', file_path], capture_output=True, text=True)
            windows_path = result.stdout.strip()

        try:
            os.startfile(windows_path)
            return
        except AttributeError:
            pass
        except FileNotFoundError:
            messagebox.showinfo("Error", "The specified Windows path does not exist.")
            return
        except Exception as e:
            messagebox.showinfo("Error", f"An unexpected error occurred: {e}")
            return

        try:
            verify_command = f'powershell.exe Test-Path "{windows_path}"'
            result = subprocess.run(verify_command, shell=True, capture_output=True, text=True)
            if result.stdout.strip() == "False":
                messagebox.showinfo("Error", "The specified Windows path does not exist.")
                return

            open_command = f'powershell.exe Start-Process "{windows_path}"'
            result = subprocess.run(open_command, shell=True, capture_output=True, text=True)
            result.check_returncode()
        except subprocess.CalledProcessError as e:
            messagebox.showinfo("Error", f"Failed to open the file: {e}")
        except Exception as e:
            messagebox.showinfo("Error", f"An unexpected error occurred: {e}")
    elif os_name == 'Linux':  # Linux
        subprocess.run(['xdg-open', file_path])
    elif os_name == 'Windows':  # Windows
        try:
            os.startfile(file_path)
        except FileNotFoundError:
            messagebox.showinfo("Error", "The file 'filtering_stats.txt' was not found in the directory.")
        except Exception as e:
            messagebox.showinfo("Error", f"An unexpected error occurred: {e}")
    else:
        messagebox.showinfo("Unsupported OS", "This functionality is supported only on macOS, Linux, WSL, and Windows.")
