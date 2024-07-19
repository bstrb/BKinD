# show_output_folder.py

# Standard library imports
import os
import platform
import subprocess

# Third-party imports
from tkinter import messagebox

def show_output_folder(self):
    os_name = platform.system()
    if os_name == 'Darwin':
        subprocess.call(['open', self.output_folder])
    elif os_name == 'Linux' and 'microsoft' in platform.uname().release.lower():
        # Convert the WSL path to a Windows path
        if self.output_folder.startswith('/mnt/'):
            drive_letter = self.output_folder[5]
            windows_path = f"{drive_letter.upper()}:" + self.output_folder[6:].replace('/', '\\')
        else:
            result = subprocess.run(['wslpath', '-w', self.output_folder], capture_output=True, text=True)
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
            messagebox.showinfo("Error", f"Failed to open the folder: {e}")
        except Exception as e:
            messagebox.showinfo("Error", f"An unexpected error occurred: {e}")
    else:
        messagebox.showinfo("Unsupported OS", "This functionality is supported only on macOS and WSL.")