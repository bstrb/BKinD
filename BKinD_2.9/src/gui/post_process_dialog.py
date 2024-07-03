# post_process_dialog.py

# Standard library imports
import os
import shutil
import platform
import subprocess

# Third-party imports
import tkinter as tk
from tkinter import messagebox, ttk, Toplevel

# Plot Imports
from plot.plot_dfm_vs_frame import plot_DFM_vs_Frame
from plot.plot_res_vs_dfm import plot_Resolution_vs_DFM
from plot.plot_dfm_distribution import plot_DFM_distribution
from plot.plot_r_vs_completeness import plot_R1_Rint_vs_completeness
from plot.plot_fvar_npd_vs_completeness import plot_FVAR_NPD_vs_completeness

# GUI Imports
from gui.tooltip import ToolTip

class PostProcessDialog(tk.Toplevel):
    """A dialog for post-processing options after filtering."""

    def __init__(self, parent, output_folder, style, DFM_plot=False):
        """Initialize the PostProcessDialog.

        Parameters:
        parent : tk.Widget
            The parent widget.
        output_folder : str
            Path to the output folder.
        style : ttk.Style
            The style used for the dialog's widgets.
        """
        super().__init__(parent)
        self.style = style
        self.output_folder = output_folder
        self.title("BKinD Filtering Results")

        ttk.Label(
            self, 
            text=("""Filtering has been successfully completed. You can find the filtering statistics in
the filtering_stats.txt file located in the output folder. Please select an action:"""),
            style="TLabel"
        ).pack(pady=10)

        ttk.Button(self, text="Show Output Folder", command=self.show_output_folder, style="TButton").pack(fill='x', expand=True, padx=20, pady=5)
        if DFM_plot:
            self.create_button("Plot DFM vs Frame", self.plot_data, plot_DFM_vs_Frame, "Interactive Plot of Filtered Data as DFM vs Frame.")
        self.create_button("Plot Resolution vs DFM", self.plot_data, plot_Resolution_vs_DFM, "Interactive Plot of Filtered Data as Resolution vs DFM.")
        self.create_button("Plot DFM Distribution", self.plot_data, plot_DFM_distribution, "Interactive Plot of Filtered Data DFM Distribution.")
        self.create_button("Plot R1, Rint and Remaining Data Percentage vs ASU", self.plot_data, plot_R1_Rint_vs_completeness, "Plot R1 and Rint along with Remaining Data Percentage and Average Multiplicity vs Completeness.")
        self.create_button("Plot FVAR, NPD vs ASU", self.plot_data, plot_FVAR_NPD_vs_completeness, "Plot FVAR and Number of NPDs vs Completeness.")
        
        self.clean_output_folder_button = ttk.Button(self, text="Clean Output Folder", command=self.clean_output_folder, style="TButton")
        self.clean_output_folder_button.pack(fill='x', expand=True, padx=20, pady=5)
        self.create_tooltip(self.clean_output_folder_button, """Clean output folder from all but .txt with filtering stats,
eventual plots(.hmtl-files) and folder with filtered data(.csv-format).""")
        
        ttk.Button(self, text="Close", command=self.destroy, style="TButton").pack(pady=20)

    def create_button(self, text, command, plot_func, tooltip_text):
        """Create a button with a tooltip.

        Parameters:
        text : str
            The text to display on the button.
        command : function
            The function to call when the button is clicked.
        plot_func : function
            The plotting function to call, if applicable.
        tooltip_text : str
            The text to display in the tooltip.
        """
        btn_command = lambda: command(plot_func) if plot_func else command
        button = ttk.Button(self, text=text, command=btn_command, style="TButton")
        button.pack(fill='x', expand=True, padx=20, pady=5)
        self.create_tooltip(button, tooltip_text)

    def create_tooltip(self, widget, text):
        """Create a tooltip for a widget.

        Parameters:
        widget : tk.Widget
            The widget to attach the tooltip to.
        text : str
            The text to display in the tooltip.
        """
        tooltip = ToolTip(widget, self.style)
        widget.bind('<Enter>', lambda event: tooltip.showtip(text))
        widget.bind('<Leave>', lambda event: tooltip.hidetip())
   
    os_name = platform.system()
    
    if os_name == 'Darwin':
        
        def show_output_folder(self):
            os.startfile(self.output_folder) if os.name == 'nt' else subprocess.call(['open', self.output_folder])

        def clean_output_folder(self):
            for filename in os.listdir(self.output_folder):
                file_path = os.path.join(self.output_folder, filename)

                # Skip deletion if the file is a .txt or .html file
                if filename.endswith('.txt') or filename.endswith('.html'):
                    continue

                # Skip deletion if the directory is named 'aggregated_filtered'
                if os.path.isdir(file_path) and filename == 'aggregated_filtered':
                    continue

                # Delete files or directories that do not meet the above conditions
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

            messagebox.showinfo("Cleanup", "Output folder has been cleaned, preserving .txt, .html files and the 'aggregated_filtered' folder.")

    elif os_name == 'Linux' and 'microsoft' in platform.uname().release.lower():
        
        def show_output_folder(self):
            """Open the output folder in the Windows file explorer."""
            output_folder = self.output_folder

            # Convert the WSL path to a Windows path
            if output_folder.startswith('/mnt/'):
                drive_letter = output_folder[5]
                windows_path = f"{drive_letter.upper()}:" + output_folder[6:].replace('/', '\\')
            else:
                result = subprocess.run(['wslpath', '-w', output_folder], capture_output=True, text=True)
                windows_path = result.stdout.strip()

            # Try to open the folder using os.startfile (simpler method)
            try:
                os.startfile(windows_path)
                return
            except AttributeError:
                # os.startfile is not available on non-Windows systems
                pass
            except FileNotFoundError:
                messagebox.showinfo("Error", "The specified Windows path does not exist.")
                return
            except Exception as e:
                messagebox.showinfo("Error", f"An unexpected error occurred: {e}")
                return

            # Fallback to PowerShell if os.startfile fails
            try:
                # Verify the existence of the directory using PowerShell
                verify_command = f'powershell.exe Test-Path "{windows_path}"'
                result = subprocess.run(verify_command, shell=True, capture_output=True, text=True)
                if result.stdout.strip() == "False":
                    messagebox.showinfo("Error", "The specified Windows path does not exist.")
                    return

                # Attempt to open the folder using PowerShell
                open_command = f'powershell.exe Start-Process "{windows_path}"'
                result = subprocess.run(open_command, shell=True, capture_output=True, text=True)
                result.check_returncode()  # This will raise CalledProcessError if the command failed
            except subprocess.CalledProcessError as e:
                messagebox.showinfo("Error", f"Failed to open the folder: {e}")
            except Exception as e:
                messagebox.showinfo("Error", f"An unexpected error occurred: {e}")

        def clean_output_folder(self):
            # Check if the output folder path exists
            if not os.path.exists(self.output_folder):
                messagebox.showinfo("Error", "Output folder does not exist.")
                return

            # Iterate over each file in the directory
            for filename in os.listdir(self.output_folder):
                file_path = os.path.join(self.output_folder, filename)

                # Skip deletion if the file is a .txt or .html file
                if filename.endswith('.txt') or filename.endswith('.html'):
                    continue

                # Skip deletion if the directory is named 'aggregated_filtered'
                if os.path.isdir(file_path) and filename == 'aggregated_filtered':
                    continue

                # Delete files or directories that do not meet the above conditions
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

            messagebox.showinfo("Cleanup", "Output folder has been cleaned, preserving .txt, .html files and the 'aggregated_filtered' folder.")
    else:
        print("Unsupported OS. This script supports only macOS and WSL.")

    def plot_data(self, plot_function):
        try:
            # Set to a known safe directory
            os.chdir(os.path.expanduser("~"))  # Change to home directory
            plot_function(self.output_folder)
            messagebox.showinfo("Plotting", "Plotting complete. The plot has been generated successfully and saved in output folder.")
        except Exception as e:
            messagebox.showerror("Error", f"Error during plotting: {e}")