# add_cbi.py

import tkinter as tk
from tkinter import filedialog, messagebox
import csv
import numpy as np
import os

def read_cbi_from_csv(csv_file_path):
    """Reads CBI values from a CSV file and returns them as a dictionary with frame numbers as keys."""
    cbi_dict = {}
    with open(csv_file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            frame = int(row[0])
            cbi = float(row[1])
            cbi_dict[frame] = cbi
    return cbi_dict

def update_hkl_file_with_cbi(hkl_filepath, cbi_dict):
    """Updates the XDS_ASCII.HKL file with CBI values based on z_obs."""
    updated_lines = []
    with open(hkl_filepath, 'r') as file:
        header = True
        for line in file:
            if header:
                updated_lines.append(line)
                if line.startswith('!END_OF_HEADER'):
                    header = False
                    updated_lines.append("!ITEM_CBI=13\n")  # Adding the CBI header
            else:
                if line.strip() and not line.startswith('!'):  # Process only data lines
                    parts = line.split()
                    h, k, l = int(parts[0]), int(parts[1]), int(parts[2])
                    intensity = float(parts[3])
                    sigma = float(parts[4])
                    xd, yd, zd = float(parts[5]), float(parts[6]), float(parts[7])
                    rlp = float(parts[8])
                    peak = int(parts[9])
                    corr = int(parts[10])
                    psi = float(parts[11])
                    z_obs = float(parts[7])  # zd as z_obs
                    z_obs_index = int(round(z_obs))  # Round to the nearest integer

                    # Get the closest CBI value based on z_obs_index
                    cbi_value = cbi_dict.get(z_obs_index, np.nan)

                    # Reconstruct the line with all original parts and append the CBI
                    updated_line = (
                        f"{h:4} {k:4} {l:4} {intensity:12.4e} {sigma:12.4e} {xd:8.1f} {yd:8.1f} {zd:8.1f} "
                        f"{rlp:8.4f} {peak:4d} {corr:4d} {psi:8.2f} {cbi_value:12.4f}\n"
                    )
                    updated_lines.append(updated_line)
                else:
                    updated_lines.append(line)

    output_filepath = hkl_filepath.replace("XDS_ASCII.HKL", "XDS_ASCII_CBI.HKL")
    with open(output_filepath, 'w') as file:
        file.writelines(updated_lines)

    return output_filepath

class AddCBIGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Add CBI to XDS_ASCII.HKL")
        
        # Initialize file paths
        self.hkl_file_path = ""
        self.cbi_csv_path = ""
        
        # Create GUI components
        self.create_widgets()

    def create_widgets(self):
        # Button to browse and select the HKL file
        self.browse_hkl_button = tk.Button(self.root, text="Browse HKL File", command=self.browse_hkl_file)
        self.browse_hkl_button.pack(pady=10)

        # Label to display the selected HKL file name
        self.hkl_file_label = tk.Label(self.root, text="No HKL file selected", fg="blue")
        self.hkl_file_label.pack(pady=5)

        # Button to browse and select the CSV file
        self.browse_csv_button = tk.Button(self.root, text="Browse CSV File", command=self.browse_csv_file)
        self.browse_csv_button.pack(pady=10)

        # Label to display the selected CSV file name
        self.csv_file_label = tk.Label(self.root, text="No CSV file selected", fg="blue")
        self.csv_file_label.pack(pady=5)

        # Button to add CBI to the HKL file
        self.add_cbi_button = tk.Button(self.root, text="Add CBI to HKL", command=self.add_cbi_to_hkl, state=tk.DISABLED)
        self.add_cbi_button.pack(pady=20)

    def browse_hkl_file(self):
        """Browse and select the HKL file."""
        self.hkl_file_path = filedialog.askopenfilename(filetypes=[("HKL files", "*.HKL")])
        if self.hkl_file_path:
            self.hkl_file_label.config(text=os.path.basename(self.hkl_file_path))
            self.check_ready_to_add_cbi()

    def browse_csv_file(self):
        """Browse and select the CSV file."""
        self.cbi_csv_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.cbi_csv_path:
            self.csv_file_label.config(text=os.path.basename(self.cbi_csv_path))
            self.check_ready_to_add_cbi()

    def check_ready_to_add_cbi(self):
        """Enable the 'Add CBI' button if both files are selected."""
        if self.hkl_file_path and self.cbi_csv_path:
            self.add_cbi_button.config(state=tk.NORMAL)

    def add_cbi_to_hkl(self):
        """Add CBI values to the HKL file."""
        cbi_dict = read_cbi_from_csv(self.cbi_csv_path)
        updated_hkl_path = update_hkl_file_with_cbi(self.hkl_file_path, cbi_dict)
        messagebox.showinfo("Success", f"Updated HKL file saved to {updated_hkl_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AddCBIGUI(root)
    root.mainloop()
