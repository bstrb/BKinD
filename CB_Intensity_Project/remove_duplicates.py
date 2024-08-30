# remove_dupliocates.py

import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd

def read_hkl_file(hkl_filepath, iset_value):
    reflections = []
    header_lines = []
    with open(hkl_filepath, 'r') as file:
        header = True
        for line in file:
            if header:
                header_lines.append(line)
                if line.startswith('!END_OF_HEADER'):
                    header = False
            else:
                if line.strip() and not line.startswith('!'):  # Process only data lines
                    parts = line.split()
                    h, k, l = int(parts[0]), int(parts[1]), int(parts[2])
                    iobs = float(parts[3])
                    sigma = float(parts[4])
                    xd, yd, zd = float(parts[5]), float(parts[6]), float(parts[7])
                    rlp = float(parts[8])
                    peak = int(parts[9])
                    corr = int(parts[10])
                    psi = float(parts[11])
                    cbi = float(parts[12])  # Assuming CBI is the last column
                    reflections.append((h, k, l, iobs, sigma, xd, yd, zd, rlp, peak, corr, psi, iset_value, cbi))
    return pd.DataFrame(reflections, columns=['h', 'k', 'l', 'iobs', 'sigma', 'xd', 'yd', 'zd', 'rlp', 'peak', 'corr', 'psi', 'iset', 'cbi']), header_lines

def merge_and_filter_reflections(hkl_files):
    combined_data = pd.DataFrame()
    headers = {}

    # Read each HKL file, assign an iset value, and combine the reflections
    for iset_value, hkl_file in enumerate(hkl_files, start=1):
        data, header_lines = read_hkl_file(hkl_file, iset_value)
        combined_data = pd.concat([combined_data, data], ignore_index=True)
        headers[iset_value] = header_lines

    # Sort by Miller indices (h, k, l) and CBI, then remove duplicates keeping the highest CBI
    combined_data.sort_values(by=['h', 'k', 'l', 'cbi'], ascending=[True, True, True, False], inplace=True)
    filtered_data = combined_data.drop_duplicates(subset=['h', 'k', 'l'], keep='first')

    return filtered_data, headers

def save_filtered_hkl_files(filtered_data, headers, hkl_files):
    for iset_value, hkl_file in enumerate(hkl_files, start=1):
        # Filter reflections belonging to the current iset
        file_data = filtered_data[filtered_data['iset'] == iset_value]

        # Drop the 'iset' column now that we have filtered the data for this file
        file_data = file_data.drop(columns=['iset', 'cbi'])

        output_filepath = hkl_file.replace("XDS_ASCII_CBI.HKL", "XDS_ASCII_filtered.HKL")
        with open(output_filepath, 'w') as file:
            # Write the header lines specific to this file
            for line in headers[iset_value]:
                file.write(line)

            # Write the filtered reflection data without iset and cbi columns
            for _, row in file_data.iterrows():
                file.write(f"{int(row['h']):4} {int(row['k']):4} {int(row['l']):4} {row['iobs']:12.4e} {row['sigma']:12.4e} ")
                file.write(f"{row['xd']:8.1f} {row['yd']:8.1f} {row['zd']:8.1f} {row['rlp']:8.4f} {int(row['peak']):4} {int(row['corr']):4} {row['psi']:8.2f}\n")

            # Add the !END_OF_DATA line after all data has been written
            file.write("!END_OF_DATA\n")
            
        print(f"Reduced file saved to {output_filepath}")
        messagebox.showinfo("Success", f"Reduced file saved to {output_filepath}")

def main(hkl_files):
    filtered_data, headers = merge_and_filter_reflections(hkl_files)
    save_filtered_hkl_files(filtered_data, headers, hkl_files)

class HKLReducerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HKL Reducer")

        self.file_list = []

        # Create GUI components
        self.create_widgets()

    def create_widgets(self):
        # Button to browse and select HKL files
        self.browse_button = tk.Button(self.root, text="Browse HKL Files", command=self.browse_files)
        self.browse_button.pack(pady=10)

        # Listbox to display selected HKL files
        self.file_listbox = tk.Listbox(self.root, selectmode=tk.MULTIPLE, width=100)
        self.file_listbox.pack(pady=10)

        # Button to start reduction process
        self.reduce_button = tk.Button(self.root, text="Reduce Files", command=self.reduce_files, state=tk.DISABLED)
        self.reduce_button.pack(pady=10)

    def browse_files(self):
        file_paths = filedialog.askopenfilenames(title="Select XDS_ASCII_CBI.HKL Files", filetypes=[("HKL Files", "*.HKL")])
        if file_paths:
            self.file_list.extend(file_paths)
            for file_path in file_paths:
                self.file_listbox.insert(tk.END, file_path)
            self.reduce_button.config(state=tk.NORMAL)

    def reduce_files(self):
        if not self.file_list:
            messagebox.showwarning("No Files Selected", "Please select HKL files to reduce.")
            return

        main(self.file_list)

if __name__ == "__main__":
    root = tk.Tk()
    app = HKLReducerApp(root)
    root.mainloop()
