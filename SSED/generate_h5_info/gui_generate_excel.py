# gui_generate_excel.py

import tkinter as tk
from tkinter import filedialog, messagebox
import os
import pandas as pd
import h5py
from tqdm import tqdm

def check_nPeaks(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            if '/entry/data/nPeaks' in f:
                nPeaks = f['/entry/data/nPeaks'][()]
                count_ge_10 = sum(nPeaks >= 10)
                count_ge_25 = sum(nPeaks >= 25)
                count_ge_50 = sum(nPeaks >= 50)
                return len(nPeaks), count_ge_10, count_ge_25, count_ge_50
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
    return 0, 0, 0, 0

def find_h5_files(directory):
    h5_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.h5'):
                file_path = os.path.join(root, file)
                file_size_gb = os.path.getsize(file_path) / (1024 * 1024 * 1024)
                nPeaks_total, nPeaks_ge_10, nPeaks_ge_25, nPeaks_ge_50 = check_nPeaks(file_path)
                h5_files.append([root, file, file_size_gb, nPeaks_total, nPeaks_ge_10, nPeaks_ge_25, nPeaks_ge_50])
    return h5_files

def create_excel(h5_files, output_file):
    columns = ['Folder', 'File Name', 'Size (GB)', 'Frames', 'nPeaks >=10', 'nPeaks >=25', 'nPeaks >=50']
    df = pd.DataFrame(h5_files, columns=columns)
    df.to_excel(output_file, index=False)

def browse_directory():
    directory = filedialog.askdirectory()
    if directory:
        directory_entry.delete(0, tk.END)
        directory_entry.insert(0, directory)

def browse_output_file():
    output_file = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")])
    if output_file:
        output_file_entry.delete(0, tk.END)
        output_file_entry.insert(0, output_file)

def run_scan():
    directory = directory_entry.get()
    output_file = output_file_entry.get()

    if not os.path.isdir(directory):
        messagebox.showerror("Error", "Please select a valid directory to scan.")
        return

    if not output_file:
        messagebox.showerror("Error", "Please select a valid output file.")
        return

    try:
        h5_files_info = find_h5_files(directory)
        create_excel(h5_files_info, output_file)
        messagebox.showinfo("Success", f"Excel file created successfully at {output_file}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Create the GUI window
root = tk.Tk()
root.title("HDF5 File Scanner and Excel Report Generator")
root.geometry("600x200")

# Directory selection
directory_label = tk.Label(root, text="Directory to Scan:")
directory_label.grid(row=0, column=0, padx=10, pady=10, sticky='e')

directory_entry = tk.Entry(root, width=50)
directory_entry.grid(row=0, column=1, padx=10, pady=10)

browse_directory_button = tk.Button(root, text="Browse", command=browse_directory)
browse_directory_button.grid(row=0, column=2, padx=10, pady=10)

# Output file selection
output_file_label = tk.Label(root, text="Output Excel File:")
output_file_label.grid(row=1, column=0, padx=10, pady=10, sticky='e')

output_file_entry = tk.Entry(root, width=50)
output_file_entry.grid(row=1, column=1, padx=10, pady=10)

browse_output_file_button = tk.Button(root, text="Browse", command=browse_output_file)
browse_output_file_button.grid(row=1, column=2, padx=10, pady=10)

# Run button
run_button = tk.Button(root, text="Run Scan", command=run_scan)
run_button.grid(row=2, column=1, padx=10, pady=20)

# Start the main event loop
root.mainloop()