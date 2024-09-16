import os

import tkinter as tk
from tkinter import Toplevel, messagebox

from browse import browse_directory
from browse import browse_file

from generate_lst_file import generate_lst_file  # Import from generate_lst_file.py

# Function to create a popup window for .lst file generation
def open_lst_file_popup(lst_file_entry):
    popup = Toplevel()
    popup.title("Generate .lst File")

    tk.Label(popup, text="Enter .lst File Name").grid(row=0, column=0)
    lst_file_name_popup_entry = tk.Entry(popup)
    lst_file_name_popup_entry.grid(row=0, column=1)

    tk.Label(popup, text="List File Directory").grid(row=1, column=0)
    lst_file_directory_entry = tk.Entry(popup)
    lst_file_directory_entry.grid(row=1, column=1)
    tk.Button(popup, text="Browse", command=lambda: browse_directory(lst_file_directory_entry)).grid(row=1, column=2)

    tk.Label(popup, text="Mask File").grid(row=2, column=0)
    mask_file_path_entry = tk.Entry(popup)
    mask_file_path_entry.grid(row=2, column=1)
    tk.Button(popup, text="Browse", command=lambda: browse_file(mask_file_path_entry)).grid(row=2, column=2)

    tk.Label(popup, text="Processed HDF5 File").grid(row=3, column=0)
    processed_h5_file_path_entry = tk.Entry(popup)
    processed_h5_file_path_entry.grid(row=3, column=1)
    tk.Button(popup, text="Browse", command=lambda: browse_file(processed_h5_file_path_entry)).grid(row=3, column=2)

    # Function to handle lst file generation from popup
    def generate_lst_from_popup():
        lst_file_name = lst_file_name_popup_entry.get()
        lst_file_directory = lst_file_directory_entry.get()
        mask_file_path = mask_file_path_entry.get()
        processed_h5_file_path = processed_h5_file_path_entry.get()
    
        try:
            # Generate the .lst file
            generate_lst_file(lst_file_name, lst_file_directory, mask_file_path, processed_h5_file_path)
            lst_file_path = os.path.join(lst_file_directory, lst_file_name + ".lst")
            messagebox.showinfo("Success", f".lst file generated successfully at {lst_file_path}.")
            
            # Automatically update the .lst file path entry in the main window
            lst_file_entry.delete(0, tk.END)
            lst_file_entry.insert(0, lst_file_path)
            popup.destroy()  # Close the popup window
        except Exception as e:
            messagebox.showerror("Error", str(e))

    tk.Button(popup, text="Generate .lst File", command=generate_lst_from_popup).grid(row=4, column=0, columnspan=2, pady=10)