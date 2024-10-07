# gui_remove_frames.py

import tkinter as tk
from tkinter import filedialog, messagebox, Label, Entry, Button
import os

from remove_frames_below_peak_number import remove_frames_below_peak_number

def select_input_file():
    file_path = filedialog.askopenfilename(filetypes=[("HDF5 files", "*.h5"), ("All files", "*")])
    if file_path:
        input_file_entry.delete(0, tk.END)
        input_file_entry.insert(0, file_path)
        update_output_info()

def update_output_info():
    input_file = input_file_entry.get()
    if input_file:
        input_file_dir = os.path.dirname(input_file)
        threshold = threshold_entry.get()
        if threshold.isdigit() and int(threshold) >= 1:
            output_file = os.path.join(input_file_dir, f"min_{threshold}_peak.h5")
            output_info_label.config(text=f"Output file will be named '{os.path.basename(output_file)}' and placed in the same folder.")
        else:
            output_info_label.config(text="")
    else:
        output_info_label.config(text="")

def run_remove_frames():
    input_file = input_file_entry.get()
    threshold = threshold_entry.get()

    if not input_file or not os.path.isfile(input_file):
        messagebox.showerror("Error", "Please select a valid input file.")
        return

    if not threshold.isdigit() or int(threshold) < 1:
        messagebox.showerror("Error", "Please enter a valid threshold (positive integer).")
        return

    threshold = int(threshold)

    try:
        remove_frames_below_peak_number(input_file, threshold)
        messagebox.showinfo("Success", f"Filtered HDF5 file created with frames having at least {threshold} peaks.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Create the GUI window
root = tk.Tk()
root.title("Remove Frames Below Peak Number")
root.geometry("500x250")

# Input file selection
input_file_label = Label(root, text="Input File:")
input_file_label.grid(row=0, column=0, padx=10, pady=10, sticky='e')

input_file_entry = Entry(root, width=40)
input_file_entry.grid(row=0, column=1, padx=10, pady=10)

select_file_button = Button(root, text="Browse", command=select_input_file)
select_file_button.grid(row=0, column=2, padx=10, pady=10)

# Threshold input
threshold_label = Label(root, text="Threshold (number of peaks):")
threshold_label.grid(row=1, column=0, padx=10, pady=10, sticky='e')

threshold_entry = Entry(root, width=10)
threshold_entry.grid(row=1, column=1, padx=10, pady=10, sticky='w')
threshold_entry.insert(0, "1")  # Default value
threshold_entry.bind("<KeyRelease>", lambda event: update_output_info())

# Output information label
output_info_label = Label(root, text="")
output_info_label.grid(row=2, column=0, columnspan=3, padx=10, pady=10)

# Run button
run_button = Button(root, text="Run", command=run_remove_frames)
run_button.grid(row=3, column=1, padx=10, pady=20)

# Start the main event loop
root.mainloop()