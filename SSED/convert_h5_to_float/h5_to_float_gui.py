import tkinter as tk
from tkinter import filedialog, messagebox
import os
from conv_h5_to_float import convert_hdf5_images_to_floats

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")])
    if file_path:
        entry_file_path.delete(0, tk.END)
        entry_file_path.insert(0, file_path)

def start_conversion():
    input_file = entry_file_path.get()
    if not os.path.isfile(input_file):
        messagebox.showerror("Error", "Please select a valid HDF5 file.")
        return

    try:
        convert_hdf5_images_to_floats(input_file)
        messagebox.showinfo("Success", "Conversion completed successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during conversion: {e}")

# Create the main window
root = tk.Tk()
root.title("HDF5 Image Converter")
root.geometry("400x200")

# Create and place widgets
frame = tk.Frame(root)
frame.pack(pady=20)

label = tk.Label(frame, text="Select HDF5 File:")
label.grid(row=0, column=0, padx=10, pady=10)

entry_file_path = tk.Entry(frame, width=40)
entry_file_path.grid(row=0, column=1, padx=10, pady=10)

button_browse = tk.Button(frame, text="Browse", command=browse_file)
button_browse.grid(row=0, column=2, padx=10, pady=10)

button_convert = tk.Button(root, text="Convert to Float", command=start_conversion)
button_convert.pack(pady=20)

# Start the main loop
root.mainloop()