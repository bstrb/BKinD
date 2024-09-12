import tkinter as tk
from tkinter import filedialog, messagebox
from generate_lst_file import generate_lst_file  # Import from generate_lst_file.py
from generate_bash_script import generate_bash_script  # Import from generate_bash_file.py

# Function to generate bash file
def generate_bash():
    # Gather inputs from user
    bash_file_name = bash_file_name_entry.get()
    bash_file_directory = filedialog.askdirectory(title="Select Directory for Bash File")
    process_output_name = process_output_name_entry.get()
    process_file_directory = filedialog.askdirectory(title="Select Process File Directory")
    num_threads = num_threads_entry.get()
    
    geom_file = filedialog.askopenfilename(title="Select .geom File", filetypes=[("Geom Files", "*.geom")])
    lst_file = filedialog.askopenfilename(title="Select .lst File", filetypes=[("List Files", "*.lst")])
    cell_file = filedialog.askopenfilename(title="Select .cell File", filetypes=[("Cell Files", "*.cell")])
    
    try:
        generate_bash_script(bash_file_name, bash_file_directory, process_output_name, process_file_directory, num_threads, geom_file, lst_file, cell_file)
        messagebox.showinfo("Success", "Bash file generated successfully.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Function to generate .lst file first, then generate bash
def generate_lst_then_bash():
    lst_file_name = lst_file_name_entry.get()
    lst_file_directory = filedialog.askdirectory(title="Select Directory for .lst File")
    mask_file_path = filedialog.askopenfilename(title="Select Mask File (.h5)", filetypes=[("HDF5 Files", "*.h5")])
    processed_h5_file_path = filedialog.askopenfilename(title="Select Processed Diffraction File (.h5)", filetypes=[("HDF5 Files", "*.h5")])
    
    try:
        # First generate .lst file
        generate_lst_file(lst_file_name, lst_file_directory, mask_file_path, processed_h5_file_path)
        messagebox.showinfo("Success", ".lst file generated successfully.")
        
        # Now generate the bash file using the .lst
        generate_bash()
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create the main window
root = tk.Tk()
root.title("Generate Bash & .lst File")

# Create labels and entry fields
tk.Label(root, text="Bash File Name").grid(row=0, column=0)
bash_file_name_entry = tk.Entry(root)
bash_file_name_entry.grid(row=0, column=1)

tk.Label(root, text="Process Output Name").grid(row=1, column=0)
process_output_name_entry = tk.Entry(root)
process_output_name_entry.grid(row=1, column=1)

tk.Label(root, text="Number of Threads").grid(row=2, column=0)
num_threads_entry = tk.Entry(root)
num_threads_entry.grid(row=2, column=1)

tk.Label(root, text=".lst File Name").grid(row=3, column=0)
lst_file_name_entry = tk.Entry(root)
lst_file_name_entry.grid(row=3, column=1)

# Create buttons
generate_bash_button = tk.Button(root, text="Generate Bash File", command=generate_bash)
generate_bash_button.grid(row=4, column=0, pady=10)

generate_lst_bash_button = tk.Button(root, text="Generate .lst and Bash File", command=generate_lst_then_bash)
generate_lst_bash_button.grid(row=4, column=1, pady=10)

# Start the GUI event loop
root.mainloop()
