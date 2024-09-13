import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel
from generate_lst_file import generate_lst_file  # Import from generate_lst_file.py
from generate_bash_script import generate_bash_script  # Import from generate_bash_file.py
import os

# Function to browse for directories and update the corresponding entry field
def browse_directory(entry_widget):
    directory = filedialog.askdirectory()
    if directory:
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, directory)

# Function to browse for files and update the corresponding entry field
def browse_file(entry_widget):
    file = filedialog.askopenfilename()
    if file:
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, file)

# Function to generate bash file
def generate_bash():
    bash_file_name = bash_file_name_entry.get()
    stream_files_directory = stream_files_directory_entry.get()
    integration_output_name = integration_output_name_entry.get()
    integration_output_directory = integration_output_directory_entry.get()
    num_threads = num_threads_entry.get()
    
    lst_file = lst_file_entry.get()  # Get the path for the .lst file
    geom_file = geom_file_entry.get()
    cell_file = cell_file_entry.get()
    sol_file = cell_file_entry.get()
    
    try:
        generate_bash_script(bash_file_name, stream_files_directory, integration_output_name, integration_output_directory, num_threads, geom_file, lst_file, cell_file, sol_file)
        messagebox.showinfo("Success", "Bash file generated successfully.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Function to create a popup window for .lst file generation
def open_lst_file_popup():
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

# Create the main window
root = tk.Tk()
root.title("Generate Bash & .lst File for fast integration")

# Create labels and entry fields for bash file
tk.Label(root, text="Bash File Name").grid(row=0, column=0)
bash_file_name_entry = tk.Entry(root)
bash_file_name_entry.grid(row=0, column=1)

tk.Label(root, text="Bash and Stream Files Directory").grid(row=1, column=0)
stream_files_directory_entry = tk.Entry(root)
stream_files_directory_entry.grid(row=1, column=1)
tk.Button(root, text="Browse", command=lambda: browse_directory(stream_files_directory_entry)).grid(row=1, column=2)

tk.Label(root, text="Integration Output Name").grid(row=2, column=0)
integration_output_name_entry = tk.Entry(root)
integration_output_name_entry.grid(row=2, column=1)

tk.Label(root, text="Integration Output Directory").grid(row=3, column=0)
integration_output_directory_entry = tk.Entry(root)
integration_output_directory_entry.grid(row=3, column=1)
tk.Button(root, text="Browse", command=lambda: browse_directory(integration_output_directory_entry)).grid(row=3, column=2)

tk.Label(root, text="Number of Threads").grid(row=4, column=0)
num_threads_entry = tk.Entry(root)
num_threads_entry.grid(row=4, column=1)

tk.Label(root, text=".lst File").grid(row=5, column=0)
lst_file_entry = tk.Entry(root)
lst_file_entry.grid(row=5, column=1)
tk.Button(root, text="Browse", command=lambda: browse_file(lst_file_entry)).grid(row=5, column=2)

tk.Label(root, text=".geom File").grid(row=6, column=0)
geom_file_entry = tk.Entry(root)
geom_file_entry.grid(row=6, column=1)
tk.Button(root, text="Browse", command=lambda: browse_file(geom_file_entry)).grid(row=6, column=2)

tk.Label(root, text=".cell File").grid(row=7, column=0)
cell_file_entry = tk.Entry(root)
cell_file_entry.grid(row=7, column=1)
tk.Button(root, text="Browse", command=lambda: browse_file(cell_file_entry)).grid(row=7, column=2)

tk.Label(root, text=".sol File").grid(row=8, column=0)
sol_file_entry = tk.Entry(root)
sol_file_entry.grid(row=8, column=1)
tk.Button(root, text="Browse", command=lambda: browse_file(sol_file_entry)).grid(row=8, column=2)

# Create buttons for generating .lst and bash files
generate_lst_button = tk.Button(root, text="Generate .lst File", command=open_lst_file_popup)
generate_lst_button.grid(row=9, column=0, pady=10)

generate_bash_button = tk.Button(root, text="Generate Bash File", command=generate_bash)
generate_bash_button.grid(row=9, column=1, pady=10)

# Start the GUI event loop
root.mainloop()
