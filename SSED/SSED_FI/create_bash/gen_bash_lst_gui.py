import os
import sys
import tkinter as tk
from tkinter import messagebox

# Ensure the script directory is in the Python path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Go one directory up
sys.path.append(project_root)

from create_bash.generate_bash_script import generate_bash_script
from create_bash.open_lst_file_popup import open_lst_file_popup
from create_bash.browse import browse_directory, browse_file

# Flag to check if the script is running as standalone
is_standalone = __name__ == "__main__"

# Function to create the "Generate Bash & .lst File" GUI
def open_generate_bash_gui(parent=None):
    # If no parent is given, create a new root window
    if parent:
        window = tk.Toplevel(parent)
    else:
        window = tk.Tk()  # Create the root window here only if parent is None
        print("Creating a root window.")

    window.title("Generate Bash & .lst File for Fast Integration")

    # Handling window close event to destroy the window correctly
    def on_close():
        print("Closing window.")
        # Use quit() and destroy() to ensure the Tkinter event loop is stopped
        window.destroy()
        
        if is_standalone:  # Only force exit if running as a standalone script
            print("Root window closed.")
            os._exit(0)  # Forcefully exit the application to handle hanging processes

    window.protocol("WM_DELETE_WINDOW", on_close)

    # Create labels and entry fields for bash file
    tk.Label(window, text="Bash File Name").grid(row=0, column=0)
    bash_file_name_entry = tk.Entry(window)
    bash_file_name_entry.grid(row=0, column=1)

    tk.Label(window, text="Bash and Stream Files Directory").grid(row=1, column=0)
    stream_files_directory_entry = tk.Entry(window)
    stream_files_directory_entry.grid(row=1, column=1)
    tk.Button(window, text="Browse", command=lambda: browse_directory(stream_files_directory_entry)).grid(row=1, column=2)

    tk.Label(window, text="Number of Threads").grid(row=4, column=0)
    num_threads_entry = tk.Entry(window)
    num_threads_entry.grid(row=4, column=1)

    tk.Label(window, text=".lst File").grid(row=5, column=0)
    lst_file_entry = tk.Entry(window)
    lst_file_entry.grid(row=5, column=1)
    tk.Button(window, text="Browse", command=lambda: browse_file(lst_file_entry)).grid(row=5, column=2)

    tk.Label(window, text=".geom File").grid(row=6, column=0)
    geom_file_entry = tk.Entry(window)
    geom_file_entry.grid(row=6, column=1)
    tk.Button(window, text="Browse", command=lambda: browse_file(geom_file_entry)).grid(row=6, column=2)

    tk.Label(window, text=".cell File").grid(row=7, column=0)
    cell_file_entry = tk.Entry(window)
    cell_file_entry.grid(row=7, column=1)
    tk.Button(window, text="Browse", command=lambda: browse_file(cell_file_entry)).grid(row=7, column=2)

    tk.Label(window, text=".sol File").grid(row=8, column=0)
    sol_file_entry = tk.Entry(window)
    sol_file_entry.grid(row=8, column=1)
    tk.Button(window, text="Browse", command=lambda: browse_file(sol_file_entry)).grid(row=8, column=2)

    # Function to generate bash file
    def generate_bash():
        bash_file_name = bash_file_name_entry.get()
        stream_files_directory = stream_files_directory_entry.get()
        num_threads = num_threads_entry.get()
        lst_file = lst_file_entry.get()
        geom_file = geom_file_entry.get()
        cell_file = cell_file_entry.get()
        sol_file = sol_file_entry.get()

        try:
            generate_bash_script(bash_file_name, stream_files_directory, num_threads, geom_file, lst_file, cell_file, sol_file)
            messagebox.showinfo("Success", "Bash file generated successfully.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # Create buttons for generating .lst and bash files
    generate_lst_button = tk.Button(window, text="Generate .lst File", command=lambda: open_lst_file_popup(lst_file_entry))
    generate_lst_button.grid(row=9, column=0, pady=10)

    generate_bash_button = tk.Button(window, text="Generate Bash File", command=generate_bash)
    generate_bash_button.grid(row=9, column=1, pady=10)
    
# Allows the script to be run independently
if is_standalone:
    print("Running as a standalone script.")
    root = tk.Tk()  # Create the root window
    root.withdraw()  # Hide the root window
    open_generate_bash_gui(root)  # Pass the root window as the parent
    root.mainloop()
    print("Exited main loop.")
