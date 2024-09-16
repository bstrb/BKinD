import tkinter as tk
from tkinter import messagebox
from generate_bash_script import generate_bash_script
from open_lst_file_popup import open_lst_file_popup

from browse import browse_directory
from browse import browse_file


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
generate_lst_button = tk.Button(root, text="Generate .lst File", command=lambda: open_lst_file_popup(lst_file_entry))
generate_lst_button.grid(row=9, column=0, pady=10)

generate_bash_button = tk.Button(root, text="Generate Bash File", command=generate_bash)
generate_bash_button.grid(row=9, column=1, pady=10)

# Start the GUI event loop
root.mainloop()
