import tkinter as tk
from tkinter import filedialog, messagebox, Button, Label, Listbox, Scrollbar
import os
from merge_h5_files import merge_h5_files

def browse_folder():
    folder_path = filedialog.askdirectory()
    if folder_path:
        h5_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.h5')]
        for file in h5_files:
            if file not in selected_files_list.get(0, tk.END):
                selected_files_list.insert(tk.END, file)

def browse_directory():
    directory_path = filedialog.askdirectory()
    if directory_path:
        h5_files = []
        for root, _, files in os.walk(directory_path):
            for f in files:
                if f.endswith('.h5'):
                    h5_files.append(os.path.join(root, f))
        for file in h5_files:
            if file not in selected_files_list.get(0, tk.END):
                selected_files_list.insert(tk.END, file)

def browse_files():
    files = filedialog.askopenfilenames(filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")])
    for file in files:
        if file not in selected_files_list.get(0, tk.END):
            selected_files_list.insert(tk.END, file)

def remove_selected():
    selected_items = selected_files_list.curselection()
    for index in reversed(selected_items):
        selected_files_list.delete(index)

def merge_files():
    files = selected_files_list.get(0, tk.END)
    if not files:
        messagebox.showerror("Error", "No HDF5 files selected for merging.")
        return

    output_file = filedialog.asksaveasfilename(defaultextension=".h5", filetypes=[("HDF5 files", "*.h5")])
    if not output_file:
        return

    try:
        # Merging the files without threading or GUI progress bar, progress will show in the terminal via tqdm
        merge_h5_files(files, output_file)
        messagebox.showinfo("Success", f"Merged HDF5 file created: {output_file}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Create the GUI window
root = tk.Tk()
root.title("Merge HDF5 Files")
root.geometry("600x400")

# Browse folder button (only selects .h5 files from the selected folder)
browse_folder_button = Button(root, text="Browse Folder (only .h5 in folder)", command=browse_folder)
browse_folder_button.grid(row=0, column=0, padx=10, pady=10)

# Browse directory button (walks through all subdirectories and selects .h5 files)
browse_directory_button = Button(root, text="Browse Directory (including subfolders)", command=browse_directory)
browse_directory_button.grid(row=0, column=1, padx=10, pady=10)

# Browse files button
browse_files_button = Button(root, text="Browse Files", command=browse_files)
browse_files_button.grid(row=0, column=2, padx=10, pady=10)

# Selected files listbox
selected_files_label = Label(root, text="Selected HDF5 Files:")
selected_files_label.grid(row=1, column=0, columnspan=3, padx=10, pady=5)

selected_files_list = Listbox(root, selectmode=tk.MULTIPLE, width=80, height=15)
selected_files_list.grid(row=2, column=0, columnspan=3, padx=10, pady=5)

# Scrollbar for listbox
scrollbar = Scrollbar(root)
scrollbar.grid(row=2, column=3, sticky='ns')
selected_files_list.config(yscrollcommand=scrollbar.set)
scrollbar.config(command=selected_files_list.yview)

# Remove selected files button
remove_selected_button = Button(root, text="Remove Selected", command=remove_selected)
remove_selected_button.grid(row=3, column=0, columnspan=3, padx=10, pady=10)

# Merge files button
merge_button = Button(root, text="Merge Files", command=merge_files)
merge_button.grid(row=4, column=0, columnspan=3, padx=10, pady=20)

# Start the main event loop
root.mainloop()
