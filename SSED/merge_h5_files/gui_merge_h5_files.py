
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, Button, Label, Listbox, Scrollbar, ttk
import os

from merge_h5_files_v2 import merge_h5_files

def browse_folder():
    folder_path = filedialog.askdirectory()
    if folder_path:
        h5_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.h5')]
        add_files_to_listbox(h5_files)

def browse_directory():
    directory_path = filedialog.askdirectory()
    if directory_path:
        h5_files = []
        for root, _, files in os.walk(directory_path):
            for f in files:
                if f.endswith('.h5'):
                    h5_files.append(os.path.join(root, f))
        add_files_to_listbox(h5_files)

def browse_files():
    files = filedialog.askopenfilenames(filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")])
    add_files_to_listbox(files)

def add_files_to_listbox(files):
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

    def run_merge():
        try:
            merge_h5_files(files, output_file)
            messagebox.showinfo("Success", f"Merged HDF5 file created: {output_file}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    threading.Thread(target=run_merge).start()

root = tk.Tk()
root.title("Merge HDF5 Files")
root.geometry("600x500")

browse_folder_button = Button(root, text="Browse Folder (only .h5 in folder)", command=browse_folder)
browse_folder_button.grid(row=0, column=0, padx=10, pady=10)

browse_directory_button = Button(root, text="Browse Directory (including subfolders)", command=browse_directory)
browse_directory_button.grid(row=0, column=1, padx=10, pady=10)

browse_files_button = Button(root, text="Browse Files", command=browse_files)
browse_files_button.grid(row=0, column=2, padx=10, pady=10)

selected_files_label = Label(root, text="Selected HDF5 Files:")
selected_files_label.grid(row=1, column=0, columnspan=3, padx=10, pady=5)

selected_files_list = Listbox(root, selectmode=tk.MULTIPLE, width=80, height=15)
selected_files_list.grid(row=2, column=0, columnspan=3, padx=10, pady=5)

scrollbar = Scrollbar(root)
scrollbar.grid(row=2, column=3, sticky='ns')
selected_files_list.config(yscrollcommand=scrollbar.set)
scrollbar.config(command=selected_files_list.yview)

remove_selected_button = Button(root, text="Remove Selected", command=remove_selected)
remove_selected_button.grid(row=3, column=0, columnspan=3, padx=10, pady=10)

merge_button = Button(root, text="Merge Files", command=merge_files)
merge_button.grid(row=4, column=0, columnspan=3, padx=10, pady=20)

progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
progress_bar.grid(row=5, column=0, columnspan=3, padx=10, pady=10)

root.mainloop()
