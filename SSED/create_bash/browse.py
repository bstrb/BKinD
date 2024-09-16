import tkinter as tk
from tkinter import filedialog

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