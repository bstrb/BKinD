import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from cap_util import process_files

def browse_file(entry):
    file_path = filedialog.askopenfilename()
    entry.delete(0, tk.END)
    entry.insert(0, file_path)

def on_process():
    file_name_res_orig = entry_orig.get()
    file_name_res = entry_filtered.get()
    if not file_name_res_orig or not file_name_res:
        messagebox.showerror("Input Error", "Please select both input files.")
        return
    results, mean_difference = process_files(file_name_res_orig, file_name_res)
    text_results.delete("1.0", tk.END)
    text_results.insert(tk.END, results)

# Create the main window
root = tk.Tk()
root.title("bkind_cap GUI")

# Create the input file selection frame
frame_inputs = ttk.Frame(root, padding="10")
frame_inputs.grid(row=0, column=0, sticky=(tk.W, tk.E))

# Original file input
label_orig = ttk.Label(frame_inputs, text="Original .res file:")
label_orig.grid(row=0, column=0, sticky=tk.W)
entry_orig = ttk.Entry(frame_inputs, width=50)
entry_orig.grid(row=0, column=1, padx=5)
button_browse_orig = ttk.Button(frame_inputs, text="Browse", command=lambda: browse_file(entry_orig))
button_browse_orig.grid(row=0, column=2)

# Filtered file input
label_filtered = ttk.Label(frame_inputs, text="Filtered .res file:")
label_filtered.grid(row=1, column=0, sticky=tk.W)
entry_filtered = ttk.Entry(frame_inputs, width=50)
entry_filtered.grid(row=1, column=1, padx=5)
button_browse_filtered = ttk.Button(frame_inputs, text="Browse", command=lambda: browse_file(entry_filtered))
button_browse_filtered.grid(row=1, column=2)

# Process button
button_process = ttk.Button(root, text="Process", command=on_process)
button_process.grid(row=1, column=0, pady=10)

# Results text box
text_results = tk.Text(root, width=100, height=20)
text_results.grid(row=2, column=0, pady=10)

# Start the main event loop
root.mainloop()
