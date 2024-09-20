import os
import tkinter as tk
from tkinter import Toplevel, messagebox, filedialog


from create_bash.browse import browse_directory  # Ensure these are correctly defined
from create_bash.generate_lst_file import generate_lst_file  # Import the lst generation function

# Global variable to accumulate selected HDF5 files
accumulated_h5_files = []

# Function to browse multiple HDF5 files and add them to the list
def browse_and_add_files(file_listbox):
    files = filedialog.askopenfilenames(
        title="Select HDF5 Files",
        filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
    )
    if files:
        accumulated_h5_files.extend(files)  # Add files to the global list
        update_file_listbox(file_listbox)  # Update the listbox UI to reflect added files

# Function to browse a folder containing HDF5 files and add them to the list
def browse_and_add_folder(file_listbox):
    folder = filedialog.askdirectory(title="Select Folder Containing HDF5 Files")
    if folder:
        h5_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.h5')]
        accumulated_h5_files.extend(h5_files)  # Add folder files to the global list
        update_file_listbox(file_listbox)  # Update the listbox UI to reflect added files

# Function to update the file listbox with accumulated files
def update_file_listbox(file_listbox):
    file_listbox.delete(0, tk.END)  # Clear the listbox
    for file in accumulated_h5_files:
        file_listbox.insert(tk.END, file)  # Add each file to the listbox

# Function to remove selected file from the listbox and accumulated list
def remove_selected_file(file_listbox):
    selected_index = file_listbox.curselection()
    if selected_index:
        file_to_remove = accumulated_h5_files.pop(selected_index[0])  # Remove from the accumulated list
        update_file_listbox(file_listbox)  # Update the UI
        messagebox.showinfo("File Removed", f"Removed file: {file_to_remove}")

# Function to create a popup window for .lst file generation
def open_lst_file_popup(lst_file_entry):
    global accumulated_h5_files
    accumulated_h5_files = []  # Clear the accumulated files at the start of popup
    
    popup = Toplevel()
    popup.title("Generate .lst File")

    # Fields for .lst file name and directory
    tk.Label(popup, text="Enter .lst File Name").grid(row=0, column=0)
    lst_file_name_popup_entry = tk.Entry(popup)
    lst_file_name_popup_entry.grid(row=0, column=1)

    tk.Label(popup, text="List File Directory").grid(row=1, column=0)
    lst_file_directory_entry = tk.Entry(popup)
    lst_file_directory_entry.grid(row=1, column=1)
    tk.Button(popup, text="Browse", command=lambda: browse_directory(lst_file_directory_entry)).grid(row=1, column=2)

    # File Listbox to display accumulated HDF5 files
    tk.Label(popup, text="Selected HDF5 Files:").grid(row=2, column=0, columnspan=2)
    file_listbox = tk.Listbox(popup, height=8, width=50)
    file_listbox.grid(row=3, column=0, columnspan=2)
    
    # Buttons to browse files and folders
    tk.Button(popup, text="Browse Files", command=lambda: browse_and_add_files(file_listbox)).grid(row=4, column=0)
    tk.Button(popup, text="Browse Folder", command=lambda: browse_and_add_folder(file_listbox)).grid(row=4, column=1)

    # Button to remove selected file from the list
    tk.Button(popup, text="Remove Selected File", command=lambda: remove_selected_file(file_listbox)).grid(row=5, column=0, columnspan=2)

    # Function to handle lst file generation from popup
    def generate_lst_from_popup():
        lst_file_name = lst_file_name_popup_entry.get().strip()
        lst_file_directory = lst_file_directory_entry.get().strip()

        # Validate inputs
        if not lst_file_name or not lst_file_directory or not accumulated_h5_files:
            messagebox.showerror("Input Error", "All fields must be filled out and at least one HDF5 file must be added.")
            return

        if not os.path.isdir(lst_file_directory):
            messagebox.showerror("Directory Error", "The directory path is not valid.")
            return

        try:
            # Generate the .lst file with accumulated HDF5 files
            generate_lst_file(lst_file_name, lst_file_directory, accumulated_h5_files)
            lst_file_path = os.path.join(lst_file_directory, lst_file_name + ".lst")
            messagebox.showinfo("Success", f".lst file generated successfully at {lst_file_path}.")
            
            # Automatically update the .lst file path entry in the main window
            lst_file_entry.delete(0, tk.END)
            lst_file_entry.insert(0, lst_file_path)
            popup.destroy()  # Close the popup window
        except Exception as e:
            messagebox.showerror("Error", str(e))

    tk.Button(popup, text="Generate .lst File", command=generate_lst_from_popup).grid(row=6, column=0, columnspan=2, pady=10)

# Example usage:
# root = tk.Tk()
# lst_file_entry = tk.Entry(root)
# open_lst_file_popup(lst_file_entry)
# root.mainloop()
