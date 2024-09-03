# create_xdsconv_inp.py

import tkinter as tk
from tkinter import filedialog, messagebox
import re

class XDSConvApp:
    def __init__(self, root):
        self.root = root
        self.root.title("XDSConv Input File Creator")

        self.file_path = ""
        self.unit_cell_constants = ""
        self.space_group_number = ""

        self.create_widgets()

    def create_widgets(self):
        # Button to browse and select the input file
        self.browse_button = tk.Button(self.root, text="Browse Input File", command=self.browse_file)
        self.browse_button.pack(pady=10)

        # Label for the output file type
        self.output_type_label = tk.Label(self.root, text="Select Output File Type:")
        self.output_type_label.pack(pady=5)

        # Dropdown menu for output file types
        self.output_file_type = tk.StringVar(self.root)
        self.output_file_type.set("SHELX")
        self.output_type_menu = tk.OptionMenu(self.root, self.output_file_type, "SHELX", "CCP4_I", "CCP4_F", "CNS")
        self.output_type_menu.pack(pady=5)

        # Checkbox for Friedel's law
        self.friedel_var = tk.BooleanVar()
        self.friedel_check = tk.Checkbutton(self.root, text="FRIEDEL'S_LAW=FALSE", variable=self.friedel_var)
        self.friedel_check.pack(pady=5)

        # Checkbox for Merge
        self.merge_var = tk.BooleanVar()
        self.merge_check = tk.Checkbutton(self.root, text="MERGE=FALSE", variable=self.merge_var)
        self.merge_check.pack(pady=5)

        # Button to create the xdsconv.inp file
        self.create_button = tk.Button(self.root, text="Create xdsconv.inp", command=self.create_xdsconv_inp)
        self.create_button.pack(pady=10)

    def browse_file(self):
        # Open file dialog to select an input file
        self.file_path = filedialog.askopenfilename(filetypes=[("HKL Files", "*.HKL"), ("All Files", "*.*")], title="Select an Input File")
        
        if not self.file_path:
            messagebox.showwarning("No file selected", "Please select an input file.")
            return
        
        # Extract UNIT_CELL_CONSTANTS and SPACE_GROUP_NUMBER from the selected file
        self.extract_data_from_file()

    def extract_data_from_file(self):
        with open(self.file_path, 'r') as file:
            header_lines = []
            for line in file:
                if line.startswith('!END_OF_HEADER'):
                    break
                header_lines.append(line)

            header_content = ''.join(header_lines)

            unit_cell_match = re.search(r'!UNIT_CELL_CONSTANTS=\s*([\d\.\s]+)', header_content)
            space_group_match = re.search(r'!SPACE_GROUP_NUMBER=\s*(\d+)', header_content)

            if unit_cell_match:
                self.unit_cell_constants = unit_cell_match.group(1).strip()
            else:
                messagebox.showerror("Error", "UNIT_CELL_CONSTANTS not found in the file.")
                return

            if space_group_match:
                self.space_group_number = space_group_match.group(1).strip()
            else:
                messagebox.showerror("Error", "SPACE_GROUP_NUMBER not found in the file.")
                return

            messagebox.showinfo("Data Extracted", f"UNIT_CELL_CONSTANTS: {self.unit_cell_constants}\nSPACE_GROUP_NUMBER: {self.space_group_number}")

    def create_xdsconv_inp(self):
        if not self.file_path:
            messagebox.showwarning("No file selected", "Please select an input file.")
            return

        output_file_name = "xdsconv.inp"
        output_file_type = self.output_file_type.get()
        friedel_law = "FALSE" if self.friedel_var.get() else "TRUE"
        merge = "FALSE" if self.merge_var.get() else "TRUE"

        try:
            with open(output_file_name, 'w') as file:
                file.write(f"! UNIT_CELL_CONSTANTS= {self.unit_cell_constants}\n")
                file.write(f"! SPACE_GROUP_NUMBER= {self.space_group_number}\n\n")
                file.write(f"INPUT_FILE= {self.file_path}\n")
                file.write(f"OUTPUT_FILE=dynaml.hkl {output_file_type}\n")
                file.write(f"FRIEDEL'S_LAW={friedel_law}\n")
                file.write(f"MERGE={merge}\n")
                file.write("\n! further options see http://xds.mpimf-heidelberg.mpg.de/~kabsch/xds/html_doc/xdsconv_parameters.html\n")
            
            messagebox.showinfo("Success", f"{output_file_name} created successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while creating the file: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = XDSConvApp(root)
    root.mainloop()
