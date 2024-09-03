# extract_data_from_file.py

import re
from tkinter import messagebox

def extract_data_from_file(file_path):
    with open(file_path, 'r') as file:
        header_lines = []
        for line in file:
            if line.startswith('!END_OF_HEADER'):
                break
            header_lines.append(line)

        header_content = ''.join(header_lines)

        unit_cell_match = re.search(r'!UNIT_CELL_CONSTANTS=\s*([\d\.\s]+)', header_content)
        space_group_match = re.search(r'!SPACE_GROUP_NUMBER=\s*(\d+)', header_content)

        if unit_cell_match:
            unit_cell_constants = unit_cell_match.group(1).strip()
        else:
            messagebox.showerror("Error", "UNIT_CELL_CONSTANTS not found in the file.")
            return None, None

        if space_group_match:
            space_group_number = space_group_match.group(1).strip()
        else:
            messagebox.showerror("Error", "SPACE_GROUP_NUMBER not found in the file.")
            return None, None

        # messagebox.showinfo("Data Extracted", f"UNIT_CELL_CONSTANTS: {unit_cell_constants}\nSPACE_GROUP_NUMBER: {space_group_number}")
        return unit_cell_constants, space_group_number
