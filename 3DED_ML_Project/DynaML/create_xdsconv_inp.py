# create_xdsconv_inp.py

from tkinter import messagebox

def create_xdsconv_inp(file_path, unit_cell_constants, space_group_number, output_file_type, friedel_var, merge_var):
    if not file_path:
        messagebox.showwarning("No file selected", "Please select an input file.")
        return

    output_file_name = "xdsconv.inp"
    friedel_law = "FALSE" if friedel_var else "TRUE"
    merge = "FALSE" if merge_var else "TRUE"

    try:
        with open(output_file_name, 'w') as file:
            file.write(f"! UNIT_CELL_CONSTANTS= {unit_cell_constants}\n")
            file.write(f"! SPACE_GROUP_NUMBER= {space_group_number}\n\n")
            file.write(f"INPUT_FILE= {file_path}\n")
            file.write(f"OUTPUT_FILE=dynaml.hkl SHELX\n")
            file.write(f"FRIEDEL'S_LAW={friedel_law}\n")
            file.write(f"MERGE={merge}\n")
            file.write("\n! further options see http://xds.mpimf-heidelberg.mpg.de/~kabsch/xds/html_doc/xdsconv_parameters.html\n")
        
        # messagebox.showinfo("Success", f"{output_file_name} created successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while creating the file: {e}")
