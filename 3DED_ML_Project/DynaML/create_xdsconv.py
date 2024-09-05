# create_xdsconv.py

import os
from tkinter import messagebox

def create_xdsconv(unit_cell_constants, space_group_number, output_folder):
    """
    Create the xdsconv.inp file in the specified output folder.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Define output file path
    output_file_name = "xdsconv.inp"
    output_file_path = os.path.join(output_folder, output_file_name)

    input_file_name = 'XDS_ASCII_filtered.HKL'
    input_file_path = os.path.join(output_folder,input_file_name)
    

    # Define other parameters
    friedel_law = "FALSE"
    merge = "FALSE" 

    try:
        # Create the file in the output folder
        with open(output_file_path, 'w') as file:
            file.write(f"! UNIT_CELL_CONSTANTS= {unit_cell_constants}\n")
            file.write(f"! SPACE_GROUP_NUMBER= {space_group_number}\n\n")
            file.write(f"INPUT_FILE= {input_file_path}\n")
            file.write(f"OUTPUT_FILE= dynaml.hkl SHELX\n")
            file.write(f"FRIEDEL'S_LAW={friedel_law}\n")
            file.write(f"MERGE={merge}\n")
            file.write("\n! Further options see http://xds.mpimf-heidelberg.mpg.de/~kabsch/xds/html_doc/xdsconv_parameters.html\n")
        
        # Notify success
        messagebox.showinfo("Success", f"{output_file_name} created successfully in {output_folder}.")
    except Exception as e:
        # Notify error
        messagebox.showerror("Error", f"An error occurred while creating the file: {e}")
