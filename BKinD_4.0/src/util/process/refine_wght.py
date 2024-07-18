# refine_wght_new.py

# Standard library imports
import os
import shutil

# Custom module imports
from util.process.run_process import run_process

def format_float(value):
    """ Helper to format the float to three significant digits for comparison. """
    return float(f"{float(value):.3g}")

def should_replace(line1, line2):
    """ Determine if the first line should be replaced with the second based on their numeric values. """
    values1 = line1.split()[1:]  # Split and ignore the 'WGHT'
    values2 = line2.split()[1:]  # Split and ignore the 'WGHT'
    
    # If the first line has only one value, replace unconditionally
    if len(values1) == 1:
        values1.append('0')
    
    # Compare each corresponding pair of values
    for v1, v2 in zip(values1, values2):
        if abs(format_float(v1) - format_float(v2)) > 0.001:  # More than 0.001 difference
            return True
    return False

def process_res_file(folder_path):
    """
    Processes the .res file to exchange lines starting with 'WGHT' under certain conditions.

    Args:
    folder_path (str): The path to the directory containing the .res file.
    """
    
    file_path = None

    # Locate the .res file
    for filename in os.listdir(folder_path):
        if filename.endswith('.res'):
            file_path = os.path.join(folder_path, filename)
            break

    if not file_path or not os.path.exists(file_path):
        print("No .res file found in the directory.")
        return
    
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find all lines starting with 'WGHT'
    wght_lines = [line for line in lines if line.strip().startswith('WGHT')]
    if len(wght_lines) < 2:
        return
    
    # Check if lines should be replaced
    if should_replace(wght_lines[0], wght_lines[1]):
        # Find original indices
        index1 = lines.index(wght_lines[0])
        index2 = lines.index(wght_lines[1])
        
        # Replace the first line with the second one
        lines[index1] = lines[index2]

        # Modify the second line if it exists and looks like a path
        if len(lines) > 1:
            parts = lines[1].split('/')
            if len(parts) > 1:
                lines[1] = '    ' + parts[-1].strip() + '\n'  # Replace the whole line with just the last part (filename)
        
        # Write back to the file
        with open(file_path, 'w') as file:
            file.writelines(lines)

        # Rename the .res file to .ins
        new_ins_file_path = os.path.join(folder_path, os.path.basename(file_path).replace('.res', '.ins'))
        shutil.move(file_path, new_ins_file_path)

        run_process(["shelxl"], folder_path, input_file='.ins', suppress_output=True)
        return True
    else:
        return False

def refine_wght(folder_path):
    max_attempts = 10
    attempts = 0
    
    while attempts < max_attempts:
        attempts += 1
        result = process_res_file(folder_path)
        # print(f"Attempt {attempts}: Function returned {result}")
        if not result:
            return False
    return True
