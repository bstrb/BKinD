# extract_unit_cell_from_fcf.py

def extract_unit_cell_from_fcf(file_path):
    """
    Extracts unit cell parameters and resolution from a given FCF file.
    
    Args:
    file_path (str): Path to the FCF file.
    
    Returns:
    list: Unit cell parameters as a list of floats.
    float: Resolution of the data.
    """
    unit_cell_params = []
    
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('_cell_length_a'):
                unit_cell_params.append(float(line.split()[1]))
            elif line.startswith('_cell_length_b'):
                unit_cell_params.append(float(line.split()[1]))
            elif line.startswith('_cell_length_c'):
                unit_cell_params.append(float(line.split()[1]))
            elif line.startswith('_cell_angle_alpha'):
                unit_cell_params.append(float(line.split()[1]))
            elif line.startswith('_cell_angle_beta'):
                unit_cell_params.append(float(line.split()[1]))
            elif line.startswith('_cell_angle_gamma'):
                unit_cell_params.append(float(line.split()[1]))
                
    if len(unit_cell_params) != 6:
        print(f"Error: Invalid unit cell parameters extracted: {unit_cell_params}")
        return None, None

    return unit_cell_params