# calculate_asu_and_resolutions.py

# Third-Party Imports
import pandas as pd

# CCTBX Imports
from cctbx import crystal, miller
from cctbx.uctbx import unit_cell
from cctbx.sgtbx import space_group_info
from cctbx.array_family import flex


def calculate_asu_and_resolutions(file_path, unit_cell_params, space_group_number):
    """
    Calculate the ASU and resolution for each data point in the FCF file using CCTBX.

    Args:
    file_path (str): Path to the FCF file containing reflection data.
    unit_cell_params (list): Unit cell parameters as a list of floats.
    space_group_number (int): Space group number.

    Returns:
    pd.DataFrame: DataFrame containing grouped Miller indices, ASU indices, and their corresponding resolutions.
    """
    reflections = []
    
    with open(file_path, 'r') as file:
        data_started = False
        for line in file:
            if line.startswith('loop_'):
                data_started = True
                continue
            if data_started and not line.startswith('_'):
                parts = line.split()
                if len(parts) >= 7:  # Ensure the line has the expected number of columns
                    hkl = tuple(map(int, parts[:3]))
                    reflections.append(hkl)
    
    # Create a DataFrame
    df = pd.DataFrame(reflections, columns=['h', 'k', 'l'])
    
    # Convert Miller indices to list of tuples
    miller_indices = list(zip(df['h'], df['k'], df['l']))
    
    # Define the crystal symmetry
    try:
        crystal_symmetry = crystal.symmetry(
            unit_cell=unit_cell(unit_cell_params),
            space_group_info=space_group_info(space_group_number)
        )
    except Exception as e:
        print(f"Error in creating crystal symmetry: {e}")
        return None
    
    # Create a Miller set from the list of tuples
    try:
        miller_set = miller.set(
            crystal_symmetry=crystal_symmetry,
            indices=flex.miller_index(miller_indices),
            anomalous_flag=False
        )
    except Exception as e:
        print(f"Error in creating Miller set: {e}")
        return None

    # Calculate ASU
    try:
        asu_miller_array = miller_set.map_to_asu()
        asu_indices = asu_miller_array.indices()
        df['asu'] = [(h, k, l) for h, k, l in asu_indices]
    except Exception as e:
        print(f"Error in calculating ASU: {e}")
        return None
    
    # Calculate resolutions
    try:
        d_spacings = list(miller_set.d_spacings().data())
        df['Resolution'] = d_spacings
    except Exception as e:
        print(f"Error in calculating d-spacings: {e}")
        return None

    # Group Miller indices and ASU indices
    df['Miller'] = df.apply(lambda row: (row['h'], row['k'], row['l']), axis=1)
    df = df[['Miller', 'asu', 'Resolution']]

    return df