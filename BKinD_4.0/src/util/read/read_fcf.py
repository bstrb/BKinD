# read_fcf.py

# Third-Party Imports
import pandas as pd

def read_fcf(file_path):
    """
    Reads an .fcf file and extracts the Miller indices, calculated intensities (Fc^2),
    observed intensities (Fo^2), and sigmas.

    Parameters:
    - file_path: str, the path to the .fcf file.

    Returns:
    - miller_indices: list of tuples, each containing the (h, k, l) indices.
    - Fc2: list of floats, each containing the calculated intensity (Fc^2).
    - Fo2: list of floats, each containing the observed intensity (Fo^2).
    - sigmas: list of floats, each containing the sigma value for Fo^2.
    """
    miller_indices = []
    Fc2 = []
    Fo2 = []
    sigmas = []
    header_end = False

    with open(file_path, 'r') as file:
        for line in file:
            if header_end:
                parts = line.split()
                if len(parts) >= 6:
                    h, k, l = map(int, parts[:3])
                    Fc2_value, Fo2_value, sigma_value = map(float, parts[3:6])
                    miller_indices.append((h, k, l))
                    Fc2.append(Fc2_value)
                    Fo2.append(Fo2_value)
                    sigmas.append(sigma_value)
            elif line.startswith('loop_'):
                header_end = True

    # Create a DataFrame from the data
    sample_df = pd.DataFrame({
        'Miller': miller_indices,
        'Fo^2': Fo2,
        'Fc^2': Fc2,
        'Fo^2_sigma': sigmas
    })

    return sample_df