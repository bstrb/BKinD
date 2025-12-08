# create_xdsconv.py
# %%

# Standard Library Imports
import os

def create_xdsconv(target_directory):
    """
    Creates an xdsconv.inp file with specific settings in the target directory.

    Parameters:
    - target_directory: str, the directory where the xdsconv.inp file will be saved.

    Returns:
    - The path to the created XDSCONV.INP file.
    """
    xdsconv_file_path = os.path.join(target_directory, "XDSCONV.INP")
    
    with open(xdsconv_file_path, 'w') as xdsconv_file:
        xdsconv_file.write("INPUT_FILE=XDS_ASCII.HKL\n")
        xdsconv_file.write(f"OUTPUT_FILE={os.path.basename(target_directory)}.hkl SHELX ! or CCP4_I or CCP4_F or SHELX or CNS\n")
        xdsconv_file.write("FRIEDEL'S_LAW=FALSE ! store anomalous signal in output file even if weak\n")
        xdsconv_file.write("MERGE=FALSE\n")

# # Example usage
# target_directory = "/Users/xiaodong/Downloads/3DED-DATA/FEACAC/FEACACm"
# xdsconv_file_path = create_xdsconv(target_directory)
# # %%
