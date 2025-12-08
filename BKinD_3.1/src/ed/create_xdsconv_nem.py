# create_xdsconv.py

# Standard Library Imports
import os

def create_xdsconv_nem(target_directory):
    """
    Creates an xdsconv.inp file with specific settings.

    Parameters:
    - source_directory: str, the directory containing the '.ins' file.
    - target_directory: str, the directory where the modified '.ins' file and 'xdsconv.inp' will be saved.

    Returns:
    - None, but creates file in the target directory.
    """
    # xdsconv_file_path = os.path.join(target_directory, 'xdsconv.inp')
    xdsconv_file_path = os.path.join(target_directory, "XDSCONV.INP")

    # Create the xdsconv.inp file
    with open(xdsconv_file_path, 'w') as xdsconv_file:
        xdsconv_file.write("INPUT_FILE=XDS_ASCII_NEM.HKL\n")
        xdsconv_file.write("OUTPUT_FILE=bkind.hkl SHELX ! or CCP4_I or CCP4_F or SHELX or CNS\n")
        xdsconv_file.write("FRIEDEL'S_LAW=FALSE ! store anomalous signal in output file even if weak\n")
        xdsconv_file.write("MERGE=FALSE\n")
