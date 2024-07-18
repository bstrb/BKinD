# prepare.py

# Standard Library Imports
import os

# Data Processing Imports
from data_processing.process_sample_data import process_sample_data

# ED Imports
from ed.create_xds_ascii_nem import create_xds_ascii_nem
from ed.create_xdsconv_nem import create_xdsconv_nem

# Utility File Imports
from util.file.create_subfolder import create_subfolder
from util.file.manage_files import manage_files
from util.file.modify_ins_file import modify_ins_file
from util.file.remove_macos_artifacts import remove_macos_artifacts

# Utility Process Imports
from util.process.run_process import run_process

from util.read.extract_space_group_symbol_from_cif import extract_space_group_symbol_from_cif

def prepare(shelx_directory, output_dir, crystal_name, target_completeness, xds_directory=None, xray=False, update_progress=None):
    """
    Prepares files and folders for processing based on the specified parameters.
    
    Parameters:
    - shelx_directory (str): The directory containing SHELX files.
    - output_dir (str): The directory where the output subfolder will be created.
    - crystal_name (str): The name of the crystal.
    - target_completeness (int or float): The target ASU percentage.
    - xds_directory (str, optional): The directory containing XDS files. Required if xray is False.
    - xray (bool): If True, process as X-ray data.
    - update_progress (function, optional): Function to update progress during preparation.
    """
    output_folder = create_subfolder(output_dir, crystal_name, target_completeness, xray)
    
    remove_macos_artifacts(shelx_directory)
    
    steps = [
        "Initial Setup",
        "Running SHELXL",
        "Running SHELXT",
        "Processing Sample Data"
    ]
    
    total_steps = len(steps)
    step_progress = 100 / total_steps

    for i, step in enumerate(steps):
        if step == "Initial Setup":
            manage_files('copy', shelx_directory, output_folder, extension='.ins', new_filename='bkind.ins')
            modify_ins_file(os.path.join(output_folder, 'bkind.ins'))

            if xray:
                manage_files('copy', shelx_directory, output_folder, extension='.hkl', new_filename='bkind.hkl')
            else:
                manage_files('copy', xds_directory, output_folder, filename='INTEGRATE.HKL')
                create_xds_ascii_nem(xds_directory, output_folder)
                create_xdsconv_nem(output_folder)
                run_process(["xdsconv"], output_folder, suppress_output=True)

        elif step == "Running SHELXL":
            run_process(["shelxl"], output_folder, input_file='.ins', suppress_output=True)

        elif step == "Running SHELXT":
            sgs = extract_space_group_symbol_from_cif(output_folder)
            run_process(["shelxt"], output_folder, input_file='.ins', suppress_output=True, additional_command=f'-s{sgs}')

        elif step == "Processing Sample Data":
            process_sample_data(output_folder, xray)

        if update_progress:
            update_progress((i + 1) * step_progress)
