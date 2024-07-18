# compute_start_completeness.py

# Standard Library Imports
import os

# Third-Party Imports
import pandas as pd

# Utility File Imports
from util.file.find_file import find_file

# Utility Read Imports
from util.read.extract_unit_cell_from_fcf import extract_unit_cell_from_fcf
from util.read.extract_space_group_number_from_cif import extract_space_group_number_from_cif

# Utility Data Imports
from util.data.compute_completeness_from_df_sgn_uc import compute_completeness_from_df_sgn_uc


def compute_start_completeness(output_folder):

    df_path = os.path.join(output_folder, 'sample_df.csv')
    
    df = pd.read_csv(df_path)

    fcf_path = find_file(output_folder, '.fcf')
    
    uc = extract_unit_cell_from_fcf(fcf_path)
    
    sgn = extract_space_group_number_from_cif(output_folder)

    start_completeness = 100 * compute_completeness_from_df_sgn_uc(df, sgn, uc)

    return start_completeness