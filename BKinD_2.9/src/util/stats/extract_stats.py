# extract_stats.py

# Standard library imports
import os
import shutil

# Third-party imports
import pandas as pd

def extract_stats(wdir):
    refinement = pd.DataFrame()
    data_dict = {
        'R1': [],
        'Rint': [],
        'FVAR': [],
        'NPD': [],
        'highest_diff_peak': [],  # List for highest difference peaks
        'deepest_hole': [],       # List for deepest holes
        'one_sigma_level': [] 
    }

    # Iterate through files in the working directory
    for file_name in os.listdir(wdir):
        file_path = os.path.join(wdir, file_name)
        if file_name.endswith(".lst"):
            process_lst_file(file_path, data_dict)
        elif file_name.endswith(".res"):
            process_res_file(file_path, data_dict)

    # Convert numeric values and populate DataFrame
    for key, values in data_dict.items():
        refinement[key] = pd.to_numeric(values, errors='coerce')  # Coerce errors to NaN for numeric fields

    save_refinement_stats(refinement, wdir)

def process_lst_file(fp, data_dict):
    with open(fp, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "R(int)" in line:
                data_dict['Rint'].append(line.split()[2])
            if "atoms NPD" in line:
                try:
                    npd_count = int(line.split("atoms NPD")[0].split()[-1])
                    data_dict['NPD'].append(npd_count)
                except ValueError:
                    data_dict['NPD'].append(0)

def process_res_file(fp, data_dict):
    with open(fp, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "REM R1 =" in line:
                # Ensure we catch the correct R1 value by accurately splitting the string
                data_dict['R1'].append(line.split("=")[1].split()[0])
            if line.startswith('FVAR'):
                data_dict['FVAR'].append(round(float(line.split()[1]), 1))
            if "Highest difference peak" in line:
                try:
                    parts = line.split(',')
                    highest_diff_peak = float(parts[0].split()[-1])
                    deepest_hole = float(parts[1].split()[-1])
                    one_sigma_level = float(parts[2].split()[-1])
                    data_dict['highest_diff_peak'].append(highest_diff_peak)
                    data_dict['deepest_hole'].append(deepest_hole)
                    data_dict['one_sigma_level'].append(one_sigma_level)
                except ValueError:
                    data_dict['highest_diff_peak'].append(pd.NA)
                    data_dict['deepest_hole'].append(pd.NA)
                    data_dict['one_sigma_level'].append(pd.NA)

def save_refinement_stats(refinement, wdir):
    stats_dir = os.path.join(wdir, "REFINEMENT_STATISTICS")
    if os.path.exists(stats_dir):
        shutil.rmtree(stats_dir)
    os.makedirs(stats_dir)
    summary_path = os.path.join(stats_dir, "summary.csv")
    refinement.to_csv(summary_path, index=False)
