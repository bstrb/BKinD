#!/usr/bin/env python3
import argparse
import os, sys

import numpy as np
import shutil
import pandas as pd
import plotly.express as px
from scipy.optimize import minimize

# CCTBX imports
from iotbx.reflection_file_reader import any_reflection_file
from iotbx.shelx.hklf import miller_array_export_as_shelx_hklf as hklf
from scitbx.array_family import flex

class Integrate():
    
    def __init__(self, file_name):
        """ 
        Class for easy access and manipulation of INTEGRATE.HKL file. 
        
        Args:
            file_name: path to INTEGRATE.HKL file
        """
        self.inte= any_reflection_file(file_name)\
            .as_miller_arrays(merge_equivalents=False)
        self.inte0 = self.inte[0]
        # print(self.inte0)
        self.inte1 = self.inte[1]
        # print(self.inte1)
        self.inte2 = self.inte[2]
        # print(self.inte2)
        self.df = pd.DataFrame()
        
    def set_resolution(self, dmax: float, dmin: float):
        """ 
        Resoltion filter
        y
        
        Args:
            dmax: maximum resolution
            dmin: minimum resolution
        
        """
        self.inte0 = self.inte[0].resolution_filter(dmax, dmin)
        self.inte1 = self.inte[1].resolution_filter(dmax, dmin)
        self.inte2 = self.inte[2].resolution_filter(dmax, dmin)
    
    def indices(self):
        """ 
        Returns indices
        """
        return(list(self.inte0.indices()))
    
    def hex_bravais(self):
        h = np.array(self.indices())[:,0]
        k = np.array(self.indices())[:,1]
        l = np.array(self.indices())[:,2]
        i = -(h+k)
        
        return list(zip(h, k, i, l))
    
    def data(self):
        """ 
        Returns intensities
        """
        return list(self.inte0.data())
    
    def sigmas(self):
        """ 
        Returns sigma(intensity)
        """
        return list(self.inte0.sigmas())
    
    def xobs(self):
        """ 
        Returns observed frame number
        """
        return np.array(self.inte1.data())[:, 0]
    
    def yobs(self):
        """ 
        Returns observed frame number
        """
        return np.array(self.inte1.data())[:, 1]
    
    def zobs(self):
        """ 
        Returns observed frame number
        """
        return np.array(self.inte1.data())[:, 2]
    
    def d_spacings(self):
        """ 
        Returns resolution
        """
        return list(self.inte0.d_spacings().data())
    
    def asus(self):
        """ 
        Returns asymmetric group
        """
        return list(self.inte0.map_to_asu().indices())

    def as_df(self) -> pd.DataFrame:
        """ 
        Converts into a DataFrame
        """
        self.df = pd.DataFrame()
        self.df['Miller'] = self.indices()
        self.df['asu'] = self.asus()
        self.df['Intensity'] = self.data()
        self.df['Sigma'] = self.sigmas()
        self.df['I/Sigma'] = self.df['Intensity']/self.df['Sigma']
        self.df['Resolution'] = self.d_spacings()
        self.df['xobs'] = self.xobs()
        self.df['yobs'] = self.yobs()
        self.df['zobs'] = self.zobs()
        self.df['Index_INTE'] = np.arange(0, len(self.df.index), 1)
        
        return self.df

    def as_df_hex(self):
        self.as_df()
        self.df['brav'] = self.df['Miller'].apply(
            lambda x: (x[0], x[1], -(x[0]+x[1]), x[2]))
        column_order = ['Miller', 'brav', 'asu', 'Intensity', 'Sigma',
                        'I/Sigma', 'Resolution', 'xobs', 'yobs', 'zobs', 
                        'Index_INTE']
        self.df = self.df.reindex(columns=column_order)
        
        return self.df
        
    def sele_idx(self, sel: np.array):
        """ 
        Select from INTEGRATE.HKL given corresponding indices
        sel = np.asarray(df.loc[df['Column'] > condition]['Index_INTE'])
        """
        sel = flex.size_t(sel)
        self.inte0 = self.inte0.select(sel)
        self.inte1 = self.inte1.select(sel)
        self.inte2 = self.inte2.select(sel)
    
    def sele_idx_as_df(self, sel: np.ndarray) -> pd.DataFrame:
        self.sele_idx(sel)
        
        return self.as_df()
    
    def sele_bool(self, sel: np.ndarray):
        """ 
        Select from INTEGRATE.HKL given Boolean indices
        sel = np.asarray(df['Column'] > condition)
        """
        sel = flex.bool(sel)
        self.inte0 = self.inte0.select(sel)
        self.inte1 = self.inte1.select(sel)
        self.inte2 = self.inte2.select(sel)
        
    def sele_bool_as_df(self, sel: np.ndarray) -> pd.DataFrame:
        self.sele_bool(sel)
        
        return self.as_df()
    
    def print_hklf4(self, outp_name: str):
        """ 
        Converts iotbx.any_reflection_file integratehkl to SHELX HKLF4 format 
        and output to outp_name.hkl
        
        Args:
            path_to_integrate_hkl (str): a path to INTERGATE.HKL file
            outp_name (str): desired output filename
        """
        # Process INTEGRATE.HKL 
        stdout_obj = sys.stdout
        sys.stdout = open(outp_name+'.hkl', 'w')
        hklf(self.inte0)
        sys.stdout = stdout_obj

    def print_hklf4_df(self, outp_name: str):

        self.print_hklf4()


def format_line(parts, column_widths):
    # Format each part according to specified widths
    formatted_parts = []
    for part, width in zip(parts, column_widths):
        formatted_part = f"{part:>{width}}"
        formatted_parts.append(formatted_part)
    return ''.join(formatted_parts)

def create_xds_ascii_nem(xds_dir, target_dir):
    xds_path = os.path.join(xds_dir, 'XDS_ASCII.HKL')
    integrate_path = os.path.join(xds_dir, 'INTEGRATE.HKL')
    output_path = os.path.join(target_dir, 'XDS_ASCII_NEM.HKL')

    # Define column widths for output formatting
    column_widths = [6, 6, 6, 11, 11, 8, 8, 9, 10, 4, 4, 8]

    try:
        with open(xds_path, 'r') as file:
            xds_lines = [line.strip() for line in file if not line.startswith('!')]

        with open(integrate_path, 'r') as file:
            integrate_lines = [line.strip() for line in file if not line.startswith('!')]

        # Map Miller indices from INTEGRATE.HKL to their corresponding entries
        integrate_dict = {}
        for line in integrate_lines:
            parts = line.split()
            miller_indices = tuple(parts[:3])  # first three columns are Miller indices
            integrate_values = [float(part) for part in parts]
            integrate_dict[miller_indices] = integrate_values

        # Prepare the output data including headers
        output_data = []
        with open(xds_path, 'r') as file:
            for line in file:
                if line.startswith('!'):
                    output_data.append(line)  # Preserve headers
                else:
                    parts = line.split()
                    miller_indices = tuple(parts[:3])
                    if miller_indices in integrate_dict and len(parts) > 4:
                        # Apply the given formula to modify the 5th column
                        try:
                            xds_value = float(parts[3])
                            integrate_values = integrate_dict[miller_indices]
                            calculated_value = (xds_value / integrate_values[3]) * integrate_values[4]
                            parts[4] = f"{calculated_value:.3E}"  # Modify the 5th column with scientific notation
                        except (IndexError, ZeroDivisionError, ValueError):
                            pass  # Skip modification if any error occurs
                    output_line = format_line(parts, column_widths)
                    output_data.append(output_line + '\n')

        # Write to the new file with appropriate formatting
        with open(output_path, 'w') as file:
            file.writelines(output_data)

        # print(f"File created successfully at {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

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

def objective_function(u, sample_df):
    """
    Objective function to minimize the absolute difference between mean and median of DFM.
    """
    dfm_values = calculate_dfm(u, sample_df)
    return abs(np.mean(dfm_values) - np.median(dfm_values))

def calculate_dfm(u, sample_df):
    """
    Calculates the DFM values based on the given instability factor 'u'.
    """
    return (sample_df['Fo^2'] - sample_df['Fc^2']) / np.sqrt(sample_df['Fo^2_sigma']**2 + (2*u * sample_df['Fc^2'])**2)
    
    # print('Using DFM equation with F_o instead of F_c in denominator')
    # return (sample_df['Fo^2'] - sample_df['Fc^2']) / np.sqrt(sample_df['Fo^2_sigma']**2 + (2*u * sample_df['Fo^2'])**2) # sqrt(P) from Jana

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
def manage_files(action, source_directory, target_directory, filename=None, new_filename=None, extension=None):
    """
    Manages files by moving or copying them, with options to rename or find files by extension.

    Parameters:
    - action: The action to perform - 'move' or 'copy'.
    - source_directory: The directory containing the source file.
    - target_directory: The directory where the file will be moved/copied.
    - filename: The name of the file to be moved/copied. If not provided, extension must be specified.
    - new_filename: The new name for the file in the target directory. Optional.
    - extension: The file extension to search for if filename is not provided. Optional.
    """

    source_file_path = None

    if filename:
        source_file_path = os.path.join(source_directory, filename)
    elif extension:
        # Find the first file with the given extension in the source directory
        for file in os.listdir(source_directory):
            if file.endswith(extension):
                source_file_path = os.path.join(source_directory, file)
                break
        if not source_file_path:
            print(f"No file with extension '{extension}' found in '{source_directory}'.")
            return False
    else:
        print("Either 'filename' or 'extension' must be provided.")
        return False

    if not new_filename:
        new_filename = os.path.basename(source_file_path)
        
    target_file_path = os.path.join(target_directory, new_filename)
    
    # Ensure the target directory exists
    os.makedirs(target_directory, exist_ok=True)

    try:
        if action == 'move':
            shutil.move(source_file_path, target_file_path)
            # print(f"File '{os.path.basename(source_file_path)}' moved to '{target_file_path}'.")
        elif action == 'copy':
            shutil.copy(source_file_path, target_file_path)
            # print(f"File '{os.path.basename(source_file_path)}' copied to '{target_file_path}'.")
        else:
            print(f"Invalid action '{action}'. Use 'move' or 'copy'.")
            return False
        return True
    except FileNotFoundError:
        print(f"Error: '{source_file_path}' not found.")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def modify_ins_file(file_path):
    """
    Modifies the .ins file to:
    1. Search lines between those containing 'UNIT' and 'FVAR' (case insensitive).
    2. Remove lines containing 'merg' and 'fmap' (case insensitive).
    3. Replace lines starting with 'list' with 'LIST 4\nMERG 0\nFMAP 2' (case insensitive).

    Parameters:
    - file_path: Path to the .ins file to modify.
    """
    modified_lines = []
    is_between_unit_fvar = False

    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        for line in lines:
            line_lower = line.lower().strip()
            if 'unit' in line_lower:
                is_between_unit_fvar = True
                modified_lines.append(line)
                continue
            elif 'fvar' in line_lower:
                is_between_unit_fvar = False
                modified_lines.append(line)
                continue

            if is_between_unit_fvar:
                if 'merg' in line_lower and 'merge' not in line_lower:
                    continue
                if 'fmap' in line_lower:
                    continue
                if 'acta' in line_lower:
                    continue
                if line_lower.startswith('list'):
                    modified_lines.append('LIST 4\nMERG 0\nFMAP 2\nACTA\n')
                    continue

            modified_lines.append(line)

        # Write the modified lines back to the file
        with open(file_path, 'w') as file:
            file.writelines(modified_lines)

    except Exception as e:
        print(f"An error occurred: {e}")

def find_file(path, pattern):
    """
    Generic file search to find a specific file pattern in the given directory.

    Args:
    path (str): The directory path where to search for the file.
    pattern (str): The file pattern to search for (e.g., '.fcf').

    Returns:
    str: The path to the file if found, otherwise None.
    """
    for file in os.listdir(path):
        if file.endswith(pattern):
            return os.path.join(path, file)
    # print(f"No {pattern} file found in the source directory.")
    return None

def die(msg: str, code: int = 2):
    print(f"ERROR: {msg}")
    raise SystemExit(code)


def prepare_ed_dataset(shelx_dir: str, xds_dir: str, out_dir: str):
    """
    Reproduce the ED 'Initial Setup' + 'Running SHELXL' steps from BKinD prepare.py
    for a single dataset, in a headless way.
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) Initial Setup: copy/modify bkind.ins and prepare NEM + bkind.hkl
    manage_files("copy", shelx_dir, out_dir, extension=".ins", new_filename="bkind.ins")
    modify_ins_file(os.path.join(out_dir, "bkind.ins"))

    # Copy INTEGRATE.HKL into out_dir
    manage_files("copy", xds_dir, out_dir, filename="INTEGRATE.HKL")

    # Create NEM XDS_ASCII in out_dir and XDSCONV.INP, run xdsconv → bkind.hkl
    create_xds_ascii_nem(xds_dir, out_dir)
    create_xdsconv_nem(out_dir)

    # Run xdsconv in out_dir
    import subprocess
    p = subprocess.run(["xdsconv"], cwd=out_dir)
    if p.returncode != 0:
        die(f"xdsconv failed with exit code {p.returncode}")

    bkind_hkl = os.path.join(out_dir, "bkind.hkl")
    if not os.path.exists(bkind_hkl):
        die("xdsconv did not produce bkind.hkl")

    # 2) Run SHELXL in out_dir (uses bkind.ins)
    p = subprocess.run(["shelxl", "bkind"], cwd=out_dir)
    if p.returncode != 0:
        die(f"shelxl failed with exit code {p.returncode}")

    bkind_fcf = os.path.join(out_dir, "bkind.fcf")
    if not os.path.exists(bkind_fcf):
        die("shelxl did not produce bkind.fcf")

    return bkind_fcf, os.path.join(out_dir, "INTEGRATE.HKL")


def build_sample_df_no_filter(out_dir: str, initial_u: float = 0.01) -> pd.DataFrame:
    """
    Reproduce the ED branch of process_sample_data(..., xray=False)
    using BKinD's own read_fcf + Integrate.as_df + DFM logic.
    """
    # Find bkind.fcf and INTEGRATE.HKL in out_dir
    fcf_path = find_file(out_dir, ".fcf")
    if fcf_path is None:
        die(f"No .fcf file found in {out_dir}")

    inte_path = find_file(out_dir, "INTEGRATE.HKL")
    if inte_path is None:
        die(f"No INTEGRATE.HKL file found in {out_dir}")

    # 1) FCF → sample_fcf_df (Miller, Fc^2, Fo^2, Fo^2_sigma)
    sample_fcf_df = read_fcf(fcf_path)

    # 2) INTEGRATE → sample_inte_df (Miller, asu, Resolution, xobs, yobs, zobs, ...)
    sample_inte_df = Integrate(inte_path).as_df()

    # 3) Merge exactly as in process_sample_data (xray=False)
    sample_df = sample_fcf_df.merge(sample_inte_df, on="Miller", how="inner")

    # Debug: show overlap
    fcf_set = set(sample_fcf_df["Miller"])
    inte_set = set(sample_inte_df["Miller"])
    print("FCF unique HKLs:", len(fcf_set))
    print("INTEGRATE unique HKLs:", len(inte_set))
    print("HKL intersection:", len(fcf_set & inte_set))

    if sample_df.empty:
        die("Merge produced 0 rows. FCF and INTEGRATE.HKL do not match in BKinD ED logic.")

    # 4) Keep exactly the columns BKinD expects for DFM, no filtering
    required = ["Miller", "asu", "Fc^2", "Fo^2", "Fo^2_sigma", "Resolution", "xobs", "yobs", "zobs"]
    missing = [c for c in required if c not in sample_df.columns]
    if missing:
        die(f"Missing expected columns after merge: {missing}")

    sample_df = sample_df[required].copy()

    # 5) Optimize u with BKinD's objective_function
    res = minimize(objective_function, initial_u, args=(sample_df,), method="Nelder-Mead")
    optimal_u = float(res.x[0])
    if abs(initial_u - optimal_u) > initial_u + 0.001:
        optimal_u = initial_u  # BKinD guard

    sample_df["DFM"] = calculate_dfm(optimal_u, sample_df)
    print(f"Optimal u used: {optimal_u:.3g}")

    # Save sample_df for inspection
    csv_path = os.path.join(out_dir, "sample_df_no_filter.csv")
    sample_df.to_csv(csv_path, index=False)
    print(f"Saved merged sample_df with DFM to: {csv_path}")

    return sample_df


def main():
    ap = argparse.ArgumentParser(
        description="Produce BKinD-style DFM vs frame (zobs) for one ED dataset, no filtering."
    )
    ap.add_argument("--shelx-dir", required=True, help="Folder with .ins (template) and possibly .hkl")
    ap.add_argument("--xds-dir", required=True, help="Folder with XDS_ASCII.HKL and INTEGRATE.HKL")
    ap.add_argument("--out-dir", required=True, help="Output folder for NEM/SHELXL/output CSV and plot")
    ap.add_argument("--initial-u", type=float, default=0.01, help="Initial u (default 0.01)")
    args = ap.parse_args()

    shelx_dir = os.path.abspath(args.shelx_dir)
    xds_dir = os.path.abspath(args.xds_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # 1) Reproduce BKinD's ED setup → bkind.fcf + INTEGRATE.HKL in out_dir
    fcf_path, inte_path = prepare_ed_dataset(shelx_dir, xds_dir, out_dir)
    print("Using FCF:", fcf_path)
    print("Using INTEGRATE:", inte_path)

    # 2) Build sample_df and DFM exactly as BKinD ED logic, no filtering
    sample_df = build_sample_df_no_filter(out_dir, initial_u=args.initial_u)

    # 3) Plot DFM vs frame (zobs)
    fig = px.scatter(
        sample_df,
        x="zobs",
        y="DFM",
        title="DFM vs Frame (BKinD ED, no filtering)",
        labels={"zobs": "Frame (zobs)", "DFM": "DFM"},
    )
    out_html = os.path.join(out_dir, "DFM_vs_frame_no_filter.html")
    fig.write_html(out_html)
    print(f"Saved plot: {out_html}")


if __name__ == "__main__":
    main()

# python dfm_vs_frame_from_bkind.py --shelx-dir /Users/xiaodong/Desktop/3DED-DATA/LTA/LTA1/shelx --xds-dir   /Users/xiaodong/Desktop/3DED-DATA/LTA/LTA1/xds --out-dir   /Users/xiaodong/Desktop/DFM_vs_frame_nem_output