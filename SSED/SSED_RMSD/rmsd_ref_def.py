# chunk_ref_def.py

import subprocess
import os
import glob


def run_ctruncate(hklin, hklout):
    ctruncate_command = f"ctruncate -hklin {hklin} -hklout {hklout} -colin '/*/*/[I,SIGI]'"
    subprocess.run(ctruncate_command, shell=True)

def run_freerflag(hklin, hklout):
    freerflag_command = f"""freerflag hklin {hklin} hklout {hklout} << EOF
freerfrac 0.05
EOF
"""
    subprocess.run(freerflag_command, shell=True)

def run_refmac5(base_dir, pdb_file, mtz_file, output_file, res_max=20, res_min=1.5, ncycles=30, bins=10):
    refmac_command = f"""refmac5 xyzin {pdb_file} xyzout {base_dir}/output.pdb hklin {mtz_file} hklout {base_dir}/ref_output.mtz << EOF
    ncyc {ncycles}
    bins {bins}
    refi TYPE RESTRAINED RESOLUTION {res_max} {res_min}
    make HYDROGENS ALL
    make HOUT No
    scat ELEC
    scal TYPE BULK
    labin FP=F SIGFP=SIGF FREE=FreeR_flag
    END
    EOF"""

    with open(output_file, "w") as f:
        subprocess.run(refmac_command, shell=True, stdout=f, stderr=f)

# def process_folder(folder_path, pdb_filename, mtz_orig_filename, output_filename, bins):
def process_folder(folder_path, pdb_filename, bins):
    print(f"Processing folder: {folder_path}")

    pdb_file = os.path.join(os.path.dirname(folder_path), pdb_filename)
    mtz_file = os.path.join(folder_path, "output.mtz")
    ctruncate_mtz_file = os.path.join(folder_path, "output_ctruncate.mtz")
    ctruncatefr_mtz_file = os.path.join(folder_path, "output_ctruncatefr.mtz")
    output_file = os.path.join(folder_path, f"output_bins_{bins}.txt")

    # Run ctruncate and freerflag
    run_ctruncate(mtz_file, ctruncate_mtz_file)
    run_freerflag(ctruncate_mtz_file, ctruncatefr_mtz_file)

    # Run refmac5 with the output of freerflag as input
    run_refmac5(folder_path, pdb_file, ctruncatefr_mtz_file, output_file, bins=bins)

import time

# Adjusted function to process folders in increasing numerical order and report processing time
def process_run_rmsd_folders(base_path, pdb_filename, bins=10):
    pattern = os.path.join(base_path, "merge_rmsd_*")
    folder_paths = glob.glob(pattern)  # Use glob to match folders with pattern
    
    # Extract numerical part from folder names and sort them
    sorted_folder_paths = sorted(folder_paths, key=lambda x: int(x.split("_")[-1]))
    print(f'Found and sorted folders: {sorted_folder_paths}')
    
    # Process each folder and track time
    for folder_path in sorted_folder_paths:
        start_time = time.time()
        
        # Call to your processing function (assumed to be defined elsewhere)
        process_folder(folder_path, pdb_filename, bins)
        
        end_time = time.time()
        processing_time = end_time - start_time
        print(f'Processed folder {folder_path} in {processing_time:.2f} seconds')
