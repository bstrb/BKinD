# full_ref_def.py

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

def run_refmac5(base_dir, pdb_file, mtz_file, output_file, res_max=20, res_min=0.85, ncycles=30, bins=10):
    refmac_command = f"""refmac5 xyzin {pdb_file} xyzout {base_dir}/output.pdb hklin {mtz_file} hklout {base_dir}/input.mtz << EOF
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

def process_folder(folder_path, pdb_filename, mtz_orig_filename, output_filename):
    print(f"Processing folder: {folder_path}")

    pdb_file = os.path.join(os.path.dirname(folder_path), pdb_filename)
    mtz_file = os.path.join(folder_path, mtz_orig_filename)
    ctruncate_mtz_file = os.path.join(folder_path, "output_ctruncate.mtz")
    ctruncatefr_mtz_file = os.path.join(folder_path, "output_ctruncatefr.mtz")
    output_file = os.path.join(folder_path, output_filename)

    # Run ctruncate and freerflag
    run_ctruncate(mtz_file, ctruncate_mtz_file)
    run_freerflag(ctruncate_mtz_file, ctruncatefr_mtz_file)

    # Run refmac5 with the output of freerflag as input
    run_refmac5(folder_path, pdb_file, ctruncatefr_mtz_file, output_file)

def process_run_folders(base_path, run_number):
    pattern = os.path.join(base_path, f"merge-*-run{run_number}")
    folder_paths = glob.glob(pattern)  # Use glob to match folders with pattern
    print(f'Found folders: {folder_paths}')
    
    for folder_path in folder_paths:
        process_folder(folder_path)