# rcis_ref_def.py

import subprocess
import os
import glob
from tqdm import tqdm


def run_ctruncate(hklin, hklout):
    ctruncate_command = f"ctruncate -hklin {hklin} -hklout {hklout} -colin '/*/*/[I,SIGI]'"
    with open(os.devnull, 'w') as fnull:
        subprocess.run(ctruncate_command, shell=True, stdout=fnull, stderr=fnull)

def run_freerflag(hklin, hklout):
    freerflag_command = f"""freerflag hklin {hklin} hklout {hklout} << EOF
freerfrac 0.05
EOF
"""
    with open(os.devnull, 'w') as fnull:
        subprocess.run(freerflag_command, shell=True, stdout=fnull, stderr=fnull)

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

    with open(output_file, "w") as f, open(os.devnull, 'w') as fnull:
        process = subprocess.Popen(refmac_command, shell=True, stdout=subprocess.PIPE, stderr=f, stdin=subprocess.DEVNULL, universal_newlines=True)
        pbar = tqdm(total=ncycles, desc="Refmac5 CGMAT cycles", unit="cycle")
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if "CGMAT cycle number" in line:
                pbar.update(1)
            f.write(line)
        pbar.close()


def process_folder(folder_path, pdb_file, bins):
    print(f"Processing folder: {folder_path}")

    mtz_file = os.path.join(folder_path, "output.mtz")
    ctruncate_mtz_file = os.path.join(folder_path, "output_ctruncate.mtz")
    ctruncatefr_mtz_file = os.path.join(folder_path, "output_ctruncatefr.mtz")
    output_file = os.path.join(folder_path, f"output_bins_{bins}.txt")

    # Run ctruncate and freerflag
    run_ctruncate(mtz_file, ctruncate_mtz_file)
    run_freerflag(ctruncate_mtz_file, ctruncatefr_mtz_file)

    # Run refmac5 with the output of freerflag as input
    run_refmac5(folder_path, pdb_file, ctruncatefr_mtz_file, output_file, bins=bins)

def process_run_folders(base_path, pdb_file, bins):
    folder_paths = [f.path for f in os.scandir(base_path) if f.is_dir()]
    
    for folder_path in folder_paths:
        process_folder(folder_path, pdb_file, bins)

# Example usage
if __name__ == "__main__":
    base_path = "/home/buster/UOXm/5x5_0-01/fast_int_RCIS_3_3_1"  # Replace with your actual base directory path
    pdb_file = "/home/buster/UOXm/5x5_0-01/UOX.pdb"  # Replace with your actual pdb file path
    bins = 20
    process_run_folders(base_path, pdb_file, bins)