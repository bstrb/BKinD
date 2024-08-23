# add_cbi.py

# %%

import fabio
import numpy as np
import os
import glob

def process_images(img_list, row, column, sqr):
    sum_list = []

    for img_file in img_list:
        img = fabio.open(img_file)
        img_data = np.array(img.data)

        cut = img_data[row - sqr:row + sqr + 1, column - sqr:column + sqr + 1]
        sum_list.append(cut.sum())

    if len(sum_list) == 0:
        print("No images processed or sum_list is empty. Please check your input files and coordinates.")
        return None

    average = sum(sum_list) / len(sum_list)
    sum_list = np.array(sum_list / average)

    return sum_list
def update_hkl_file(hkl_filepath, cbi):
    updated_lines = []
    with open(hkl_filepath, 'r') as file:
        header = True
        for line in file:
            if header:
                updated_lines.append(line)
                if line.startswith('!END_OF_HEADER'):
                    header = False
            else:
                if line.strip() and not line.startswith('!'):  # Process only data lines
                    parts = line.split()
                    h, k, l = int(parts[0]), int(parts[1]), int(parts[2])
                    intensity = float(parts[3])
                    sigma = float(parts[4])
                    xd, yd, zd = float(parts[5]), float(parts[6]), float(parts[7])
                    rlp = float(parts[8])
                    peak = int(parts[9])
                    corr = int(parts[10])
                    psi = float(parts[11])
                    z_obs = float(parts[7])  # zd as z_obs
                    z_obs_index = int(round(z_obs))  # Round to the nearest integer
                    cbi_value = cbi[z_obs_index - 1] if z_obs_index - 1 < len(cbi) else np.nan

                    # Reconstruct the line with all original parts and append the CBI
                    updated_line = (
                        f"{h:4} {k:4} {l:4} {intensity:12.4e} {sigma:12.4e} {xd:8.1f} {yd:8.1f} {zd:8.1f} "
                        f"{rlp:8.4f} {peak:4d} {corr:4d} {psi:8.2f} {cbi_value:12.4f}\n"
                    )
                    updated_lines.append(updated_line)
                else:
                    updated_lines.append(line)

    output_filepath = hkl_filepath.replace("XDS_ASCII.HKL", "XDS_ASCII_CBI.HKL")
    with open(output_filepath, 'w') as file:
        file.writelines(updated_lines)

    return output_filepath


def main(img_directories, center, square_size, hkl_files):
    # Parse center coordinates
    center = center.split()
    row = int(center[1])
    column = int(center[0])
    sqr = square_size // 2

    for img_directory, hkl_file in zip(img_directories, hkl_files):
        print(f"Processing directory: {img_directory}")
        
        # List of all .img files in the directory
        img_list = glob.glob(os.path.join(img_directory, '*.img'))

        if not img_list:
            print(f"No .img files found in the directory {img_directory}.")
            continue

        # Process images to get center beam intensity
        cbi = process_images(img_list, row, column, sqr)

        if cbi is None:
            continue

        # Update the corresponding XDS_ASCII.HKL file
        print(f"Updating {hkl_file}...")
        update_hkl_file(hkl_file, cbi)

if __name__ == "__main__":
    img_directories = [
        '/mnt/c/Users/bubl3932/Desktop/3DED-DATA/FeAcAc/FeAcAC_crushed_11/SMV',
        '/mnt/c/Users/bubl3932/Desktop/3DED-DATA/FeAcAc/FeAcAC_crushed_12/SMV',
        '/mnt/c/Users/bubl3932/Desktop/3DED-DATA/FeAcAc/FeAcAC_crushed_13/SMV'
    ]
    center = '1124 1124'  # Example center coordinates
    square_size = 100  # Example square size
    hkl_files = [
        '/mnt/c/Users/bubl3932/Desktop/3DED-DATA/FeAcAc/FeAcAC_crushed_11/XDS_ASCII.HKL',
        '/mnt/c/Users/bubl3932/Desktop/3DED-DATA/FeAcAc/FeAcAC_crushed_12/XDS_ASCII.HKL',
        '/mnt/c/Users/bubl3932/Desktop/3DED-DATA/FeAcAc/FeAcAC_crushed_13/XDS_ASCII.HKL'
    ]

    main(img_directories, center, square_size, hkl_files)

# %%