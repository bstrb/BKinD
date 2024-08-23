# %%

import fabio
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from skimage import exposure

def process_images(img_list, row, column, sqr):
    inverse_sum_list = []

    for img_file in img_list:
        img = fabio.open(img_file)
        img_data = np.array(img.data)

        # Increase contrast using histogram equalization
        img_data = exposure.equalize_hist(img_data)
        img_data = img_data * 0.000001  # Scale back to 0-255 after equalization

        # Create a mask for the central square region
        mask = np.ones_like(img_data, dtype=bool)
        mask[row - sqr:row + sqr + 1, column - sqr:column + sqr + 1] = False

        # Avoid division by zero by setting a minimum threshold
        safe_img_data = np.where(img_data > 0, img_data, 1e-10)

        # Calculate the inverse of the intensities outside the central square
        inverse_intensities = np.where(mask, 1.0 / safe_img_data, 0)

        # Sum the inverse intensities
        inverse_sum = np.sum(inverse_intensities[mask])
        inverse_sum_list.append(inverse_sum)

        # Print the inverse sum for debugging
        print(f"File: {img_file}, Inverse Sum: {inverse_sum}")

        # Plot the processed image with the masked region
        if img_file == img_list[0]:  # Only plot for the first image
            plt.imshow(img_data, cmap='gray')
            plt.colorbar()
            plt.title("Processed Image with Masked Region")
            plt.gca().add_patch(plt.Rectangle((column - sqr, row - sqr), 2 * sqr, 2 * sqr, 
                                               linewidth=1, edgecolor='r', facecolor='none'))
            plt.show()

    if len(inverse_sum_list) == 0:
        print("No images processed or inverse_sum_list is empty. Please check your input files and coordinates.")
        return None

    # Directly return inverse sums without normalization
    return np.array(inverse_sum_list)

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

                    # Reconstruct the line with all original parts and append the normalized CBI
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

def plot_cbi_vs_frame(cbi):
    plt.figure(figsize=(10, 6))
    plt.plot(cbi, marker='o', linestyle='-', color='b')
    plt.title('Normalized CBI vs. Frame')
    plt.xlabel('Frame Number')
    plt.ylabel('Normalized CBI')
    plt.grid(True)
    plt.show()

# def plot_cbi_vs_frame(cbi):
#     plt.figure(figsize=(10, 6))
#     plt.plot(cbi[:-1], marker='o', linestyle='-', color='b')  # Exclude the last value
#     plt.title('Normalized CBI vs. Frame')
#     plt.xlabel('Frame Number')
#     plt.ylabel('Normalized CBI')
#     plt.grid(True)
#     plt.show()

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

        # Plot the first image with the masked square region
        first_img = fabio.open(img_list[0])
        first_img_data = np.array(first_img.data)
        
        # Create a mask for the central square region
        mask = np.ones_like(first_img_data, dtype=bool)
        mask[row - sqr:row + sqr + 1, column - sqr + sqr + 1] = False

        # Plot the image
        plt.imshow(first_img_data, cmap='gray')
        plt.colorbar()
        plt.title("First Image with Masked Region")
        
        # Plot the square on the image
        plt.gca().add_patch(plt.Rectangle((column - sqr, row - sqr), square_size, square_size, 
                                           linewidth=1, edgecolor='r', facecolor='none'))
        plt.show()

        # Process images to get center beam intensity
        cbi = process_images(img_list, row, column, sqr)

        if cbi is None:
            continue

        # Normalize the CBI values by dividing by the mean CBI value
        cbi = cbi / np.mean(cbi)

        # Plot CBI vs. Frame
        plot_cbi_vs_frame(cbi)

        # Update the corresponding XDS_ASCII.HKL file
        print(f"Updating {hkl_file}...")
        update_hkl_file(hkl_file, cbi)


if __name__ == "__main__":
    img_directories = [
        '/mnt/c/Users/bubl3932/Desktop/3DED-DATA/LTA_re/LTA_t1/converted/images'
        # '/mnt/c/Users/bubl3932/Desktop/3DED-DATA/FeAcAc/FeAcAC_crushed_13/SMV'
    ]
    center = '248 248'  # Example center coordinates
    square_size = 50  # Example square size
    hkl_files = [
        '/mnt/c/Users/bubl3932/Desktop/3DED-DATA/LTA_re/LTA_t1/converted/common-miller/XDS_ASCII.no_error_model_applied.HKL'
        # '/mnt/c/Users/bubl3932/Desktop/3DED-DATA/FeAcAc/FeAcAC_crushed_13/XDS_ASCII.HKL'
    ]

    main(img_directories, center, square_size, hkl_files)

# %%
