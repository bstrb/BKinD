import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import h5py

# Pseudo-Voigt function definition with multiple components
def multi_pseudo_voigt(r, A1, mu1, sigma1, gamma1, eta1, A2, mu2, sigma2, gamma2, eta2, A3, mu3, sigma3, gamma3, eta3):
    """
    Multi-component Pseudo-Voigt function to model background with multiple peaks.
    
    Parameters:
    - r: Radial distance array.
    - A1, mu1, sigma1, gamma1, eta1: Parameters for the first Pseudo-Voigt component.
    - A2, mu2, sigma2, gamma2, eta2: Parameters for the second Pseudo-Voigt component.
    - A3, mu3, sigma3, gamma3, eta3: Parameters for the third Pseudo-Voigt component.
    """
    gaussian1 = np.exp(-((r - mu1) ** 2) / (2 * sigma1 ** 2))
    lorentzian1 = gamma1 ** 2 / ((r - mu1) ** 2 + gamma1 ** 2)
    pv1 = A1 * (eta1 * lorentzian1 + (1 - eta1) * gaussian1)
    
    gaussian2 = np.exp(-((r - mu2) ** 2) / (2 * sigma2 ** 2))
    lorentzian2 = gamma2 ** 2 / ((r - mu2) ** 2 + gamma2 ** 2)
    pv2 = A2 * (eta2 * lorentzian2 + (1 - eta2) * gaussian2)
    
    gaussian3 = np.exp(-((r - mu3) ** 2) / (2 * sigma3 ** 2))
    lorentzian3 = gamma3 ** 2 / ((r - mu3) ** 2 + gamma3 ** 2)
    pv3 = A3 * (eta3 * lorentzian3 + (1 - eta3) * gaussian3)
    
    return pv1 + pv2 + pv3

# Function to read the mask file
def read_mask_file(mask_file_path):
    with h5py.File(mask_file_path, 'r') as mask_file:
        # Assuming the mask is stored under '/mask'
        if '/mask' not in mask_file:
            raise ValueError("Mask dataset '/mask' not found in the mask file.")
        mask_dataset = mask_file['/mask']
        mask = mask_dataset[:]
    return mask

# Main processing function
def process_frame(h5_file_path, mask_file_path):
    # Read the mask
    mask = read_mask_file(mask_file_path)
    print(f"Mask shape: {mask.shape}")
    print(f"Mask unique values before conversion: {np.unique(mask)}")
    
    # Convert mask to binary values (assuming 1 is valid and 2 is masked)
    mask = np.where(mask == 2, 0, mask)  # Convert 2 to 0 (masked), retain 1 as valid
    mask = mask.astype(bool)  # Ensure the mask is boolean
    print(f"Mask unique values after conversion: {np.unique(mask)}")
    
    # Open the HDF5 file
    with h5py.File(h5_file_path, 'r') as h5_file:
        # Access the datasets
        images_dataset = h5_file['/entry/data/images']
        center_x_dataset = h5_file['/entry/data/center_x']
        center_y_dataset = h5_file['/entry/data/center_y']
        
        # Get the number of images
        num_images = images_dataset.shape[0]
        print(f"The dataset contains {num_images} images.")
        
        # Prompt the user to select an image index
        image_index = int(input(f"Enter the index of the image to process (0 to {num_images - 1}): "))
        if image_index < 0 or image_index >= num_images:
            raise ValueError("Invalid image index.")
        
        # Extract the image and center coordinates
        image = images_dataset[image_index, :, :].astype(np.float64)
        center_x = center_x_dataset[image_index]
        center_y = center_y_dataset[image_index]
        
        print(f"Using center coordinates: center_x = {center_x}, center_y = {center_y}")
        
        # Apply the mask from the mask file
        if mask.shape != image.shape:
            raise ValueError("Mask shape does not match image shape.")
        print(f"Mask applied. Number of unmasked pixels: {np.sum(mask)}")
        
        # Apply the mask to the image
        image_masked = np.where(mask, image, np.nan)
        
        # Verify the masked image
        print(f"Masked image stats - min: {np.nanmin(image_masked)}, max: {np.nanmax(image_masked)}, count of valid pixels: {np.sum(~np.isnan(image_masked))}")
        
        # Compute radial distances from the center
        y_indices, x_indices = np.indices(image.shape)
        
        # Compute radial distances
        radii = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)
        
        # Limit the radius to 450 pixels
        radius_limit = 450
        within_radius_mask = (radii <= radius_limit) & mask
        print(f"Number of pixels within radius and unmasked: {np.sum(within_radius_mask)}")
        
        # Flatten arrays and remove masked values
        masked_image = image[within_radius_mask]
        masked_radii = radii[within_radius_mask]
        
        # Verify the radial distances and masked image
        print(f"Masked radial distances - min: {masked_radii.min()}, max: {masked_radii.max()}, count: {len(masked_radii)}")
        print(f"Masked image intensities - min: {masked_image.min()}, max: {masked_image.max()}, count: {len(masked_image)}")
        
        # Bin the data
        num_bins = 100  # Adjust the number of bins as needed
        bins = np.linspace(0, masked_radii.max(), num_bins)
        bin_indices = np.digitize(masked_radii, bins)
        
        # Compute mean intensity for each bin
        radial_means = []
        radial_distances = []
        for i in range(1, len(bins)):
            bin_mask = bin_indices == i
            if np.any(bin_mask):
                mean_intensity = masked_image[bin_mask].mean()
                radial_means.append(mean_intensity)
                radial_distances.append((bins[i] + bins[i - 1]) / 2)  # Use bin center
        
        radial_means = np.array(radial_means)
        radial_distances = np.array(radial_distances)
        
        # Fit a multi-component Pseudo-Voigt curve to the radial means
        # Initial guess for the parameters (adjust as needed)
        initial_guess = [
            np.nanmax(radial_means),      # A1
            180,                          # mu1 (first bump position)
            20,                           # sigma1
            20,                           # gamma1
            0.5,                          # eta1
            np.nanmax(radial_means) / 2,  # A2
            320,                          # mu2 (second bump position)
            20,                           # sigma2
            20,                           # gamma2
            0.5,                          # eta2
            np.nanmax(radial_means) / 4,  # A3
            np.nanmedian(radial_distances),  # mu3 (center peak)
            50,                           # sigma3
            50,                           # gamma3
            0.5                           # eta3
        ]
        
        # Boundaries for the parameters to ensure physical meaningfulness
        param_bounds = (
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # Lower bounds
            [np.inf, np.inf, np.inf, np.inf, 1, np.inf, np.inf, np.inf, np.inf, 1, np.inf, np.inf, np.inf, np.inf, 1]  # Upper bounds
        )
        
        # Perform the curve fitting
        try:
            popt, pcov = curve_fit(
                multi_pseudo_voigt,
                radial_distances,
                radial_means,
                p0=initial_guess,
                bounds=param_bounds
            )
        except RuntimeError as e:
            print("Curve fitting failed:", e)
            popt = initial_guess  # Use initial guess if fitting fails
        
        # Generate the background model over the entire image within the radius limit
        background = multi_pseudo_voigt(radii, *popt)
        background[~within_radius_mask] = 0
        
        # Subtract the background from the original image
        corrected_image = image - background
        
        # Plot the original and corrected images
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(image, cmap='gray', origin='lower')
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        plt.title('Background Corrected Image')
        plt.imshow(corrected_image, cmap='gray', origin='lower')
        plt.colorbar()
        
        plt.show()
        
        # Plot the radial means and the fitted background model
        plt.figure()
        plt.title('Radial Intensity Profile')
        plt.plot(radial_distances, radial_means, 'bo', label='Radial Means')
        r_fit = np.linspace(0, radius_limit, 1000)
        plt.plot(r_fit, multi_pseudo_voigt(r_fit, *popt), 'r-', label='Fitted Background')
        plt.xlabel('Radius (pixels)')
        plt.ylabel('Mean Intensity')
        plt.legend()
        plt.show()
        
        # Optionally, save the corrected image
        save_option = input("Do you want to save the corrected image? (y/n): ").lower()
        if save_option == 'y':
            output_filename = input("Enter the filename to save the corrected image (e.g., corrected_image.png): ")
            # Normalize the corrected image for saving
            corrected_image_scaled = (corrected_image - np.nanmin(corrected_image))
            corrected_image_scaled /= np.nanmax(corrected_image_scaled)
            corrected_image_uint8 = (corrected_image_scaled * 255).astype(np.uint8)
            plt.imsave(output_filename, corrected_image_uint8, cmap='gray')
            print(f"Corrected image saved as {output_filename}")

# Paths to the files
h5_file_path = '/home/buster/UOX1/UOX_His_MUA_450nm_spot4_ON_20240311_0928.h5'
mask_file_path = '/home/buster/UOX1/mask/pxmask.h5'

# Call the main processing function
process_frame(h5_file_path, mask_file_path)
