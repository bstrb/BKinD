import h5py
import matplotlib.pyplot as plt

def display_h5_mask(file_path, dataset_path):
    """
    Display a mask stored in an HDF5 file.

    Parameters:
    - file_path: Path to the HDF5 file.
    - dataset_path: Path to the dataset containing the mask within the HDF5 file.
    """
    try:
        # Open the HDF5 file
        with h5py.File(file_path, 'r') as f:
            # Load the mask data
            mask_data = f[dataset_path][()]

            # Display the mask using matplotlib
            plt.imshow(mask_data, cmap='gray')
            plt.title('Mask')
            plt.axis('off')
            plt.show()

    except Exception as e:
        print(f"Error: {e}")

# Example usage
file_path = '/mnt/c/Users/bubl3932/Desktop/testmask.h5'
dataset_path = '/mask'  # Replace with the correct dataset path in your file
display_h5_mask(file_path, dataset_path)
