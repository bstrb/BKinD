import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt

def process_frame(file_path, frame_index=1, radius=50, angle_range=(-20, 20)):
    with h5py.File(file_path, 'r') as hdf:
        images = hdf['/entry/data/images']
        
        # Select a frame for analysis
        image = images[frame_index, :, :]

        # Save the image to understand the beam stopper's position
        plt.imshow(image, cmap='gray')
        plt.title(f"Frame {frame_index}")
        plt.colorbar()
        plt.savefig(f"frame_{frame_index}.png")
        plt.close()

        # Define a mask (example parameters, adjust as needed)
        center = (image.shape[0] // 2, image.shape[1] // 2)
        
        # Create a circular mask
        y, x = np.ogrid[:image.shape[0], :image.shape[1]]
        circular_mask = ((x - center[0])**2 + (y - center[1])**2) >= radius**2

        # Create a wedge mask (simple angular selection)
        theta = np.arctan2(y - center[1], x - center[0]) * 180 / np.pi
        wedge_mask = (theta < angle_range[0]) | (theta > angle_range[1])

        # Combine the masks
        mask = circular_mask & wedge_mask
        
        # Apply mask to the image
        masked_image = np.where(mask, image, 0)

        # Save the masked image
        plt.imshow(masked_image, cmap='gray')
        plt.title("Masked Image")
        plt.colorbar()
        plt.savefig(f"masked_frame_{frame_index}.png")
        plt.close()

# Example usage
file_path = "/home/buster/UOX1/UOX_His_MUA_450nm_spot4_ON_20240311_0928.h5"
process_frame(file_path, frame_index=1500, radius=50, angle_range=(-40, 10))
