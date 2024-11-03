import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt


def find_beam_center(file_path, mask_radius, mask_angle_range, save_plot=False):
    with h5py.File(file_path, 'r') as hdf:
        images = hdf['/entry/data/images']
        num_frames = images.shape[0]

        # Initialize variables to accumulate radial intensity distributions
        x_sum = 0
        y_sum = 0
        intensity_sum = 0

        for frame_index in range(num_frames):
            image = images[frame_index, :, :]

            # Define the center (geometric center initially)
            center = (image.shape[0] // 2, image.shape[1] // 2)

            # Create a circular mask
            y, x = np.ogrid[:image.shape[0], :image.shape[1]]
            circular_mask = ((x - center[1])**2 + (y - center[0])**2) >= mask_radius**2

            # Create a wedge mask
            theta = np.arctan2(y - center[0], x - center[1]) * 180 / np.pi
            wedge_mask = (theta < mask_angle_range[0]) | (theta > mask_angle_range[1])

            # Combine masks to apply to the image
            mask = circular_mask & wedge_mask

            # Apply mask to the image
            masked_image = np.where(mask, image, 0)

            # Calculate the intensity-weighted center of mass for the masked image
            total_intensity = masked_image.sum()
            y_indices, x_indices = np.indices(image.shape)
            x_center = (masked_image * x_indices).sum() / total_intensity
            y_center = (masked_image * y_indices).sum() / total_intensity

            # Accumulate the coordinates for averaging
            x_sum += x_center * total_intensity
            y_sum += y_center * total_intensity
            intensity_sum += total_intensity

        # Calculate the average center based on all frames
        avg_x_center = x_sum / intensity_sum
        avg_y_center = y_sum / intensity_sum

        # Save the average center position if needed
        if save_plot:
            plt.imshow(images[num_frames // 2, :, :], cmap='gray')
            plt.scatter(avg_x_center, avg_y_center, color='red', marker='+', s=100)
            plt.title(f"Average Beam Center: ({avg_x_center:.2f}, {avg_y_center:.2f})")
            plt.savefig("average_beam_center.png")
            plt.close()

        return avg_x_center, avg_y_center


# Example usage
file_path = "/home/buster/UOX1/UOX_His_MUA_450nm_spot4_ON_20240311_0928.h5"
mask_radius = 50
mask_angle_range = (-40, 10)
avg_center = find_beam_center(file_path, mask_radius, mask_angle_range, save_plot=True)
print(f"Average Beam Center: {avg_center}")
