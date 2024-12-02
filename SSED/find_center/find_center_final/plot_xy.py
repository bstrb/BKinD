import h5py
import matplotlib.pyplot as plt

# File path to the processed HDF5 file
def plot_xy(beam_center_h5_file):# = '/home/buster/UOX1/UOX1_original/UOX1_bg_removed.h5'

    # Read the processed HDF5 file
    with h5py.File(beam_center_h5_file, 'r') as h5_file:
    # Extract x and y positions
        x_positions = h5_file['entry/data/center_x'][:]
        y_positions = h5_file['entry/data/center_y'][:]

    # Generate indices (image numbers)
    image_indices = range(len(x_positions))

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(image_indices, x_positions, label='X Position', marker='o', linestyle='-')
    plt.plot(image_indices, y_positions, label='Y Position', marker='s', linestyle='--')
    plt.xlabel('Image Index')
    plt.ylabel('Beam Center Position')
    plt.title('Beam Center Positions vs Image Index')
    plt.legend()
    plt.grid(True)
    plt.show()
