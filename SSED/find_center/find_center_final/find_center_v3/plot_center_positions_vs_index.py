import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle

def read_center_and_index_data(h5_file):
    """
    Reads center_x, center_y, and index datasets from the given HDF5 file.

    Parameters:
        h5_file (str): Path to the HDF5 file.

    Returns:
        tuple: (center_x, center_y, index) as numpy arrays.

    Raises:
        KeyError: If the required datasets are not found.
        IOError: If the file cannot be opened.
    """
    try:
        with h5py.File(h5_file, 'r') as f:
            center_x = f['entry/data/center_x'][:]
            center_y = f['entry/data/center_y'][:]
            index = f['entry/data/index'][:]
    except KeyError as e:
        raise KeyError(f"Dataset not found in {h5_file}: {e}")
    except Exception as e:
        raise IOError(f"Error reading {h5_file}: {e}")
    
    return center_x, center_y, index

def plot_center_positions(h5_paths, selected_indices=None, labels=None, save_path=None):
    """
    Plots center_x and center_y positions vs index for selected indices from multiple HDF5 files.

    Parameters:
        h5_paths (list of str): List of paths to HDF5 files.
        selected_indices (list or array-like, optional): List of specific indices to plot. 
            If None, all indices are plotted.
        labels (list of str, optional): List of labels for each HDF5 file. If None, file names are used.
        save_path (str, optional): If provided, saves the plot to the given path instead of displaying it.

    Raises:
        ValueError: If the lengths of h5_paths and labels do not match.
    """
    if labels and len(labels) != len(h5_paths):
        raise ValueError("Length of labels must match length of h5_paths.")

    # Convert selected_indices to a numpy array for efficient processing
    if selected_indices is not None:
        selected_indices = np.array(selected_indices)
        # Ensure selected_indices are unique and sorted (optional)
        selected_indices = np.unique(selected_indices)
    
    file_data = {}
    for idx, h5_file in enumerate(h5_paths):
        if not os.path.isfile(h5_file):
            print(f"Warning: File not found - {h5_file}. Skipping.")
            continue
        try:
            center_x, center_y, index = read_center_and_index_data(h5_file)
            if len(center_x) == 0 or len(center_y) == 0 or len(index) == 0:
                print(f"Warning: Empty datasets in {h5_file}. Skipping.")
                continue

            # Filter to ensure all arrays have the same length
            valid_mask = np.isfinite(center_x) & np.isfinite(center_y) & np.isfinite(index)
            center_x, center_y, index = center_x[valid_mask], center_y[valid_mask], index[valid_mask]

            if len(index) == 0:
                print(f"Warning: No valid data in {h5_file}. Skipping.")
                continue

            # If selected_indices is specified, filter the data
            if selected_indices is not None:
                # Create a mask where index is in selected_indices
                # Assuming index contains integer values; adjust if necessary
                mask = np.isin(index, selected_indices)
                center_x, center_y, index = center_x[mask], center_y[mask], index[mask]
                
                if len(index) == 0:
                    print(f"Warning: No matching selected indices in {h5_file}. Skipping.")
                    continue

            label = labels[idx] if labels else os.path.basename(h5_file)
            file_data[label] = (center_x, center_y, index)
            print(f"Successfully read {h5_file} with {len(index)} selected indices")
        except Exception as e:
            print(f"Error reading {h5_file}: {e}")

    if not file_data:
        print("No valid data to plot.")
        return

    # Define color and linestyle cycles for better differentiation
    color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    linestyle_cycle = cycle(['-', '--', '-.', ':'])

    # Create separate figures for center_x and center_y
    plt.figure(figsize=(14, 10))
    
    # Top subplot for center_x
    plt.subplot(2, 1, 1)
    plt.title('Center X Positions vs Index', fontsize=16)

    for label, (center_x, _, index) in file_data.items():
        color = next(color_cycle)
        linestyle = next(linestyle_cycle)
        plt.plot(index, center_x, label=label, linestyle=linestyle, marker='o', color=color)

    plt.xlabel('Index', fontsize=14)
    plt.ylabel('Center X Position', fontsize=14)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Reset color and linestyle cycles for the next plot
    color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    linestyle_cycle = cycle(['-', '--', '-.', ':'])

    # Bottom subplot for center_y
    plt.subplot(2, 1, 2)
    plt.title('Center Y Positions vs Index', fontsize=16)

    for label, (_, center_y, index) in file_data.items():
        color = next(color_cycle)
        linestyle = next(linestyle_cycle)
        plt.plot(index, center_y, label=label, linestyle=linestyle, marker='x', color=color)

    plt.xlabel('Index', fontsize=14)
    plt.ylabel('Center Y Position', fontsize=14)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

# List of HDF5 file paths
# h5_files = [
#     'data/file1.h5',
#     'data/file2.h5',
#     'data/file3.h5'
# ]


# Optional labels for each file
# labels = ['Sample 1', 'Sample 2', 'Sample 3']

# Define the indices you want to plot
# selected_indices = list(range(0, 100)) 

# Call the plotting function
# plot_center_positions(
#     h5_paths=h5_files,
#     selected_indices=selected_indices,
#     # labels=labels,
#     # save_path='selected_indices_plot.png'  # Optional: specify a path to save the plot
# )
