"""
Module: plot_center_positions

This module provides a function to plot center_x and center_y positions from multiple HDF5 files
against frame numbers.

Functions:
    plot_center_positions(h5_paths, labels=None, save_path=None)
"""

import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle

def read_center_data(h5_file):
    """
    Reads center_x and center_y datasets from the given HDF5 file.

    Parameters:
        h5_file (str): Path to the HDF5 file.

    Returns:
        tuple: (center_x, center_y) as numpy arrays.

    Raises:
        KeyError: If the required datasets are not found.
        IOError: If the file cannot be opened.
    """
    try:
        with h5py.File(h5_file, 'r') as f:
            center_x = f['entry/data/center_x'][:]
            center_y = f['entry/data/center_y'][:]
    except KeyError as e:
        raise KeyError(f"Dataset not found in {h5_file}: {e}")
    except Exception as e:
        raise IOError(f"Error reading {h5_file}: {e}")
    
    return center_x, center_y

def plot_center_positions(h5_paths, labels=None, save_path=None):
    """
    Plots center_x and center_y positions vs frame number for multiple HDF5 files.

    Parameters:
        h5_paths (list of str): List of paths to HDF5 files.
        labels (list of str, optional): List of labels for each HDF5 file. If None, file names are used.
        save_path (str, optional): If provided, saves the plot to the given path instead of displaying it.

    Raises:
        ValueError: If the lengths of h5_paths and labels do not match.
    """
    if labels and len(labels) != len(h5_paths):
        raise ValueError("Length of labels must match length of h5_paths.")

    file_data = {}
    for idx, h5_file in enumerate(h5_paths):
        if not os.path.isfile(h5_file):
            print(f"Warning: File not found - {h5_file}. Skipping.")
            continue
        try:
            center_x, center_y = read_center_data(h5_file)
            label = labels[idx] if labels else os.path.basename(h5_file)
            file_data[label] = (center_x, center_y)
            print(f"Successfully read {h5_file}")
        except Exception as e:
            print(f"Error reading {h5_file}: {e}")

    if not file_data:
        print("No valid data to plot.")
        return

    plt.figure(figsize=(14, 8))

    # Define color and linestyle cycles for better differentiation
    color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    linestyle_cycle = cycle(['-', '--', '-.', ':'])

    for label, (center_x, center_y) in file_data.items():
        frames = np.arange(1, len(center_x) + 1)
        color = next(color_cycle)
        linestyle_x = next(linestyle_cycle)
        linestyle_y = next(linestyle_cycle)
        
        plt.plot(frames, center_x, label=f"{label} - center_x", linestyle=linestyle_x, marker='o', color=color)
        plt.plot(frames, center_y, label=f"{label} - center_y", linestyle=linestyle_y, marker='x', color=color)

    plt.xlabel('Frame Number', fontsize=14)
    plt.ylabel('Center Position', fontsize=14)
    plt.title('Center X and Y Positions vs Frame', fontsize=16)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
