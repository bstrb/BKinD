import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import gc
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Use 'Agg' backend to optimize for saving without rendering
matplotlib.use('Agg')

def generate_heatmap(serial_number, group, x_coords, y_coords, rmsd_min, rmsd_max, global_mean_rmsd, output_folder):
    """
    Function to generate and save a heatmap for a given serial number.
    
    Parameters:
    - serial_number (int): The serial number to process.
    - group (pd.DataFrame): The DataFrame group for this serial number.
    - x_coords (list): List of sorted x coordinates.
    - y_coords (list): List of sorted y coordinates.
    - rmsd_min (float): Minimum RMSD value for color scaling.
    - rmsd_max (float): Maximum RMSD value for color scaling.
    - output_folder (str): Folder to save the heatmap.
    """

    # Preallocate 2D arrays with NaN values
    heatmap_rmsd = np.full((len(y_coords), len(x_coords)), np.nan)  # Initialize with NaNs
    
    # Fill the heatmap array with RMSD values
    for _, row in group.iterrows():
        x_index = x_coords.index(row['x_coord'])
        y_index = y_coords.index(row['y_coord'])
        heatmap_rmsd[y_index, x_index] = row['rmsd']  # y_index first due to row-major order

    # Plot the RMSD heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(
        heatmap_rmsd, 
        cmap='viridis', 
        origin='lower', 
        extent=[min(x_coords), max(x_coords), min(y_coords), max(y_coords)],
        vmin=0,  # Set global minimum RMSD value for color scaling
        vmax=2*global_mean_rmsd   # Set global maximum RMSD value for color scaling
    )
    plt.colorbar(label='RMSD', ticks=np.linspace(0, 2*global_mean_rmsd, 10))  # Limit colorbar ticks
    plt.title(f'RMSD Heatmap for Serial Number {serial_number}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.tight_layout()

    # Save the plot to the output folder with reduced DPI for faster saving
    output_path = os.path.join(output_folder, f"rmsd_heatmap_{serial_number}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=100)
    plt.close()  # Close the plot to free up memory

    # return f"Saved heatmap for serial number {serial_number} to {output_path}"
    return

def plot_and_save_rmsd_heatmaps(csv_file_path, max_workers=4):
    """
    Function to plot and save RMSD heatmaps for each serial number in a CSV file with parallel processing.
    
    Parameters:
    - csv_file_path (str): Path to the input CSV file.
    - max_workers (int): Number of parallel workers (default is 4).
    """

    # Get the directory of the input CSV file
    csv_file_dir = os.path.dirname(csv_file_path)
    
    # Hardcode the output folder name and place it in the same directory as the input CSV file
    output_folder = os.path.join(csv_file_dir, "rmsd_heatmaps")
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load the entire CSV data as a DataFrame
    df = pd.read_csv(csv_file_path)
    
    # Remove the minus sign if needed (making all values positive)
    df['x_coord'] = df['x_coord'].abs()
    df['y_coord'] = df['y_coord'].abs()
    
    # Sort and get unique x and y coordinates once globally
    x_coords = sorted(df['x_coord'].unique())
    y_coords = sorted(df['y_coord'].unique())

    # Find the global min and max values for RMSD (for consistent color scaling)
    rmsd_min, rmsd_max = df['rmsd'].min(), df['rmsd'].max()
    global_mean_rmsd = df['rmsd'].mean()  # Calculate the mean RMSD for the entire dataset

    # Group the data by serial_number
    grouped = df.groupby('serial_number')

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for serial_number, group in grouped:
            # Submit each task (heatmap generation) to the pool
            futures.append(
                executor.submit(
                    generate_heatmap, 
                    serial_number, group, x_coords, y_coords, rmsd_min, rmsd_max, global_mean_rmsd, output_folder
                )
            )
        
        # Display progress using tqdm
        for future in tqdm(futures, desc="Generating heatmaps"):
            future.result() # Print the result from each future

    # Clean up memory after processing
    gc.collect()

# Example usage:
# csv_file_path = '/path/to/your/rmsd_data.csv'
# plot_and_save_rmsd_heatmaps(csv_file_path, max_workers=4)  # Adjust max_workers based on available CPU cores
