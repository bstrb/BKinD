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

def generate_heatmap(event_number, group, x_coords, y_coords, combined_metric_min, combined_metric_max, global_mean_combined_metric, output_folder):
    """
    Function to generate and save a heatmap for a given event number.
    
    Parameters:
    - event_number (int): The event number to process.
    - group (pd.DataFrame): The DataFrame group for this event number.
    - x_coords (list): List of sorted x coordinates.
    - y_coords (list): List of sorted y coordinates.
    - combined_metric_min (float): Minimum combined metric value for color scaling.
    - combined_metric_max (float): Maximum combined metric value for color scaling.
    - output_folder (str): Folder to save the heatmap.
    """

    # Preallocate 2D arrays with NaN values
    heatmap_combined_metric = np.full((len(y_coords), len(x_coords)), np.nan)  # Initialize with NaNs
    
    # Fill the heatmap array with combined metric values
    for _, row in group.iterrows():
        x_index = x_coords.index(row['x_coord'])
        y_index = y_coords.index(row['y_coord'])
        heatmap_combined_metric[y_index, x_index] = row['combined_metric']  # y_index first due to row-major order

    # Plot the combined metric heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(
        heatmap_combined_metric, 
        cmap='viridis', 
        origin='lower', 
        extent=[min(x_coords), max(x_coords), min(y_coords), max(y_coords)],
        vmin=0,  # Set global minimum combined metric value for color scaling
        vmax=2*global_mean_combined_metric   # Set global maximum combined metric value for color scaling
    )
    plt.colorbar(label='Combined Metric', ticks=np.linspace(0, 2*global_mean_combined_metric, 10))  # Limit colorbar ticks
    plt.title(f'Combined Metric Heatmap for Event Number {event_number}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.tight_layout()

    # Save the plot to the output folder with reduced DPI for faster saving
    output_path = os.path.join(output_folder, f"combined_metric_heatmap_{event_number}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=100)
    plt.close()  # Close the plot to free up memory

    return

def plot_and_save_combined_metric_heatmaps(csv_file_path, max_workers=4):
    """
    Function to plot and save combined metric heatmaps for each event number in a CSV file with parallel processing.
    
    Parameters:
    - csv_file_path (str): Path to the input CSV file.
    - max_workers (int): Number of parallel workers (default is 4).
    """

    # Get the directory of the input CSV file
    csv_file_dir = os.path.dirname(csv_file_path)
    
    # Hardcode the output folder name and place it in the same directory as the input CSV file
    output_folder = os.path.join(csv_file_dir, "combined_metric_heatmaps")
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load the entire CSV data as a DataFrame
    df = pd.read_csv(csv_file_path)
    
    # # Remove the minus sign if needed (making all values positive)
    # df[['x_coord', 'y_coord']] = df['stream_file'].str.extract(r'UOX1_([\-0-9\.]+)_([\-0-9\.]+)').replace(r'\.$', '', regex=True).astype(float)

    df[['x_coord', 'y_coord']] = (
        df['stream_file']
        .str.extract(r'([-0-9.]+)_([-0-9.]+)\.stream$')  # Match coordinates right before ".stream"
        .astype(float)
)

    # Sort and get unique x and y coordinates once globally
    x_coords = sorted(df['x_coord'].unique())
    y_coords = sorted(df['y_coord'].unique())

    # Find the global min and max values for combined metric (for consistent color scaling)
    combined_metric_min, combined_metric_max = df['combined_metric'].min(), df['combined_metric'].max()
    global_mean_combined_metric = df['combined_metric'].mean()  # Calculate the mean combined metric for the entire dataset

    # Group the data by event_number
    grouped = df.groupby('event_number')

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for event_number, group in grouped:
            # Submit each task (heatmap generation) to the pool
            futures.append(
                executor.submit(generate_heatmap, event_number, group, x_coords, y_coords, combined_metric_min, combined_metric_max, global_mean_combined_metric, output_folder)
            )
        
        # Display progress using tqdm
        for future in tqdm(futures, desc="Generating combined metric heatmaps"):
            future.result() # Print the result from each future

    # Clean up memory after processing
    gc.collect()

# Example usage:
csv_file_path = "/home/buster/UOX1/UOX1_original/UOX1_original_backup/combined_metrics_IQM_SUM_12_12_10_-12_12_-15_10_13_-13.csv"
plot_and_save_combined_metric_heatmaps(csv_file_path, max_workers=23)  # Adjust max_workers based on available CPU cores
