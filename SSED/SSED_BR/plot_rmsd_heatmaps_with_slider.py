import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import interact, IntSlider

def plot_rmsd_heatmaps_with_slider(csv_file_path):
    # Load the CSV data
    df = pd.read_csv(csv_file_path)
    
    # Remove the minus sign if needed (making all values positive)
    df['x_coord'] = df['x_coord'].abs()
    df['y_coord'] = df['y_coord'].abs()
    
    # Find the maximum and minimum coordinates for the heatmap dimensions
    max_x = df['x_coord'].max()
    min_x = df['x_coord'].min()
    max_y = df['y_coord'].max()
    min_y = df['y_coord'].min()
    
    # Create the range arrays for x and y coordinates
    x_range = np.arange(min_x, max_x + 0.01, 0.01)  # Adjust step size as needed
    y_range = np.arange(min_y, max_y + 0.01, 0.01)
    
    # Group the data by serial_number and store in a dictionary
    grouped = {serial_number: group for serial_number, group in df.groupby('serial_number')}
    serial_numbers = list(grouped.keys())

    # Define a plotting function to be used with the slider
    def plot_heatmap(serial_number_index):
        serial_number = serial_numbers[serial_number_index]
        group = grouped[serial_number]
        
        # Pivot the data to create a matrix suitable for heatmap plotting
        pivot_df = group.pivot_table(index='y_coord', columns='x_coord', values='rmsd', aggfunc='mean')
        
        # Reindex to ensure all heatmaps have the same dimensions, filling NaN with a suitable value (e.g., 0 or np.nan)
        pivot_df = pivot_df.reindex(index=y_range, columns=x_range)
        
        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_df, cmap="viridis", annot=False, fmt=".2f", cbar_kws={'label': 'RMSD'})
        plt.title(f"Heatmap of RMSD Values for Serial Number {serial_number}")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        
        # Show the heatmap
        plt.show()
    
    # Create an interactive slider
    interact(plot_heatmap, serial_number_index=IntSlider(min=0, max=len(serial_numbers) - 1, step=1, value=0))

# Example usage:
# plot_rmsd_heatmaps_with_slider("path_to_your_csv_file.csv")
