import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

def plot_rmsd_heatmap_from_csv(csv_file_path):
    # Load CSV data into a DataFrame
    df = pd.read_csv(csv_file_path)
    
    # Get unique frame numbers
    frame_numbers = sorted(df['serial_number'].unique())
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(left=0.1, bottom=0.25)
    
    # Create the initial heatmap
    initial_frame = frame_numbers[0]
    heatmap_data = df[df['serial_number'] == initial_frame]
    
    x_coords = sorted(heatmap_data['x_coord'].unique())
    y_coords = sorted(heatmap_data['y_coord'].unique())
    
    # Create a 2D array for RMSD values
    heatmap_rmsd = np.full((len(y_coords), len(x_coords)), np.nan)
    
    for _, row in heatmap_data.iterrows():
        x_idx = x_coords.index(row['x_coord'])
        y_idx = y_coords.index(row['y_coord'])
        heatmap_rmsd[y_idx, x_idx] = row['rmsd']
    
    # Display the heatmap
    im = ax.imshow(heatmap_rmsd, cmap='viridis', origin='lower',
                   extent=[min(x_coords), max(x_coords), min(y_coords), max(y_coords)])
    plt.colorbar(im, ax=ax, label='RMSD')
    ax.set_title(f'RMSD Heatmap for Frame {initial_frame}')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')

    # Create a slider axis and slider
    ax_slider = plt.axes([0.1, 0.1, 0.8, 0.05], facecolor='lightgoldenrodyellow')
    frame_slider = Slider(ax_slider, 'Frame', frame_numbers[0], frame_numbers[-1], valinit=initial_frame, valstep=1)

    # Update function for the slider
    def update(val):
        frame_number = int(frame_slider.val)
        ax.set_title(f'RMSD Heatmap for Frame {frame_number}')
        
        # Update the heatmap data
        heatmap_data = df[df['serial_number'] == frame_number]
        heatmap_rmsd.fill(np.nan)  # Reset heatmap
        
        for _, row in heatmap_data.iterrows():
            x_idx = x_coords.index(row['x_coord'])
            y_idx = y_coords.index(row['y_coord'])
            heatmap_rmsd[y_idx, x_idx] = row['rmsd']
        
        im.set_data(heatmap_rmsd)
        fig.canvas.draw_idle()

    # Attach the update function to the slider
    frame_slider.on_changed(update)

