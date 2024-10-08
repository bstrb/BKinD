
# Standard library imports
import platform
import subprocess

from plotly.subplots import make_subplots
import plotly.io as pio

from pathlib import Path


def open_plot(fig, plot_filename):
    os_name = platform.system()
    
    if os_name == 'Darwin':
        # macOS: Just show the figure
        fig.show()
    elif os_name == 'Linux' and 'microsoft' in platform.uname().release.lower():
        # For WSL
        windows_path = plot_filename
        
        # Check if the path is under /mnt (e.g., /mnt/c)
        if windows_path.startswith('/mnt/'):
            # Convert /mnt/c/path/to/file to C:\path\to\file
            drive_letter = windows_path[5].upper()  # Extract the drive letter
            windows_path = f'{drive_letter}:\\' + windows_path[7:].replace('/', '\\')
        elif windows_path.startswith('/home'):
            # Convert /home/user/path to \\wsl.localhost\Ubuntu\home\user\path
            windows_path = r'\\\wsl.localhost\Ubuntu' + windows_path.replace('/', '\\')

        # Use Windows command to open the file
        # Make sure the path is enclosed in double quotes
        command = f'cmd.exe /c start \"{windows_path}\"'
        try:
            subprocess.run(command, shell=True, check=True, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"Failed to open the plot. Error: {e.stderr.decode()}")
    else:
        print("Unsupported OS. This script supports only macOS and WSL.")


def combine_and_plot_jsons(json_paths, combined_plot_path, title = None, subplot_titles=None, x_axis_titles=None, y_axis_title="Y-axis"):
    # If titles are not provided, generate default ones
    if subplot_titles is None:
        subplot_titles = [f"Plot {i + 1}" for i in range(len(json_paths))]
    if x_axis_titles is None:
        x_axis_titles = ["X-axis" for _ in range(len(json_paths))]

    # Create combined figure with subplots
    combined_fig = make_subplots(
        rows=1, cols=len(json_paths),
        subplot_titles=subplot_titles,
        shared_yaxes=True
    )

    # Iterate through the JSON files and add traces to the appropriate subplot
    for index, json_path in enumerate(json_paths):
        path = Path(json_path)
        if not path.is_file():
            print(f"Warning: File {json_path} does not exist. Skipping this plot.")
            continue

        try:
            fig = pio.read_json(json_path)

            # Add each trace to the appropriate subplot
            for trace in fig.data:
                combined_fig.add_trace(trace, row=1, col=index + 1)

            # Update subplot x-axis labels based on user input
            combined_fig.update_xaxes(
                title_text=x_axis_titles[index],
                row=1, col=index + 1
            )

        except Exception as e:
            print(f"Error loading or processing {json_path}: {e}")

    # Update the combined figure layout, including the y-axis title
    combined_fig.update_layout(
        title=title,
        yaxis_title=y_axis_title,  # Set the y-axis title
        template='plotly_dark',
        showlegend=True
    )

    # Write the combined plot to a new HTML file
    combined_fig.write_html(combined_plot_path)
    print(f"Combined plot saved to: {combined_plot_path}")

    # Optionally, open the plot
    open_plot(combined_fig, combined_plot_path)

# Example usage:
plot1 = "/home/buster/UOXm/5x5_0-01/fast_int_3-4-7/chunked_1000/Final_Rfree_vs_Frames.json"
plot2 = "/home/buster/UOXm/5x5_0-01/fast_int_3-4-7/high_rmsd_removed_20_x_5.0_percentage_units/Final_Rfree_vs_Frames.json"
json_paths = [plot1, plot2]
title = 'Rfree vs Indexed Frames and Percentage Lowest RMSD Frames'
combined_plot_path = "/mnt/c/Users/bubl3932/Desktop/combined_plot_subplots.html"

# User-defined titles and x-axis labels
subplot_titles = [
    "Chunked 1000 - Final Rfree vs Indexed Frames", 
    "High RMSD Removed - Final Rfree vs Remaining Percentage Lowest RMSD Frames"
]
x_axis_titles = [
    "Indexed Frames", 
    "Remaining Percentage Lowest RMSD Frames"
]
y_axis_title = "Final Rfree Value"

# Call the function with the y-axis title
combine_and_plot_jsons(json_paths, combined_plot_path, title, subplot_titles, x_axis_titles, y_axis_title)
