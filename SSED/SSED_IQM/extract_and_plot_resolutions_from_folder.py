import os
import re
import plotly.graph_objects as go
from open_plot import open_plot

def extract_resolutions_and_plot(input_stream_file):
    event_numbers = []
    peak_resolutions = []
    diffraction_resolutions = []
    resolution_differences = []

    current_event = None
    current_peak_resolution = None
    current_diffraction_resolution = None

    try:
        with open(input_stream_file, 'r') as stream_file:
            for line in stream_file:
                if line.startswith('Event:'):
                    current_event = int(re.search(r'//(\d+)', line).group(1))
                elif 'peak_resolution' in line:
                    match = re.search(r'or ([\d\.]+) A', line)
                    if match:
                        current_peak_resolution = float(match.group(1))
                elif 'diffraction_resolution_limit' in line:
                    match = re.search(r'or ([\d\.]+) A', line)
                    if match:
                        current_diffraction_resolution = float(match.group(1))
                elif line.startswith('----- End chunk -----'):
                    if current_event is not None and current_peak_resolution is not None and current_diffraction_resolution is not None:
                        event_numbers.append(current_event)
                        peak_resolutions.append(current_peak_resolution)
                        diffraction_resolutions.append(current_diffraction_resolution)
                        resolution_differences.append(current_diffraction_resolution - current_peak_resolution)
                    # Reset for next chunk
                    current_event = None
                    current_peak_resolution = None
                    current_diffraction_resolution = None

    except Exception as e:
        print(f"An exception occurred while extracting resolutions from {input_stream_file}: {e}")

    # Plotting peak resolution, diffraction resolution, and their difference against event number
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=event_numbers, y=peak_resolutions, mode='markers', name='Peak Resolution (Å)'))
    fig.add_trace(go.Scatter(x=event_numbers, y=diffraction_resolutions, mode='markers', name='Diffraction Resolution Limit (Å)'))
    fig.add_trace(go.Scatter(x=event_numbers, y=resolution_differences, mode='markers', name='Difference (Diffraction - Peak) (Å)'))
    fig.update_layout(
        title='Peak and Diffraction Resolution vs Event Number',
        xaxis_title='Event Number',
        yaxis_title='Resolution (Å)',
        legend_title='Resolution Types'
    )

    # Save plot to HTML and open it
    plotname = "peak_diff_res.html"
    plot_filename = os.path.join(os.path.dirname(input_stream_file), plotname)
    fig.write_html(plot_filename)
    print(f"Plot saved as {plot_filename}")
    # os.system(f"xdg-open {plot_filename}")  # Adjust as needed for your environment
    open_plot(fig, plot_filename)

# Function to extract resolutions from all stream files in a folder and plot
def extract_and_plot_resolutions_from_folder(input_folder):
    try:
        for file_name in os.listdir(input_folder):
            if file_name.endswith('.stream'):
                file_path = os.path.join(input_folder, file_name)
                extract_resolutions_and_plot(file_path)
    except Exception as e:
        print(f"An exception occurred while processing the folder {input_folder}: {e}")

# Example usage
if __name__ == "__main__":
    input_folder = '/home/buster/UOX123/EMP_1_2_3_-1_1_1'  # Replace with your actual folder path
    extract_and_plot_resolutions_from_folder(input_folder)
