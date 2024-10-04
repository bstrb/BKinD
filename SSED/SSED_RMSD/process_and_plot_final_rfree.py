# process_and_plot_final_rfree.py

import os
import re
import plotly.graph_objects as go
from open_plot import open_plot  # Ensure this function is correctly defined in the open_plot module

def extract_final_rfree(content):
    """
    Extracts the final Rfree value from the result section at the bottom of the output file.
    """
    # Regular expression pattern to match the final Rfree line
    rfree_pattern = re.compile(r'R free\s+\d+\.\d+\s+(\d+\.\d+)', re.MULTILINE)

    # Search for the pattern
    match = rfree_pattern.search(content)
    
    if match:
        final_rfree = float(match.group(1))
        return final_rfree
    else:
        print("No final Rfree value found.")
        return None
    
def process_and_plot_final_rfree(base_path):
    fig = go.Figure()
    frame_numbers = []
    rfree_values = []

    # Walk through the directory to find the .txt file
    for root, dirs, files in os.walk(base_path):
        txt_files = [f for f in files if f.endswith('.txt')]
        
        if len(txt_files) == 1:
            file_path = os.path.join(root, txt_files[0])
            print(f"Processing file: {file_path}")

            # Extract just the numeric value from the folder name
            folder_name = os.path.basename(os.path.dirname(file_path))
            frame_number_match = re.search(r'\d+', folder_name)

            if frame_number_match:
                frame_number = int(frame_number_match.group(0))

                with open(file_path, 'r') as file:
                    content = file.read()

                final_rfree = extract_final_rfree(content)

                if final_rfree is not None:
                    frame_numbers.append(frame_number)
                    rfree_values.append(final_rfree)

    # Sort the data based on frame numbers
    sorted_data = sorted(zip(frame_numbers, rfree_values))
    frame_numbers, rfree_values = zip(*sorted_data)

    # Plot the data
    fig.add_trace(go.Scatter(
        x=frame_numbers,
        y=rfree_values,
        mode='lines+markers',
        name='Final Rfree'
    ))

    fig.update_layout(
        title=f'Final Rfree vs Percentage lowest RMSD for data in {base_path}',
        xaxis_title='Percentage lowest RMSD',
        yaxis_title='Final Rfree',
        template='plotly_dark',
        showlegend=True
    )
    
    plotname = "Final_Rfree_vs_Frames.html"
    plot_filename = os.path.join(base_path, plotname)
    fig.write_html(plot_filename)

    open_plot(fig, plot_filename)


# # Usage example
# base_path = "/home/buster/UOX1/fast_int/MERGE_4-5-9"
# process_and_plot_final_rfree(base_path)
