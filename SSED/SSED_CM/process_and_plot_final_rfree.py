import os
import re
import plotly.graph_objects as go
from open_plot import open_plot  # Ensure this function is correctly defined in the open_plot module
from scipy.optimize import curve_fit  # Import curve fitting library
import numpy as np
import json

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
    
def process_and_plot_final_rfree(base_path, fit_exponential=False):
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
    frame_numbers = np.array(frame_numbers, dtype=float)  # Convert to numpy array of type float

    # Plot the data
    fig.add_trace(go.Scatter(
        x=frame_numbers,
        y=rfree_values,
        mode='lines+markers',
        name='Final Rfree'
    ))

    # Fit a decreasing exponential curve if requested
    if fit_exponential:
        def decreasing_exponential_func(x, a, b, c):
            return a * np.exp(-b * x) + c

        # Perform the curve fitting
        try:
            # Set initial guesses automatically based on data
            initial_a = rfree_values[0] - rfree_values[-1]  # Difference between first and last values
            initial_b = 1.0 / (frame_numbers[-1] - frame_numbers[0])  # Estimate decay rate
            initial_c = rfree_values[-1]  # Final value
            initial_guesses = [initial_a, initial_b, initial_c]
            bounds = ([0, 0, 0], [np.inf, np.inf, rfree_values[0]])
            params, _ = curve_fit(decreasing_exponential_func, frame_numbers, rfree_values, p0=initial_guesses, bounds=bounds, maxfev=10000)
            fitted_y = decreasing_exponential_func(frame_numbers, *params)

            # Add the fitted curve to the plot
            fig.add_trace(go.Scatter(
                x=frame_numbers,
                y=fitted_y,
                mode='lines',
                name='Decreasing Exponential Fit',
                line=dict(dash='dash')  # Use dashed lines for the fit
            ))

        except RuntimeError:
            print("Exponential fit failed to converge.")
        except OverflowError:
            print("Overflow error encountered during exponential fitting.")
        except TypeError as e:
            print(f"Type error encountered: {e}")
        except ValueError as e:
            print(f"Value error encountered: {e}")

    # Update the figure layout
    fig.update_layout(
        title=f'Final Rfree vs Frames for data in {base_path}',
        xaxis_title='Indexed Frames',
        yaxis_title='Final Rfree',
        template='plotly_dark',
        showlegend=True
    )
    
    # Save the plot as HTML
    plotname = "Final_Rfree_vs_Frames.html"
    plot_filename = os.path.join(base_path, plotname)
    fig.write_html(plot_filename)

    # Save the figure as JSON
    fig_json = fig.to_json()
    plotname_json = "Final_Rfree_vs_Frames.json"
    plot_filename_json = os.path.join(base_path, plotname_json)
    with open(plot_filename_json, 'w') as json_file:
        json_file.write(fig_json)

    # Optionally open the plot
    open_plot(fig, plot_filename)

# # Usage example
# base_path = "/home/buster/UOXm/5x5_0-01/fast_int_3-4-7/high_rmsd_removed_20_x_5.0_percentage_units"
base_path = "/home/buster/UOXm/5x5_0-01/fast_int_3-4-7/chunked_1000"
process_and_plot_final_rfree(base_path, fit_exponential=True)
