# plot_DFM_distribution.py

# Standard library imports
import os

# Third-party imports
import plotly.express as px
import pandas as pd

# Plot Imports
from plot.open_plot import open_plot
from plot.append_percentage import append_percentage

def plot_DFM_distribution(output_folder, bin_size=1000):
    """
    Creates an interactive plot of the distribution of 'DFM' from all CSV files in the specified folder.

    Parameters:
    - output_folder: str, path to the base folder where the 'aggregated_filtered' folder is located.
    - bin_size: int, the size of the bins to use for the histogram.
    """

    folder_path = os.path.join(output_folder, "aggregated_filtered")

    data_list = []
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    for file in files:
        full_path = os.path.join(folder_path, file)
        try:
            df = pd.read_csv(full_path)
            if 'DFM' not in df.columns:
                print(f"Warning: 'DFM' not found in {file}. Skipping this file.")
                continue

            # Label the data by its source file
            source_label = file.replace('.csv', '').replace('_', ' ')
            source_label = source_label.replace('filtered', 'Filtered to')
            source_label = append_percentage(source_label).title()
            df['Source'] = source_label
            
            # Exclude columns where all entries are NA before adding to the list
            df = df.dropna(axis=1, how='all')
            data_list.append(df)
        except Exception as e:
            print(f"Failed to load {file}: {e}")
            continue

    if not data_list:
        print("No valid data found to plot.")
        return

    # Concatenate the cleaned DataFrames
    combined_df = pd.concat(data_list, ignore_index=True)

    # Create an interactive histogram using Plotly
    fig = px.histogram(combined_df, x='DFM', color='Source',
                       title='Distribution of DFM from all CSV files',
                       labels={'DFM': 'DFM'},
                       nbins=bin_size,
                       category_orders={"Source": sorted(combined_df['Source'].unique(), reverse=True)})
    
    fig.update_traces(opacity=0.75)

    # Add interactivity features
    fig.update_layout(
        title=dict(font=dict(size=24)),
        xaxis=dict(title=dict(font=dict(size=18)), tickfont=dict(size=14)),
        yaxis=dict(title=dict(font=dict(size=18)), tickfont=dict(size=14)),
        legend=dict(title=dict(font=dict(size=18)), font=dict(size=14)),
        barmode='overlay'
    )
    
    plotname="DFM_Distribution.html"

    # Save the plot as an HTML file
    plot_filename = os.path.join(os.path.dirname(folder_path), plotname)
    fig.write_html(plot_filename)

    open_plot(fig, plot_filename)
# %%
