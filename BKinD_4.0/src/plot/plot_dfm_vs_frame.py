# plot_DFM_vs_Frame.py

# Standard library imports
import os
import pandas as pd

# Third-party imports
import plotly.express as px

# Plot Imports
from plot.open_plot import open_plot
from plot.append_percentage import append_percentage

def plot_DFM_vs_Frame(output_folder):
    """
    Creates an interactive plot of 'DFM' vs 'zobs' from all CSV files in the specified folder,
    allowing for dynamic color changes and prioritization of data sets.

    Parameters:
    - output_folder: str, path to the base folder where the 'aggregated_filtered' folder is located.
    """

    folder_path = os.path.join(output_folder, "aggregated_filtered")

    data_list = []
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    for file in files:
        full_path = os.path.join(folder_path, file)
        try:
            df = pd.read_csv(full_path)
            if 'DFM' not in df.columns or 'zobs' not in df.columns:
                print(f"Warning: 'DFM' or 'zobs' not found in {file}. Skipping this file.")
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

    # Create an interactive scatter plot using Plotly
    fig = px.scatter(combined_df, x='zobs', y='DFM', color='Source', 
                     title='DFM vs Frame Number for Filtered Data. Click the labels in the legend to hide/unhide data sets.', 
                     labels={'zobs': 'Frame', 'DFM': 'DFM'},
                     category_orders={"Source": sorted(combined_df['Source'].unique(), reverse=True)})
    
    fig.update_traces(marker=dict(size=8, line=dict(width=0,
                                                    color='DarkSlateGrey')),
                      selector=dict(mode='markers'))

        
    # # Add interactivity features
    fig.update_layout(
        title=dict(font=dict(size=32)),
        xaxis=dict(title=dict(font=dict(size=32)), tickfont=dict(size=24)),
        yaxis=dict(title=dict(font=dict(size=32)), tickfont=dict(size=24)),
        legend=dict(title=dict(font=dict(size=32)), font=dict(size=24)),
        legend_title_text='   Data Sets'
    )
    
    plotname="DFM_vs_Frame.html"

    # Save the plot as an HTML file
    plot_filename = os.path.join(os.path.dirname(folder_path), plotname)
    fig.write_html(plot_filename)

    open_plot(fig, plot_filename)
