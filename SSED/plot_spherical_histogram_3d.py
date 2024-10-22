import numpy as np
import plotly.graph_objects as go
from open_plot import open_plot
import re
import os

# Function to parse orientation matrices from the stream file
def parse_stream_file(stream_file_path):
    with open(stream_file_path, 'r') as file:
        content = file.read()

    # Regular expression to match orientation matrices
    chunk_pattern = re.compile(r'----- Begin chunk -----.*?--- Begin crystal.*?astar = (.*?) nm\^\-1\nbstar = (.*?) nm\^\-1\ncstar = (.*?) nm\^\-1\n', re.S)
    matches = chunk_pattern.findall(content)

    orientation_matrices = []
    for match in matches:
        astar = np.array([float(x) for x in match[0].split()])
        bstar = np.array([float(x) for x in match[1].split()])
        cstar = np.array([float(x) for x in match[2].split()])
        orientation_matrices.append((astar, bstar, cstar))

    return orientation_matrices

# Function to create a 3D spherical heatmap of orientation matrices
def plot_spherical_heatmap_3d(orientation_matrices, plot_path=None):
    # Calculate vector sums of a, b, and c
    vector_sums = [astar + bstar + cstar for astar, bstar, cstar in orientation_matrices]

    # Normalize the vector sums to project onto a unit sphere
    points = np.array([v / np.linalg.norm(v) for v in vector_sums])

    # Convert Cartesian coordinates to spherical coordinates
    theta = np.arccos(points[:, 2])  # Polar angle
    phi = np.arctan2(points[:, 1], points[:, 0])  # Azimuthal angle

    # Create a 2D histogram in spherical coordinates
    histogram, theta_edges, phi_edges = np.histogram2d(theta, phi, bins=[36, 72])

    # Calculate bin centers for plotting
    theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2
    phi_centers = (phi_edges[:-1] + phi_edges[1:]) / 2

    # Create meshgrid for spherical coordinates
    theta_grid, phi_grid = np.meshgrid(theta_centers, phi_centers, indexing='ij')

    # Convert spherical coordinates to Cartesian for plotting on the sphere
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)

    # Flatten the histogram for color mapping
    intensity = histogram.flatten()

    # Create a 3D surface plot to represent the heatmap on the sphere
    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        surfacecolor=intensity.reshape(theta_grid.shape),
        colorscale='Viridis',
        cmin=0,
        cmax=intensity.max(),
        opacity=0.8,
        showscale=True
    ))

    # Set labels and title
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ),
        title='3D Spherical Heatmap of Orientation Matrices'
    )

    # Save the 3D spherical heatmap plot as an HTML file
    if plot_path is None:
        plot_path = f"{os.path.dirname(stream_file_path)}/{os.path.basename(stream_file_path).replace('.stream', '_spherical_heatmap_orientation_matrices.html')}"
    fig.write_html(plot_path)

    open_plot(fig, plot_path)

if __name__ == "__main__":
    # Path to the stream file
    stream_file_path = "/home/buster/UOX3/best_results_EMP_1_2_3_-1_1_1.stream"

    # Parse the orientation matrices from the stream file
    orientation_matrices = parse_stream_file(stream_file_path)

    # Plot the 3D spherical heatmap of orientation matrices
    plot_spherical_heatmap_3d(orientation_matrices, plot_path=None)
