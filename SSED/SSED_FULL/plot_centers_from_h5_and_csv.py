import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

""""""


# File paths
h5_file_path = "/home/bubl3932/files/UOX1/UOX_His_MUA_450nm_spot4_ON_20240311_0928.h5"
csv_file_path =  "/home/bubl3932/files/UOX1/UOX1_0-02_2nd_index_run/beam_centers.csv"

# Load data from .h5 file
with h5py.File(h5_file_path, 'r') as h5_file:
    center_x_h5 = h5_file['entry/data/center_x'][:]
    center_y_h5 = h5_file['entry/data/center_y'][:]

# Load data from CSV file
csv_data = pd.read_csv(csv_file_path)
center_x_csv = csv_data['Xc_px'].values
center_y_csv = csv_data['Yc_px'].values
image_serial_csv = csv_data['Image_Serial_Number'].values

# Compute median and ranges for plotting
median_Xc = np.median(center_x_csv)
median_Yc = np.median(center_y_csv)
# After the debug prints, ADD these lines:
print("\nCleaning data...")
center_x_h5_clean = center_x_h5[~np.isnan(center_x_h5)]
center_y_h5_clean = center_y_h5[~np.isnan(center_y_h5)]

print("Original H5 shape:", center_x_h5.shape)
print("Cleaned H5 shape:", center_x_h5_clean.shape)

# Use cleaned data for all calculations
combined_x = np.concatenate([center_x_csv, center_x_h5_clean])
combined_y = np.concatenate([center_y_csv, center_y_h5_clean])

std_x = np.std(combined_x)
std_y = np.std(combined_y)

# ... rest of your code, but update the plotting sections to use center_x_h5_clean and center_y_h5_clean ...
# Print debug information
print(f"std_x: {std_x}, std_y: {std_y}")
print(f"median_Xc: {median_Xc}, median_Yc: {median_Yc}")

# Add safety checks to prevent NaN/Inf
plotrange_x = max(std_x/10, 0.5) if not np.isnan(std_x) and not np.isinf(std_x) else 0.8
plotrange_y = max(std_y/10, 0.5) if not np.isnan(std_y) and not np.isinf(std_y) else 0.8


y_min_X = median_Xc - plotrange_x
y_max_X = median_Xc + plotrange_x
y_min_Y = median_Yc - plotrange_y
y_max_Y = median_Yc + plotrange_y

# Plotting X Centers
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(image_serial_csv, center_x_csv, label='CSV Centers', color='red', alpha=0.7)
plt.scatter(image_serial_csv[0] + np.arange(len(center_x_h5)), center_x_h5, label='H5 Centers', alpha=0.5)
plt.xlabel('Image Index / Serial Number')
plt.ylabel('X Center (px)')
plt.title(f'X Centers Comparison\nMedian: {median_Xc:.2f} px, Std: {std_x:.2f} px')
plt.ylim(y_min_X, y_max_X)
plt.legend()
plt.grid(True)

# Plotting Y Centers
plt.subplot(1, 2, 2)
plt.scatter(image_serial_csv, center_y_csv, label='CSV Centers', color='red', alpha=0.7)
plt.scatter(image_serial_csv[0] + np.arange(len(center_y_h5)), center_y_h5, label='H5 Centers', alpha=0.5)
plt.xlabel('Image Index / Serial Number')
plt.ylabel('Y Center (px)')
plt.title(f'Y Centers Comparison\nMedian: {median_Yc:.2f} px, Std: {std_y:.2f} px')
plt.ylim(y_min_Y, y_max_Y)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()