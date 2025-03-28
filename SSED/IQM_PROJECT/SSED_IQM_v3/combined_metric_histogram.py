import pandas as pd
import matplotlib.pyplot as plt

# Path to your CSV file
csv_path = "/home/bubl3932/files/UOX1/UOX1_original_IQM/combined_metrics_IQM_SUM_10_12_10_-12_12_-15_10_13_-13.csv"
# Load the CSV file into a DataFrame
df = pd.read_csv(csv_path)

# Ensure the columns are as expected:
# The CSV has the following columns:
# stream_file,event_number,combined_metric
#
# If your CSV has headers and matches these names, this should work directly.
# Otherwise, specify headers or column names appropriately.
#
# df = pd.read_csv(csv_path, names=["stream_file", "event_number", "combined_metric"], header=None) 
# if the CSV does not have headers. But it seems it does.

# Group by event_number and find the minimum combined_metric per event_number
grouped = df.groupby("event_number")["combined_metric"].min()

# `grouped` is now a Series containing the minimal combined_metric value for each event_number
# Plot a histogram of these minimal combined_metric values
plt.figure(figsize=(10,6))
plt.hist(grouped, bins=30, edgecolor='black')
plt.title("Histogram of Minimum Combined Metric per Event Number")
plt.xlabel("Combined Metric")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()
