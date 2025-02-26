import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load your data
csv_path = "/home/bubl3932/files/UOX1/UOX1_original_IQM/combined_metrics_IQM_SUM_10_10_10_-10_10_-10_10_10_-10.csv"

df = pd.read_csv(csv_path)

# Group by event_number and get the minimum combined_metric
grouped = df.groupby("event_number")["combined_metric"].min().values.reshape(-1, 1)

# Apply K-Means with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(grouped)
labels = kmeans.labels_
centers = kmeans.cluster_centers_.flatten()

# Sort the centers
sorted_centers = np.sort(centers)
cutoff = np.mean(sorted_centers)

print(f"Optimal cutoff determined by K-Means: {cutoff:.4f}")

# Plot the histogram with cutoff
plt.figure(figsize=(10,6))
plt.hist(grouped, bins=100, edgecolor='black', alpha=0.6, label='Data')
plt.axvline(cutoff, color='red', linestyle='dashed', linewidth=2, label=f'Cutoff = {cutoff:.2f}')
plt.title("Histogram with K-Means Cutoff")
plt.xlabel("Combined Metric")
plt.ylabel("Frequency")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
