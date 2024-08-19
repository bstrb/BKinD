# %%

import matplotlib.pyplot as plt
import re
import os

def extract_data_from_file(file_path):
    target_completeness = []
    highest_diff_peak = []
    
    with open(file_path, 'r') as file:
        for line in file:
            # Extract Target Completeness
            if "Target Completeness:" in line:
                completeness = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
                target_completeness.append(completeness)
            
            # Extract Highest Diff Peak
            if "Highest Diff Peak:" in line:
                diff_peak = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
                highest_diff_peak.append(diff_peak)
    
    return target_completeness, highest_diff_peak

def plot_data(datasets, labels):
    plt.figure(figsize=(10, 6))
    
    for i, data in enumerate(datasets):
        target_completeness, highest_diff_peak = data
        plt.plot(target_completeness, highest_diff_peak, marker='o', label=labels[i])
    
    plt.gca().invert_xaxis()  # Reverse the x-axis so higher completeness is on the left
    plt.xlabel('Target Completeness (%)')
    plt.ylabel('Highest Diff Peak')
    plt.title('Highest Diff Peak vs Target Completeness')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
file_paths = ['LTA4.txt','LTA4_atom_removed.txt']  # Add more file names if you have multiple datasets
labels = ['LTA4','LTA4 atom removed']  # Corresponding labels for each file

# Extract data for each dataset
datasets = [extract_data_from_file(file_path) for file_path in file_paths]

# Plot the data
plot_data(datasets, labels)

# %%
