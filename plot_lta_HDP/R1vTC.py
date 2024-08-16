# %%

import matplotlib.pyplot as plt
import re

def extract_data_from_file(file_path):
    target_completeness = []
    highest_diff_peak = []
    
    with open(file_path, 'r') as file:
        for line in file:
            # Extract Target Completeness
            if "Target Completeness" in line:
                completeness = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
                target_completeness.append(completeness)
            
            # Extract Highest Diff Peak value
            if "Highest Diff Peak:" in line:
                diff_peak_match = re.search(r"Highest Diff Peak:\s*([-+]?\d*\.\d+|\d+)", line)
                if diff_peak_match:
                    diff_peak = float(diff_peak_match.group(1))
                    highest_diff_peak.append(diff_peak)
    
    return target_completeness, highest_diff_peak

def plot_data(datasets, labels, annotations):
    plt.figure(figsize=(10, 6))
    
    for i, data in enumerate(datasets):
        target_completeness, highest_diff_peak = data
        plt.plot(target_completeness, highest_diff_peak, marker='o', label=labels[i])
    
    # Reversing the x-axis
    plt.gca().invert_xaxis()
    
    # Adding annotations
    arrowprops = dict(arrowstyle="->", color='black', lw=2)
    
    for annotation in annotations:
        label, xy, xytext = annotation
        plt.annotate(label, xy=xy, xytext=xytext, arrowprops=arrowprops, fontsize=10, ha='center', va='center')
    
    plt.xlabel('Target Completeness (%)')
    plt.ylabel('Highest Diff Peak')
    plt.title('Highest Diff Peak vs Target Completeness for Multiple Datasets')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
file_paths = ['LTA1.txt', 'LTA2.txt', 'LTA3.txt', 'LTA4.txt', 'LTA_SCXRD.txt']
labels = ['LTA1', 'LTA2', 'LTA3', 'LTA4', 'LTA-SCXRD']

# Extract data for each dataset
datasets = [extract_data_from_file(file_path) for file_path in file_paths]

# Annotations (You can adjust these based on the content)
annotations = [
    ('LTA1: Structure Solvable for\nRemoved Data with 59.0% Completeness', (97, 0.485), (97, 0.6)),
    ('LTA2: Structure Solvable for\nRemoved Data with 67.5% Completeness', (96.7, 0.273), (94.5, 0.4)),
    ('LTA3: Structure Solvable for\nRemoved Data with 83.3% Completeness', (95.3, 0.257), (93, 0.35)),
    ('LTA4: Structure Solvable for\nRemoved Data with 89.5% Completeness', (93.8, 0.302), (91.7, 0.4)),
    ('LTA-SCXRD: Structure Solvable for\nRemoved Data with 43.2% Completeness', (98.8, 0.485), (97.8, 0.6))
]

annotations = []
# Plot the data
plot_data(datasets, labels, annotations)

# %%
