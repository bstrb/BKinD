# %%

import matplotlib.pyplot as plt
import re

def extract_data_from_file(file_path):
    resulting_completeness = []
    r1_values = []
    
    with open(file_path, 'r') as file:
        for line in file:
            # Extract Resulting Completeness
            if "Resulting Completeness" in line:
                completeness = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
                resulting_completeness.append(completeness)
            
            # Extract R1 value
            if "R1:" in line:
                r1_match = re.search(r"R1:\s*([-+]?\d*\.\d+|\d+)", line)
                if r1_match:
                    r1 = float(r1_match.group(1))
                    r1_values.append(r1)
    
    return resulting_completeness, r1_values

def plot_data(datasets, labels, annotations):
    plt.figure(figsize=(10, 6))
    
    for i, data in enumerate(datasets):
        resulting_completeness, r1_values = data
        plt.plot(resulting_completeness, r1_values, marker='o', label=labels[i])
    
    # Reversing the x-axis
    plt.gca().invert_xaxis()
    
    # Adding annotations
    arrowprops = dict(arrowstyle="->", color='black', lw=2)
    
    for annotation in annotations:
        label, xy, xytext = annotation
        plt.annotate(label, xy=xy, xytext=xytext, arrowprops=arrowprops, fontsize=10, ha='center', va='center')
    
    plt.xlabel('Resulting Completeness (%)')
    plt.ylabel('R1')
    plt.title('R1 vs Resulting Completeness for Multiple Datasets')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
file_paths = ['FEACAC11.txt','FEACAC12.txt','FEACAC13.txt', 'FEACACm.txt']
labels = ['FEACAC11','FEACAC12','FEACAC13', 'FEACAC merged']

# Extract data for each dataset
datasets = [extract_data_from_file(file_path) for file_path in file_paths]

# Annotations (You can adjust these based on the content)
annotations = [
    ('FEACAC11: Structure Solvable for\nRemoved Data with 50.3% Completeness', (73, 0.11), (74, 0.04)),
    ('FEACAC12: Structure Solvable for\nRemoved Data with 49.5% Completeness', (78.5, 0.12), (78.2, 0.065)),
    ('FEACAC13: Structure Solvable for\nRemoved Data with 37.7% Completeness', (83.8, 0.135), (84.5, 0.09)),
    ('FEACAC merged: Structure Solvable for\nRemoved Data with 74.6% Completeness', (91, 0.1579), (85, 0.19))
]

# Plot the data
plot_data(datasets, labels, annotations)

# %%
