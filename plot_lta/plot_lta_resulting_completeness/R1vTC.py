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
file_paths = ['LTA1.txt', 'LTA2.txt', 'LTA3.txt', 'LTA4.txt']
labels = ['LTA1', 'LTA2', 'LTA3', 'LTA4']

# Extract data for each dataset
datasets = [extract_data_from_file(file_path) for file_path in file_paths]

# Annotations (You can adjust these based on the content)
annotations = [
    ('LTA1: Structure Solvable for\nRemoved Data with 59.0% Completeness', (97, 0.1396), (96.9, 0.32)),
    ('LTA2: Structure Solvable for\nRemoved Data with 67.5% Completeness', (96.7, 0.1331), (94.5, 0.28)),
    ('LTA3: Structure Solvable for\nRemoved Data with 83.3% Completeness', (95.3, 0.1447), (93, 0.23)),
    ('LTA4: Structure Solvable for\nRemoved Data with 89.5% Completeness', (93.8, 0.1684), (91.7, 0.19))
]

# Plot the data
plot_data(datasets, labels, annotations)

# %%
