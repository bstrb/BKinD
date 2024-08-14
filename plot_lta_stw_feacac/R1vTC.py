import matplotlib.pyplot as plt
import re

def extract_data_from_file(file_path):
    target_completeness = []
    r1_values = []
    
    with open(file_path, 'r') as file:
        for line in file:
            # Extract Target Completeness
            if "Target Completeness" in line:
                completeness = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
                target_completeness.append(completeness)
            
            # Extract R1 value
            if "R1:" in line:
                r1_match = re.search(r"R1:\s*([-+]?\d*\.\d+|\d+)", line)
                if r1_match:
                    r1 = float(r1_match.group(1))
                    r1_values.append(r1)
    
    return target_completeness, r1_values

def plot_data(datasets, labels, annotations):
    plt.figure(figsize=(10, 6))
    
    for i, data in enumerate(datasets):
        target_completeness, r1_values = data
        plt.plot(target_completeness, r1_values, marker='o', label=labels[i])
    
    # Reversing the x-axis
    plt.gca().invert_xaxis()
    
    # Adding annotations
    arrowprops = dict(arrowstyle="->", color='black', lw=2)
    
    for annotation in annotations:
        label, xy, xytext = annotation
        plt.annotate(label, xy=xy, xytext=xytext, arrowprops=arrowprops, fontsize=10, ha='center', va='center')
    
    plt.xlabel('Target Completeness (%)')
    plt.ylabel('R1')
    plt.title('R1 vs Target Completeness for Multiple Datasets')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
file_paths = ['LTA1.txt', 'STW1.txt', 'FEACACm.txt']
labels = ['LTA1', 'STW1', 'FEACAC merged']

# Extract data for each dataset
datasets = [extract_data_from_file(file_path) for file_path in file_paths]

# Annotations (You can adjust these based on the content)
annotations = [
('LTA1: Structure Solvable for\nRemaining Data with 64.3% Completeness', (96.9, 0.1396), (96, 0.32)),
    ('LTA2: Structure Solvable for\nRemaining Data with 70.0% Completeness', (96.5, 0.1331), (91, 0.28)),
    ('LTA3: Structure Solvable for\nRemaining Data with 81.7% Completeness', (95.5, 0.1447), (87, 0.23)),
    ('LTA4: Structure Solvable for\nRemaining Data with 88.2% Completeness', (94.0, 0.1684), (85, 0.19)),
    ('FEACAC13: Structure Solvable for\nRemaining Data with 40.6% Completeness', (83.5, 0.135), (81.5, 0.16)),
    ('FEACAC merged: Structure Solvable for\nRemaining Data with 74.6% Completeness', (90.0, 0.08), (83.7, 0.07))
]

annotations = []

# Plot the data
plot_data(datasets, labels, annotations)
