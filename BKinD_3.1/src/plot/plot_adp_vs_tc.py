# plot_adp_vs_tc.py
import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict

def extract_values_from_res(file_path):
    values = {}
    with open(file_path, 'r') as file:
        for line in file:
            if re.match(r'^[A-Za-z]+\d+', line) and not line.startswith("Q"):
                parts = line.split()
                atom_name = parts[0]
                last_value = float(parts[-1])
                values[atom_name] = last_value
    return values

def plot_results(data):
    plt.figure(figsize=(10, 6))
    
    for atom_name, values in data.items():
        plt.scatter(values['completeness'], values['values'], label=atom_name)
    
    plt.xlabel('Target Completeness (%)')
    plt.ylabel('ADP (isotropic radius)')
    plt.title('Atomic Displacement Parameters vs Target Completeness')
    plt.legend()
    plt.grid(True)
    plt.gca().invert_xaxis()  # Reverse the x-axis
    plt.show()

def main(folder_path):
    data = defaultdict(lambda: {'completeness': [], 'values': []})
    
    for subfolder in os.listdir(folder_path):
        match = re.match(r'filtered_(\d+\.\d+)', subfolder)
        if match:
            completeness = float(match.group(1))
            res_file = os.path.join(folder_path, subfolder, f'removed_data_{completeness:.1f}', f'removed_data_{completeness:.1f}.res')
            
            if os.path.exists(res_file):
                values = extract_values_from_res(res_file)
                for atom_name, value in values.items():
                    data[atom_name]['completeness'].append(completeness)
                    data[atom_name]['values'].append(value)
    
    plot_results(data)

if __name__ == "__main__":
    folder_path = input("Enter the path to the folder: ")
    main(folder_path)
