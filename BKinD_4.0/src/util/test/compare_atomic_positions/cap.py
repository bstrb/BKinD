import numpy as np
from scipy.optimize import linear_sum_assignment
from itertools import permutations
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

def read_coordinates(file_name, start_keyword, end_keyword):
    coordinates = []
    reading = False

    with open(file_name, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith(end_keyword):
                break
            if reading and line and not line.startswith(start_keyword):
                entries = line.split()
                if len(entries) > 4:  # Ensure there are enough entries in the line
                    try:
                        coords = [float(entries[2]), float(entries[3]), float(entries[4])]
                        coordinates.append(coords)
                    except ValueError:
                        continue
            if line.startswith(start_keyword):
                reading = True

    return np.array(coordinates)

def compare_matrices(matrix1, matrix2, large_cost=1e6):
    n, m = matrix1.shape
    n2 = matrix2.shape[0]
    cost_matrix = np.zeros((n2, n2))

    for i in range(n2):
        for j in range(n2):
            if i < n and j < n:
                diff_matrix = np.minimum(np.abs(matrix1[i] - matrix2[j]), np.abs(1 - matrix1[i] - matrix2[j]))
                cost_matrix[i, j] = np.sum(diff_matrix)
            else:
                # Assign a large finite cost for dummy rows
                cost_matrix[i, j] = large_cost

    return cost_matrix

# Function to drop rows with NaN values
def drop_nan_rows(matrix):
    return matrix[~np.isnan(matrix).any(axis=1)]

def calculate_differences(matrix1, matrix2):
    differences = np.minimum(np.abs(matrix1 - matrix2), np.abs(1 - matrix1 - matrix2))
    differences = drop_nan_rows(differences)
    return differences

def process_files(file_name_res_orig, file_name_res):
    # Read coordinates and extract relevant columns
    coords_res_orig = read_coordinates(file_name_res_orig, 'PLAN', 'HKLF')
    coords_res = read_coordinates(file_name_res, 'PLAN', 'HKLF')

    min_cost = float('inf')
    best_permutation_orig = None
    best_permutation_filtered = None

    # Consider all column permutations
    for perm in permutations(range(3)):
        permuted_coords_res = coords_res[:, perm]

        # Compute cost matrix
        cost_matrix = compare_matrices(coords_res_orig, permuted_coords_res)

        # Solve the assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Get the cost for this permutation
        cost = cost_matrix[row_ind, col_ind].sum()

        if cost < min_cost:
            min_cost = cost
            best_permutation_orig = np.pad(coords_res_orig[row_ind[row_ind < len(coords_res_orig)]], ((0, len(row_ind) - len(coords_res_orig)), (0, 0)), 'constant', constant_values=np.nan)
            best_permutation_filtered = permuted_coords_res[col_ind]

    # Drop rows with NaN values
    best_permutation_orig = drop_nan_rows(best_permutation_orig)
    best_permutation_filtered = drop_nan_rows(best_permutation_filtered)

    # Calculate differences
    differences = calculate_differences(best_permutation_orig, best_permutation_filtered)

    # Display results
    results = (
        f"Minimum cost (sum of minimized differences): {min_cost}\n"
        f"Original .res coords:\n{best_permutation_orig}\n"
        f"Best fit after filtering .res coords:\n{best_permutation_filtered}\n"
        f"Differences between original and best fit:\n{differences}\n"
        f"Mean of differences:\n{np.mean(differences)}"
    )
    return results

def browse_file(entry):
    file_path = filedialog.askopenfilename()
    entry.delete(0, tk.END)
    entry.insert(0, file_path)

def on_process():
    file_name_res_orig = entry_orig.get()
    file_name_res = entry_filtered.get()
    if not file_name_res_orig or not file_name_res:
        messagebox.showerror("Input Error", "Please select both input files.")
        return
    results = process_files(file_name_res_orig, file_name_res)
    text_results.delete("1.0", tk.END)
    text_results.insert(tk.END, results)

# Create the main window
root = tk.Tk()
root.title("bkind_cap GUI")

# Create the input file selection frame
frame_inputs = ttk.Frame(root, padding="10")
frame_inputs.grid(row=0, column=0, sticky=(tk.W, tk.E))

# Original file input
label_orig = ttk.Label(frame_inputs, text="Original .res file:")
label_orig.grid(row=0, column=0, sticky=tk.W)
entry_orig = ttk.Entry(frame_inputs, width=50)
entry_orig.grid(row=0, column=1, padx=5)
button_browse_orig = ttk.Button(frame_inputs, text="Browse", command=lambda: browse_file(entry_orig))
button_browse_orig.grid(row=0, column=2)

# Filtered file input
label_filtered = ttk.Label(frame_inputs, text="Filtered .res file:")
label_filtered.grid(row=1, column=0, sticky=tk.W)
entry_filtered = ttk.Entry(frame_inputs, width=50)
entry_filtered.grid(row=1, column=1, padx=5)
button_browse_filtered = ttk.Button(frame_inputs, text="Browse", command=lambda: browse_file(entry_filtered))
button_browse_filtered.grid(row=1, column=2)

# Process button
button_process = ttk.Button(root, text="Process", command=on_process)
button_process.grid(row=1, column=0, pady=10)

# Results text box
text_results = tk.Text(root, width=100, height=20)
text_results.grid(row=2, column=0, pady=10)

# Start the main event loop
root.mainloop()
