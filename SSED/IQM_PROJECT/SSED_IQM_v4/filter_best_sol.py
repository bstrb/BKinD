import pandas as pd
import sys
import re

# -----------------------------
# User Inputs
# -----------------------------

# Path to the CSV file with combined_metrics
csv_path = "/home/buster/UOX1/UOX1_min_10/CF_intensity_copy5/UOX1_min_10_no_bg_beam_centers19/combined_metrics_IQM_SUM_12_12_10_-12_12_-15_10_13_-13.csv"

# Path to the .sol file to be filtered
sol_file_path = "/home/buster/UOX1/UOX1_min_10/CF_intensity_copy5/UOX1_min_10_no_bg_beam_centers19/best_results_IQM_SUM_12_12_10_-12_12_-15_10_13_-13.sol"

# Path to save the filtered .sol file
filtered_sol_file_path = "/home/buster/UOX1/UOX1_min_10/CF_intensity_copy5/UOX1_min_10_no_bg_beam_centers19/filtered_best_results.sol"

# Cutoff value obtained from K-Means
cutoff_value = 10.5728  # Replace with your actual cutoff value

# -----------------------------
# Load and Process Combined Metrics
# -----------------------------

try:
    # Load the CSV file into a DataFrame
    metrics_df = pd.read_csv(csv_path)
    print("Combined metrics CSV loaded successfully.")
except Exception as e:
    print(f"Error loading CSV file: {e}")
    sys.exit(1)

# Ensure necessary columns exist
required_columns = {'stream_file', 'event_number', 'combined_metric'}
if not required_columns.issubset(metrics_df.columns):
    print(f"CSV file is missing required columns. Required columns: {required_columns}")
    sys.exit(1)

# Group by event_number and find the minimum combined_metric for each
grouped_metrics = metrics_df.groupby("event_number")["combined_metric"].min()
print("Grouped metrics by event_number and extracted minimum combined_metric.")

# Identify event_numbers to keep (combined_metric <= cutoff_value)
events_to_keep = set(grouped_metrics[grouped_metrics <= cutoff_value].index)
print(f"Number of events to keep: {len(events_to_keep)}")

# -----------------------------
# Process the .sol File
# -----------------------------

# Regular expression to extract event_number after '//'
# It assumes the event_number is an integer following '//'
event_number_pattern = re.compile(r'//(\d+)')

# Initialize counters
total_lines = 0
kept_lines = 0
removed_lines = 0

try:
    with open(sol_file_path, 'r') as infile, open(filtered_sol_file_path, 'w') as outfile:
        for line in infile:
            total_lines += 1
            # Search for the event_number using regex
            match = event_number_pattern.search(line)
            if match:
                event_number = int(match.group(1))
                if event_number in events_to_keep:
                    outfile.write(line)
                    kept_lines += 1
                else:
                    removed_lines += 1
            else:
                # If no event_number is found, decide whether to keep or discard
                # Here, we'll keep such lines but you can modify as needed
                outfile.write(line)
                kept_lines += 1
except FileNotFoundError:
    print(f".sol file not found at path: {sol_file_path}")
    sys.exit(1)
except Exception as e:
    print(f"Error processing .sol file: {e}")
    sys.exit(1)

print(f"Total lines processed: {total_lines}")
print(f"Lines kept: {kept_lines}")
print(f"Lines removed: {removed_lines}")
print(f"Filtered .sol file saved to: {filtered_sol_file_path}")
