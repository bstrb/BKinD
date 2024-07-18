# merge_stats_sections.py

# Standard Library Imports
import os

def merge_stats_sections(main_directory):
    file_path = os.path.join(main_directory, "filtering_stats.txt")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    with open(file_path, 'r') as file:
        lines = file.readlines()

    merged_data = {}
    current_target = None
    initial_data = []

    # Parse lines and group by Target ASU %, keeping initial data separate
    for line in lines:
        line = line.strip()
        if (line.startswith("Original Data Count:") or
            line.startswith("Initial Completeness:") or
            line.startswith("Optimal Instability Factor (u):") or
            line.startswith("Space Group Number:") or
            line.startswith("Unit Cell Parameters:") or
            line.startswith("Filtering")):
            initial_data.append(line)
        elif line.startswith("Target Completeness:"):
            current_target = line.split(':')[1].strip()  # Extract just the percentage
            if current_target.endswith('%'):
                current_target = current_target[:-1].strip()  # Remove the '%' symbol if present
            if current_target not in merged_data:
                merged_data[current_target] = {
                    "header": line,
                    "content": []
                }
        if current_target and line and not line.startswith("Original"):
            if not line.startswith("Target"):
                merged_data[current_target]["content"].append(line)

    # Format and write the merged data to the file
    with open(file_path, 'w') as file:
        # Write initial overall data first
        file.write("\n".join(initial_data) + "\n")
        file.write("-------------------------\n")
        file.write("-------------------------\n")  # Separator for readability after initial data

        # Write each Target completeness % section formatted
        for _, data in sorted(merged_data.items(), key=lambda x: float(x[0]), reverse=True):
            file.write(data["header"] + "\n")
            file.write("\n".join(data["content"]) + "\n")
            file.write("-------------------------\n")
            file.write("-------------------------\n")  # Separator for readability between sections

