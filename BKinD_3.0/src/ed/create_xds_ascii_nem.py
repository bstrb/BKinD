# create_xds_ascii_nem.py

# Standard library imports
import os

def format_line(parts, column_widths):
    # Format each part according to specified widths
    formatted_parts = []
    for part, width in zip(parts, column_widths):
        formatted_part = f"{part:>{width}}"
        formatted_parts.append(formatted_part)
    return ''.join(formatted_parts)

def create_xds_ascii_nem(xds_dir, target_dir):
    xds_path = os.path.join(xds_dir, 'XDS_ASCII.HKL')
    integrate_path = os.path.join(xds_dir, 'INTEGRATE.HKL')
    output_path = os.path.join(target_dir, 'XDS_ASCII_NEM.HKL')

    # Define column widths for output formatting
    column_widths = [6, 6, 6, 11, 11, 8, 8, 9, 10, 4, 4, 8]

    try:
        with open(xds_path, 'r') as file:
            xds_lines = [line.strip() for line in file if not line.startswith('!')]

        with open(integrate_path, 'r') as file:
            integrate_lines = [line.strip() for line in file if not line.startswith('!')]

        # Map Miller indices from INTEGRATE.HKL to their corresponding entries
        integrate_dict = {}
        for line in integrate_lines:
            parts = line.split()
            miller_indices = tuple(parts[:3])  # first three columns are Miller indices
            integrate_values = [float(part) for part in parts]
            integrate_dict[miller_indices] = integrate_values

        # Prepare the output data including headers
        output_data = []
        with open(xds_path, 'r') as file:
            for line in file:
                if line.startswith('!'):
                    output_data.append(line)  # Preserve headers
                else:
                    parts = line.split()
                    miller_indices = tuple(parts[:3])
                    if miller_indices in integrate_dict and len(parts) > 4:
                        # Apply the given formula to modify the 5th column
                        try:
                            xds_value = float(parts[3])
                            integrate_values = integrate_dict[miller_indices]
                            calculated_value = (xds_value / integrate_values[3]) * integrate_values[4]
                            parts[4] = f"{calculated_value:.3E}"  # Modify the 5th column with scientific notation
                        except (IndexError, ZeroDivisionError, ValueError):
                            pass  # Skip modification if any error occurs
                    output_line = format_line(parts, column_widths)
                    output_data.append(output_line + '\n')

        # Write to the new file with appropriate formatting
        with open(output_path, 'w') as file:
            file.writelines(output_data)

        # print(f"File created successfully at {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
