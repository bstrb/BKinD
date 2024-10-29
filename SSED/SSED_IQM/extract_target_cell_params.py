import re

# Function to extract target unit cell parameters from the header
def extract_target_cell_params(header):
    target_cell_params_match = re.search(r'a = ([\d.]+) A\nb = ([\d.]+) A\nc = ([\d.]+) A\nal = ([\d.]+) deg\nbe = ([\d.]+) deg\nga = ([\d.]+) deg', header)
    if target_cell_params_match:
        target_a, target_b, target_c = map(float, target_cell_params_match.groups()[:3])
        target_al, target_be, target_ga = map(float, target_cell_params_match.groups()[3:])
        return target_a, target_b, target_c, target_al, target_be, target_ga
    else:
        raise ValueError("Target unit cell parameters not found in the header.")
