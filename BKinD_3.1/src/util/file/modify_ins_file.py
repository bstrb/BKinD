# modify_ins_file.py

def modify_ins_file(file_path):
    """
    Modifies the .ins file to:
    1. Search lines between those containing 'UNIT' and 'FVAR' (case insensitive).
    2. Remove lines containing 'merg' and 'fmap' (case insensitive).
    3. Replace lines starting with 'list' with 'LIST 4\nMERG 0\nFMAP 2' (case insensitive).

    Parameters:
    - file_path: Path to the .ins file to modify.
    """
    modified_lines = []
    is_between_unit_fvar = False

    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        for line in lines:
            line_lower = line.lower().strip()
            if 'unit' in line_lower:
                is_between_unit_fvar = True
                modified_lines.append(line)
                continue
            elif 'fvar' in line_lower:
                is_between_unit_fvar = False
                modified_lines.append(line)
                continue

            if is_between_unit_fvar:
                if 'merg' in line_lower and 'merge' not in line_lower:
                    continue
                if 'fmap' in line_lower:
                    continue
                if 'acta' in line_lower:
                    continue
                if line_lower.startswith('list'):
                    modified_lines.append('LIST 4\nMERG 0\nFMAP 2\nACTA\n')
                    continue

            modified_lines.append(line)

        # Write the modified lines back to the file
        with open(file_path, 'w') as file:
            file.writelines(modified_lines)

    except Exception as e:
        print(f"An error occurred: {e}")
