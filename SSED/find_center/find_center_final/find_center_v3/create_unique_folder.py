from pathlib import Path

def create_unique_folder(base_dir, folder_name="beam_centers"):
    base_path = Path(base_dir)
    path = base_path / folder_name
    counter = 1
    while path.exists():
        path = base_path / f"{folder_name}{counter}"
        counter += 1
    path.mkdir(parents=True)
    print(f"Folder created: {path}")
    return path

# Example usage:
base_directory = "/path/to/your/directory"  # Replace with your target directory
created_folder = create_unique_folder(base_directory)
