def extract_lattice_info(stream_file_path):
    lattice_type, centering = None, None

    with open(stream_file_path, 'r') as file:
        for line in file:
            if line.startswith('lattice_type'):
                lattice_type = line.split('=')[1].strip()[0].lower()
            elif line.startswith('centering'):
                centering = line.split('=')[1].strip()[0]
            
            if lattice_type and centering:
                break

    return f"{lattice_type}{centering}"

if __name__ == "__main__":
    stream_file_path = '/home/buster/UOX1/different_index_params/5x5_retry/best_results_IQM.stream'
    lattice_info = extract_lattice_info(stream_file_path)
    print(f"Lattice Info: {lattice_info}")
