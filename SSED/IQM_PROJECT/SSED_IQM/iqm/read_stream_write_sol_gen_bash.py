import os

from generate_bash_script import generate_bash_script

def process_block(block, output_file, lattice):
    try:
        image = ''
        event = ''
        det_shift_x = ''
        det_shift_y = ''
        astar_values = ''
        bstar_values = ''
        cstar_values = ''

        for line in block:
            if line.startswith('Image filename:'):
                image = line.split(':', 1)[1].strip()
            elif line.startswith('Event:'):
                event = line.split(':', 1)[1].strip()
            elif 'predict_refine/det_shift' in line:
                det_shifts = line.split()
                if len(det_shifts) >= 7:
                    det_shift_x = det_shifts[3]
                    det_shift_y = det_shifts[6]
            elif 'astar' in line:
                astar_values = ' '.join(line.split()[2:5])
            elif 'bstar' in line:
                bstar_values = ' '.join(line.split()[2:5])
            elif 'cstar' in line:
                cstar_values = ' '.join(line.split()[2:5])

        if image and event and astar_values and bstar_values and cstar_values and det_shift_x and det_shift_y:
            det_shifts = f"{det_shift_x} {det_shift_y}"
            output_line = f"{image} {event} {astar_values} {bstar_values} {cstar_values} {det_shifts} {lattice}\n"
            output_file.write(output_line)
            return 1
        return 0

    except Exception as e:
        print(f"Error processing block: {e}")
        return 0
        
def read_stream_write_sol(stream_file_path, lattice):
    base,streamfile_name = os.path.split(stream_file_path)
    filename_without_extension, _ = os.path.splitext(streamfile_name)
    solfilename = filename_without_extension + '.sol'
    solfile_path = os.path.join(base,solfilename)

    lines_written = 0
    with open(stream_file_path, 'r') as stream_file, open(solfile_path, 'w') as output_file:
        current_block = []
        for line in stream_file:
            line = line.strip()
            if line == '----- End chunk -----':  # End of a block
                if current_block:
                    # Process the current block before moving to the next one
                    lines_written += process_block(current_block, output_file, lattice)
                current_block = []
            else:
                # Store each line in the current block
                current_block.append(line)

        # Process the last block
        if current_block:
            lines_written += process_block(current_block, output_file, lattice)

    return solfilename, lines_written

# Example usage
if __name__ == "__main__":
    # Example input stream file path and lattice value
    stream_file_folder = '/home/buster/UOX1/different_index_params/3x3'
    stream_file_path = '/home/buster/UOX1/different_index_params/3x3/best_results_IQM_1_2_3_-1_1_-1_1_1.stream'  # Update this with your actual file path
    lattice = 'oI'  # Example lattice value, replace with the appropriate lattice for your data
    num_threads = 23

    # Process the stream file and write output to the .sol file
    best_results_sol, num_index = read_stream_write_sol(stream_file_path, lattice)
    print(f"Finished processing. Total lines written to .sol file: {num_index}")

    # Generate bash script for fast integration
    bash_file_name = f"IQM_1_2_3_-1_1_-1_1_1"
    
    generate_bash_script(bash_file_name, stream_file_folder, num_threads, best_results_sol)
