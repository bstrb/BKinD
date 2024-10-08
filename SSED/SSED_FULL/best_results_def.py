# best_results_definitions.py

import re
import os
import fnmatch
import numpy as np
from scipy.spatial import KDTree

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def parse_stream_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    chunks = []
    current_chunk = {}
    chunk_text = []
    serial_number = None

    for line in lines:
        linestrip = line.strip()
        if linestrip == '----- Begin chunk -----':
            current_chunk = {'peaks': [], 'reflections': [], 'text': []}
            chunk_text = [line]
            serial_number = None
        elif linestrip == '----- End chunk -----':
            current_chunk['text'] = chunk_text + [line]
            chunks.append(current_chunk)
        else:
            chunk_text.append(line)
            if linestrip.startswith('Image serial number:'):
                serial_number = linestrip.split(':')[-1].strip()
                current_chunk['serial'] = serial_number
            elif linestrip.startswith('Event:'):
                current_event = int(re.search(r'\d+', linestrip).group())
                current_chunk['frame'] = current_event
            elif linestrip == 'Peaks from peak search':
                current_chunk['peaks_section'] = True
            elif linestrip == 'End of peak list':
                current_chunk['peaks_section'] = False
            elif linestrip == 'Reflections measured after indexing':
                current_chunk['reflections_section'] = True
            elif linestrip == 'End of reflections':
                current_chunk['reflections_section'] = False
            elif 'peaks_section' in current_chunk and current_chunk['peaks_section']:
                parts = linestrip.split()
                if len(parts) == 5 and all(is_float(part) for part in parts[:4]):
                    current_chunk['peaks'].append((float(parts[0]), float(parts[1]), float(parts[3])))
            elif 'reflections_section' in current_chunk and current_chunk['reflections_section']:
                parts = linestrip.split()
                if len(parts) == 10 and is_float(parts[-3]) and is_float(parts[-2]):
                    current_chunk['reflections'].append((float(parts[-3]), float(parts[-2])))

    return chunks

def find_nearest_neighbours(peaks, reflections, n=50):
    if not peaks or not reflections:
        return []
    if len(peaks) > n:
        peaks = sorted(peaks, key=lambda x: x[2], reverse=True)[:n]
    peak_coords = np.array([(peak[0], peak[1]) for peak in peaks])
    reflection_coords = np.array([(reflection[0], reflection[1]) for reflection in reflections])
    if peak_coords.size == 0 or reflection_coords.size == 0:
        return []
    tree = KDTree(reflection_coords)
    distances, _ = tree.query(peak_coords)
    rmsd = np.sqrt(np.mean(distances**2))
    return rmsd

def rmsd_analysis(file_paths, n):
    best_results = {}
    stream_stats = {}

    for file_path in file_paths:
        chunks = parse_stream_file(file_path)
        sum_rmsd = 0
        rmsd_count = 0
        indexed_patterns_count = 0  # Initialize counter for indexed patterns

        for chunk in chunks:
            rmsd = find_nearest_neighbours(chunk['peaks'], chunk['reflections'], n)
            if rmsd:  # Ensure there's a valid rmsd value
                sum_rmsd += rmsd
                rmsd_count += 1
                if chunk['serial'] not in best_results or best_results[chunk['serial']]['rmsd'] > rmsd:
                    best_results[chunk['serial']] = {'file_path': file_path, 'chunk': chunk, 'rmsd': rmsd}
            if chunk['reflections']:  # Check if the chunk has reflections, indicating indexing
                indexed_patterns_count += 1

        # Calculate and store statistics for this stream file
        if rmsd_count > 0:
            avg_rmsd = sum_rmsd / rmsd_count
        else:
            avg_rmsd = None
        stream_stats[file_path] = {
            'avg_rmsd': avg_rmsd,
            'chunk_count': len(chunks),
            'indexed_patterns_count': indexed_patterns_count  # Store the count of indexed patterns
        }

    # Print statistics for each stream file, including the number of indexed patterns
    for file_path, stats in stream_stats.items():
        print(f"File: {file_path}, Average RMSD: {stats['avg_rmsd']}, Chunk Count: {stats['chunk_count']}, Indexed Patterns: {stats['indexed_patterns_count']}")

    return best_results

def write_best_results_to_stream(best_results, output_file_path):
# def write_best_results_to_stream(best_results, folder_path):
    with open(output_file_path, 'w') as outfile:
        for serial, data in sorted(best_results.items(), key=lambda x: x[1]['rmsd']):
            chunk = data['chunk']
            for line in chunk['text']:
                outfile.write(line.rstrip('\r\n') + '\n')

def find_stream_files(directory):
    # Check if the directory is a valid directory
    if not os.path.isdir(directory):
        raise ValueError("Provided path is not a valid directory")

    # List all files in the directory
    all_files = os.listdir(directory)

    # Filter files that end with 'stream'
    stream_files = fnmatch.filter(all_files, '*.stream')

    full_paths = [os.path.join(directory, file) for file in stream_files]

    return full_paths

def print_output_file_statistics(output_file_path, best_results):
    chunk_count = len(best_results)
    indexed_patterns_count = sum(1 for data in best_results.values() if data['chunk']['reflections'])  # Count chunks with reflections
    
    if chunk_count > 0:
        total_rmsd = sum(data['rmsd'] for data in best_results.values())
        average_rmsd = total_rmsd / chunk_count
    else:
        average_rmsd = "N/A"  # No chunks or RMSD data available
    
    # Adjusted print format for consistency
    print(f"File: {output_file_path}, Average RMSD: {average_rmsd:.3f}, Chunk Count: {chunk_count}, Indexed Patterns: {indexed_patterns_count}")

def find_and_copy_header(inputfolder_path, output_path):
    print(f"Adding header to {output_path}...")
    _, output_file = os.path.split(output_path)
    found_file_path = None

    for root, dirs, files in os.walk(inputfolder_path):
        for file in files:
            if file.endswith('.stream') and file != output_file:
                found_file_path = os.path.join(root, file)
                break
        if found_file_path:
            print(found_file_path)
            break

    with open(found_file_path, 'r', newline='') as source_file:
        lines = []
        for line in source_file:
            if "----- Begin chunk -----" in line:
                break
            lines.append(line)

    # Read existing best_results file
    with open(output_path, 'r', newline='') as target_file:
        existing_content = target_file.readlines()

    # Combine the new header with the existing content and write back
    combined_content = lines + existing_content


    with open(output_path, 'w', newline='') as target_file:
        target_file.writelines(combined_content)
    
    print(f"Header copied and added to {output_path}")

def find_best_results(folder_path, output_path):

    file_paths = find_stream_files(folder_path)

    print(file_paths)

    best_results = rmsd_analysis(file_paths, n=10)

    output_path = os.path.join(folder_path, "best_results.stream")

    write_best_results_to_stream(best_results, output_path)

    print_output_file_statistics(output_path, best_results)

    find_and_copy_header(folder_path, output_path)

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
                #print(astar_values)
            elif 'bstar' in line:
                bstar_values = ' '.join(line.split()[2:5])
                #print(bstar_values)
            elif 'cstar' in line:
                cstar_values = ' '.join(line.split()[2:5])
                #print(cstar_values)

        if image and event and astar_values and bstar_values and cstar_values and det_shift_x and det_shift_y:
            det_shifts = f"{det_shift_x} {det_shift_y}"
            output_line = f"{image} {event} {astar_values} {bstar_values} {cstar_values} {det_shifts} {lattice}\n"
            output_file.write(output_line)

    except Exception as e:
        print(f"Error processing block: {e}")
        
def read_stream_write_sol(stream_file_path, lattice):
    base,streamfile_name = os.path.split(stream_file_path)
    filename_without_extension, _ = os.path.splitext(streamfile_name)
    solfilename = filename_without_extension + '.sol'
    solfile_path = os.path.join(base,solfilename)

    with open(stream_file_path, 'r') as stream_file, open(solfile_path, 'w') as output_file:
        current_block = []
        for line in stream_file:
            line = line.strip()
            if line == '----- End chunk -----':  # End of a block
                if current_block:
                    # Process the current block before moving to the next one
                    process_block(current_block, output_file, lattice)
                current_block = []
            else:
                # Store each line in the current block
                current_block.append(line)

        # Process the last block
        if current_block:
            process_block(current_block, output_file)