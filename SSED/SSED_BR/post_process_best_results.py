# post_process_best_results.py

import os

def write_best_results_to_stream(best_results, output_file_path):
    with open(output_file_path, 'w') as outfile:
        for serial, data in sorted(best_results.items(), key=lambda x: x[1]['rmsd']):
            chunk = data['chunk']
            for line in chunk['text']:
                outfile.write(line.rstrip('\r\n') + '\n')

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