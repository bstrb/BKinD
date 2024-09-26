import os
import shutil 

def split_stream_file_with_step(stream_file, output_dir, step_size, tolerance_ratio=0.4):
    """
    Split a stream file into multiple files with increasing chunk sizes, containing the header.
    
    Parameters:
    - stream_file: The path to the original stream file.
    - output_dir: Directory where the split files will be saved.
    - step_size: The step size for increasing chunks (e.g., 5000).
    - tolerance_ratio: The ratio of step size to decide if the next chunk size should be skipped if it's within tolerance.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Read the entire stream file content
    with open(stream_file, 'r') as file:
        lines = file.readlines()

    # Identify where chunks start and end using the delimiter ----- Begin chunk ----- and ----- End chunk -----
    chunk_start_indices = [i for i, line in enumerate(lines) if "----- Begin chunk -----" in line]
    chunk_end_indices = [i for i, line in enumerate(lines) if "----- End chunk -----" in line]

    if len(chunk_start_indices) != len(chunk_end_indices):
        print("Mismatch in the number of 'Begin chunk' and 'End chunk' lines.")
        return

    # Calculate the total indexed chunks
    indexed_chunk_indices = []
    for start_idx, end_idx in zip(chunk_start_indices, chunk_end_indices):
        chunk_lines = lines[start_idx:end_idx + 1]
        if any("Reflections measured after indexing" in line for line in chunk_lines):
            indexed_chunk_indices.append((start_idx, end_idx))

    total_indexed_chunks = len(indexed_chunk_indices)
    print(f"Total indexed chunks in the original file: {total_indexed_chunks}")

    # Extract the header (everything before the first "----- Begin chunk -----")
    header_end_index = chunk_start_indices[0]
    header = lines[:header_end_index]

    # Generate files with increasing chunk sizes
    current_chunk_size = step_size
    while current_chunk_size <= total_indexed_chunks:
        # Check if the remaining indexed chunks are within the tolerance ratio based on the step size
        if total_indexed_chunks - current_chunk_size <= step_size * tolerance_ratio:
            print(f"Skipping generation of further files as the next chunk size is within tolerance ratio.")
            break

        merged_lines = header[:]  # Start with the header for each split file
        
        # Collect the specified number of indexed chunks
        for i in range(current_chunk_size):
            start_index, end_index = indexed_chunk_indices[i]
            merged_lines.extend(lines[start_index:end_index + 1])

        # Write the collected chunks to a new stream file
        output_stream_file = os.path.join(output_dir, f"chunk_{current_chunk_size}.stream")
        with open(output_stream_file, 'w') as temp_file:
            temp_file.writelines(merged_lines)

        print(f"Created stream file with {current_chunk_size} indexed chunks: {output_stream_file}")

        # Increase chunk size for the next iteration
        current_chunk_size += step_size

    # Create a copy of the original stream file named after the total number of indexed chunks
    copied_stream_file = os.path.join(output_dir, f"chunk_{total_indexed_chunks}.stream")
    shutil.copy2(stream_file, copied_stream_file)
    print(f"Copied the original stream file to: {copied_stream_file}")

    print("Completed generating split stream files.")


# # Example usage
# stream_file = "/home/buster/UOXm/fast_int_347/fast_int_rings_3-4-7.stream"  # Path to your original stream file
# output_dir = "/home/buster/UOXm/fast_int_347"             # Directory where the split files will be saved
# step_size = 5000                                     # Step size for increasing chunks
# tolerance_ratio = 0.4                                # Tolerance ratio based on step size (50% in this case)

# # Call the function with the specified parameters
# split_stream_file_with_step(stream_file, output_dir, step_size, tolerance_ratio)
