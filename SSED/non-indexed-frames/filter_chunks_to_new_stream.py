import re
import os
from tqdm import tqdm

def filter_chunks_to_new_stream(stream_file_path, min_peaks=15):
    # Regular expressions for identifying chunk start, end, event, number of peaks, and indexing status
    chunk_start_pattern = re.compile(r'^----- Begin chunk -----')
    chunk_end_pattern = re.compile(r'^----- End chunk -----')
    event_pattern = re.compile(r'^Event: //(?P<event_number>\d+)')
    num_peaks_pattern = re.compile(r'^num_peaks = (?P<num_peaks>\d+)')
    indexing_status_pattern = re.compile(r'^Reflections measured after indexing')

    filtered_chunks = []
    current_chunk = []
    current_event_number = None
    current_num_peaks = 0
    in_chunk = False
    has_indexing = False
    header = []
    header_captured = False

    # Read through the stream file line by line
    with open(stream_file_path, 'r') as stream_file:
        lines = stream_file.readlines()
        total_lines = len(lines)
        
        for line in tqdm(lines, total=total_lines, desc="Processing stream file"):
            line = line.rstrip()

            # Capture header until the first chunk starts
            if not header_captured:
                if chunk_start_pattern.match(line):
                    header_captured = True
                else:
                    header.append(line)
                    continue

            # Check for beginning of a chunk
            if chunk_start_pattern.match(line):
                in_chunk = True
                current_chunk = [line]
                current_event_number = None
                current_num_peaks = 0
                has_indexing = False
                continue

            # Check for end of a chunk
            if chunk_end_pattern.match(line):
                if in_chunk:
                    current_chunk.append(line)
                    if current_num_peaks > min_peaks and not has_indexing:
                        filtered_chunks.extend(current_chunk)
                in_chunk = False
                continue

            if in_chunk:
                # Add line to current chunk
                current_chunk.append(line)

                # Extract the event number
                event_match = event_pattern.match(line)
                if event_match:
                    current_event_number = int(event_match.group('event_number'))

                # Extract the number of peaks
                num_peaks_match = num_peaks_pattern.match(line)
                if num_peaks_match:
                    current_num_peaks = int(num_peaks_match.group('num_peaks'))

                # Check for indexing status
                if indexing_status_pattern.search(line):
                    has_indexing = True

    # Output the filtered chunks to a new stream file
    output_file_path = os.path.join(os.path.dirname(stream_file_path), 'filtered_stream.stream')
    with open(output_file_path, 'w') as output_file:
        for line in header:
            output_file.write(f'{line}\n')
        for chunk_line in filtered_chunks:
            output_file.write(f'{chunk_line}\n')

    print(f'Filtered stream file written to {output_file_path}')

# Example usage
stream_file_path = '/home/buster/UOX_tot/UOX_-512_-512_pushres5_rings_4-5-9.stream'
filter_chunks_to_new_stream(stream_file_path, min_peaks=15)