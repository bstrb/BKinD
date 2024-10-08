import re
import os

def scan_stream_file(stream_file_path, min_peaks=15):
    # Regular expressions for identifying chunk start, end, event, number of peaks, and indexing status
    chunk_start_pattern = re.compile(r'^----- Begin chunk -----')
    chunk_end_pattern = re.compile(r'^----- End chunk -----')
    event_pattern = re.compile(r'^Event: //(?P<event_number>\d+)')
    num_peaks_pattern = re.compile(r'^num_peaks = (?P<num_peaks>\d+)')
    indexing_status_pattern = re.compile(r'^Reflections measured after indexing')

    events_no_indexing = []
    current_event_number = None
    current_num_peaks = 0
    in_chunk = False
    has_indexing = False

    # Read through the stream file line by line
    with open(stream_file_path, 'r') as stream_file:
        for line in stream_file:
            line = line.strip()

            # Check for beginning of a chunk
            if chunk_start_pattern.match(line):
                in_chunk = True
                current_event_number = None
                current_num_peaks = 0
                has_indexing = False
                continue

            # Check for end of a chunk
            if chunk_end_pattern.match(line):
                if in_chunk and current_num_peaks > min_peaks and not has_indexing:
                    if current_event_number is not None:
                        events_no_indexing.append(current_event_number)
                in_chunk = False
                continue

            if in_chunk:
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

    # Output the list of event numbers to a file
    output_file_path = os.path.join(os.path.dirname(stream_file_path), 'events_no_indexing.txt')
    with open(output_file_path, 'w') as output_file:
        for event_number in events_no_indexing:
            output_file.write(f'Event: //{event_number}\n')

    print(f'Output written to {output_file_path}')

# Example usage
stream_file_path = '/home/buster/UOX1/0-05-step-indexing/41x41_0-05_indexing/fast_int/fast_int_rings_3-4-7.stream'
scan_stream_file(stream_file_path, min_peaks=15)