# parse_stream_file.py

import re

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
