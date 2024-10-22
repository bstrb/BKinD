# Function to parse the stream file and extract chunks
def parse_stream_file(stream_file_path):
    with open(stream_file_path, 'r') as f:
        content = f.read()
    chunks = content.split("----- Begin chunk -----")
    return chunks[0], chunks
