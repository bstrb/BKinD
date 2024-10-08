import os

# Function to write combined metrics to an output file
def write_combined_metrics(metrics, stream_file_path):
    output_file_path = os.path.join(os.path.dirname(stream_file_path), 'combined_metrics.txt')
    with open(output_file_path, 'w') as output_file:
        for event_number, combined_metric in metrics:
            output_file.write(f'Event #{event_number}: Combined metric value = {combined_metric}\n')
    print(f'Combined metrics written to {output_file_path}')
