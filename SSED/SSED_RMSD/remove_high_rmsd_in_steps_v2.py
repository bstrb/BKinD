from parse_stream_file import parse_stream_file
from find_nearest_neighbours import find_nearest_neighbours
import os

def remove_high_rmsd_in_steps(stream_file, step_fraction, num_steps):
    """
    Removes high RMSD frames in successive steps.
    
    Parameters:
    - stream_file (str): Path to the input stream file.
    - step_fraction (float): Fraction of total frames to remove in each step.
    - num_steps (int): Number of steps to perform.
    - output_folder (str): Path to the output folder where modified stream files will be saved.
    """
    
    # Get the directory of the original stream file
    stream_file_dir = os.path.dirname(stream_file)
    
    # Create the output folder in the same directory as the input stream file
    output_folder = os.path.join(stream_file_dir, f"high_rmsd_removed_{num_steps}_x_{100*step_fraction}_percentage_units")
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Parse the stream file once to get chunks and calculate RMSD
    chunks = parse_stream_file(stream_file)
    
    # Read the header from the stream file (before "----- Begin chunk -----")
    header = []
    with open(stream_file, 'r') as file:
        for line in file:
            header.append(line)
            if line.strip() == '----- Begin chunk -----':
                break
    
    # Calculate RMSD for each chunk
    rmsd_list = []
    for chunk in chunks:
        rmsd = find_nearest_neighbours(chunk['peaks'], chunk['reflections'], n=50)
        if rmsd:
            rmsd_list.append((chunk, rmsd))
    
    # Sort chunks by RMSD (highest first)
    rmsd_list.sort(key=lambda x: x[1], reverse=True)
    
    total_chunks = len(rmsd_list)
    num_to_remove_per_step = int(step_fraction * total_chunks)
    
    print(f"Total frames: {total_chunks}")
    print(f"Removing {num_to_remove_per_step} highest RMSD frames per step for {num_steps} steps.")
    
    # Start with the full RMSD list and progressively remove high RMSD frames
    current_rmsd_list = rmsd_list.copy()
    
    for step in range(1, num_steps + 1):
        # Determine how many frames to remove in this step
        num_to_remove = min(num_to_remove_per_step, len(current_rmsd_list))
        
        # Split the RMSD list into chunks to remove and chunks to keep
        chunks_to_remove = current_rmsd_list[:num_to_remove]
        chunks_to_keep = current_rmsd_list[num_to_remove:]
        
        # Update the list for the next step
        current_rmsd_list = chunks_to_keep
        
        # Prepare the modified stream content (starting with the header)
        modified_stream = header.copy()

        # Prepare the modified stream content (starting with the header)
        removed_stream = header.copy()
        
        # Add chunks that are to be kept
        for chunk, rmsd in chunks_to_keep:
            modified_stream.extend(chunk['text'])
            
        # Add chunks that are to be kept
        for chunk, rmsd in chunks_to_remove:
            removed_stream.extend(chunk['text'])
        
        # Format the output file name based on the step
        output_file = os.path.join(output_folder, f"{100 - step * step_fraction * 100:.0f}_percent.stream")
        
        # Write the modified stream file
        with open(output_file, 'w') as file:
            file.writelines(modified_stream)
        
        # Format the output file name based on the step
        removed_output_file = os.path.join(output_folder, f"{step * step_fraction * 100:.0f}_percent_highest_rmsd.stream")

        # Write the modified stream file
        with open(removed_output_file, 'w') as file:
            file.writelines(removed_stream)
        
        print(f"Step {step}: Removed {num_to_remove} highest RMSD frames. Remaining frames: {len(chunks_to_keep)}")
        print(f"Modified stream file saved to {output_file}")
    
    print(f"All steps completed. Stream files saved to {output_folder}.")
