import numpy as np
import h5py
import json 
import os
from tqdm import tqdm
from process_image import process_image
from update_detector_shifts import update_detector_shifts
from create_unique_folder import create_unique_folder
from plot_center_positions_vs_index import plot_center_positions
import multiprocessing

def load_mask(mask_file_path):
    """
    Load the mask from an HDF5 file.
    
    Parameters:
    - mask_file_path (str): Path to the HDF5 mask file.
    
    Returns:
    - mask (numpy.ndarray): 2D array representing the mask.
    """
    with h5py.File(mask_file_path, 'r') as h5_file:
        mask = h5_file['/mask'][()]
    return mask

def init_worker(mask_file_path):
    """
    Initialize worker process by loading the mask into a global variable.
    
    Parameters:
    - mask_file_path (str): Path to the HDF5 mask file.
    """
    global global_mask
    global_mask = load_mask(mask_file_path)

def process_image_worker(args):
    """
    Worker function to process a single image.
    
    Parameters:
    - args (tuple): Contains all necessary arguments for processing.
    
    Returns:
    - tuple: (image_index, beam_center) where beam_center is a tuple (x, y).
    """
    (
        image_index,
        h5_file_path,
        plot,
        verbose,
        median_filter_size,
        top_intensity_exc,
        downsample_factor,
        center_initial,
        radial_bins,
        num_slices
    ) = args

    try:
        with h5py.File(h5_file_path, 'r') as h5_file:
            images_dataset = h5_file['/entry/data/images']
            image = images_dataset[image_index, :, :].astype(np.int16)
        
        mask = global_mask

        # Pass the additional parameters to process_image
        beam_center = process_image(
            image=image,
            image_index=image_index,
            mask=mask,
            median_filter_size=median_filter_size,
            top_intensity_exc=top_intensity_exc,
            downsample_factor=downsample_factor,
            center_initial=center_initial,
            radial_bins=radial_bins,
            num_slices=num_slices,
            plot=plot,
            verbose=verbose
        )
        return (image_index, beam_center)
    except Exception as e:
        print(f"Error processing image {image_index}: {e}", flush=True)
        return (image_index, None)

def find_center(
    h5_file_path,
    mask_path,
    selected_indices=None,
    plot=False,
    verbose=True,
    median_filter_size=3,
    top_intensity_exc=0,
    downsample_factor=0.5,
    center_initial=None,
    radial_bins=None,
    num_slices=3
):
    """
    Process multiple images to find their centers and update detector shifts.
    
    Parameters:
    - h5_file_path (str): Path to the HDF5 file containing images and datasets.
    - mask_path (str): Path to the HDF5 mask file.
    - selected_indices (list or None): List of image indices to process. If None, process all images.
    - plot (bool): Whether to generate plots for each image.
    - verbose (bool): Whether to print detailed logs.
    - median_filter_size (int): Size of the median filter window.
    - top_intensity_exc (float): Percentage to exclude top-intensity pixels.
    - downsample_factor (float): Factor by which to downsample images and masks.
    - center_initial (list or tuple or None): Initial guess for the center coordinates [x, y].
    - radial_bins (int or None): Number of radial bins. If None, defaults to 100.
    - num_slices (int): Number of angular slices to divide the half-circle into.
    
    Returns:
    - None
    """
    if selected_indices is None:
        with h5py.File(h5_file_path, 'r') as h5_file:
            images_dataset = h5_file['/entry/data/images']
            num_images = images_dataset.shape[0]
            selected_indices = range(num_images)
    else:
        with h5py.File(h5_file_path, 'r') as h5_file:
            num_images = h5_file['/entry/data/images'].shape[0]
            selected_indices = [idx for idx in selected_indices if 0 <= idx < num_images]
    
    print(f"Processing {len(selected_indices)} images.", flush=True)
    
    # Prepare arguments for each worker
    args_list = [
        (
            image_index,
            h5_file_path,
            plot,
            verbose,
            median_filter_size,
            top_intensity_exc,
            downsample_factor,
            center_initial,
            radial_bins,
            num_slices
        )
        for image_index in selected_indices
    ]
    
    # Initialize multiprocessing pool with the mask loaded
    pool = multiprocessing.Pool(initializer=init_worker, initargs=(mask_path,))
    try:
        results = pool.imap_unordered(process_image_worker, args_list)
    
        beam_centers_dict = {}  # Use a dict for direct indexing

        # Initialize tqdm progress bar
        with tqdm(total=len(selected_indices), desc="Processing Images", unit="image") as pbar:
            for result in results:
                image_index, beam_center = result
                if beam_center is None:
                    # print(f"Skipping image {image_index} due to processing error.", flush=True)
                    pass
                else:
                    beam_centers_dict[image_index] = beam_center
                    # print(f"Center for image {image_index} found at {beam_center}", flush=True)
                pbar.update(1)  # Update the progress bar by one
    finally:
        pool.close()
        pool.join()
    
    # Overwrite beam centers in the original HDF5 file
    with h5py.File(h5_file_path, 'r+') as h5_file:
        center_x_dataset = h5_file['/entry/data/center_x']
        center_y_dataset = h5_file['/entry/data/center_y']
    
        for image_index in selected_indices:
            if image_index in beam_centers_dict:
                beam_center = beam_centers_dict[image_index]
                center_x_dataset[image_index] = beam_center[0]
                center_y_dataset[image_index] = beam_center[1]
            else:
                # Set to -1 if processing failed
                center_x_dataset[image_index] = -1
                center_y_dataset[image_index] = -1
    
        # Get the framesize for detector shifts
        images_dataset = h5_file['/entry/data/images']
        framesize = images_dataset.shape[1]  # Assuming square images
    
    # Update detector shifts based on the newly found beam centers
    update_detector_shifts(h5_file_path, beam_centers_dict, selected_indices, framesize, pixels_per_meter=17857.14285714286)
    
    print("Processing completed.", flush=True)

    # === Begin Snippet to Save Beam Centers ===

    # Prepare data to save
    image_indices = list(beam_centers_dict.keys())
    x_positions = [beam_centers_dict[idx][0] for idx in image_indices]
    y_positions = [beam_centers_dict[idx][1] for idx in image_indices]

    # Create a dictionary of input parameters for naming
    params = {
        'median_filter_size': median_filter_size,
        'top_intensity_exc': top_intensity_exc,
        'downsample_factor': downsample_factor,
        'radial_bins': radial_bins,
        'num_slices': num_slices,
        # Add other parameters if needed
    }

    # Generate a unique file name based on input parameters
    # Convert parameters to a JSON string and replace problematic characters
    params_str = json.dumps(params, sort_keys=True)
    params_str_clean = params_str.replace(" ", "").replace(":", "").replace(",", "_").replace("{", "").replace("}", "").replace("\"", "")
    output_h5_filename = f"beam_centers_{params_str_clean}.h5"

    # Determine the directory of the input h5 file
    h5_dir = os.path.dirname(h5_file_path)
    output_dir = create_unique_folder(h5_dir)
    output_h5_path = os.path.join(output_dir, output_h5_filename)

    txt_file_name = f"params_{params_str_clean}.txt"
    txt_file_path = os.path.join(output_dir, txt_file_name)

    with open(txt_file_path, 'w') as txt_file:
        txt_file.write("Input Parameters and Processing Time\n")
        txt_file.write("====================================\n\n")

    # Save the beam centers to the new HDF5 file
    with h5py.File(output_h5_path, 'w') as out_h5:
        out_h5.create_dataset('entry/data/center_x', data=np.array(x_positions))
        out_h5.create_dataset('entry/data/center_y', data=np.array(y_positions))
        out_h5.create_dataset('entry/data/index', data=np.array(image_indices))
    
    print(f"Beam centers saved to {output_h5_path}", flush=True)

    # === End Snippet ===