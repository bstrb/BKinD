import numpy as np
import h5py
from process_image import process_image
import multiprocessing

def load_mask(mask_file_path):
    with h5py.File(mask_file_path, 'r') as h5_file:
        mask = h5_file['/mask'][()]
    # print(f"Loaded mask with shape {mask.shape} and dtype {mask.dtype}", flush=True)
    return mask

def init_worker(mask_file_path):
    """
    Initializes each worker process by loading the mask into a global variable.
    """
    global global_mask
    global_mask = load_mask(mask_file_path)

def process_image_worker(args):
    image_index, h5_file_path, plot, verbose = args
    try:
        # Each worker opens the HDF5 file independently
        with h5py.File(h5_file_path, 'r') as h5_file:
            images_dataset = h5_file['/entry/data/images']
            image = images_dataset[image_index, :, :].astype(np.float32)

        # Use the global mask loaded during worker initialization
        mask = global_mask

        # Process the image
        beam_center = process_image(image, image_index, mask, plot=plot, verbose=verbose)

        return (image_index, beam_center)
    except Exception as e:
        print(f"Error processing image {image_index}: {e}", flush=True)
        return (image_index, None)

def find_center(h5_file_path, new_h5_file_path, mask_path, selected_indices=None, plot=False, verbose=True):
    if selected_indices is None:
        # Get the number of images from the HDF5 file
        with h5py.File(h5_file_path, 'r') as h5_file:
            images_dataset = h5_file['/entry/data/images']
            num_images = images_dataset.shape[0]
            selected_indices = range(num_images)
    else:
        # Validate selected indices
        with h5py.File(h5_file_path, 'r') as h5_file:
            num_images = h5_file['/entry/data/images'].shape[0]
            selected_indices = [idx for idx in selected_indices if 0 <= idx < num_images]

    print(f"Processing {len(selected_indices)} selected images.", flush=True)
    total_images = len(selected_indices)

    # Prepare arguments for worker processes
    args_list = [(image_index, h5_file_path, plot, verbose) for image_index in selected_indices]

    # Initialize multiprocessing Pool with the mask loaded in each worker
    pool = multiprocessing.Pool(initializer=init_worker, initargs=(mask_path,))
    try:
        # Use imap_unordered for efficient processing
        results = pool.imap_unordered(process_image_worker, args_list)

        # Collect results
        beam_centers = [None] * total_images  # Preallocate list
        for result in results:
            image_index, beam_center = result
            idx_in_dataset = selected_indices.index(image_index)
            if beam_center is None:
                print(f"Skipping image {image_index} due to processing error.", flush=True)
                continue
            beam_centers[idx_in_dataset] = beam_center
            print(f"Center for image {image_index} found at {beam_center}", flush=True)
    finally:
        pool.close()
        pool.join()

    # Write beam centers to output HDF5 file
    with h5py.File(new_h5_file_path, 'w') as new_h5_file:
        beam_centers_dataset = new_h5_file.create_dataset(
            'beam_centers',
            shape=(total_images, 2),
            dtype='float64'
        )
        for idx, beam_center in enumerate(beam_centers):
            if beam_center is not None:
                beam_centers_dataset[idx, :] = beam_center
            else:
                beam_centers_dataset[idx, :] = [-1, -1]  # Indicator for failed processing

    print("Processing completed.", flush=True)

# Example usage
if __name__ == '__main__':
    h5_file_path = '/home/buster/UOX1/UOX1_min_50/UOX1_min_50_peak.h5'
    new_h5_file_path = '/home/buster/UOX1/UOX1_min_50/UOX1_min_50_peak_centerfound.h5'
    mask_path = '/home/buster/mask/pxmask.h5'
    selected_indices = list(range(1800, 1830))  # Adjust indices as needed

    # Set plot=False to prevent conflicts in multiprocessing
    find_center(h5_file_path, new_h5_file_path, mask_path, selected_indices=selected_indices, plot=False, verbose=True)
