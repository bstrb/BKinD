import numpy as np
import h5py
from process_image import process_image
import multiprocessing

def load_mask(mask_file_path):
    with h5py.File(mask_file_path, 'r') as h5_file:
        mask = h5_file['/mask'][()]
    return mask

def init_worker(mask_file_path):
    global global_mask
    global_mask = load_mask(mask_file_path)

def process_image_worker(args):
    image_index, h5_file_path, plot, verbose = args
    try:
        with h5py.File(h5_file_path, 'r') as h5_file:
            images_dataset = h5_file['/entry/data/images']
            image = images_dataset[image_index, :, :].astype(np.float32)
        mask = global_mask
        beam_center = process_image(image, image_index, mask, plot=plot, verbose=verbose)
        return (image_index, beam_center)
    except Exception as e:
        print(f"Error processing image {image_index}: {e}", flush=True)
        return (image_index, None)

def find_center(h5_file_path, mask_path, selected_indices=None, plot=False, verbose=True):
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

    args_list = [(image_index, h5_file_path, plot, verbose) for image_index in selected_indices]

    pool = multiprocessing.Pool(initializer=init_worker, initargs=(mask_path,))
    try:
        results = pool.imap_unordered(process_image_worker, args_list)

        beam_centers_dict = {}  # Use a dict for direct indexing
        for result in results:
            image_index, beam_center = result
            if beam_center is None:
                print(f"Skipping image {image_index} due to processing error.", flush=True)
                continue
            beam_centers_dict[image_index] = beam_center
            print(f"Center for image {image_index} found at {beam_center}", flush=True)
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
    update_detector_shifts(h5_file_path, beam_centers_dict, selected_indices, framesize)

    print("Processing completed.", flush=True)

def update_detector_shifts(h5_file_path, beam_centers_dict, selected_indices, framesize):
    pixels_per_meter = 17857.14285714286
    presumed_center = framesize / 2  # Half of the framesize

    # Open the HDF5 file and prepare to update datasets
    with h5py.File(h5_file_path, 'r+') as h5_file:
        # Ensure datasets exist
        for ds_name in ['entry/data/det_shift_x_mm', 'entry/data/det_shift_y_mm']:
            if ds_name not in h5_file:
                # Initialize with zeros or desired default values
                data_shape = (h5_file['/entry/data/center_x'].shape[0],)
                h5_file.create_dataset(ds_name, data=np.zeros(data_shape), maxshape=(None,), dtype='float64')

        det_shift_x_mm_dataset = h5_file['/entry/data/det_shift_x_mm']
        det_shift_y_mm_dataset = h5_file['/entry/data/det_shift_y_mm']

        for image_index in selected_indices:
            if image_index in beam_centers_dict:
                beam_center = beam_centers_dict[image_index]
                det_shift_x = -((beam_center[0] - presumed_center) / pixels_per_meter) * 1000
                det_shift_y = -((beam_center[1] - presumed_center) / pixels_per_meter) * 1000
                det_shift_x_mm_dataset[image_index] = det_shift_x
                det_shift_y_mm_dataset[image_index] = det_shift_y
            else:
                # Set to -1 if processing failed
                det_shift_x_mm_dataset[image_index] = -1
                det_shift_y_mm_dataset[image_index] = -1

    print('Updated detector shifts written to HDF5 file')

# Example usage
if __name__ == '__main__':
    h5_file_path = '/home/buster/UOX1/UOX1_min_50/UOX1_min_50_peak.h5'
    mask_path = '/home/buster/mask/pxmask.h5'
    selected_indices = list(range(10, 100))  # Adjust indices as needed

    # Set plot=False to prevent conflicts in multiprocessing
    find_center(h5_file_path, mask_path, selected_indices=selected_indices, plot=False, verbose=True)
