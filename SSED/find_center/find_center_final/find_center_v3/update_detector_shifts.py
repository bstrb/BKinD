import h5py
import numpy as np

def update_detector_shifts(
    h5_file_path,
    beam_centers_dict,
    selected_indices,
    framesize,
    pixels_per_meter=17857.14285714286
):
    """
    Update detector shift datasets based on computed beam centers.
    
    Parameters:
    - h5_file_path (str): Path to the HDF5 file containing datasets.
    - beam_centers_dict (dict): Dictionary mapping image indices to beam centers [x, y].
    - selected_indices (list): List of image indices processed.
    - framesize (int): Size of the image frame (assuming square images).
    - pixels_per_meter (float): Conversion factor from pixels to meters.
    
    Returns:
    - None
    """
    presumed_center = framesize / 2  # Half of the framesize
    
    # Open the HDF5 file and prepare to update datasets
    with h5py.File(h5_file_path, 'r+') as h5_file:
        # Ensure datasets exist; if not, create them
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
                # Calculate detector shifts in millimeters
                det_shift_x = -((beam_center[0] - presumed_center) / pixels_per_meter) * 1000  # Convert meters to mm
                det_shift_y = -((beam_center[1] - presumed_center) / pixels_per_meter) * 1000  # Convert meters to mm
                det_shift_x_mm_dataset[image_index] = det_shift_x
                det_shift_y_mm_dataset[image_index] = det_shift_y
            else:
                # Set to -1 if processing failed
                det_shift_x_mm_dataset[image_index] = -1
                det_shift_y_mm_dataset[image_index] = -1
    
    print('Updated detector shifts written to HDF5 file', flush=True)