import h5py
import numpy as np
from diffractem.peakfinder8_extension import peakfinder_8
from tqdm import tqdm

def load_mask(mask_h5_path):
    """
    Load the mask from an HDF5 file.
    """
    with h5py.File(mask_h5_path, 'r') as mask_h5:
        mask = mask_h5['/mask'][:]  # Adjust the dataset path as needed
    return mask.astype(np.int8)  # Convert to int8 for compatibility with peakfinder_8

def find_peaks_in_frame(frame, x0, y0, mask=None, adc_thresh=100, min_snr=6, min_pix_count=3, max_pix_count=20, local_bg_radius=3, min_res=0, max_res=None):
    """
    Uses peakfinder8 to find peaks in a given frame.
    Returns the number of peaks found.
    """
    if max_res is None:
        max_res = np.max(frame.shape)  # Default max_res if not provided

    # Create radial distance array
    X, Y = np.meshgrid(range(frame.shape[1]), range(frame.shape[0]))
    R = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2).astype(np.float32)
    
    # Create mask if none provided
    if mask is None:
        mask = np.ones_like(frame, dtype=np.int8)
    mask[R > max_res] = 0
    mask[R < min_res] = 0

    # Call peakfinder_8
    peaks = peakfinder_8(
        500,  # Maximum number of peaks
        frame.astype(np.float32),
        mask,
        R,  # Radial distance array
        frame.shape[1], frame.shape[0],
        1, 1,  # ASIC configuration (dummy values)
        adc_thresh, min_snr,
        min_pix_count, max_pix_count, local_bg_radius
    )

    return len(peaks[0])  # Return number of peaks

def is_bad_frame(frame, x0, y0, mask=None, adc_thresh=100, min_snr=6, min_pix_count=3, max_pix_count=20, local_bg_radius=3, min_peaks=5, min_res=0, max_res=None):
    """
    Determine if a frame is 'bad' based on peakfinding results.
    Returns True if the frame is considered bad.
    """
    num_peaks = find_peaks_in_frame(frame, x0, y0, mask, adc_thresh, min_snr, min_pix_count, max_pix_count, local_bg_radius, min_res, max_res)
    return num_peaks < min_peaks
   
def remove_bad_frames(input_h5_path, output_h5_path, mask_h5_path=None, adc_thresh=100, min_snr=6, min_pix_count=3, max_pix_count=20, local_bg_radius=3, min_peaks=5, min_res=0, max_res=None):
    """
    Reads an HDF5 file, removes bad frames based on peak finding, and writes the good frames to a new HDF5 file.
    """
    # Load the mask if provided
    mask = load_mask(mask_h5_path) if mask_h5_path else None

    # Placeholder values for beam center
    x0, y0 = 0.5, 0.5  # Update these values based on your specific detector/beam setup

    with h5py.File(input_h5_path, 'r') as input_h5:
        dataset = input_h5['/entry/data/images']
        total_frames, height, width = dataset.shape

        # Create the output HDF5 file and dataset
        with h5py.File(output_h5_path, 'w') as output_h5:
            output_dataset = output_h5.create_dataset('/entry/data/images', shape=(0, height, width), maxshape=(None, height, width), dtype='float32')

            # Process each frame and write to the output HDF5 file if it passes the peakfinding criteria
            for i in tqdm(range(total_frames), desc="Processing frames"):
                frame = dataset[i, :, :]

                # Check if the frame is 'bad'
                if not is_bad_frame(frame, x0, y0, mask, adc_thresh, min_snr, min_pix_count, max_pix_count, local_bg_radius, min_peaks, min_res, max_res):
                    # Resize the output dataset to accommodate the new frame
                    output_dataset.resize(output_dataset.shape[0] + 1, axis=0)
                    output_dataset[-1] = frame  # Add the frame to the dataset

            print(f"Total frames: {total_frames}")
            print(f"Number of good frames saved: {output_dataset.shape[0]}")

# Example usage
input_h5_path = '/home/buster/wsl-TEST1/R2aOx.h5'
output_h5_path = '/home/buster/wsl-TEST1/R2aOx_cleaned.h5'
mask_h5_path = '/home/buster/wsl-TEST1/mask/pxmask.h5'

adc_thresh = 100
min_snr = 6
min_pix_count = 3
max_pix_count = 20
local_bg_radius = 3
min_peaks = 5
min_res = 0
max_res = None

remove_bad_frames(input_h5_path, output_h5_path, mask_h5_path, adc_thresh, min_snr, min_pix_count, max_pix_count, local_bg_radius, min_peaks, min_res, max_res)
