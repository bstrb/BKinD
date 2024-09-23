import h5py
import numpy as np
from diffractem.peakfinder8_extension import peakfinder_8
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def copy_metadata(input_h5, output_h5):
    """
    Copies all groups, datasets, and attributes from the input HDF5 file to the output HDF5 file,
    excluding the dataset '/entry/data/images'.
    """
    def copy_attrs(input_obj, output_obj):
        for key, value in input_obj.attrs.items():
            output_obj.attrs[key] = value

    def copy_group(input_group, output_group):
        for name, item in input_group.items():
            # Skip copying the dataset '/entry/data/images' and anything under '/entry/data/'
            if name == 'images' and input_group.name == '/entry/data':
                continue

            if isinstance(item, h5py.Group):
                new_group = output_group.create_group(name)
                copy_attrs(item, new_group)
                copy_group(item, new_group)
            elif isinstance(item, h5py.Dataset):
                # Copy datasets with the same settings as in the original file
                output_group.create_dataset(name, data=item[...], chunks=item.chunks, compression=item.compression)
                copy_attrs(item, output_group[name])

    copy_attrs(input_h5, output_h5)  # Copy attributes at the root level
    copy_group(input_h5, output_h5)  # Copy groups and datasets except for '/entry/data/images'


def load_mask(mask_h5_path):
    """
    Load the mask from an HDF5 file.
    """
    with h5py.File(mask_h5_path, 'r') as mask_h5:
        mask = mask_h5['/mask'][:]
    return mask.astype(np.int8)

def find_peaks_in_frame(frame, x0, y0, mask=None, adc_thresh=100, min_snr=6, min_pix_count=3, max_pix_count=20, local_bg_radius=3, min_res=0, max_res=None):
    if max_res is None:
        max_res = np.max(frame.shape)

    X, Y = np.meshgrid(range(frame.shape[1]), range(frame.shape[0]))
    R = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2).astype(np.float32)

    if mask is None:
        mask = np.ones_like(frame, dtype=np.int8)
    mask[R > max_res] = 0
    mask[R < min_res] = 0

    peaks = peakfinder_8(
        500,
        frame.astype(np.float32),
        mask,
        R,
        frame.shape[1], frame.shape[0],
        1, 1,
        adc_thresh, min_snr,
        min_pix_count, max_pix_count, local_bg_radius
    )

    return len(peaks[0])

def is_bad_frame(frame, x0, y0, mask=None, adc_thresh=100, min_snr=6, min_pix_count=3, max_pix_count=20, local_bg_radius=3, min_peaks=5, min_res=0, max_res=None):
    num_peaks = find_peaks_in_frame(frame, x0, y0, mask, adc_thresh, min_snr, min_pix_count, max_pix_count, local_bg_radius, min_res, max_res)
    return num_peaks < min_peaks

def process_frame(index, dataset, mask, x0, y0, adc_thresh, min_snr, min_pix_count, max_pix_count, local_bg_radius, min_peaks, min_res, max_res):
    """
    Process a single frame to check if it is 'bad'. Returns a tuple (result, frame).
    """
    frame = dataset[index, :, :]
    result = not is_bad_frame(frame, x0, y0, mask, adc_thresh, min_snr, min_pix_count, max_pix_count, local_bg_radius, min_peaks, min_res, max_res)
    return result, frame

def remove_bad_frames(input_h5_path, output_h5_path, mask_h5_path=None, adc_thresh=100, min_snr=6, min_pix_count=3, max_pix_count=20, local_bg_radius=3, min_peaks=5, min_res=0, max_res=None, num_threads=4):
    mask = load_mask(mask_h5_path) if mask_h5_path else None
    x0, y0 = 0.5, 0.5

    with h5py.File(input_h5_path, 'r') as input_h5:
        dataset = input_h5['/entry/data/images']
        total_frames, height, width = dataset.shape

        with h5py.File(output_h5_path, 'w') as output_h5:
            # Copy the metadata (everything except the main dataset '/entry/data/images')
            copy_metadata(input_h5, output_h5)


            # Create the output dataset for cleaned frames
            output_dataset = output_h5.create_dataset('/entry/data/images', shape=(0, height, width), maxshape=(None, height, width), dtype='float32')

            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [
                    executor.submit(process_frame, i, dataset, mask, x0, y0, adc_thresh, min_snr, min_pix_count, max_pix_count, local_bg_radius, min_peaks, min_res, max_res)
                    for i in range(total_frames)
                ]

                # Process results as they complete
                for future in tqdm(futures, desc="Processing frames"):
                    result, frame = future.result()
                    if result:
                        output_dataset.resize(output_dataset.shape[0] + 1, axis=0)
                        output_dataset[-1] = frame

            print(f"Total frames: {total_frames}")
            print(f"Number of good frames saved: {output_dataset.shape[0]}")

# Example usage
input_h5_path = '/home/buster/wsl-TEST1/R2aOx.h5'
output_h5_path = '/home/buster/wsl-TEST1/R2aOx_cleaned.h5'
mask_h5_path = '/home/buster/wsl-TEST1/mask/pxmask.h5'

adc_thresh = 50
min_snr = 7
min_pix_count = 1
max_pix_count = 20
local_bg_radius = 9
min_peaks = 15
min_res = 0
max_res = None
num_threads = 16  # Adjust based on the number of available CPU cores (8 faster than 24 so maybe a number betwee 8 and 16 (no of cores))

remove_bad_frames(input_h5_path, output_h5_path, mask_h5_path, adc_thresh, min_snr, min_pix_count, max_pix_count, local_bg_radius, min_peaks, min_res, max_res, num_threads)
