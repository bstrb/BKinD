# Peak Finder Function Definitions

import os
import h5py
import fnmatch
import numpy as np

from diffractem.peakfinder8_extension import peakfinder_8


def findpeaks(h5_path, x0, y0, threshold, min_snr, min_pix_count, max_pix_count, local_bg_radius, min_res, max_res):

    """
    Find peaks in an image stack and save the results to an HDF5 file.

    Parameters:
    h5_path (str): The path to the HDF5 file containing the image stack.
    x0 (float): The x-coordinate of the center of the image.
    y0 (float): The y-coordinate of the center of the image.
    threshold (float): The threshold for peak detection.
    min_snr (float): The minimum signal-to-noise ratio for peak detection.
    min_pix_count (int): The minimum number of pixels for peak detection.
    max_pix_count (int): The maximum number of pixels for peak detection.
    local_bg_radius (int): The radius of the local background region.
    min_res (float): The minimum resolution for peak detection.
    max_res (float): The maximum resolution for peak detection.
    """

    all_results = []
    
    with h5py.File(h5_path, 'r+') as workingfile:  # Open the file in read/write mode
        print(f"started processing {os.path.basename(h5_path)}")
        stack_shape = workingfile['entry/data/images'].shape

        # Delete datasets if they already exist
        for dataset_name in ['nPeaks', 'peakTotalIntensity', 'peakXPosRaw', 'peakYPosRaw', 'index']:
            full_name = f'entry/data/{dataset_name}'
            if full_name in workingfile:
                del workingfile[full_name]
                print('peak datasets already exist - deleting old data')

        # Create new datasets
        for dataset_name in ['nPeaks', 'index']:
            full_name = f'entry/data/{dataset_name}'
            if full_name not in workingfile:
                workingfile.create_dataset(full_name, shape=(stack_shape[0],), dtype=int)

        for dataset_name in ['peakTotalIntensity', 'peakXPosRaw', 'peakYPosRaw']:
            full_name = f'entry/data/{dataset_name}'
            if full_name not in workingfile:
                workingfile.create_dataset(full_name, shape=(stack_shape[0], 500), dtype=float)  # 2D shape
        
        print('new datasets created')

        
        for i in range(stack_shape[0]):
                image_data = workingfile['entry/data/images'][i]
                result = findpeaks_single_frame(i, image_data, x0, y0, threshold, min_snr, min_pix_count, max_pix_count, local_bg_radius, min_res, max_res)
                all_results.append(result)

        # Populate the data after sorting the results
        all_results = sorted(all_results, key=lambda x: x['index'])
        print(workingfile['entry/data/peakTotalIntensity'].shape)     
         
        for res in all_results:
            idx = res['index']
            workingfile['entry/data/nPeaks'][idx] = res['nPeaks']
            truncated_peak_intensity = res['peakTotalIntensity'][:500]
            workingfile['entry/data/peakTotalIntensity'][idx] = truncated_peak_intensity
            truncated_peakXPosRaw = res['peakXPosRaw'][:500]
            workingfile['entry/data/peakXPosRaw'][idx] = truncated_peakXPosRaw
            truncated_peakYPosRaw = res['peakYPosRaw'][:500]
            workingfile['entry/data/peakYPosRaw'][idx] = truncated_peakYPosRaw
            workingfile['entry/data/index'][idx] = idx

    print(f"finished processing {os.path.basename(h5_path)}")

def findpeaks_single_frame(i, image_data, x0, y0, threshold, min_snr, min_pix_count, max_pix_count, local_bg_radius, min_res, max_res):
    
    """
    Find peaks in a single frame of an image stack.

    Parameters:
    i (int): The index of the frame.
    image_data (numpy.ndarray): The image data.
    x0 (float): The x-coordinate of the center of the image.
    y0 (float): The y-coordinate of the center of the image.
    threshold (float): The threshold for peak detection.
    min_snr (float): The minimum signal-to-noise ratio for peak detection.
    min_pix_count (int): The minimum number of pixels for peak detection.
    max_pix_count (int): The maximum number of pixels for peak detection.
    local_bg_radius (int): The radius of the local background region.
    min_res (float): The minimum resolution for peak detection.
    max_res (float): The maximum resolution for peak detection.
    """

    if i % 1000 == 0:
       print(str(i) + ' frames processed') 

    nPeaks = 0
    X, Y = np.meshgrid(range(image_data.shape[1]), range(image_data.shape[0]))
    R = np.sqrt((X-x0)**2 + (Y-y0)**2).astype(np.float32)
    
    mask = np.ones_like(image_data, dtype=np.int8)
    mask[R > max_res] = 0
    mask[R < min_res] = 0

    pks = peakfinder_8(500, image_data.astype(np.float32), mask, R, image_data.shape[1], image_data.shape[0], 1, 1, threshold, min_snr, min_pix_count, max_pix_count, local_bg_radius)
    nPeaks = len(pks[0])   

    if pks is None or len(pks[0]) == 0:
        fill = [0] * (500)
        return {
            
            'index': i,
            'nPeaks': 0,
            'peakTotalIntensity': np.array(fill),
            'peakXPosRaw': np.array(fill),
            'peakYPosRaw': np.array(fill),
        }
    fill = [0] * (500 - nPeaks)

    return {
        'index': i,
        'nPeaks': nPeaks,
        'peakTotalIntensity': np.array(pks[2] + fill),
        'peakXPosRaw': np.array(pks[0] + fill),
        'peakYPosRaw': np.array(pks[1] + fill),
    }

# Now only h5 files in given folder
def find_files_and_run_peakfinding(folder_path, x0, y0, threshold, min_snr, min_pix_count, max_pix_count, local_bg_radius, min_res, max_res):
    """
    Find peaks in all HDF5 files in a given folder.

    Parameters:
    folder_path (str): The path to the folder containing the HDF5 files.
    x0 (float): The x-coordinate of the center of the image.
    y0 (float): The y-coordinate of the center of the image.
    threshold (float): The threshold for peak detection.
    min_snr (float): The minimum signal-to-noise ratio for peak detection.
    min_pix_count (int): The minimum number of pixels for peak detection.
    max_pix_count (int): The maximum number of pixels for peak detection.
    local_bg_radius (int): The radius of the local background region.
    min_res (float): The minimum resolution for peak detection.
    max_res (float): The maximum resolution for peak detection.
    """
    for filename in fnmatch.filter(os.listdir(folder_path), '*.h5'):
        filepath = os.path.join(folder_path, filename)
        print(f"Processing {filepath}")
        # Call your processing function
        findpeaks(filepath, x0, y0, threshold, min_snr, min_pix_count, max_pix_count, local_bg_radius, min_res, max_res)
