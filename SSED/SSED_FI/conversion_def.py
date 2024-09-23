# conversion_definitions.py

import os
import gc
import h5py
import shutil
import fnmatch
import numpy as np
import hyperspy.api as hs

def find_path_in_h5(h5file):
    base_path = '/Data/Image/'
    if base_path in h5file:
        image_group = h5file[base_path]
        for subfolder in image_group:
            print(subfolder)
            data_path = f'{base_path}{subfolder}/Data'
            if data_path in h5file:
                print(data_path)
                return data_path

def velox_conversion(h5file_path):
    print('converting ' + str(h5file_path))
    backup_file_path = h5file_path + ".backup"
    base_name = os.path.splitext(h5file_path)[0]
    new_file_path = base_name + ".h5"
    
    with h5py.File(h5file_path, 'r+') as workingfile:
        framepath = find_path_in_h5(workingfile)

    print(f'original datset located in: {framepath}')
    new_framepath = 'entry/data/images'

    # Rename the original file
    shutil.move(h5file_path, backup_file_path)
    print("original file renamed to backup")

    # Create a new file with the original name
    with h5py.File(new_file_path, 'w') as new_file, h5py.File(backup_file_path, 'r') as backup_file:
        print("new file created with the original name")

        # Chunk and write the frames
        print(f"started copying image data")    
        dataset = backup_file[framepath]
        x_dim, y_dim, z_dim = dataset.shape
        chunk_size = (1000, y_dim, x_dim)
        new_dataset_name = new_framepath

        chunked_dataset = new_file.create_dataset(new_dataset_name, shape=(z_dim, y_dim, x_dim),dtype=dataset.dtype, chunks=chunk_size)
        for z in range(z_dim):
            chunked_dataset[z, :, :] = dataset[:, :, z]
            
        print("chunked dataset created and data copied")

    # Delete original file
    os.remove(backup_file_path)
    print("conversion successful, backup file deleted")

def find_and_process_velox_files(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for filename in fnmatch.filter(files, '*.emd'):
            filepath = os.path.join(root, filename)
            velox_conversion(filepath)

def ser_conversion(originalfile_path):
    print('converting ' + str(originalfile_path))
    emi_data = hs.load(originalfile_path, only_valid_data=True)
    n_frames, height, width = emi_data.data.shape

    chunk_size = (1000, height, width)

    base_name = os.path.splitext(originalfile_path)[0]
    output_h5_file = base_name + ".h5"

    # Backup and rename existing file
    if os.path.exists(output_h5_file):
        backup_file_path = output_h5_file + ".backup"
        shutil.move(output_h5_file, backup_file_path)

    # Create new HDF5 file for output
    with h5py.File(output_h5_file, 'w') as h5f:
        # Create dataset for storing frames
        new_framepath = 'entry/data/images'
        dset = h5f.create_dataset(new_framepath, shape=(n_frames, height, width), dtype=np.float32, chunks=chunk_size)

        print(f"started processing and chunking EMI data into {output_h5_file}")
        # Process and write the data in chunks
        for start in range(0, n_frames, chunk_size[0]):
            end = min(start + chunk_size[0], n_frames)
            # Load a chunk
            chunk_data = emi_data.data[start:end, :, :].astype(np.float32)
            # Write the chunk
            dset[start:end, :, :] = chunk_data
            del chunk_data
            gc.collect()

        print("SER file successfully converted and chunked into HDF5 format")

    # Cleanup
    del emi_data
    gc.collect()
    os.remove(originalfile_path)
    print("conversion successful, backup file deleted")

def find_and_process_ser_files(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for filename in fnmatch.filter(files, '*.ser'):
            filepath = os.path.join(root, filename)
            ser_conversion(filepath)