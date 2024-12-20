import h5py
# import numpy as np

# Input file and output file
input_file = "/home/bubl3932/files/UOX1/UOX1_original/UOX_His_MUA_450nm_spot4_ON_20240311_0928.h5"    
output_file = "/home/bubl3932/files/UOX1/UOX1_original/UOX1_subset.h5"

# Indices of the entries you want to copy, e.g. a list of specific frames
# This can be user input, e.g. chosen_indices = list(map(int, input("Enter indices: ").split()))
chosen_indices = [  87, 88,
                    92,
                    108, #109,
                    4400, 4401, 4402, #4403, 
                    4404, #4405,
                    6339, 6340, 6341, 6342, 6343, 6344,
                    6355, 6356,
                    6368, 6369, 6370,
                    6375, 6376,
                    6408, 6409, 
                    9548, 9549, 9550,
                    9554,
                    9568, 9569,
                    9571,
                    9575,
                    9584, 9585, 9586,
                    9590, 9591, 9592,
                    10011, 10012,
                    10018, 10019, 10020, 10021,
                    10072, 10073,
                    10712, 10713, 10714, 10715, 10716,
                    12150, 12151,
                    13106, 13107,
                    13773,
                    13777, 13778,
                    14813, 14814, 14815, 14816, 14817, 14818,
                    15513, 15514,
                    15526,
                    15534, 15535, 15536,
                    15546, 15547, 15548,
                    15554, 15555,
                    15561, 15562, 15563, 15564, 15565, 15566, 
                    15569, 15570,
                    16188, 16189, 16190, 16191, 16192, 
                    16196, 16197,
                    16202, 
                    16205, 16206, 
                    16211, 16212, 16213,
                    16230,
                    18390, 18391,
                    25780, 25781
                    ]  # Example: pick entries at indices 0, 5, and 10
chosen_indices.sort()  # Add this line after defining chosen_indices

with h5py.File(input_file, 'r') as f_in, h5py.File(output_file, 'w') as f_out:
    
    # Function to copy attributes from one object to another
    def copy_attributes(source_obj, target_obj):
        for attr_name, attr_value in source_obj.attrs.items():
            target_obj.attrs.create(attr_name, attr_value)

    # Copy group structure
    # We'll create /entry and /entry/data groups in the output file
    entry_group_out = f_out.create_group("entry")
    copy_attributes(f_in["/entry"], entry_group_out)

    data_group_out = entry_group_out.create_group("data")
    copy_attributes(f_in["/entry/data"], data_group_out)
    
    # List of datasets to copy. Each dataset's first dimension matches the indexing dimension.
    # According to your structure, all datasets under /entry/data are indexed by the first dimension,
    # except they vary in shape. We'll handle them individually.
    
    # We'll define a helper function to copy a dataset subset.
    def copy_dataset_subset(dset_name):
        dset_in = f_in["/entry/data/" + dset_name]
        
        # Original shape and dtype
        orig_shape = dset_in.shape
        orig_dtype = dset_in.dtype
        
        # Calculate appropriate chunk size for the new dataset
        if dset_in.chunks:
            new_chunks = list(dset_in.chunks)
            # Ensure first dimension of chunk is not larger than output size
            new_chunks[0] = min(new_chunks[0], len(chosen_indices))
            new_chunks = tuple(new_chunks)
        else:
            new_chunks = None
        
        if len(orig_shape) == 1:
            # Shape like (N,)
            # We'll create a new dataset with shape (len(chosen_indices),)
            new_shape = (len(chosen_indices),)
            dset_out = data_group_out.create_dataset(
                dset_name,
                shape=new_shape,
                dtype=orig_dtype,
                chunks=new_chunks,  # Use adjusted chunk size
                compression=dset_in.compression,
                compression_opts=dset_in.compression_opts,
                shuffle=(dset_in.shuffle if hasattr(dset_in, 'shuffle') else None),
                fletcher32=dset_in.fletcher32
            )
            # Copy data
            dset_out[:] = dset_in[chosen_indices]
        
        elif len(orig_shape) == 2:
            # Shape like (N, M)
            # We'll only slice the first dimension by chosen_indices, keep second dimension intact.
            new_shape = (len(chosen_indices), orig_shape[1])
            dset_out = data_group_out.create_dataset(
                dset_name,
                shape=new_shape,
                dtype=orig_dtype,
                chunks=dset_in.chunks,
                compression=dset_in.compression,
                compression_opts=dset_in.compression_opts,
                shuffle=(dset_in.shuffle if hasattr(dset_in, 'shuffle') else None),
                fletcher32=dset_in.fletcher32
            )
            # Copy data
            dset_out[:] = dset_in[chosen_indices, :]
        
        elif len(orig_shape) == 3:
            # For the images dataset (N, X, Y)
            new_shape = (len(chosen_indices), orig_shape[1], orig_shape[2])
            
            # Adjust chunks for 3D dataset
            if dset_in.chunks:
                new_chunks = list(dset_in.chunks)
                new_chunks[0] = min(new_chunks[0], len(chosen_indices))  # Ensure first dimension isn't larger than output
                new_chunks = tuple(new_chunks)
            else:
                new_chunks = None
            
            dset_out = data_group_out.create_dataset(
                dset_name,
                shape=new_shape,
                dtype=orig_dtype,
                chunks=new_chunks,  # Use adjusted chunks instead of original
                compression=dset_in.compression,
                compression_opts=dset_in.compression_opts,
                shuffle=(dset_in.shuffle if hasattr(dset_in, 'shuffle') else None),
                fletcher32=dset_in.fletcher32
            )
            dset_out[:] = dset_in[chosen_indices, :, :]

        else:
            # If there are other shapes not covered, handle accordingly.
            raise ValueError(f"Unexpected dataset shape for {dset_name}: {orig_shape}")

        # Copy attributes
        copy_attributes(dset_in, dset_out)
    
    # Now copy each dataset using the helper function:
    datasets_to_copy = [
        "center_x",
        "center_y",
        "det_shift_x_mm",
        "det_shift_y_mm",
        "images",
        "index",
        "nPeaks",
        "peakTotalIntensity",
        "peakXPosRaw",
        "peakYPosRaw"
    ]

    for dname in datasets_to_copy:
        copy_dataset_subset(dname)
    
    # After this, the output HDF5 should have the same structure, datatypes, and
    # attributes, but only contain the chosen entries.
    print("Subset copying complete. Output saved to:", output_file)
