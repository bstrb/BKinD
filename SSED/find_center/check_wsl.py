import os
import h5py

# File paths
uox_file = "/home/buster/UOX1/UOX1_original/UOX1.h5"
mask_file = "/home/buster/mask/pxmask.h5"

# Check existence
print("Checking file existence...")
print(f"UOX1.h5 exists: {os.path.exists(uox_file)}")
print(f"pxmask.h5 exists: {os.path.exists(mask_file)}")

# Check HDF5 file access
try:
    with h5py.File(uox_file, 'r') as f:
        print("Successfully accessed UOX1.h5")
        print("Keys:", list(f.keys()))
except Exception as e:
    print(f"Error accessing UOX1.h5: {e}")

try:
    with h5py.File(mask_file, 'r') as f:
        print("Successfully accessed pxmask.h5")
        print("Keys:", list(f.keys()))
except Exception as e:
    print(f"Error accessing pxmask.h5: {e}")
