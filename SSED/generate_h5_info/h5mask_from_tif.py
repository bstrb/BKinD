import numpy as np
from PIL import Image
import h5py
from pathlib import Path

# -------------------------------------------------
# 1.  Load the TIFF (white = valid, black = invalid)
# -------------------------------------------------
tif_path = Path('/Users/xiaodong/Desktop/mask_2.tif')
img = Image.open(tif_path).convert('L')       # force single‑channel, 0‑255
arr = np.array(img)

# White (255) → 1,  black (0) → 0
mask = (arr > 0).astype(np.uint8)            # uint8, values {0, 1}

# -------------------------------------------------
# 2.  Write it to /mask in pxmask.h5
# -------------------------------------------------
out_h5 = Path('/Users/xiaodong/Desktop/pxmask.h5')
with h5py.File(out_h5, 'w') as f:
    dset = f.create_dataset(
        'mask',               # path = /mask
        data=mask,            # 1024×1024, uint8
        dtype='uint8',
        compression=None,     # to match your reference object
        chunks=None,          # ditto
        fillvalue=0
    )
    # (Optional) provenance
    dset.attrs['source'] = str(tif_path.name)
    dset.attrs['description'] = 'Binary ED mask: 1=valid, 0=masked'

print('✓ wrote', out_h5.resolve())
