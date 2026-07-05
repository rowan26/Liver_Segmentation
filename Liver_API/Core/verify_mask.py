# verify_mask.py
import nibabel as nib
import numpy as np

mask = nib.load("mask_output.nii.gz").get_fdata()

print(f"Shape              : {mask.shape}")
print(f"Classes présentes  : {np.unique(mask)}")
print(f"Voxels background  : {np.sum(mask == 0)}")
print(f"Voxels foie        : {np.sum(mask == 1)}")
print(f"Voxels tumeur      : {np.sum(mask == 2)}")
print(f"% foie             : {np.sum(mask == 1) / mask.size * 100:.2f}%")
print(f"% tumeur           : {np.sum(mask == 2) / mask.size * 100:.2f}%")