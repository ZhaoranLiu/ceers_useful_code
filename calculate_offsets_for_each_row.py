from astropy.io import fits
import numpy as np

from astropy.io import fits
import numpy as np
import scipy.ndimage

def calculate_offsets_and_resample(galaxy_id, seg_map_path, grism_image_row):
    hdu_seg = fits.open(seg_map_path)
    seg_map = hdu_seg[0].data
    hdu_seg.close()

    center_row = seg_map.shape[0] // 2
    offsets = np.zeros(seg_map.shape[0])

    # Calculate offsets for each row
    for row_index in range(seg_map.shape[0]):
        galaxy_pixels = np.where(seg_map[row_index, :] == galaxy_id)[0]
        if len(galaxy_pixels) > 0:
            center = galaxy_pixels[0] + len(galaxy_pixels) // 2
            offsets[row_index] = center - seg_map.shape[1] // 2
        else:
            offsets[row_index] = 0

    # Resample offsets to match Grism image rows
    resampled_offsets = scipy.ndimage.zoom(offsets, grism_image_row / seg_map.shape[0], order=0)
    resampled_offsets = np.rint(resampled_offsets).astype(int)
    
    return resampled_offsets