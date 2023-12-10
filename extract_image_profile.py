import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import sys
import os
from astropy.cosmology import FlatLambdaCDM
from astropy.modeling.functional_models import Sersic1D as Ser
import matplotlib.cm as cm
import math
pi = np.pi
from scipy.ndimage import gaussian_filter1d
from astropy.stats import sigma_clip
from astropy.cosmology import FlatLambdaCDM
import numpy as np

import numpy as np
import matplotlib
from scipy import stats
import math
import matplotlib.pyplot as mplt
import pandas as pd
from astropy.cosmology import WMAP9 as cosmo
from astropy import constants as const
from astropy.coordinates import SkyCoord
from astropy.coordinates import match_coordinates_sky
from astropy import units as u
from astropy.table import Table
from astropy.table import Table
from matplotlib import rcParams
from matplotlib.pyplot import MultipleLocator


## The seg map and sci map must have same size & wcs ##
def extract_light_profile(id, sci, seg, xcen, ycen, half_box_size, direction, sigma=1.0, sigma_clip_threshold=5.0):

    hdu_sci = fits.open(sci)
    hdu_seg = fits.open(seg)
    sci_cut = hdu_sci[0].data[int(ycen-box_half_size):int(ycen+box_half_size), int(xcen-box_half_size):int(xcen+box_half_size)]
    seg_cut = hdu[0].data[int(ycen-box_half_size):int(ycen+box_half_size), int(xcen-box_half_size):int(xcen+box_half_size)]
    
    for k in range(len(seg_cut[0])):
        for j in range(len(seg_cut)):
            if seg_cut[j][k] != id:
                #seg_cut[j][k] = 0
                sci_cut[j][k] = 0

    if direction == 'R':
        profile = np.sum(box, axis=1)
    elif direction == 'C':
        profile = np.sum(box, axis=0)
    else:
        raise ValueError("Invalid direction. Use 'R' or 'C'.")

    # Sigma clipping
    clipped_profile = sigma_clip(profile, sigma=sigma_clip_threshold, masked=True)

    smoothed_profile = gaussian_filter1d(clipped_profile, sigma=sigma, mode='nearest')
    #normalized_profile = smoothed_profile / np.max(smoothed_profile)

    return smoothed_profile