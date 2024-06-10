# my_grism/core.py

from .config import (
    DEFAULT_MODA_LW_POM_PATH,
    DEFAULT_MODB_LW_POM_PATH,
    DEFAULT_MATPLT_STYLE,
    DEFAULT_PS_PATH_TEMPLATE,
    DEFAULT_POM_PATH_TEMPLATE,
    DEFAULT_SPEC_DISPER,
    DEFAULT_SPEC_SENSI,
    USER_RATE_FILES_PATH,
    USER_FILTER,
    USER_CALIBRATED_DIR,
    USER_RATE_LV15_FILES_PATH,
    USER_SW_RATE_FILES_PATH,
    USER_ASTROMETRIC_CATALOG_PATH,
    USER_TB_F444W_PATH,
    USER_OUTPUT_SPECTRA_PATH,
    USER_PATH_LW,
    USER_TARGET
)

# Astropy modules
from tqdm import tqdm
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.io import fits, ascii
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import MinMaxInterval, PercentileInterval, SqrtStretch, ZScaleInterval
from astropy import units as u
from astropy import constants as c
from astropy import wcs, table
from astropy.coordinates import SkyCoord
from astropy.table import Table, Column, vstack
from astropy.time import Time
from astropy.modeling.models import custom_model
from astropy.cosmology import FlatLambdaCDM
import os
# Photutils modules
from photutils.detection import find_peaks, DAOStarFinder, IRAFStarFinder
from photutils import psf
from photutils.aperture import CircularAperture, aperture_photometry, CircularAnnulus, RectangularAperture, RectangularAnnulus
from photutils.background import Background2D, MedianBackground, SExtractorBackground
from photutils.segmentation import detect_sources, SourceFinder, SourceCatalog

# Photutils background estimator
sigma_clip = SigmaClip(sigma=3.)
bkg_estimator = MedianBackground()

# Numpy and Scipy modules
import numpy as np
from scipy import interpolate, integrate, optimize, stats, ndimage, signal

# Matplotlib modules
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import matplotlib.animation as animation

# Optional modules
import glob
import os
import subprocess
import time
import re

# Multiprocessing
from multiprocessing import Pool, Lock, get_context

# External packages
import grismconf
import pysiaf
import asdf
# os.environ['CRDS_PATH'] = '/Volumes/Extreme_S4/zhaoran_wfss/crds_cache'

os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'
import crds
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
import jwst
import json
import asdf
from jwst.pipeline import Detector1Pipeline
from jwst.pipeline import calwebb_image2, calwebb_image3
from jwst import assign_wcs, datamodels, tweakreg
from jwst.flatfield import flat_field
## https://github.com/spacetelescope/crds
import crds 
## https://tshirt.readthedocs.io/en/latest/index.html
max_retries = 5
for i in range(max_retries):
    try:
        from tshirt.pipeline.instrument_specific import rowamp_sub
        print("Successfully imported rowamp_sub from tshirt.pipeline.instrument_specific")
        break
    except ImportError as e:
        print(f"Attempt {i+1}/{max_retries} failed: {e}")
        time.sleep(1)
else:
    raise ImportError("Failed to import rowamp_sub from tshirt.pipeline.instrument_specific after several attempts")
## idk why but I need to import at least two times to get this module loaded 
#from tshirt.pipeline.instrument_specific import rowamp_sub
# Filter warnings
import warnings
warnings.filterwarnings("ignore", message="'obsfix' made the change")
warnings.filterwarnings("ignore", message="'datfix' made the change")
warnings.filterwarnings("ignore", message="Card is too long")
warnings.filterwarnings("ignore", message="divide by zero")
warnings.filterwarnings("ignore", message="invalid value encountered")
warnings.filterwarnings("ignore", message="Input data contains invalid values")

# Default nircam zeropoint with native pixel size in LW (0.0629 arcsec/pixel)
nircam_LW_ZP = -2.5 * np.log10((u.MJy / u.sr * (0.0629*u.arcsec)**2 / (3631 * u.Jy)).cgs.value)

'''
Matplotlib Drawing:
'''
plt.rcParams.update({'font.size': 16})  ## larger font size

def get_corner_pos(ax, loc = 1, edge = 0.02):
    '''get (x, y) for the corner of an axes at given quadrant (1, 2, 3, 4)'''
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xscale, yscale = ax.get_xscale(), ax.get_yscale()
    if loc not in [1, 2, 3, 4]: raise ValueError('The parameter `loc` must be 1 or 2 or 3 or 4.')
    vec_edge = np.array([1-edge, edge])
    vec_x = np.array([[xlim[1], xlim[0], xlim[0], xlim[1]][loc-1], [xlim[0], xlim[1], xlim[1], xlim[0]][loc-1]])
    vec_y = np.array([[ylim[1], ylim[1], ylim[0], ylim[0]][loc-1], [ylim[0], ylim[0], ylim[1], ylim[1]][loc-1]])
    if xscale == 'linear': x_text = np.dot(vec_edge, vec_x)
    else: x_text = 10 ** np.dot(vec_edge, np.log10(vec_x))
    if yscale == 'linear': y_text = np.dot(vec_edge, vec_y)
    else: y_text = 10 ** np.dot(vec_edge, np.log10(vec_y))
    return (x_text, y_text)

def corner_text(ax, s ='', loc = 1, edge = 0.02, **kwargs):
    '''Add text at the corner of an axes'''
    if loc not in [1, 2, 3, 4]: raise ValueError('The parameter `loc` must be 1 or 2 or 3 or 4.')
    ha = ['right', 'left', 'left', 'right']
    va = ['top', 'top', 'bottom', 'bottom']
    return ax.text(*get_corner_pos(ax, loc, edge), s=s, ha = ha[loc-1], va = va[loc-1], **kwargs)

def rotate(axis, angle):
    '''Fundamental rotation matrices.
    Rotate by angle measured in degrees, about axis 1 2 or 3'''
    if axis not in list(range(1, 4)):
        print ('Axis must be in range 1 to 3')
        return
    r = np.zeros((3, 3))
    ax0 = axis-1 #Allow for zero offset numbering
    theta = angle * np.pi / 180. # radians(angle)
    r[ax0, ax0] = 1.0
    ax1 = (ax0+1) % 3
    ax2 = (ax0+2) % 3
    r[ax1, ax1] = np.cos(theta)
    r[ax2, ax2] = np.cos(theta)
    r[ax1, ax2] = - np.sin(theta)
    r[ax2, ax1] = np.sin(theta)
    return r

def attitude(v2, v3, ra, dec, pa):
    '''This will make a rotation matrix which rotates a unit vector representing a v2, v3 position
    to a unit vector representing an RA, Dec pointing with an assigned position angle
    Described in JWST-STScI-001550, SM-12, section 6.1'''
    # v2, v3 in arcsec, ra, dec and position angle in degrees
    v2d = v2/3600.0
    v3d = v3/3600.0

    # Get separate rotation matrices
    mv2 = rotate(3, -v2d)
    mv3 = rotate(2, v3d)
    mra = rotate(3, ra)
    mdec = rotate(2, -dec)
    mpa = rotate(1, -pa)

    # Combine as mra*mdec*mpa*mv3*mv2
    m = np.dot(mv3, mv2)
    m = np.dot(mpa, m)
    m = np.dot(mdec, m)
    m = np.dot(mra, m)

    return m

'''This is the Grism Dispersion function, will be detailed later'''
def fit_disp_order32(data, 
                     a01, a02, a03, a04, a05, a06, 
                     b01, b02, b03, b04, b05, b06, 
                     c01, c02, c03, #c04, c05, c06,
                     d01, #d02, d03, d04, d05, d06,
                    ):
    ## data is an numpy array of the shape (3, N)
    ##     - data[0]:  x_pixel      --> fit with second-degree polynomial
    ##     - data[1]:  y_pixel      --> fit with second-degree polynomial
    ##     - data[2]:  wavelength   --> fit with third-degree polynomial
    xpix, ypix, dx = data[0] - 1024, data[1] - 1024, data[2] - 3.95
    ## return dx = dx(x_pixel, y_pixel, lambda)
    return ((a01 + (a02 * xpix + a03 * ypix) + (a04 * xpix**2 + a05 * xpix * ypix + a06 * ypix**2)
            ) + 
            (b01 + (b02 * xpix + b03 * ypix) + (b04 * xpix**2 + b05 * xpix * ypix + b06 * ypix**2)
            ) * dx +
            (c01 + (c02 * xpix + c03 * ypix) #+ (c04 * xpix**2 + c05 * xpix * ypix + c06 * ypix**2)
            ) * dx**2 + 
            (d01 #+ (d02 * xpix + d03 * ypix) + (d04 * xpix**2 + d05 * xpix * ypix + d06 * ypix**2)
            ) * dx**3
           ) 

func_fit_wave = fit_disp_order32


'''This is the Spectral Tracing function, will be detailed later'''
def fit_disp_order23(data, 
                     a01, a02, a03, a04, a05, a06, a07, a08, a09, a10,
                     b01, b02, b03, b04, b05, b06, b07, b08, b09, b10,
                     c01, c02, c03, c04, c05, c06, c07, c08, c09, c10,
                    ):
    ## data is an numpy array of the shape (3, N)
    ##     - data[0]:  x_pixel  --> fit with second-degree polynomial
    ##     - data[1]:  y_pixel  --> fit with second-degree polynomial
    ##     - data[2]:  dx     --> fit with second-degree polynomial
    xpix, ypix, dx = data[0] - 1024, data[1] - 1024, data[2]
    ## return dy = dy(x_pixel, y_pixel, d_x)
    return ((a01 + (a02 * xpix + a03 * ypix) + (a04 * xpix**2 + a05 * xpix * ypix + a06 * ypix**2)
             + (a07 * xpix**3 + a08 * xpix**2 * ypix + a09 * xpix * ypix**2 + a10 * ypix**3)
            ) + 
            (b01 + (b02 * xpix + b03 * ypix) + (b04 * xpix**2 + b05 * xpix * ypix + b06 * ypix**2)
             + (b07 * xpix**3 + b08 * xpix**2 * ypix + b09 * xpix * ypix**2 + b10 * ypix**3)
            ) * dx +
            (c01 + (c02 * xpix + c03 * ypix) + (c04 * xpix**2 + c05 * xpix * ypix + c06 * ypix**2)
             + (c07 * xpix**3 + c08 * xpix**2 * ypix + c09 * xpix * ypix**2 + c10 * ypix**3)
            ) * dx**2
           ) 


def grism_conf_preparation(x0 = 1024, y0 = 1024, pupil = 'R', 
                           fit_opt_fit = np.zeros(30), w_opt = np.zeros(16)):
    '''
    Prepare grism configuration, dxs, dys, wavelengths based on input (x0, y0) pixel postion
        and filter/pupil/module information.
    -----------------------------------------------
        Parameters
        ----------  
        x0, y0 : float
            Reference position (i.e., in direct image)
        
        pupil: 'R' or 'C'
            pupil of grism ('R' or 'C')
        
        fit_opt_fit: numpy.ndarray, shape: (30,)
            polynomial parameters in the perpendicular direction, used by function `fit_disp_order23` 
        
        w_opt: numpy.ndarray, shape: (16,)
            polynomial parameters in the dispersed direction, used by function `fit_disp_order32` 
        
        Returns
        -------
        
        dxs, dys : `~numpy.ndarray`
            offset of spectral pixel from the direct imaging position
        
        wavs: `~numpy.ndarray`
            Array of wavelengths corresponding to dxs and dys
    '''
    # Load the Grism Configuration file
    # GConf = grismconf.Config(os.environ['MIRAGE_DATA'] + "/nircam/GRISM_NIRCAM/V3/" +
    #                          "NIRCAM_%s_mod%s_%s.conf" % (filter, module, pupil))
    wave_space = np.arange(2.39, 5.15, 0.01)
    disp_space = func_fit_wave(np.vstack((x0 * np.ones_like(wave_space), y0 * np.ones_like(wave_space), wave_space)), *w_opt)
    inverse_wave_disp = interpolate.UnivariateSpline(disp_space[np.argsort(disp_space)], wave_space[np.argsort(disp_space)], s = 0, k = 1)

    if pupil == 'R':
        dxs = np.arange(int(np.min(disp_space)), int(np.max(disp_space))) - x0%1
        # Compute wavelength of each of the pixels
        wavs = inverse_wave_disp(dxs)
        # Compute the dys values for the same pixels
        dys = fit_disp_order23(np.vstack((x0* np.ones_like(dxs), y0 * np.ones_like(dxs), dxs)), *fit_opt_fit)
        # dxs = np.arange(-1800, 1800, 1) - x0%1
        ## Compute the t values corresponding to the exact offsets (with old grism config)
        # ts = GConf.INVDISPX(order = '+1', x0 = x0, y0 = y0, dx = dxs)
        # dys = GConf.DISPY('+1', x0, y0, ts)
        # wavs = GConf.DISPL('+1', x0, y0, ts)
        # tmp_aper = np.max([0.2, 1.5 * tmp_re_maj * np.cos(tmp_pa), 1.5 * tmp_re_min * np.sin(tmp_pa)])
    elif pupil == 'C':
        # Compute the dys values for the same pixels
        dys = np.arange(int(np.min(disp_space)), int(np.max(disp_space))) - y0%1
        # Compute wavelength of each of the pixels
        wavs = inverse_wave_disp(dys)
        dxs = fit_disp_order23(np.vstack((x0* np.ones_like(dys), y0 * np.ones_like(dys), dys)), *fit_opt_fit)
        # dys = np.arange(-1800, 1800, 1) - y0%1
        ## Compute the t values corresponding to the exact offsets (with old grism config)
        # ts = GConf.INVDISPY(order = '+1', x0 = x0, y0 = y0, dy = dys)
        # dxs = GConf.DISPX('+1', x0, y0, ts)
        # wavs = GConf.DISPL('+1', x0, y0, ts)
    return (dxs, dys, wavs)

'''Lienar Function'''
linear = lambda x, k, b: x * k + b
'''gaussian function'''
gauss = lambda x, x0, flux, FWHM : (flux / FWHM / 1.064467) * np.exp(-(x - x0)**2 / (2 * (FWHM/2.354820)**2))
'''gaussian function + continuum'''
gauss_cont_prof = lambda x, x0, flux, FWHM, k, b : (flux / FWHM / 1.064467) * np.exp(-(x - x0)**2 / (2 * (FWHM/2.354820)**2)) + k * (x - 1024) + b


modA_LW_POM_fits = fits.open(DEFAULT_MODA_LW_POM_PATH)
modB_LW_POM_fits = fits.open(DEFAULT_MODB_LW_POM_PATH)
modA_LW_POM_trans = modA_LW_POM_fits[1].data
modB_LW_POM_trans = modB_LW_POM_fits[1].data
xy_start_POM_modA = modA_LW_POM_fits[1].header['NOMXSTRT'], modA_LW_POM_fits[1].header['NOMYSTRT']
xy_start_POM_modB = modB_LW_POM_fits[1].header['NOMXSTRT'], modA_LW_POM_fits[1].header['NOMYSTRT']

modA_LW_POM_fits.close()
modB_LW_POM_fits.close()

def is_pickoff(x, y, module='A'):
    '''
    Input x/y pixel(s) and module, find whether the source will be picked off by NIRCam POM.
    '''
    if module.upper() == 'A': xy_start, POM_trans = xy_start_POM_modA, modA_LW_POM_trans
    elif module.upper() == 'B': xy_start, POM_trans = xy_start_POM_modB, modB_LW_POM_trans
    else: raise ValueError("Only \"A\" or \"B\" is allowed for module.")
    if type(x) == np.ndarray: tmp_x, tmp_y = np.int32(x), np.int32(y)
    elif type(x) == list: tmp_x, tmp_y = np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)
    elif type(x) == int: tmp_x, tmp_y = np.array([x], dtype=np.int32), np.array([y], dtype=np.int32)
    elif type(x) == float: tmp_x, tmp_y = np.array([x], dtype=np.int32), np.array([y], dtype=np.int32)
    else: raise TypeError("Only integers/floats or their list/np.array() are allowed.")
    if len(tmp_x) != len(tmp_y): raise ValueError("x, y list should be of the same length.")
    tmp_x = tmp_x + xy_start[0]
    tmp_y = tmp_y + xy_start[1]
    arr_trans = []
    for j in range(len(tmp_x)):
        if (tmp_x[j] < 0) | (tmp_y[j] < 0) | (tmp_x[j] > POM_trans.shape[1] - 1) | (tmp_y[j] > POM_trans.shape[0] - 1):
            arr_trans.append(0)
        else:
            arr_trans.append(POM_trans[tmp_x[j], tmp_x[j]])
    return np.array(arr_trans, dtype=int)

def is_pickoff_PS(x, y, module='A', filter='F444W', pupil='R'):
    '''
    Input x/y pixel(s) and module, 
    find whether the source will be picked off by NIRCam POM and yield partial/complete spectra.
    '''
    path_PS = DEFAULT_PS_PATH_TEMPLATE % (filter, module, pupil)
    path_POM = DEFAULT_POM_PATH_TEMPLATE % module
    if os.path.isfile(path_PS) == False: 
        raise FileNotFoundError("Spectral Coverage file %s not found!" % path_PS)
    spec_cov_fits = fits.open(path_PS)
    xy_start = (spec_cov_fits[0].header['NOMXSTRT'], spec_cov_fits[0].header['NOMYSTRT'])
    POM_trans = spec_cov_fits[0].data
    spec_cov_fits.close()
    if type(x) == np.ndarray: tmp_x, tmp_y = np.int32(x), np.int32(y)
    elif type(x) == list: tmp_x, tmp_y = np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)
    elif type(x) == int: tmp_x, tmp_y = np.array([x], dtype=np.int32), np.array([y], dtype=np.int32)
    elif type(x) == float: tmp_x, tmp_y = np.array([x], dtype=np.int32), np.array([y], dtype=np.int32)
    else: raise TypeError("Only integers/floats or their list/np.array() are allowed.")
    if len(tmp_x) != len(tmp_y): raise ValueError("x, y list should be of the same length.")
    tmp_x = tmp_x + xy_start[0]
    tmp_y = tmp_y + xy_start[1]
    arr_trans = []
    for j in range(len(tmp_x)):
        if (tmp_x[j] < 0) | (tmp_y[j] < 0) | (tmp_x[j] > POM_trans.shape[1] - 1) | (tmp_y[j] > POM_trans.shape[0] - 1):
            arr_trans.append(0)
        elif tmp_y[j] > 2250:
            arr_trans.append(0)
        else:
            arr_trans.append(POM_trans[tmp_y[j], tmp_x[j]])
    return np.array(arr_trans, dtype=float)
    
    
#now we grab the LW grism files
list_rate = np.array(glob.glob(USER_RATE_FILES_PATH))
list_rate.sort()
list_filter = []
list_pupil = []
list_module = []
list_target = []
list_exp = []

for x in list_rate:
    tmp_hd = fits.getheader(x)
    list_filter.append(tmp_hd['FILTER'])
    list_pupil.append(tmp_hd['PUPIL'])
    list_module.append(tmp_hd['MODULE'])
    list_target.append(tmp_hd['TARGPROP'])
    list_exp.append(tmp_hd['EFFEXPTM'])

list_filter = np.array(list_filter)
list_pupil = np.array(list_pupil)
list_module = np.array(list_module)
list_target = np.array(list_target)
list_exp = np.array(list_exp)

print(np.unique(list_filter), '\n',  np.unique(list_pupil), '\n', 
      np.unique(list_module), '\n',  np.unique(list_target))
 # Stage 2.1 Background subtraction + hot pixel subtraction
# 2.1.1 get list of rate.fits by module and pupil
tmp_filter = USER_FILTER 
list_rate_this_band = list_rate[(list_filter == tmp_filter) & (list_pupil != 'CLEAR')]
print(len(list_rate_this_band), 'files,\n', list_rate_this_band[0])

calibrated_dir = USER_CALIBRATED_DIR
if not os.path.isdir(calibrated_dir): 
    os.mkdir(calibrated_dir)
if not os.path.isdir(os.path.join(calibrated_dir, 'plots/')): 
    os.mkdir(os.path.join(calibrated_dir, 'plots/'))

def background_grism_stage2(arr_grism_rate, tmp_module, tmp_pupil, tmp_filter):
    print('\n>> Create Median Background for %s - module %s - grism %s' % (tmp_filter, tmp_module, tmp_pupil))
    tmp_med_bkg = sigma_clipped_stats(arr_grism_rate, axis=-1, sigma_upper=2.0, maxiters=10)[1]
    tmp_med_bkg_path = os.path.join(calibrated_dir, 'median_bkg_%s_mod%s_%s.fits' % (tmp_filter, tmp_module, tmp_pupil))
    tmp_med_bkg_fits = fits.HDUList(fits.PrimaryHDU(tmp_med_bkg))
    tmp_med_bkg_fits[0].header['object'] = 'BKG_%s_mod%s_%s' % (tmp_filter, tmp_module, tmp_pupil)
    tmp_med_bkg_fits[0].header['filter'] = tmp_filter
    tmp_med_bkg_fits[0].header['module'] = tmp_module
    tmp_med_bkg_fits[0].header['pupil'] = tmp_pupil
    tmp_med_bkg_fits.writeto(tmp_med_bkg_path, overwrite=True)
    print('>> Saved background models to %s' % tmp_med_bkg_path)

def assignwcs_grism_stage2(rate_grism_file):
    rate_grism_fits = fits.open(rate_grism_file)
    tmp_grism_hd = rate_grism_fits[0].header
    siaf_file = crds.getreferences(tmp_grism_hd, reftypes=['distortion'], ignore_cache=False)['distortion']
    tmp_filter, tmp_module, tmp_pupil = tmp_grism_hd['filter'], tmp_grism_hd['module'], tmp_grism_hd['pupil']
    print('   read %s' % rate_grism_file, '(%s - %s - %s)' % (tmp_filter, tmp_module, tmp_pupil))
    grism_wcs_step = assign_wcs.assign_wcs_step.AssignWcsStep(override_distortion=siaf_file)
    rate_grism_fits[0].header['EXP_TYPE'] = 'NRC_IMAGE'
    grism_image = datamodels.image.ImageModel(rate_grism_fits)
    grism_with_wcs = grism_wcs_step(grism_image)
    tmp_flat_path = crds.getreferences({
        'INSTRUME': tmp_grism_hd['INSTRUME'], 'READPATT': tmp_grism_hd['READPATT'], 
        'SUBARRAY': tmp_grism_hd['SUBARRAY'], 
        'DATE-OBS': tmp_grism_hd['DATE'].split('T')[0], 'TIME-OBS': tmp_grism_hd['DATE'].split('T')[1],
        'DETECTOR': tmp_grism_hd['DETECTOR'], 'CHANNEL': tmp_grism_hd['CHANNEL'], 
        'MODULE':  tmp_grism_hd['MODULE'], 'EXP_TYPE': 'NRC_IMAGE',
        'FILTER': tmp_grism_hd['FILTER'], 'PUPIL': 'CLEAR'
    })['flat']
    flat_field.do_flat_field(grism_with_wcs, datamodels.FlatModel(tmp_flat_path))
    grism_save_path = rate_grism_file.replace('rate.fits', 'rate_lv1.5.fits')
    grism_save_path = os.path.join(calibrated_dir, os.path.basename(grism_save_path))
    print("Saving file with proper WCS in", grism_save_path)
    grism_with_wcs.save(grism_save_path)

def get_crds_dict_from_fits_header(tmp_hd):
    crds_dict = {}
    crds_dict['INSTRUME'] = tmp_hd['INSTRUME'].upper()
    crds_dict['READPATT'], crds_dict['SUBARRAY'] = tmp_hd['readpatt'].upper(), tmp_hd['SUBARRAY'].upper()
    crds_dict['DATE-OBS'], crds_dict['TIME-OBS'] = tmp_hd['DATE-OBS'], tmp_hd['TIME-OBS']
    crds_dict['DETECTOR'] = tmp_hd['DETECTOR']
    if crds_dict['INSTRUME'] == 'NIRCAM':
        if crds_dict['DETECTOR'] in ['NRCALONG', 'NRCBLONG']:
            crds_dict['CHANNEL'] = 'LONG'
        else:
            crds_dict['CHANNEL'] = 'SHORT'
    crds_dict['MODULE'] = tmp_hd['module']
    crds_dict['EXP_TYPE'], crds_dict['FILTER'], crds_dict['PUPIL'] = tmp_hd['EXP_TYPE'], tmp_hd['filter'], tmp_hd['pupil']
    return crds_dict

def process_rate_files():
    """
    """
    list_rate_files = np.array(glob.glob(USER_RATE_FILES_PATH))
    list_rate_files.sort()

    list_filter = []
    list_pupil = []
    for rate_file in list_rate_files:
        tmp_hd = fits.getheader(rate_file)
        list_filter.append(tmp_hd['FILTER'])
        list_pupil.append(tmp_hd['PUPIL'])
    list_filter = np.array(list_filter)
    list_pupil = np.array(list_pupil)

    tmp_filter = USER_FILTER
    list_rate_this_band = list_rate_files[(list_filter == tmp_filter) & (list_pupil != 'CLEAR')]

    for rate_file in tqdm(list_rate_this_band, desc="Processing files", unit="file"):
        assignwcs_grism_stage2(rate_file)
    
    all_rate_list = np.array([os.path.join(USER_CALIBRATED_DIR, os.path.basename(x).replace('rate.fits', 'rate_lv1.5.fits'))
                              for x in list_rate_files])
    return all_rate_list

def get_lv1_5_rate_files():
    list_rate_1p5 = np.array(glob.glob(USER_RATE_LV15_FILES_PATH))
    list_rate_1p5.sort()
    if len(list_rate_1p5) == 0:
        raise FileNotFoundError("Reduced frames not found. Please run 'mygrism super_background' first or check the 'USER_RATE_LV15_FILES_PATH' in config.py")
    return list_rate_1p5


def background_grism_stage2(arr_grism_rate, tmp_module, tmp_pupil, tmp_filter):
    tmp_med_bkg = sigma_clipped_stats(arr_grism_rate, axis=-1, sigma_upper=2.0, maxiters=10)[1]
    tmp_med_bkg_path = os.path.join(USER_CALIBRATED_DIR, f'median_bkg_{tmp_filter}_mod{tmp_module}_{tmp_pupil}.fits')
    tmp_med_bkg_fits = fits.HDUList(fits.PrimaryHDU(tmp_med_bkg))
    tmp_med_bkg_fits[0].header['object'] = f'BKG_{tmp_filter}_mod{tmp_module}_{tmp_pupil}'
    tmp_med_bkg_fits[0].header['filter'] = tmp_filter
    tmp_med_bkg_fits[0].header['module'] = tmp_module
    tmp_med_bkg_fits[0].header['pupil'] = tmp_pupil
    tmp_med_bkg_fits.writeto(tmp_med_bkg_path, overwrite=True)
    print(f'Saved background models to {tmp_med_bkg_path}')
    del arr_grism_rate
    del tmp_med_bkg
    tmp_med_bkg_fits.close()

def generate_sky_background(data_type):
    list_rate_1p5 = get_lv1_5_rate_files()
    if len(list_rate_1p5) == 0:
        print("Error: No rate_lv1.5 files found. Please run 'wcs_flat_fielding' first.")
        return
    
    print(f"- Background + Hot pixel construction for {USER_FILTER} -")
    count_ac, count_bc, count_ar, count_br = 0, 0, 0, 0

    for i, tmp_path_rate in enumerate(list_rate_1p5):
        tmp_grism_hd = fits.getheader(tmp_path_rate)
        tmp_filter, tmp_module, tmp_pupil = tmp_grism_hd['filter'], tmp_grism_hd['module'], tmp_grism_hd['pupil']
        print(f'>>> [{i:3d}] read {tmp_path_rate} ({tmp_filter} {tmp_module} {tmp_pupil})')
        tmp_grism_rate = fits.getdata(tmp_path_rate, 'sci')
        if (tmp_module == 'A') & (tmp_pupil == 'GRISMC'):
            if count_ac == 0:  arr_grism_rate_ac = tmp_grism_rate
            else: arr_grism_rate_ac = np.dstack((arr_grism_rate_ac, tmp_grism_rate))
            count_ac += 1
        if (tmp_module == 'B') & (tmp_pupil == 'GRISMC'):
            if count_bc == 0:  arr_grism_rate_bc = tmp_grism_rate
            else: arr_grism_rate_bc = np.dstack((arr_grism_rate_bc, tmp_grism_rate))
            count_bc += 1
        if (tmp_module == 'A') & (tmp_pupil == 'GRISMR'):
            if count_ar == 0:  arr_grism_rate_ar = tmp_grism_rate
            else: arr_grism_rate_ar = np.dstack((arr_grism_rate_ar, tmp_grism_rate))
            count_ar += 1
        if (tmp_module == 'B') & (tmp_pupil == 'GRISMR'):
            if count_br == 0:  arr_grism_rate_br = tmp_grism_rate
            else: arr_grism_rate_br = np.dstack((arr_grism_rate_br, tmp_grism_rate))
            count_br += 1
    print(count_ar, count_ac, count_br, count_bc)

    if data_type == 'GRISMC':
        list_arr_grism_rate = [arr_grism_rate_ac, arr_grism_rate_bc]
        list_arr_grism_module = ['A', 'B']
        list_arr_grism_pupil = ['GRISMC', 'GRISMC']
    elif data_type == 'GRISMR':
        list_arr_grism_rate = [arr_grism_rate_ar, arr_grism_rate_br]
        list_arr_grism_module = ['A', 'B']
        list_arr_grism_pupil = ['GRISMR', 'GRISMR']
    elif data_type == 'BOTH':
        list_arr_grism_rate = [arr_grism_rate_ar, arr_grism_rate_ac, arr_grism_rate_br, arr_grism_rate_bc]
        list_arr_grism_module = ['A', 'A', 'B', 'B']
        list_arr_grism_pupil = ['GRISMR', 'GRISMC', 'GRISMR', 'GRISMC']
    else:
        raise ValueError("Invalid data type. Use 'GRISMC', 'GRISMR', or 'BOTH'.")

    n_procs = 8
    ctx = get_context("fork")  # 使用 'fork' 启动方法
    with ctx.Pool(np.min([n_procs, len(list_rate_1p5)])) as pool:
        try:
            pool.starmap(background_grism_stage2, zip(list_arr_grism_rate, list_arr_grism_module, 
                                                      list_arr_grism_pupil, [USER_FILTER]*len(list_arr_grism_rate)))
        finally:
            pool.close()
            pool.join()
            del pool

def my_grism_bkg_subtraction(rate_grism_file):
    import matplotlib.pyplot as plt 
    print('>>> subtract background for %s ' % rate_grism_file)
    tmp_grism_fits = fits.open(rate_grism_file)
    tmp_grism_img = tmp_grism_fits[1].data
    tmp_grism_hd = tmp_grism_fits[0].header
    tmp_grism_bkg = fits.getdata(os.path.join(USER_CALIBRATED_DIR, f'median_bkg_{tmp_grism_hd["filter"]}_mod{tmp_grism_hd["module"]}_{tmp_grism_hd["pupil"]}.fits'))
    plt.close()
    fig, ax = plt.subplots(2, 2, figsize=(15, 15))
    ax = ax.flatten()
    '''ax-0: flat-fielded grism image'''
    ax[0].imshow(np.nan_to_num(tmp_grism_img), origin='lower',
                 vmin=np.nanpercentile(tmp_grism_img, 2.5), vmax=np.nanpercentile(tmp_grism_img, 97.5))
    corner_text(ax[0], s='(1) Flat-Fielded', color='w', loc=2, weight='bold', fontsize=20)

    '''ax-1: subtract sigma-clipped median sky background'''

    tmp_grism_img = tmp_grism_img - tmp_grism_bkg
    tmp_grism_fits[0].header['HISTORY'] = 'sigma-clipped median sky background subtracted on %s' % time.strftime("%Y/%m/%d",  time.localtime())
    ax[1].imshow(np.nan_to_num(tmp_grism_img), origin='lower',
                 vmin=np.nanpercentile(tmp_grism_img, 2.5), vmax=np.nanpercentile(tmp_grism_img, 97.5))
    corner_text(ax[1], s='(2) Median Sky BKG Subtracted', color='w', loc=2, weight='bold', fontsize=20)
    if (tmp_grism_hd['filter'], tmp_grism_hd['module'], tmp_grism_hd['pupil'][-1]) == ('F322W2', 'B', 'R'):
        corner_text(ax[1], s='(Not applied for modB - GRISMR)', color='w', loc=4, weight='bold', fontsize=20)

    # mask-1: from segment map
    _, tmp_med, tmp_rms = sigma_clipped_stats(tmp_grism_img[100:700, 100:700])
    segment_map = detect_sources(tmp_grism_img - tmp_med, tmp_rms * 1.0, npixels=100)

    ### further background subtraction?
    sigma_clip = SigmaClip(sigma=2.5)
    bkg_estimator = SExtractorBackground()
    if (tmp_grism_hd['filter'], tmp_grism_hd['module'], tmp_grism_hd['pupil'][-1]) == ('F322W2', 'B', 'R'):
        tmp_mask = segment_map.data != 0
        bkg = Background2D(tmp_grism_img, (24, 24), filter_size=(5, 5), mask=tmp_mask,
                           sigma_clip=sigma_clip, bkg_estimator=bkg_estimator, exclude_percentile=50.)
        tmp_grism_img = tmp_grism_img - bkg.background
        tmp_grism_fits[0].header['HISTORY'] = 'SExtractor-modeled 2D background subtracted on %s' % time.strftime("%Y/%m/%d",  time.localtime())
    else:
        tmp_grism_img = tmp_grism_img
        tmp_grism_fits[0].header['HISTORY'] = 'No other 2D background subtracted'
        tmp_mask = np.zeros_like(segment_map.data)

    '''ax-2: subtract modeled 2D sky background'''
    ax[2].imshow(np.nan_to_num(tmp_grism_img), origin='lower',
                 vmin=np.nanpercentile(tmp_grism_img, 2.5), vmax=np.nanpercentile(tmp_grism_img, 97.5))
    corner_text(ax[2], s='(3) Modeled 2D BKG Subtracted (Not applied)', color='w', loc=2, weight='bold', fontsize=20)

    ### for grism C: subtract 1/f noise
    if tmp_grism_hd['pupil'][-1] == 'C':
        _, tmp_med, tmp_rms = sigma_clipped_stats(tmp_grism_img[100:1700, 100:1700].flatten()[::10])
        segment_map = detect_sources(tmp_grism_img, tmp_rms * 5.0, npixels=20)
        rowSub, modelImg = rowamp_sub.do_backsub(tmp_grism_img,
                                                 amplifiers=4, backgMask=segment_map.data == 0)
        tmp_grism_img = rowSub
        tmp_grism_fits[0].header['HISTORY'] = '1/f noise subtracted on %s' % time.strftime("%Y/%m/%d",  time.localtime())

    '''ax-3: subtract 1/f noise'''
    ax[3].imshow(np.nan_to_num(tmp_grism_img), origin='lower',
                 vmin=np.nanpercentile(tmp_grism_img, 2.5), vmax=np.nanpercentile(tmp_grism_img, 97.5))
    corner_text(ax[3], s='(4) 1/f Noise Subtracted', color='w', loc=2, weight='bold', fontsize=20)
    if tmp_grism_hd['pupil'][-1] == 'R':
        corner_text(ax[3], s='(Not applied for GRISMR)', color='w', loc=4, weight='bold', fontsize=20)

    tmp_title = '%s (%s - %s - %s)' % (os.path.basename(rate_grism_file).replace('_lv1.5.fits', ''),
                                       tmp_grism_hd['filter'], tmp_grism_hd['module'], tmp_grism_hd['pupil'])
    fig.text(0.5, 0.985, s=tmp_title, ha='center', va='top', fontsize=24,)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.01, wspace=0.06, hspace=0.04)

    for tmp_ax in ax:
        tmp_ax.set_xticks([]); tmp_ax.set_yticks([])
    plt.savefig(os.path.join(USER_CALIBRATED_DIR, 'plots', f'{tmp_grism_hd["filter"]}_mod{tmp_grism_hd["module"]}_{tmp_grism_hd["pupil"]}_{tmp_title.split(" ")[0]}.pdf'), dpi=100)
    plt.close()

    '''Write Fits File'''
    tmp_grism_fits[1].data = tmp_grism_img
    tmp_grism_fits.writeto(rate_grism_file, overwrite=True)
    tmp_grism_fits.close()
    
    
# def run_bkg_subtraction():
#     list_rate_1p5 = get_lv1_5_rate_files()
#     n_procs = 8
#     ctx = get_context("fork") 
# 
#     with ctx.Pool(np.min([n_procs, len(list_rate_1p5)])) as pool:
#         try:
#             pool.map(my_grism_bkg_subtraction, list_rate_1p5)
#         finally:
#             pool.close()
#             pool.join()
#             del pool
def run_bkg_subtraction():
    list_rate_1p5 = get_lv1_5_rate_files()
    for rate_file in list_rate_1p5:
        my_grism_bkg_subtraction(rate_file)
        
def full_pipeline(data_type):
    process_rate_files()
    generate_sky_background(data_type)
    run_bkg_subtraction()
    

def my_grism_cont_subtraction(rate_grism_file, L_box, L_mask):
    import matplotlib.pyplot as plt  
    tmp_grism_fits = fits.open(rate_grism_file)
    tmp_grism_sci = tmp_grism_fits['sci'].copy()
    tmp_grism_img = tmp_grism_sci.data
    tmp_grism_hd = tmp_grism_fits[0].header

    ### median filter the data using a box with a hole:
    if tmp_grism_hd['pupil'][-1] == 'R':
        mf_footprint = np.ones((1, L_box * 2 + 1))
        mf_footprint[:, L_box - L_mask:L_box + L_mask + 1] = 0
    elif tmp_grism_hd['pupil'][-1] == 'C':
        mf_footprint = np.ones((L_box * 2 + 1, 1))
        mf_footprint[L_box - L_mask:L_box + L_mask + 1, :] = 0
    else:
        raise KeyError('pupil should be in GRISMR or GRISMC!')
    tmp_grism_img_median = ndimage.median_filter(tmp_grism_img, footprint=mf_footprint, mode='reflect')
    tmp_grism_img_emline = tmp_grism_img - tmp_grism_img_median  ## emission line map
    ### horizontal stripe (1/f) removal
    rowSub, modelImg_horizontal = rowamp_sub.do_backsub(tmp_grism_img_emline, amplifiers=1)
    ### vertical stripe (1/f) removal
    rowSub, modelImg_vertical = rowamp_sub.do_backsub(rowSub.T, amplifiers=1)
    rowSub = rowSub.T
    print('EMLINE map produced for %s' % rate_grism_file)

    ### save as an extension in the same files
    tmp_grism_sci.header['EXTNAME'] = 'EMLINE'
    tmp_grism_sci.header['HISTORY'] = 'Continuum Subtracted by Median Filter, Line-Only Map'
    tmp_grism_sci.data = rowSub
    if len(tmp_grism_fits) == 8:
        tmp_grism_fits.append(tmp_grism_sci)
    else:
        tmp_grism_fits['EMLINE'] = tmp_grism_sci

    fig, ax = plt.subplots(1, 2, figsize=(14.5, 7))
    ax[0].imshow(np.nan_to_num(tmp_grism_img), vmin=-0.02, vmax=0.02, origin='lower')
    ax[1].imshow(np.nan_to_num(rowSub), vmin=-0.02, vmax=0.02, origin='lower')
    for tmp_ax in ax:
        tmp_ax.set_xticks([], [])
        tmp_ax.set_yticks([], [])
    plt.tight_layout()
    for tmp_ax in ax:
        tmp_ax.set_xticks([])
        tmp_ax.set_yticks([])
    plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.01, wspace=0.06, hspace=0.04)
    plt.savefig(os.path.join(USER_CALIBRATED_DIR, 'plots', f'{tmp_grism_hd["filter"]}_mod{tmp_grism_hd["module"]}_{tmp_grism_hd["pupil"]}_{os.path.basename(rate_grism_file).replace("_lv1.5.fits", "")}_EMLINE.pdf'), dpi=100)
    plt.close()

    '''Write Fits File'''
    tmp_grism_fits.writeto(rate_grism_file, overwrite=True)
    tmp_grism_fits.close()

def run_cont_subtraction(L_box, L_mask):
    list_rate_1p5 = get_lv1_5_rate_files()
    for rate_file in list_rate_1p5:
        my_grism_cont_subtraction(rate_file, L_box, L_mask)


def get_sw_rate_files():
    list_rate_sw = np.array(glob.glob(os.path.join(USER_SW_RATE_FILES_PATH, '*rate.fits')))
    list_rate_sw.sort()
    if len(list_rate_sw) == 0:
        raise FileNotFoundError("SW rate files not found. Please check the path in the config file.")
    return list_rate_sw



def process_sw_rate_files():
    list_rate_sw = get_sw_rate_files()
    print(f"Found {len(list_rate_sw)} SW rate files.")
    
    list_sw_filter = []
    list_sw_pupil = []
    list_sw_module = []
    list_sw_target = []
    list_sw_exp = []
    for x in list_rate_sw:
        tmp_hd = fits.getheader(x)
        list_sw_filter.append(tmp_hd['FILTER'])
        list_sw_pupil.append(tmp_hd['PUPIL'])
        list_sw_module.append(tmp_hd['MODULE'])
        list_sw_target.append(tmp_hd['TARGPROP'])
        list_sw_exp.append(tmp_hd['EFFEXPTM'])
    
    list_sw_filter = np.array(list_sw_filter)
    list_sw_pupil = np.array(list_sw_pupil)
    list_sw_module = np.array(list_sw_module)
    list_sw_target = np.array(list_sw_target)
    
    print("Filters:", np.unique(list_sw_filter))
    print("Modules:", np.unique(list_sw_module))
    print("Targets:", np.unique(list_sw_target))

    return list_rate_sw, list_sw_filter, list_sw_pupil, list_sw_module, list_sw_target, list_sw_exp

def reduce_img_stage2(rate_image_file, wing='sw',output_dir=USER_SW_RATE_FILES_PATH):
    '''Function to reduce jwst/nircam image at stage-2'''
    
    dict_lev2 = {"asn_type" : "image2", 
                 "asn_rule": "DMSLevel2bBase",
                 "version_id": None, "code_version": jwst.__version__,
                 "degraded_status": "No known degraded exposures in association.",
                 "constraints": "No constraints", "asn_id": "a3001", "asn_pool": "none",
                 }
    dict_products = []
    
    if wing == 'lw':
        tmp_basename = os.path.basename(rate_image_file)[:34]
    elif wing == 'sw':
        tmp_basename = os.path.basename(rate_image_file)[:31]
    tmp_dict_members = {"expname": rate_image_file, "exptype": "science"}
    tmp_dict_product = {"name": os.path.basename(rate_image_file).replace('_rate_rowsub.fits', '').replace('_rate.fits', ''), 
                        "members": [tmp_dict_members]}
    dict_products.append(tmp_dict_product)

    dict_lev2['products'] = dict_products
    json_object = json.dumps(dict_lev2, indent=4)
    
    with open(os.path.join(output_dir, f"nircam_stage2_dir_img_{tmp_basename}.json"), "w") as outfile:
        outfile.write(json_object)
        
    os.chdir(output_dir)
    asn_file = f"nircam_stage2_dir_img_{tmp_basename}.json"
    image2 = calwebb_image2.Image2Pipeline()
    image2.output_dir = './'
    image2.save_results = True
    image2.resample.skip = True
    image2.run(asn_file)
    
    os.remove(asn_file)

def run_stage2_on_sw_files():
    list_rate_sw = get_sw_rate_files()
    n_procs = 8
    ctx = get_context("fork")  
    with ctx.Pool(n_procs) as pool:
        try:
            pool.map(reduce_img_stage2, list_rate_sw)
        finally:
            pool.close()
            pool.join()
            del pool
            
def my_daofind_sw_fits(tmp_rate_sw):
    '''
    If SW cal.fits images have been astrometrically corrected and passed through stage-2 pipelines,
    we then need to restore the original astrometry and use the images provided by stage-2 pipeline
    '''
    ## assign wcs 
    rate_sw_fits = fits.open(tmp_rate_sw)
    rate_sw_hd = rate_sw_fits[0].header
    ### read distortion assignment
    siaf_file = crds.getreferences(rate_sw_hd, reftypes = ['distortion'], ignore_cache = False)['distortion']
    ### assign default WCS
    rate_sw_wcs_step = assign_wcs.assign_wcs_step.AssignWcsStep(override_distortion=siaf_file)
    rate_sw_IM = datamodels.image.ImageModel(rate_sw_fits)
    rate_sw_IM_wcs = rate_sw_wcs_step(rate_sw_IM)
    tmp_wcs = rate_sw_IM_wcs.get_fits_wcs()  ## this is the default wcs from observational parameters
    rate_sw_IM_wcs.close()
    rate_sw_fits.close()

    ### well-processed SW cal.fits files 
    direct_sw_fits = fits.open(USER_SW_RATE_FILES_PATH + os.path.basename(tmp_rate_sw).replace('_rate.fits', '_cal.fits'))
    tmp_coord_center = wcs.utils.pixel_to_skycoord(1024.5, 1024.5, tmp_wcs)
    ### construct detection image
    tmp_detect_img = np.nan_to_num(direct_sw_fits['sci'].data)
    bkg_estimator = MedianBackground()
    bkg = Background2D(tmp_detect_img, (64, 64), filter_size=(5, 5),
                       sigma_clip=SigmaClip(sigma=3.), bkg_estimator=bkg_estimator)
    tmp_detect_std = sigma_clipped_stats(tmp_detect_img[100:1500, 100:1500].flatten()[::7], 
                                         sigma=3, maxiters=10)[-1]
    tmp_detect_img = (tmp_detect_img - bkg.background) / tmp_detect_std
    
    ### run DAOStarFinder - change parameters (FWHM or Threshold if needed)
    print('run daofind on %s' % os.path.basename(tmp_rate_sw).replace('_rate.fits', '_cal.fits'))
    daofind = DAOStarFinder(fwhm=5.0, threshold=7.)  # DAOStarFinder(fwhm = 2.5, threshold = 8.)
    tmp_tb_daofind = daofind(tmp_detect_img)
    tmp_tb_daofind['detector'] = direct_sw_fits[0].header['detector'].lower()
    tmp_tb_daofind['skycoord'] = wcs.utils.pixel_to_skycoord(tmp_tb_daofind['xcentroid'], tmp_tb_daofind['ycentroid'], tmp_wcs)
    ### Save SW catalogs 
    if not os.path.exists(USER_CALIBRATED_DIR + 'astrom/'):
        os.makedirs(USER_CALIBRATED_DIR + 'astrom/')
    ascii.write(tmp_tb_daofind, USER_CALIBRATED_DIR + 'astrom/%s' % os.path.basename(tmp_rate_sw).replace('_rate.fits', '_daofind.cat'),
                format='ecsv', overwrite=True)
    direct_sw_fits.close()

def generate_sw_catalogs():
    list_rate_sw = get_sw_rate_files()
    print(f"{len(list_rate_sw)} SW rate files selected.")
    
    n_procs = 8
    ctx = get_context("fork")  
    with ctx.Pool(np.min([n_procs, len(list_rate_sw)])) as pool:
        try:
            pool.map(my_daofind_sw_fits, list_rate_sw)
        finally:
            pool.close()
            pool.join()
            del pool

    idx_unfinished = np.array([not os.path.isfile(os.path.join(USER_CALIBRATED_DIR, 'astrom', os.path.basename(x).replace('_rate.fits', '_daofind.cat')))
                               for x in list_rate_sw])
    print(np.sum(idx_unfinished), 'unfinished.')
#     if np.sum(idx_unfinished) > 0:
#         with ctx.Pool(np.min([n_procs, len(list_rate_sw[idx_unfinished])])) as pool:
#             try:
#                 pool.map(my_daofind_sw_fits, list_rate_sw[idx_unfinished])
#             finally:
#                 pool.close()
#                 pool.join()
#                 del pool
                
def combined_reduce_and_generate_sw():
    run_stage2_on_sw_files()
    generate_sw_catalogs()    
    
def perform_astrometric_calculations():
    catalog_path = USER_ASTROMETRIC_CATALOG_PATH
    output_dir_astrometry = os.path.dirname(catalog_path)

    # Read astrometric catalog
    tb_LW = Table.read(catalog_path)
    tmp_RA, tmp_DEC = tb_LW['RA'].value, tb_LW['DEC'].value
    print(tb_LW[:5])

    # Get list of *_cal.fits files in the specified bands
    list_rate_sw = get_sw_rate_files()
    list_img_cal_this_band = np.array([os.path.join(USER_SW_RATE_FILES_PATH, os.path.basename(x).replace('_rate.fits', '_cal.fits')) 
                                       for x in list_rate_sw])
    
    # Initialize astrometry table
    list_img_cal_this_band.sort()
    tb_sw_astrometry = Table(names=['expName', 'N_match', 'dRA', 'dDEC', 'dRA_err', 'dDEC_err', 'theta', 'theta_err'], 
                             dtype=['S40', 'i4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4'])
    tb_sw_astrometry.meta['comments'] = [
        'RA/Dec offsets are in arcsec and absolute',
        'i.e., cos(Dec) projection has been considered.',
        '  >> dRA  =  RA_daofind -  RA_gaia',
        '  >> dDEC = DEC_daofind - DEC_gaia',
        'generated on %s' % time.strftime("%Y/%m/%d", time.localtime())
    ]
    for colname in tb_sw_astrometry.colnames[2:]:
        tb_sw_astrometry[colname].info.format = '.4f'
    sigma_clip = SigmaClip(sigma=2.)

    # Run the loop to compute astrometric offsets for each group of SW exposures
    print('>>>  Astrometry (RA/DEC) offsets calculation for %3d SW Frames' % len(list_img_cal_this_band))
    for k in range(len(list_img_cal_this_band) // 4):
        # For each cal.fits, get all the paths of cal.fits from all four detectors
        tmp_img_cal_path = list_img_cal_this_band[k * 4]
        tmp_img_cal_path_base = os.path.basename(tmp_img_cal_path)[:30]
        tmp_obsnum = int(tmp_img_cal_path_base[7:10])
        tmp_grism_LW_path = os.path.join(USER_CALIBRATED_DIR, tmp_img_cal_path_base + 'long_rate_lv1.5.fits')
        if not os.path.isfile(tmp_grism_LW_path):
            continue
        tmp_grism_hd_sci = fits.getheader(tmp_grism_LW_path, 1)
        print('%3d >> %s ' % (k, tmp_img_cal_path_base))
        tmp_img_cal_hd = fits.getheader(tmp_img_cal_path, 0)
        tmp_detector = tmp_img_cal_hd['DETECTOR'].lower()
        tmp_img_cal_sci_hd = fits.getheader(tmp_img_cal_path, 'sci')
        tmp_time_obs = Time(tmp_img_cal_sci_hd['MJD-AVG'], format='mjd')
        tmp_detectors = np.array(['%s%d' % (tmp_detector[:-1], tmp_det) for tmp_det in [1, 2, 3, 4]])
        tmp_img_cal_paths = np.array([tmp_img_cal_path.replace(tmp_detector, x) for x in tmp_detectors])

        # Run DAOFIND to get a star catalog
        daofind = DAOStarFinder(fwhm=3.0, threshold=8.0)
        for j, tmp_img_cal_path in enumerate(tmp_img_cal_paths):
            if not os.path.isfile(tmp_img_cal_path):
                print('%s not found!' % tmp_img_cal_path)
                continue
            tmp_img_cal_sci_hd = fits.getheader(tmp_img_cal_path, 'sci')
            # Directly use prepared SW DAOFIND catalog
            tmp_tb_daofind = ascii.read(os.path.join(USER_CALIBRATED_DIR, 'astrom', os.path.basename(tmp_img_cal_path).replace('_cal.fits', '_daofind.cat').replace('_rate.fits', '_daofind.cat')),
                                        format='ecsv')
            if j == 0:
                tb_daofind = tmp_tb_daofind
            else:
                tb_daofind = vstack((tb_daofind, tmp_tb_daofind))
        tb_daofind = tb_daofind[np.argsort(tb_daofind['peak'])][-200:]
        tmp_coord_daofind = tb_daofind['skycoord']

        # Astrometric Source RA & DECs
        tmp_coord_LW = SkyCoord(tmp_RA, tmp_DEC, unit=(u.deg, u.deg))
        # Rough center of the frame
        tmp_coord_center = SkyCoord(np.median(tmp_coord_daofind.ra), np.median(tmp_coord_daofind.dec))
        # Only select sources close to the center
        tmp_coord_LW = tmp_coord_LW[tmp_coord_center.separation(tmp_coord_LW) < 4 * u.arcmin]

        # Cross match DAOFind Catalog with Gaia Catalog
        idx_daofind, d2d, _ = tmp_coord_LW.match_to_catalog_sky(tmp_coord_daofind)
        idx_LW = np.where(d2d < 0.50 * u.arcsec)[0]
        idx_daofind = idx_daofind[idx_LW]

        # Compute RA/DEC offset
        tmp_ra_offset = ((tmp_coord_daofind[idx_daofind].ra - tmp_coord_LW[idx_LW].ra) 
                         * np.cos(tmp_coord_LW[idx_LW].dec)).to(u.arcsec).value
        tmp_dec_offset = (tmp_coord_daofind[idx_daofind].dec - tmp_coord_LW[idx_LW].dec).to(u.arcsec).value

        # Find center of clustering, sigma-clipped RA/DEC offset
        dbin = 0.1
        tmp_ra_bins = np.arange(tmp_ra_offset.min()-0.01 - dbin/2, tmp_ra_offset.max() + dbin + 0.01, dbin)
        tmp_dec_bins = np.arange(tmp_dec_offset.min()-0.01- dbin/2, tmp_dec_offset.max() + dbin + 0.01, dbin)
        offset_hist2d = np.histogram2d(tmp_ra_offset, tmp_dec_offset, bins=(tmp_ra_bins, tmp_dec_bins))[0]
        idx_ra, idx_dec = np.where(offset_hist2d == np.nanmax(offset_hist2d))
        idx_ra, idx_dec = idx_ra[0], idx_dec[0]
        arg_clipped = np.where((tmp_ra_offset > tmp_ra_bins[idx_ra] - 1.5 * dbin) & 
                               (tmp_ra_offset < tmp_ra_bins[idx_ra] + 1.5 * dbin) &
                               (tmp_dec_offset > tmp_dec_bins[idx_dec] - 1.5 * dbin) & 
                               (tmp_dec_offset < tmp_dec_bins[idx_dec] + 1.5 * dbin))[0]
        tmp_ra_offset_clipped, tmp_dec_offset_clipped = tmp_ra_offset[arg_clipped], tmp_dec_offset[arg_clipped]
        idx_daofind = idx_daofind[arg_clipped]

        tmp_ra_offset_med, tmp_ra_offset_std = sigma_clipped_stats(tmp_ra_offset_clipped)[1:]
        tmp_dec_offset_med, tmp_dec_offset_std = sigma_clipped_stats(tmp_dec_offset_clipped)[1:]

        # Compute rotation
        poly_dec_dRA = optimize.curve_fit(f=linear, xdata=tmp_coord_daofind[idx_daofind].dec, ydata=tmp_ra_offset_clipped)[0]
        poly_ra_dDEC = optimize.curve_fit(f=linear, xdata=tmp_coord_daofind[idx_daofind].ra, ydata=tmp_dec_offset_clipped)[0]
        # Iteration 1
        arg_dec_dRA = np.where(sigma_clip(np.polyval(poly_dec_dRA, tmp_coord_daofind[idx_daofind].dec.value) - tmp_ra_offset_clipped).mask == False)[0]
        poly_dec_dRA = optimize.curve_fit(f=linear, xdata=tmp_coord_daofind[idx_daofind].dec[arg_dec_dRA], ydata=tmp_ra_offset_clipped[arg_dec_dRA])[0]
        arg_ra_dDEC = np.where(sigma_clip(np.polyval(poly_ra_dDEC, tmp_coord_daofind[idx_daofind].ra.value) - tmp_dec_offset_clipped).mask == False)[0]
        poly_ra_dDEC = optimize.curve_fit(f=linear, xdata=tmp_coord_daofind[idx_daofind].ra[arg_ra_dDEC], ydata=tmp_dec_offset_clipped[arg_ra_dDEC])[0]
        # Iteration 2
        arg_dec_dRA = np.where(sigma_clip(np.polyval(poly_dec_dRA, tmp_coord_daofind[idx_daofind].dec.value) - tmp_ra_offset_clipped).mask == False)[0]
        poly_dec_dRA, pcov_dec_dRA = optimize.curve_fit(f=linear, xdata=tmp_coord_daofind[idx_daofind].dec[arg_dec_dRA], ydata=tmp_ra_offset_clipped[arg_dec_dRA])
        arg_ra_dDEC = np.where(sigma_clip(np.polyval(poly_ra_dDEC, tmp_coord_daofind[idx_daofind].ra.value) - tmp_dec_offset_clipped).mask == False)[0]
        poly_ra_dDEC, pccov_ra_dDEC = optimize.curve_fit(f=linear, xdata=tmp_coord_daofind[idx_daofind].ra[arg_ra_dDEC], ydata=tmp_dec_offset_clipped[arg_ra_dDEC])
        perr_dec_dRA = np.sqrt(np.diag(pcov_dec_dRA))
        perr_ra_dDEC = np.sqrt(np.diag(pccov_ra_dDEC))
        # Combine two directions
        cos_factor = np.cos(np.deg2rad(np.median(tmp_coord_daofind[idx_daofind].dec.value)))
        theta_as_deg = (poly_dec_dRA[0] / perr_dec_dRA[0] - poly_ra_dDEC[0] / perr_ra_dDEC[0]) / (1/perr_dec_dRA[0] + cos_factor / perr_ra_dDEC[0]) / 3600.
        theta_as_deg_err = np.sum(perr_dec_dRA[0]**2 + (perr_ra_dDEC[0] / cos_factor)**2)**0.5 / 2. / 3600.
        theta_as_deg = np.rad2deg(theta_as_deg)
        theta_as_deg_err = np.rad2deg(theta_as_deg_err)

        tmp_ra_offset_med = tmp_grism_hd_sci['CRVAL2'] * poly_dec_dRA[0] + poly_dec_dRA[1]
        tmp_dec_offset_med = tmp_grism_hd_sci['CRVAL1'] * poly_ra_dDEC[0] + poly_ra_dDEC[1]

        tmp_ra_offset_std = sigma_clipped_stats(tmp_ra_offset_clipped)[2]
        tmp_dec_offset_std = sigma_clipped_stats(tmp_dec_offset_clipped)[2]
        print('           N_match = %d' % len(arg_clipped))
        print(' RA_image -  RA_ref = %.3f"±%.3f"' % (tmp_ra_offset_med, tmp_ra_offset_std / np.sqrt(len(arg_clipped))))
        print('DEC_image - DEC_ref = %.3f"±%.3f"' % (tmp_dec_offset_med, tmp_dec_offset_std / np.sqrt(len(arg_clipped))))
        print('     rotation theta = %.3f±%.3f deg' % (theta_as_deg, theta_as_deg_err))
        tb_sw_astrometry.add_row([tmp_img_cal_path_base, len(arg_clipped),
                                  tmp_ra_offset_med, tmp_dec_offset_med, 
                                  tmp_ra_offset_std / np.sqrt(len(arg_clipped)), tmp_dec_offset_std / np.sqrt(len(arg_clipped)),
                                  theta_as_deg, theta_as_deg_err])

        plt.close()
        plt.subplots(1, 1, figsize=(5, 5))
        plt.plot(tmp_ra_offset, tmp_dec_offset, marker='.', color='gray', ls='none')
        plt.plot(tmp_ra_offset_clipped, tmp_dec_offset_clipped, marker='.', color='k', ls='none')
        plt.plot(tmp_ra_offset_med, tmp_dec_offset_med, marker='+', color='red', ms=20, mew=2, ls='none')
        plt.axhline(0, color='grey', ls='--')
        plt.axvline(0, color='grey', ls='--')
        plt.gca().set(xlabel=r'$\Delta$RA', ylabel=r'$\Delta$Dec')
        if np.max(np.abs(np.array([tmp_ra_offset_med, tmp_dec_offset_med]))) > 1:
            plt.gca().set(aspect=1, xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), 
                          xticks=np.arange(-1.5, 1.51, 0.5), yticks=np.arange(-1.5, 1.51, 0.3))
        elif  np.max(np.abs(np.array([tmp_ra_offset_med, tmp_dec_offset_med]))) > 0.5:
            plt.gca().set(aspect=1, xlim=(-1., 1.), ylim=(-1., 1.), 
                          xticks=np.arange(-0.8, 1.1, 0.4), yticks=np.arange(-1., 1.1, 0.2))
        else:
            plt.gca().set(aspect=1, xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), 
                          xticks=np.arange(-0.4, 0.51, 0.2), yticks=np.arange(-0.5, 0.51, 0.1))

        corner_text(plt.gca(), s=' RA_image -  RA_ref = %.3f"±%.3f"' % (tmp_ra_offset_med,  tmp_ra_offset_std / np.sqrt(len(arg_clipped))), loc=1, fontsize=12)
        corner_text(plt.gca(), s='DEC_image - DEC_ref = %.3f"±%.3f"' % (tmp_dec_offset_med, tmp_dec_offset_std / np.sqrt(len(arg_clipped))), loc=4, fontsize=12)
        plt.title(tmp_img_cal_path_base, fontsize=14)
        plt.tight_layout()

    # Save the astrometry table to the same directory as the input catalog
        input_basename = os.path.basename(catalog_path)
        output_basename = input_basename.replace('.fits', '_astrometry.dat')
        output_path = os.path.join(output_dir_astrometry, output_basename)
        ascii.write(tb_sw_astrometry, output_path, overwrite=True)
        print(f"Astrometry results saved to {output_path}")

#Function to plot sources on top of FRESCO footprint."""
def plot_sources_on_footprint():
    
    catalog_path = USER_ASTROMETRIC_CATALOG_PATH
    output_dir_astrometry = os.path.dirname(catalog_path)
    input_basename = os.path.basename(USER_ASTROMETRIC_CATALOG_PATH)
    output_basename = input_basename.replace('.fits', '_astrometry.dat')
    output_path = os.path.join(output_dir_astrometry, output_basename)
    tb_sw_astrometry = ascii.read(output_path)
    
    # Read the user-provided tb_f444w catalog
    tb_f444w = ascii.read(USER_TB_F444W_PATH)
    
    # Get list of 1v1.5 files
    all_rate_list = get_lv1_5_rate_files()
    print(f"{len(all_rate_list)} wcs corrected fits files")
    
    # Read filter, module, and pupil
    all_filters, all_module, all_pupil = [], [], []
    all_roll_angle, all_target = [], []
    for i, x in enumerate(all_rate_list):
        tmp_header = fits.getheader(x)
        all_filters.append(tmp_header['filter'])
        all_module.append(tmp_header['module'])
        all_pupil.append(tmp_header['pupil'])
        all_roll_angle.append(fits.getheader(x, 1)['ROLL_REF'])
        all_target.append(tmp_header['TARGPROP'])
    all_filters, all_module, all_pupil = np.array(all_filters), np.array(all_module), np.array(all_pupil)
    all_roll_angle, all_target = np.array(all_roll_angle), np.array(all_target)
    print(np.unique(all_filters), np.unique(all_module), np.unique(all_target))
    
    # Input master catalog for spectral extraction
    coord_f444w = SkyCoord(tb_f444w['RA'], tb_f444w['DEC'], unit=(u.deg, u.deg))
    print(f"{len(coord_f444w)} sources to be extracted in total")
    
    # Coordinates in Science frame
    sci_x = np.array([0, 2047, 2047, 0])
    sci_y = np.array([0, 0, 2047, 2047])
    sci_fullcov_A_x = sci_x
    sci_fullcov_B_x = sci_x
    
    # Check source position on top of FRESCO footprint
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    for i, tmp_rate_path in enumerate(all_rate_list):
        tmp_module = fits.getheader(tmp_rate_path)['module']
        if tmp_module == 'A':
            tmp_color = '#1f77b4'
            sci_x = sci_fullcov_A_x
        else:
            tmp_color = '#ff7f0e'
            sci_x = sci_fullcov_B_x
        tmp_grism_hd_sci = fits.getheader(tmp_rate_path, 'sci')
        tmp_grism_hd_wcs = wcs.WCS(tmp_grism_hd_sci)
        coords_corners = wcs.utils.pixel_to_skycoord(sci_x, sci_y, tmp_grism_hd_wcs)
        coords_corners = np.array([coords_corners.ra.value, coords_corners.dec.value]).T
        line_fullcov = patches.Polygon(coords_corners, facecolor='none', ls='--', ec=tmp_color, closed=True, alpha=1)
        ax.add_patch(line_fullcov)
    
        if i == 0:
            all_corners = coords_corners
        else:
            all_corners = np.vstack((all_corners, coords_corners))
    
    # Show sources
    ax.plot(tb_f444w['RA'], tb_f444w['DEC'], marker='o', color='r', mfc='none', mew=1, ls='none')
    for i, item in enumerate(tb_f444w[:10]):
        ax.text(item['RA'] - 0.003, item['DEC'], s=f'{item["ID"]}\n(z={item["zspec"]:.3f})', ha='left', va='center', color='r', fontsize=14)
    # Format axes
    ax.set(xlim=(np.max(all_corners[:, 0]) + 1/60, np.min(all_corners[:, 0]) - 1/60),
           ylim=(np.min(all_corners[:, 1]) - 1/60, np.max(all_corners[:, 1]) + 1/60),
           xlabel='RA [deg]', ylabel='DEC [deg]')
    print(np.percentile(all_corners[:, 0], [0, 100]), np.percentile(all_corners[:, 1], [0, 100]))
    
    output_plot_path = os.path.join(os.path.dirname(output_dir_astrometry), 'source_positions_on_footprint.png')
    plt.savefig(output_plot_path)
    plt.close()
    print(f"Plot saved to {output_plot_path}")


def generate_pom_catalogs():
    #Function to generate POM catalogs and save them in a subdirectory
    # Read the astrometric catalog
    catalog_path = USER_ASTROMETRIC_CATALOG_PATH
    output_dir_astrometry = os.path.dirname(catalog_path)
    input_basename = os.path.basename(USER_ASTROMETRIC_CATALOG_PATH)
    output_basename = input_basename.replace('.fits', '_astrometry.dat')
    output_path = os.path.join(output_dir_astrometry, output_basename)
    tb_sw_astrometry = ascii.read(output_path)
#     
#     input_basename = os.path.basename(USER_ASTROMETRIC_CATALOG_PATH)
#     output_basename = input_basename.replace('.fits', '_astrometry.dat')
#     astrometry_path = os.path.join(USER_CALIBRATED_DIR, 'astrom', output_basename)
#     tb_sw_astrometry = ascii.read(astrometry_path)
#     
    # Read the user-provided tb_f444w catalog
    tb_f444w = ascii.read(USER_TB_F444W_PATH)

    # Get list of 1v1.5 files
    all_rate_list = get_lv1_5_rate_files()
    print(f"{len(all_rate_list)} wcs corrected fits files")

    # Read filter, module, and pupil
    all_filters, all_module, all_pupil = [], [], []
    all_roll_angle, all_target = [], []
    for i, x in enumerate(all_rate_list):
        tmp_header = fits.getheader(x)
        all_filters.append(tmp_header['FILTER'])
        all_module.append(tmp_header['MODULE'])
        all_pupil.append(tmp_header['PUPIL'])
        all_roll_angle.append(fits.getheader(x, 1)['ROLL_REF'])
        all_target.append(tmp_header['TARGPROP'])
    all_filters, all_module, all_pupil = np.array(all_filters), np.array(all_module), np.array(all_pupil)
    all_roll_angle, all_target = np.array(all_roll_angle), np.array(all_target)
    print(np.unique(all_filters), np.unique(all_module), np.unique(all_target))

    # Create output directory for POM catalogs
    pom_catalog_dir = os.path.join(USER_CALIBRATED_DIR, 'POM_catalog')
    if not os.path.exists(pom_catalog_dir):
        os.makedirs(pom_catalog_dir)
    
    for i, tmp_rate_path in enumerate(all_rate_list):
        tmp_rate_path_base = os.path.basename(tmp_rate_path).split('long_rate')[0]
        print(f'[{i:3d}] {tmp_rate_path_base}')
        
        tmp_grism_hd_1st = fits.getheader(tmp_rate_path, 0)
        tmp_grism_hd_sci = fits.getheader(tmp_rate_path, 'sci')
        
        tmp_filter = tmp_grism_hd_1st['FILTER']
        tmp_loc_v2, tmp_loc_v3 = tmp_grism_hd_sci['V2_REF'], tmp_grism_hd_sci['V3_REF']
        coord_f444w = SkyCoord(tb_f444w['RA'], tb_f444w['DEC'], unit=(u.deg, u.deg))
        list_coords, tb_sex = coord_f444w, tb_f444w

        tmp_RA, tmp_DEC = list_coords.ra.to(u.deg).value, list_coords.dec.to(u.deg).value
        
        item_sw_astrom = tb_sw_astrometry[tb_sw_astrometry['expName'] == tmp_rate_path_base]
        if len(item_sw_astrom) > 0:
            item_sw_astrom = item_sw_astrom[0]
            tmp_grism_hd_sci['CRVAL1'] -= item_sw_astrom['dRA'] / np.cos(np.deg2rad(tmp_grism_hd_sci['CRVAL2'])) / 3600.
            tmp_grism_hd_sci['CRVAL2'] -= item_sw_astrom['dDEC'] / 3600. 
            tmp_cd_matrix = np.array([[tmp_grism_hd_sci['CD1_1'], tmp_grism_hd_sci['CD1_2']],
                                      [tmp_grism_hd_sci['CD2_1'], tmp_grism_hd_sci['CD2_2']]])
            tmp_rot_matrix = np.array([[np.cos(np.deg2rad(-item_sw_astrom['theta'])), - np.sin(np.deg2rad(-item_sw_astrom['theta']))],
                                       [np.sin(np.deg2rad(-item_sw_astrom['theta'])),   np.cos(np.deg2rad(-item_sw_astrom['theta']))]])
            tmp_cd_matrix = np.matmul(tmp_cd_matrix, tmp_rot_matrix)
            tmp_grism_hd_sci['CD1_1'] = tmp_cd_matrix[0][0]
            tmp_grism_hd_sci['CD1_2'] = tmp_cd_matrix[0][1]
            tmp_grism_hd_sci['CD2_1'] = tmp_cd_matrix[1][0]
            tmp_grism_hd_sci['CD2_2'] = tmp_cd_matrix[1][1]
            tmp_RA_astrom = tmp_RA
            tmp_DEC_astrom = tmp_DEC
        else:
            print(f'Warning: No SW images found for {tmp_rate_path_base}, skip astrom correction!')
            tmp_RA_astrom = tmp_RA 
            tmp_DEC_astrom = tmp_DEC 
        try:
            rotation = item_sw_astrom['theta']
        except KeyError:
            rotation = 0
        
        tmp_grism_wcs = wcs.WCS(tmp_grism_hd_sci)
        idx_this_field = np.where((np.abs((tmp_RA_astrom - tmp_grism_hd_sci['CRVAL1']) * np.cos(np.deg2rad(tmp_grism_hd_sci['CRVAL2']))) < 3 / 60.) &
                                  (np.abs(tmp_DEC_astrom - tmp_grism_hd_sci['CRVAL2']) < 3 / 60.))[0]
        pixelx, pixely = wcs.utils.skycoord_to_pixel(SkyCoord(tmp_RA_astrom[idx_this_field], 
                                                              tmp_DEC_astrom[idx_this_field], unit=(u.deg, u.deg)), 
                                                     tmp_grism_wcs)
        
        if tmp_grism_hd_1st['MODULE'] == 'A':
            dx, dy = 2.0, 0
        else:
            dx, dy = 0, 0
        pixelx, pixely = pixelx + dx, pixely + dy
        tb_sex = tb_sex[idx_this_field]
        if len(tb_sex) == 0:
            continue

        if 'F444W_mag' in tb_sex.colnames:
            tmp_mag_auto = tb_sex['F444W_mag'].data
        elif 'F160W_mag' in tb_sex.colnames:
            tmp_mag_auto = tb_sex['F160W_mag'].data
        tb_pom_applied = Table(data=[tb_sex['ID'].data, 
                                     tmp_RA[idx_this_field], tmp_DEC[idx_this_field], 
                                     pixelx, pixely, tmp_mag_auto],
                               names=['Index', 'ra', 'dec', 'pixel_x', 'pixel_y', 'MAG_AUTO'])
        arg_is_pickoff = np.where(is_pickoff_PS(pixelx, pixely, module=tmp_grism_hd_1st['MODULE'], 
                                                filter=tmp_grism_hd_1st['FILTER'], pupil=tmp_grism_hd_1st['PUPIL'][-1]) > 0.05)[0]
        tb_pom_applied = tb_pom_applied[arg_is_pickoff]
        tb_pom_applied['ra'].info.format = '.6f'
        tb_pom_applied['dec'].info.format = '.6f'
        tb_pom_applied['pixel_x'].info.format = '.3f'
        tb_pom_applied['pixel_y'].info.format = '.3f'
        tb_pom_applied['MAG_AUTO'].info.format = '.3f'

        path_tb_pom_applied = os.path.join(pom_catalog_dir, f"{tmp_rate_path_base}_{tmp_filter}_dirimg_sources.list")
        ascii.write(tb_pom_applied, path_tb_pom_applied, overwrite=True)


# def check_and_prepare_spectra_extraction():
#     """Function to check and prepare spectra extraction parameters."""
#     tmp_filter = USER_FILTER
# 
#     # Check the filter and set wavelength range
#     if tmp_filter == 'F444W':
#         WRANGE = np.array([3.8, 5.1])
#     elif tmp_filter == 'F322W2':
#         WRANGE = np.array([2.4, 4.1])
#     elif tmp_filter == 'F356W':
#         WRANGE = np.array([3.1, 4.0])
#     elif tmp_filter == 'F277W':
#         WRANGE = np.array([2.4, 3.1])
#     else:
#         raise ValueError(f"Unsupported filter: {tmp_filter}. Supported filters are: 'F444W', 'F322W2', 'F356W', 'F277W'.")
# 
#     # Set dispersion filter
#     #disp_filter = tmp_filter
#     ### Spectral tracing parameters:
#     if tmp_filter in ['F277W', 'F335M', 'F322W2', 'F356W', 'F360M']: disp_filter = 'F322W2'
#     elif tmp_filter in ['F410M', 'F444W', 'F480M']: disp_filter = 'F444W' 
# 
#     # Read the dispersion parameters
#     dir_disper = './data/FSun_cal/disper/'
#     tb_order23_fit_AR = ascii.read(os.path.join(dir_disper, f'DISP_{disp_filter}_modA_grismR.dat'))
#     fit_opt_fit_AR, fit_err_fit_AR = tb_order23_fit_AR['col0'].data, tb_order23_fit_AR['col1'].data
#     tb_order23_fit_BR = ascii.read(os.path.join(dir_disper, f'DISP_{disp_filter}_modB_grismR.dat'))
#     fit_opt_fit_BR, fit_err_fit_BR = tb_order23_fit_BR['col0'].data, tb_order23_fit_BR['col1'].data
#     tb_order23_fit_AC = ascii.read(os.path.join(dir_disper, f'DISP_{disp_filter}_modA_grismC.dat'))
#     fit_opt_fit_AC, fit_err_fit_AC = tb_order23_fit_AC['col0'].data, tb_order23_fit_AC['col1'].data
#     tb_order23_fit_BC = ascii.read(os.path.join(dir_disper, f'DISP_{disp_filter}_modB_grismC.dat'))
#     fit_opt_fit_BC, fit_err_fit_BC = tb_order23_fit_BC['col0'].data, tb_order23_fit_BC['col1'].data
# 
#     # Read the grism dispersion parameters
#     tb_fit_displ_AR = ascii.read(os.path.join(dir_disper, 'DISPL_modA_grismR.dat'))
#     w_opt_AR, w_err_AR = tb_fit_displ_AR['col0'].data, tb_fit_displ_AR['col1'].data
#     tb_fit_displ_BR = ascii.read(os.path.join(dir_disper, 'DISPL_modB_grismR.dat'))
#     w_opt_BR, w_err_BR = tb_fit_displ_BR['col0'].data, tb_fit_displ_BR['col1'].data
#     tb_fit_displ_AC = ascii.read(os.path.join(dir_disper, 'DISPL_modA_grismC.dat'))
#     w_opt_AC, w_err_AC = tb_fit_displ_AC['col0'].data, tb_fit_displ_AC['col1'].data
#     tb_fit_displ_BC = ascii.read(os.path.join(dir_disper, 'DISPL_modB_grismC.dat'))
#     w_opt_BC, w_err_BC = tb_fit_displ_BC['col0'].data, tb_fit_displ_BC['col1'].data
# 
#     # List of module/pupil and corresponding tracing/dispersion function
#     list_mod_pupil = np.array(['AR', 'BR', 'AC', 'BC'])
#     list_fit_opt_fit = np.array([fit_opt_fit_AR, fit_opt_fit_BR, fit_opt_fit_AC, fit_opt_fit_BC])
#     list_w_opt = np.array([w_opt_AR, w_opt_BR, w_opt_AC, w_opt_BC])
# 
#     # Read the sensitivity curve
#     dir_sensitivity = './data/FSun_cal/disper/sensitivity'
#     tb_sens_AR = ascii.read(os.path.join(dir_sensitivity, f'{tmp_filter}_modA_grismR_sensitivity.dat'))
#     tb_sens_BR = ascii.read(os.path.join(dir_sensitivity, f'{tmp_filter}_modB_grismR_sensitivity.dat'))
#     tb_sens_AC = ascii.read(os.path.join(dir_sensitivity, f'{tmp_filter}_modA_grismC_sensitivity.dat'))
#     tb_sens_BC = ascii.read(os.path.join(dir_sensitivity, f'{tmp_filter}_modB_grismC_sensitivity.dat'))
#     f_sens_AR = interpolate.UnivariateSpline(tb_sens_AR['wavelength'], tb_sens_AR['DN/s/Jy'], ext='zeros', k=1, s=1e2)
#     f_sens_BR = interpolate.UnivariateSpline(tb_sens_BR['wavelength'], tb_sens_BR['DN/s/Jy'], ext='zeros', k=1, s=1e2)
#     f_sens_AC = interpolate.UnivariateSpline(tb_sens_AC['wavelength'], tb_sens_AC['DN/s/Jy'], ext='zeros', k=1, s=1e2)
#     f_sens_BC = interpolate.UnivariateSpline(tb_sens_BC['wavelength'], tb_sens_BC['DN/s/Jy'], ext='zeros', k=1, s=1e2)
#     list_f_sens = [f_sens_AR, f_sens_BR, f_sens_AC, f_sens_BC]
# 
#     # List of source list (POM-applied catalog)
#     pom_catalog_dir = os.path.join(USER_CALIBRATED_DIR, 'POM_catalog')
#     list_sl = glob.glob(os.path.join(pom_catalog_dir, f'jw*_nrc[a-b]_{tmp_filter}_dirimg_sources.list'))
#     list_sl.sort()
#     list_rate_this = np.array([os.path.join(USER_CALIBRATED_DIR, os.path.basename(x).split(f'_{tmp_filter}')[0] + 'long_rate_lv1.5.fits')
#                                for x in list_sl])
# 
#     Index_interest = tb_sex['ID'].data
#     print(len(Index_interest), 'sources selected')
#     list_sl_table = [ascii.read(x) for x in list_sl]
# 
#     # Generate two dictionaries which record the grism frame ID and path for each source to be extracted
#     grism_frame_ID_per_source = dict()  # dictionary of the ID of exposure (0-95) for each source (0-5000+)
#     grism_frame_path_per_source = dict()  # dictionary of the path of exposure (*/jw*.fits) for each source (0-5000+)
#     for idx in Index_interest:
#         grism_frame_ID_per_source[f'{idx}'] = []
#         grism_frame_path_per_source[f'path_{idx}'] = []
#     for i, tmp_sl_table in enumerate(list_sl_table):
#         for idx in tmp_sl_table['Index'].data:
#             grism_frame_ID_per_source[f'{idx}'].append(i)
#             grism_frame_path_per_source[f'path_{idx}'].append(list_rate_this[i])
#     N_frame_per_source = np.array([len(grism_frame_ID_per_source[x]) for x in grism_frame_ID_per_source.keys()])
#     print(f'{np.sum(N_frame_per_source > 0)} sources may yield spectra')
    

def check_and_prepare_spectra_extraction():
    """Function to check and prepare spectra extraction parameters."""
    tmp_filter = USER_FILTER
    list_coords, tb_sex = None, None  # Placeholder, update with actual data loading

    # Check the filter and set wavelength range
    if tmp_filter == 'F444W':
        WRANGE = np.array([3.8, 5.1])
    elif tmp_filter == 'F322W2':
        WRANGE = np.array([2.4, 4.1])
    elif tmp_filter == 'F356W':
        WRANGE = np.array([3.1, 4.0])
    elif tmp_filter == 'F277W':
        WRANGE = np.array([2.4, 3.1])
    else:
        raise ValueError(f"Unsupported filter: {tmp_filter}. Supported filters are: 'F444W', 'F322W2', 'F356W', 'F277W'.")

    # Set dispersion filter
    #disp_filter = tmp_filter
    ### Spectral tracing parameters:
    if tmp_filter in ['F277W', 'F335M', 'F322W2', 'F356W', 'F360M']: disp_filter = 'F322W2'
    elif tmp_filter in ['F410M', 'F444W', 'F480M']: disp_filter = 'F444W' 
    
    # Load necessary data files
    # Read the dispersion parameters
    dir_disper =DEFAULT_SPEC_DISPER
    tb_order23_fit_AR = ascii.read(os.path.join(dir_disper, f'DISP_{disp_filter}_modA_grismR.dat'))
    fit_opt_fit_AR, fit_err_fit_AR = tb_order23_fit_AR['col0'].data, tb_order23_fit_AR['col1'].data
    tb_order23_fit_BR = ascii.read(os.path.join(dir_disper, f'DISP_{disp_filter}_modB_grismR.dat'))
    fit_opt_fit_BR, fit_err_fit_BR = tb_order23_fit_BR['col0'].data, tb_order23_fit_BR['col1'].data
    tb_order23_fit_AC = ascii.read(os.path.join(dir_disper, f'DISP_{disp_filter}_modA_grismC.dat'))
    fit_opt_fit_AC, fit_err_fit_AC = tb_order23_fit_AC['col0'].data, tb_order23_fit_AC['col1'].data
    tb_order23_fit_BC = ascii.read(os.path.join(dir_disper, f'DISP_{disp_filter}_modB_grismC.dat'))
    fit_opt_fit_BC, fit_err_fit_BC = tb_order23_fit_BC['col0'].data, tb_order23_fit_BC['col1'].data

    # Read the grism dispersion parameters
    tb_fit_displ_AR = ascii.read(os.path.join(dir_disper, 'DISPL_modA_grismR.dat'))
    w_opt_AR, w_err_AR = tb_fit_displ_AR['col0'].data, tb_fit_displ_AR['col1'].data
    tb_fit_displ_BR = ascii.read(os.path.join(dir_disper, 'DISPL_modB_grismR.dat'))
    w_opt_BR, w_err_BR = tb_fit_displ_BR['col0'].data, tb_fit_displ_BR['col1'].data
    tb_fit_displ_AC = ascii.read(os.path.join(dir_disper, 'DISPL_modA_grismC.dat'))
    w_opt_AC, w_err_AC = tb_fit_displ_AC['col0'].data, tb_fit_displ_AC['col1'].data
    tb_fit_displ_BC = ascii.read(os.path.join(dir_disper, 'DISPL_modB_grismC.dat'))
    w_opt_BC, w_err_BC = tb_fit_displ_BC['col0'].data, tb_fit_displ_BC['col1'].data

    # List of module/pupil and corresponding tracing/dispersion function
    list_mod_pupil = np.array(['AR', 'BR', 'AC', 'BC'])
    list_fit_opt_fit = np.array([fit_opt_fit_AR, fit_opt_fit_BR, fit_opt_fit_AC, fit_opt_fit_BC])
    list_w_opt = np.array([w_opt_AR, w_opt_BR, w_opt_AC, w_opt_BC])

    # Read the sensitivity curve
    dir_sensitivity = DEFAULT_SPEC_SENSI
    tb_sens_AR = ascii.read(os.path.join(dir_sensitivity, f'{tmp_filter}_modA_grismR_sensitivity.dat'))
    tb_sens_BR = ascii.read(os.path.join(dir_sensitivity, f'{tmp_filter}_modB_grismR_sensitivity.dat'))
    tb_sens_AC = ascii.read(os.path.join(dir_sensitivity, f'{tmp_filter}_modA_grismC_sensitivity.dat'))
    tb_sens_BC = ascii.read(os.path.join(dir_sensitivity, f'{tmp_filter}_modB_grismC_sensitivity.dat'))
    f_sens_AR = interpolate.UnivariateSpline(tb_sens_AR['wavelength'], tb_sens_AR['DN/s/Jy'], ext='zeros', k=1, s=1e2)
    f_sens_BR = interpolate.UnivariateSpline(tb_sens_BR['wavelength'], tb_sens_BR['DN/s/Jy'], ext='zeros', k=1, s=1e2)
    f_sens_AC = interpolate.UnivariateSpline(tb_sens_AC['wavelength'], tb_sens_AC['DN/s/Jy'], ext='zeros', k=1, s=1e2)
    f_sens_BC = interpolate.UnivariateSpline(tb_sens_BC['wavelength'], tb_sens_BC['DN/s/Jy'], ext='zeros', k=1, s=1e2)
    list_f_sens = [f_sens_AR, f_sens_BR, f_sens_AC, f_sens_BC]
    all_rate_list = get_lv1_5_rate_files()
    # Load the POM-applied catalogs
    list_sl = glob.glob(os.path.join(USER_CALIBRATED_DIR, 'POM_catalog', f'jw*_nrc[a-b]_{tmp_filter}_dirimg_sources.list'))
    list_sl.sort()
    list_rate_this = np.array([os.path.dirname(all_rate_list[0]) + '/' + os.path.basename(list_sl[i]).split('_%s' % tmp_filter)[0] + 'long_rate_lv1.5.fits' 
                           for i in range(len(list_sl))])

    tb_sex = ascii.read(USER_TB_F444W_PATH)
    Index_interest = tb_sex['ID'].data
    list_sl_table = [ascii.read(x) for x in list_sl]

    grism_frame_ID_per_source = {str(idx): [] for idx in Index_interest}
    grism_frame_path_per_source = {f'path_{idx}': [] for idx in Index_interest}

    for i, tmp_sl_table in enumerate(list_sl_table):
        for idx in tmp_sl_table['Index'].data:
            grism_frame_ID_per_source[str(idx)].append(i)
            grism_frame_path_per_source[f'path_{idx}'].append(list_rate_this[i])

    N_frame_per_source = np.array([len(grism_frame_ID_per_source[x]) for x in grism_frame_ID_per_source.keys()])
    print(f'{np.sum(N_frame_per_source > 0)} sources may yield spectra')

    return Index_interest, grism_frame_ID_per_source, grism_frame_path_per_source, list_sl_table, list_fit_opt_fit, list_w_opt, list_f_sens, tb_sex, WRANGE

def extract_2d_spec(img, WRANGE, x0, y0, dxs, dys, wave, img_wht, img_dq, header, 
                    img_line = None, aper = 10., pupil = 'R', n_procs = 4):
    '''
    Extract 2D spectrum from the Grism Image
    -----------------------------------------------
        Parameters
        ----------
        img : `~numpy.ndarray`
            2D dispersed slitless spectroscopic image
        
        GConf: `~grismconf.grismconf.Config`
            Configuration file of grism (taken from `grismconf`)
            
        x0, y0 : float
            Reference position (i.e., in direct image)
        
        dxs, dys : `~numpy.ndarray`
            First derivative of the trace x/y-coordinates with respect to 't' (taken from `grismconf`)
        
        wave: 
            Wavelength of extracted spectrum
            
        img_wht : `~numpy.ndarray`
            Weighting image (1/rms^2) of `img`
            
        img_dq : `~numpy.ndarray`
            DQ (data-quality) image (1/rms^2) of `img`
            
        header: `fits.Header()` object
            Header of 2D dispersed slitless spectroscopic image
        
        img_line : None or `~numpy.ndarray`
            EMLINE (scientific) image, continuum-subtracted version of `img`
            
        aper: float
            Aperture radius for 2D spectra extraction (unit: arcsec, default: 1.25)
        
        pix_scale: float
            Pixel scale in unit of arcsec. If None then computed from header
        
        pupil: 'R' or 'C'
            pupil of grism ('R' or 'C')
            
        Returns
        -------
        hdul : `~astropy.io.fits.HDUList()` object
            The fits files of extracted 2D spectra, including a primary frame, a sci frame,
                (scientific), a wht frame (weighting) and a dq frame (data quality).
    '''
    w_min, w_max = WRANGE
    x_on_G_img = dxs + x0
    y_on_G_img = dys + y0
    
    if pupil not in ['C', 'R']: raise KeyError('pupil of grism should be either "R" or "C"! ')
    
    aper_int_pix = int(aper)
    
    
    ### region that we can extract spectra: 
    if pupil == 'R':
        args_eff = np.where((wave >= w_min) & (wave <= w_max) 
                            & (x_on_G_img >= 5) & (x_on_G_img <= 2047 - 6)
                            & (y_on_G_img >= aper_int_pix) & (y_on_G_img <= 2047 - 6 - aper_int_pix) )
    elif pupil == 'C':
        args_eff = np.where((wave >= w_min) & (wave <= w_max) 
                            & (x_on_G_img >= aper_int_pix) & (x_on_G_img <= 2047 - 6 - aper_int_pix)
                            & (y_on_G_img >= 5) & (y_on_G_img <= 2047 - 6) )
    if np.size(args_eff) <= 20:  ## spectrum is too short
        raise ValueError('No spectrum can be extracted from the 2D Image')
    tmp_spec_2d = np.zeros((len(args_eff[0]), aper_int_pix*2+1))
    tmp_wht_2d = np.zeros((len(args_eff[0]), aper_int_pix*2+1))
    tmp_dq_2d = np.zeros((len(args_eff[0]), aper_int_pix*2+1))
    # if img_line != None: 
    tmp_line_2d = np.zeros((len(args_eff[0]), aper_int_pix*2+1))
        
    for i, j in enumerate(args_eff[0]):
        if pupil == 'R':
            tmp_x, tmp_y1, tmp_y2 = int(x_on_G_img[j]), int(y_on_G_img[j] - aper_int_pix - 1), int(y_on_G_img[j] + aper_int_pix + 2)
            img.T[tmp_x, tmp_y1:tmp_y2] = ndimage.shift(img.T[tmp_x, tmp_y1:tmp_y2], -(y_on_G_img[j]%1), order = 1, mode ='wrap')
            img_wht.T[tmp_x, tmp_y1:tmp_y2] = ndimage.shift(img_wht.T[tmp_x, tmp_y1:tmp_y2], -(y_on_G_img[j]%1), order = 1, mode ='wrap')
            img_dq.T[tmp_x, tmp_y1:tmp_y2] = ndimage.shift(img_dq.T[tmp_x, tmp_y1:tmp_y2], -(y_on_G_img[j]%1), order = 1, mode ='wrap')
            tmp_spec_2d[i] = img.T[tmp_x][tmp_y1+1:tmp_y2-1]
            tmp_wht_2d[i] = img_wht.T[tmp_x][tmp_y1+1:tmp_y2-1]
            tmp_dq_2d[i] = img_dq.T[tmp_x][tmp_y1+1:tmp_y2-1]
            # if img_line != None: 
            img_line.T[tmp_x, tmp_y1:tmp_y2] = ndimage.shift(img_line.T[tmp_x, tmp_y1:tmp_y2], -(y_on_G_img[j]%1), order = 1, mode ='wrap')
            tmp_line_2d[i] = img_line.T[tmp_x][tmp_y1+1:tmp_y2-1]
        elif pupil == 'C':
            tmp_y, tmp_x1, tmp_x2 = int(y_on_G_img[j]), int(x_on_G_img[j] - aper_int_pix - 1), int(x_on_G_img[j] + aper_int_pix + 2)
            img[tmp_y, tmp_x1:tmp_x2] = ndimage.shift(img[tmp_y, tmp_x1:tmp_x2], -(x_on_G_img[j]%1), order = 1, mode ='wrap')
            img_wht[tmp_y, tmp_x1:tmp_x2] = ndimage.shift(img_wht[tmp_y, tmp_x1:tmp_x2], -(x_on_G_img[j]%1), order = 1, mode ='wrap')
            img_dq[tmp_y, tmp_x1:tmp_x2] = ndimage.shift(img_dq[tmp_y, tmp_x1:tmp_x2], -(x_on_G_img[j]%1), order = 1, mode ='wrap')
            tmp_spec_2d[i] = img[tmp_y, tmp_x1+1:tmp_x2-1]
            tmp_wht_2d[i] = img_wht[tmp_y, tmp_x1+1:tmp_x2-1]
            tmp_dq_2d[i] = img_dq[tmp_y, tmp_x1+1:tmp_x2-1]
            # if img_line != None: 
            img_line[tmp_y, tmp_x1:tmp_x2] = ndimage.shift(img_line[tmp_y, tmp_x1:tmp_x2], -(x_on_G_img[j]%1), order = 1, mode ='wrap')
            tmp_line_2d[i] = img[tmp_y, tmp_x1+1:tmp_x2-1]
    
    tmp_spec_2d = tmp_spec_2d.T
    tmp_wht_2d = tmp_wht_2d.T
    tmp_dq_2d = tmp_dq_2d.T
    # if img_line != None: 
    tmp_line_2d = tmp_line_2d.T
        
    '''Construct fits file for 2d grism spectra'''
    ### Primary HDU
    hdu = fits.PrimaryHDU()
    hdu.header['x0'] = (np.float32(x0), 'Reference position X in direct image')
    hdu.header['y0'] = (np.float32(y0), 'Reference position Y in direct image')
    # tmp_coord = wcs.utils.pixel_to_skycoord(x0, y0, tmp_wcs)
    # hdu.header['RA0'] = (tmp_coord.ra.value, 'Reference position RA in direct image')
    # hdu.header['DEC0'] = (tmp_coord.dec.value, 'Reference position Dec in direct image')
    hdu.header['author'] = ('Zhaoran Liu', 'Author of this file')
    hdu.header['time'] = (time.strftime("%Y/%m/%d %H:%M:%S",  time.localtime()), 'Time of Creation')
    ### Scientific HDU
    hdu_sci = fits.ImageHDU(np.float32(tmp_spec_2d), name = 'SPEC2D')
    hdu_sci.header['wave_1'] = (wave[args_eff[0][1]], 'Wavelength (um) of first pixel')
    hdu_sci.header['d_wave'] = (np.mean(np.diff(wave[args_eff[0]])), 'Wavelength Difference (um) between each pixel')
    hdu_sci.header['comments'] = ('wave = wave_1 + np.arange(0, NAXIS1, 1) * d_wave')
    hdu_sci.header['aperture'] = (aper, 'Aperture radius in undispersed direction (arcsec)')
    hdu_sci.header['pupil'] = (pupil, 'Pupil of grism (R or C)')
    hdu_sci.header['module'] = (header['module'], 'Module of Detector (A or B)')
    hdu_sci.header['diff_y'] = (y_on_G_img[args_eff[0]][0], 'Y_(full)-Y_(trim)')
    hdu_sci.header['diff_x'] = (x_on_G_img[args_eff[0]][0], 'X_(full)-X_(trim)')
    ### Weight HDU
    hdu_wht = fits.ImageHDU(np.float32(tmp_wht_2d), name = 'WHT2D')
    ### Data-quality HDU
    hdu_dq = fits.ImageHDU(np.int32(tmp_dq_2d), name = 'DQ2D')
    ### Line-only HDU:
    # if img_line != None:
    hdu_line = hdu_sci.copy()
    hdu_line.header['EXTNAME'] = 'LINE2D'
    hdu_line.data = np.float32(tmp_line_2d)
    hdu_line.header['comments'] = 'extracted from continuum subtracted map'
    ### Tracing & Dispersion information
    tmp_tb_wave = Table(data = [wave[args_eff], x_on_G_img[args_eff], y_on_G_img[args_eff], dxs[args_eff], dys[args_eff]],
                    names = ['wavelength', 'xs', 'ys', 'dxs', 'dys'])
    tmp_tb_wave['wavelength'].info.format = '.6f'
    for x in tmp_tb_wave.colnames[1:]: tmp_tb_wave[x].info.format = '.3f'
    hdul = fits.HDUList([hdu, hdu_sci, hdu_wht, hdu_dq])
    # if img_line != None:  
    hdul.append(hdu_line)
    hdul.append(fits.BinTableHDU(tmp_tb_wave, name = 'WAVE'))
    return hdul


def store_all_2d_spec(fits_list, pupils, modules, paths, output = 'test.fits',
                      coord = None, filter = None, info_table = None, overwrite = True):
    '''
    Write all the individual 2D spectra of one single sources into a fits file
    -----------------------------------------------
    Parameters
        ----------  
        fits_list : list of `astropy.io.fits`
            list of fits files generated by extract_2d_spec()
        
        pupils: list of strings
            pupil list of grism ('R' or 'C')
        
        modules: list of strings
            module list of NIRCam detector ('A' or 'B')
        
        paths: list of strings
            paths of grism exposures
        
        output: string
            Output file name
        
        coord: `astropy.coordinates.SkyCoord` object
            coordinates of the source. default: None (will not record this in the header)
        
        filter: string
            Filter of grism observation (e.g., 'F444W', 'F332W', 'F356W')
            default: None (will not record this in the header)
        
        info_table: astropy.table.Row
            A row of information related to this source
            default: None (will not record this in the header)
            
        overwrite: bool
            If true, save (and overwrite) the data. default: True
            
        Returns
        -------
        ind_hdul: `~astropy.io.fits.HDUList`
            fits HDU list of individual sci, wht and dq data.
            
    '''
    ## HDU list of individual exposures:
    ind_hdul = fits.HDUList([fits.PrimaryHDU()])
    for l, x in enumerate(fits_list):
        ind_hdul.append(x[1]) 
        ind_hdul[-1].header['EXTNAME'] = 'SPEC2D-%d' % l
        for card_name in ['x0', 'y0']: # 'ra0', 'dec0']:
            ind_hdul[-1].header[card_name] = x[0].header[card_name]
        ind_hdul[-1].header['pupil'] = pupils[l]
        ind_hdul[-1].header['module'] = modules[l]
        ind_hdul[-1].header['datapath'] = paths[l].split('/')[-1]
        ind_hdul.append(x[2]) 
        ind_hdul[-1].header['EXTNAME'] = 'WHT2D-%d' % l
        ind_hdul.append(x[3])
        ind_hdul[-1].header['EXTNAME'] = 'DQ2D-%d' % l
        if len(x) == 5:  # no line image extension
            ind_hdul.append(x[4])
            ind_hdul[-1].header['EXTNAME'] = 'WAVE-%d' % l
        else:            # with line image extension
            ind_hdul.append(x[4])
            ind_hdul[-1].header['EXTNAME'] = 'LINE2D-%d' % l
            ind_hdul.append(x[5])
            ind_hdul[-1].header['EXTNAME'] = 'WAVE-%d' % l
    #### stats of single exposures
    ind_hdul.append(fits.BinTableHDU(Table(names = ['id', 'pupil', 'module', 'datapath'], # 'POM', 
                                           data = [range(len(fits_list)), pupils, modules, # poms,
                                                   [tmp_path.split('/')[-1] for tmp_path in paths]]),
                                     name = 'STATS'))
    #### add other important specs
    ind_hdul[0].header['DIRNAME'] = (os.path.dirname(paths[0]), 'Directory name of original grism data')
    if coord != None:
        coord_1 = coord
        ind_hdul[0].header['RA0'] = (coord_1.ra.value, 'Reference position RA in direct image')
        ind_hdul[0].header['DEC0'] = (coord_1.dec.value, 'Reference position Dec in direct image')
    ind_hdul[0].header['N_coadd'] = (len(ind_hdul[-1].data['pupil']), 'Number of coadded frames')
    ind_hdul[0].header['N_R'] = (np.sum(ind_hdul[-1].data['pupil'] == 'R'), 'Number of R grism frames')
    ind_hdul[0].header['N_C'] = (np.sum(ind_hdul[-1].data['pupil'] == 'C'), 'Number of C grism frames')
    ind_hdul[0].header['author'] = ('Zhaoran Liu', 'Author of this file')
    ind_hdul[0].header['time'] = (time.strftime("%Y/%m/%d %H:%M:%S",  time.localtime()), 'Time of Creation')
    if filter != None:
        ind_hdul[0].header['filter'] = (tmp_filter, 'Filter name')
    if info_table != None:
        ind_hdul[0].header['COMMENTS'] = 'Belows are information taken from input catalog:'
        for x in info_table.colnames:
            if type(info_table[x]) == np.ma.core.MaskedConstant: continue
            elif type(info_table[x]) != np.str_ : 
                if np.isnan(info_table[x]) : ind_hdul[0].header['HIERARCH ' + x] = 'nan' 
                elif np.isinf(info_table[x]) : ind_hdul[0].header['HIERARCH ' + x] = 'inf' 
                else: ind_hdul[0].header['HIERARCH ' + x] = info_table[x]
            else: ind_hdul[0].header['HIERARCH ' + x] = info_table[x]
    if overwrite: ind_hdul.writeto(output, overwrite = overwrite) ## save the fits file?
    return ind_hdul

def resample_spec2d_wmin_wmax(x, i, tmp_wave_2d, tmp_ind_fits_list, wave_sample):
    '''
    Resample a series of 2D spectra at wavelegth range = tmp_w_min - tmp_w_max
    -----------------------------------------------
    Parameters
        ---------- 
        x: int
            index of the individual fits files in the list.
        
        i: int
            index of the resampled wave array
        
        tmp_wave_2d: 
            Global variable (`list(np.array())`). 
            A compilation of 1D wavelength array associated with all individual grism frames.
        
        tmp_ind_fits_list:
            Global variable (`astropy.io.fits.HDUList()`). 
            A HDU List that have extracted 2D spectra (SCI/WHT/DQ) from each individual frames
            
        wave_sample:
            Global variable (`np.array()`)
            Resampled wavelengths. 
            
        Returns
        -------
        tmp_spec_w_1: `~np.array()`
            a column of 2D SCI spectra from frame [x] that fall in the resampled wavelength range [i]
        
        tmp_wht_w_1: `~np.array()`
            a column of 2D WHT spectra at the same position as that of `tmp_spec_w_1`
        
        tmp_cov_w_1: `~np.array()`
            a column of 2D DQ spectra at the same position as that of `tmp_spec_w_1`
    '''
   # global tmp_wave_2d, tmp_ind_fits_list, wave_sample # arrays to use
    # global arr_spec_2d, arr_wht_2d, arr_cov_2d         # arrays to save; impossible with pool
    tmp_w_min, tmp_w_max = wave_sample[i], wave_sample[i+1]
    ## resampled SCI, WHT, DQ image
    arg_in_wmin_wmax = tuple([(tmp_wave_2d[x] > tmp_w_min) & (tmp_wave_2d[x] <= tmp_w_max)])
    tmp_spec_w = tmp_ind_fits_list[x][1].data.T[arg_in_wmin_wmax]
    tmp_wht_w = np.nan_to_num(tmp_ind_fits_list[x][2].data.T[arg_in_wmin_wmax], posinf = 0, neginf = 0)
    tmp_dq_w = tmp_ind_fits_list[x][3].data.T[arg_in_wmin_wmax]
    if len(tmp_ind_fits_list[x]) == 5: tmp_line_w = tmp_ind_fits_list[x][1].data.T[arg_in_wmin_wmax]
    else: tmp_line_w = tmp_ind_fits_list[x][4].data.T[arg_in_wmin_wmax]  ## line map 
    # tmp_wht_w[tmp_dq_w!=0] = 0    ### strict dq: reject all dq != 0:
    tmp_wht_w[tmp_dq_w%2==1] = 0    ### loose dq: only remove do not use pixel:
    ## stack/resample N spectra at wmin-wmax (M rows)
    tmp_spec_w_1 = np.nansum(tmp_spec_w * tmp_wht_w, axis = 0) / np.nansum(tmp_wht_w, axis = 0) #/ sum_tmp_wht_w #
    tmp_wht_w_1 = np.nansum(tmp_wht_w, axis = 0)
    tmp_cov_w_1 = np.int8(tmp_wht_w_1 != 0)
    tmp_line_w_1 = np.nansum(tmp_line_w * tmp_wht_w, axis = 0) / np.nansum(tmp_wht_w, axis = 0) #/ sum_tmp_wht_w #
    ## save resampled spectra to array
    # arr_spec_2d[i, x], arr_wht_2d[i, x], arr_cov_2d[i, x] = tmp_spec_w_1, tmp_wht_w_1, tmp_cov_w_1
    return(tmp_spec_w_1, tmp_wht_w_1, tmp_cov_w_1, tmp_line_w_1)

def extract_spectra():
    extract_mode = 'all'
    extract_path = USER_OUTPUT_SPECTRA_PATH
    start = time.time()
    if not os.path.isdir(extract_path):
        os.makedirs(extract_path)
    tmp_filter = USER_FILTER
    list_coords, tb_sex = None, None  # Placeholder, update with actual data loading
    tmp_target = USER_TARGET
    # Check the filter and set wavelength range
    if tmp_filter == 'F444W':
        WRANGE = np.array([3.8, 5.1])
    elif tmp_filter == 'F322W2':
        WRANGE = np.array([2.4, 4.1])
    elif tmp_filter == 'F356W':
        WRANGE = np.array([3.1, 4.0])
    elif tmp_filter == 'F277W':
        WRANGE = np.array([2.4, 3.1])
    else:
        raise ValueError(f"Unsupported filter: {tmp_filter}. Supported filters are: 'F444W', 'F322W2', 'F356W', 'F277W'.")

    # Set dispersion filter
    #disp_filter = tmp_filter
    ### Spectral tracing parameters:
    if tmp_filter in ['F277W', 'F335M', 'F322W2', 'F356W', 'F360M']: disp_filter = 'F322W2'
    elif tmp_filter in ['F410M', 'F444W', 'F480M']: disp_filter = 'F444W' 
    
    # Load necessary data files
    # Read the dispersion parameters
    dir_disper =DEFAULT_SPEC_DISPER
    tb_order23_fit_AR = ascii.read(os.path.join(dir_disper, f'DISP_{disp_filter}_modA_grismR.dat'))
    fit_opt_fit_AR, fit_err_fit_AR = tb_order23_fit_AR['col0'].data, tb_order23_fit_AR['col1'].data
    tb_order23_fit_BR = ascii.read(os.path.join(dir_disper, f'DISP_{disp_filter}_modB_grismR.dat'))
    fit_opt_fit_BR, fit_err_fit_BR = tb_order23_fit_BR['col0'].data, tb_order23_fit_BR['col1'].data
    tb_order23_fit_AC = ascii.read(os.path.join(dir_disper, f'DISP_{disp_filter}_modA_grismC.dat'))
    fit_opt_fit_AC, fit_err_fit_AC = tb_order23_fit_AC['col0'].data, tb_order23_fit_AC['col1'].data
    tb_order23_fit_BC = ascii.read(os.path.join(dir_disper, f'DISP_{disp_filter}_modB_grismC.dat'))
    fit_opt_fit_BC, fit_err_fit_BC = tb_order23_fit_BC['col0'].data, tb_order23_fit_BC['col1'].data

    # Read the grism dispersion parameters
    tb_fit_displ_AR = ascii.read(os.path.join(dir_disper, 'DISPL_modA_grismR.dat'))
    w_opt_AR, w_err_AR = tb_fit_displ_AR['col0'].data, tb_fit_displ_AR['col1'].data
    tb_fit_displ_BR = ascii.read(os.path.join(dir_disper, 'DISPL_modB_grismR.dat'))
    w_opt_BR, w_err_BR = tb_fit_displ_BR['col0'].data, tb_fit_displ_BR['col1'].data
    tb_fit_displ_AC = ascii.read(os.path.join(dir_disper, 'DISPL_modA_grismC.dat'))
    w_opt_AC, w_err_AC = tb_fit_displ_AC['col0'].data, tb_fit_displ_AC['col1'].data
    tb_fit_displ_BC = ascii.read(os.path.join(dir_disper, 'DISPL_modB_grismC.dat'))
    w_opt_BC, w_err_BC = tb_fit_displ_BC['col0'].data, tb_fit_displ_BC['col1'].data

    # List of module/pupil and corresponding tracing/dispersion function
    list_mod_pupil = np.array(['AR', 'BR', 'AC', 'BC'])
    list_fit_opt_fit = np.array([fit_opt_fit_AR, fit_opt_fit_BR, fit_opt_fit_AC, fit_opt_fit_BC])
    list_w_opt = np.array([w_opt_AR, w_opt_BR, w_opt_AC, w_opt_BC])

    # Read the sensitivity curve
    dir_sensitivity = DEFAULT_SPEC_SENSI
    tb_sens_AR = ascii.read(os.path.join(dir_sensitivity, f'{tmp_filter}_modA_grismR_sensitivity.dat'))
    tb_sens_BR = ascii.read(os.path.join(dir_sensitivity, f'{tmp_filter}_modB_grismR_sensitivity.dat'))
    tb_sens_AC = ascii.read(os.path.join(dir_sensitivity, f'{tmp_filter}_modA_grismC_sensitivity.dat'))
    tb_sens_BC = ascii.read(os.path.join(dir_sensitivity, f'{tmp_filter}_modB_grismC_sensitivity.dat'))
    f_sens_AR = interpolate.UnivariateSpline(tb_sens_AR['wavelength'], tb_sens_AR['DN/s/Jy'], ext='zeros', k=1, s=1e2)
    f_sens_BR = interpolate.UnivariateSpline(tb_sens_BR['wavelength'], tb_sens_BR['DN/s/Jy'], ext='zeros', k=1, s=1e2)
    f_sens_AC = interpolate.UnivariateSpline(tb_sens_AC['wavelength'], tb_sens_AC['DN/s/Jy'], ext='zeros', k=1, s=1e2)
    f_sens_BC = interpolate.UnivariateSpline(tb_sens_BC['wavelength'], tb_sens_BC['DN/s/Jy'], ext='zeros', k=1, s=1e2)
    list_f_sens = [f_sens_AR, f_sens_BR, f_sens_AC, f_sens_BC]
    all_rate_list = get_lv1_5_rate_files()
    # Load the POM-applied catalogs
    list_sl = glob.glob(os.path.join(USER_CALIBRATED_DIR, 'POM_catalog', f'jw*_nrc[a-b]_{tmp_filter}_dirimg_sources.list'))
    list_sl.sort()
    list_rate_this = np.array([os.path.dirname(all_rate_list[0]) + '/' + os.path.basename(list_sl[i]).split('_%s' % tmp_filter)[0] + 'long_rate_lv1.5.fits' 
                           for i in range(len(list_sl))])

    tb_sex = ascii.read(USER_TB_F444W_PATH)
    Index_interest = tb_sex['ID'].data
    list_sl_table = [ascii.read(x) for x in list_sl]

    grism_frame_ID_per_source = {str(idx): [] for idx in Index_interest}
    grism_frame_path_per_source = {f'path_{idx}': [] for idx in Index_interest}

    for i, tmp_sl_table in enumerate(list_sl_table):
        for idx in tmp_sl_table['Index'].data:
            grism_frame_ID_per_source[str(idx)].append(i)
            grism_frame_path_per_source[f'path_{idx}'].append(list_rate_this[i])

    for i, idx in enumerate(Index_interest[:]):
        # Your extraction code herea
        t_a = time.time()
        item_sex = tb_sex[tb_sex['ID'] == idx][0]
        if 'F444W_mag' in tb_sex.colnames: tmp_mag = item_sex['F444W_mag']
        elif 'F160W_mag' in tb_sex.colnames: tmp_mag = item_sex['F160W_mag']
        tmp_coord = SkyCoord(item_sex['RA'], item_sex['DEC'], unit = (u.deg, u.deg))
        print('-[%3d] ID%s mag=%.2f' % (i, idx, tmp_mag), end=" ")

        if len(grism_frame_ID_per_source[str(idx)]) == 0:
            print(' >> no spec found; ')
            continue
        else:
            this_list_rate_wcs = np.array(grism_frame_path_per_source['path_'+str(idx)])
            this_ID_rate_wcs = np.array(grism_frame_ID_per_source[str(idx)])

        tmp_ind_fits_list = []
        tmp_ind_fits_name_list = []
        tmp_ind_module_list = []
        tmp_ind_pupil_list = []
        tmp_ind_pom_list = []

        for j, tmp_grism in enumerate(this_list_rate_wcs):
            tmp_idx_grism = this_ID_rate_wcs[j]  
            tmp_grism_img = fits.getdata(tmp_grism, 'sci')       
            try:
                tmp_grism_img_emline = fits.getdata(tmp_grism, 'emline')
            except KeyError:
                tmp_grism_img_emline = tmp_grism_img
            tmp_grism_dq = fits.getdata(tmp_grism, 'dq')
            tmp_grism_hd = fits.getheader(tmp_grism)
            tmp_filter, tmp_module, tmp_pupil = tmp_grism_hd['filter'], tmp_grism_hd['module'], tmp_grism_hd['pupil'][-1]
            
            if extract_mode.lower() == 'r' and tmp_pupil == 'C':
                continue
            if extract_mode.lower() == 'c' and tmp_pupil == 'R':
                continue
            
            tmp_grism_hd_sci = fits.getheader(tmp_grism, 'sci')
            tmp_grism_wht_path = tmp_grism.replace('lv1.5.fits', 'wht.fits')
            
            if os.path.isfile(tmp_grism_wht_path):
                tmp_grism_wht = fits.getdata(tmp_grism_wht_path)
            else:
                tmp_grism_err = fits.getdata(tmp_grism, 'err')
                tmp_grism_err[tmp_grism_err == 0] = np.nan
                tmp_grism_wht = tmp_grism_err**-2
                tmp_grism_wht_hdul = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(tmp_grism_wht, name='WHT')])
                tmp_grism_wht_hdul[1].header = tmp_grism_hd_sci
                tmp_grism_wht_hdul.writeto(tmp_grism_wht_path, overwrite=True)
                tmp_grism_wht_hdul.close()
            
            item_sl = list_sl_table[tmp_idx_grism][list_sl_table[tmp_idx_grism]['Index'] == idx][0]
            x0 = item_sl['pixel_x']
            y0 = item_sl['pixel_y']
            print('%s(%.1f, %.1f) ' % (tmp_pupil, x0, y0), end=' ')
            
            fit_opt_fit = list_fit_opt_fit[list_mod_pupil == tmp_module + tmp_pupil][0]
            w_opt = list_w_opt[list_mod_pupil == tmp_module + tmp_pupil][0]
            dxs, dys, wavs = grism_conf_preparation(x0=x0, y0=y0, pupil=tmp_pupil, fit_opt_fit=fit_opt_fit, w_opt=w_opt)
            tmp_aper = 20.

            try:
                tmp_ind_fits = extract_2d_spec(img=tmp_grism_img, WRANGE=WRANGE, x0=x0, y0=y0, dxs=dxs, dys=dys, wave=wavs, img_wht=tmp_grism_wht, img_dq=tmp_grism_dq, img_line=tmp_grism_img_emline, header=tmp_grism_hd, pupil=tmp_pupil, aper=tmp_aper)
            except ValueError:
                continue
            
            tmp_ind_fits_list.append(tmp_ind_fits)
            tmp_ind_module_list.append(tmp_module)
            tmp_ind_fits_name_list.append(tmp_grism)
            tmp_ind_pupil_list.append(tmp_pupil)
        
        if len(tmp_ind_fits_list) == 0: 
            print(' >> no spec found; ')
            continue
        else: 
            print(' >> N=%d, x0=%.1f, y0=%.1f' % (len(tmp_ind_fits_list), x0, y0), end='')
            if extract_mode.lower() not in ['r', 'c']:
                tmp_ind_specs_name = os.path.join(extract_path, 'allspec_2d_%s_%s_ID%s.fits' % (tmp_target, tmp_filter, idx))
                store_all_2d_spec(fits_list=tmp_ind_fits_list, pupils=tmp_ind_pupil_list, modules=tmp_ind_module_list, paths=tmp_ind_fits_name_list, output=tmp_ind_specs_name, coord=tmp_coord, filter=tmp_filter, info_table=item_sex, overwrite=True)
                print(' >> save all extracted spec2d ', end='')

        for k, tmp_fits in enumerate(tmp_ind_fits_list[:]):
            tmp_modpup = tmp_fits[1].header['MODULE'] + tmp_fits[1].header['PUPIL']
            tmp_f_sens = list_f_sens[np.where(list_mod_pupil == tmp_modpup)[0][0]]
            if len(tmp_fits) == 5:
                tmp_wavelength = tmp_fits[4].data['wavelength']
            else:
                tmp_wavelength = tmp_fits[5].data['wavelength']
                tmp_ind_fits_list[k][4].header['bunit'] = ('mJy', 'Brightness Unit')
                tmp_ind_fits_list[k][4].data = tmp_fits[4].data / tmp_f_sens(tmp_wavelength) * 1e3
            tmp_ind_fits_list[k][1].header['bunit'] = ('mJy', 'Brightness Unit')
            tmp_ind_fits_list[k][1].data = tmp_fits[1].data / tmp_f_sens(tmp_wavelength) * 1e3
            tmp_ind_fits_list[k][2].data = tmp_fits[2].data * tmp_f_sens(tmp_wavelength)**2 * 1e-6
        
        tmp_wave_2d = np.array([x[-1].data['wavelength'] for x in tmp_ind_fits_list], dtype=object)
        d_wave = 0.001
        wave_sample = np.arange(WRANGE[0], WRANGE[1] + d_wave, d_wave)
        wave_sample_c = wave_sample[:-1] + d_wave / 2.
        arr_spec_2d = np.zeros((len(wave_sample)-1, len(tmp_wave_2d), len(tmp_ind_fits_list[0][1].data)))
        arr_wht_2d  = np.zeros_like(arr_spec_2d)
        arr_cov_2d  = np.zeros_like(arr_spec_2d)
        arr_line_2d = np.zeros_like(arr_spec_2d)
        np.seterr(invalid='ignore')
        
        xx, ii = np.meshgrid(range(len(tmp_wave_2d)), range(len(wave_sample) - 1))
        xx = xx.flatten()
        ii = ii.flatten()
        output = []
        for x, i in zip(xx, ii):
            result = resample_spec2d_wmin_wmax(x, i, tmp_wave_2d, tmp_ind_fits_list, wave_sample)
            output.append(result)

    # Since tmp_wave_2d, tmp_ind_fits_list, and wave_sample are the same for each call,
    # we don't need to create multiple copies of these variables; just pass them directly

        for o in range(len(output)):
            i, x = ii[o], xx[o]
            arr_spec_2d[i, x], arr_wht_2d[i, x], arr_cov_2d[i, x], arr_line_2d[i, x] = output[o]
        
        tmp_sigma_clip = SigmaClip(sigma_lower=2.0, sigma_upper=3.0)
        arg_sigclip = np.where(tmp_sigma_clip(np.nanmean(arr_spec_2d, axis=(0, 2))).mask == False)[0]
        tmp_ind_fits_list = [tmp_ind_fits_list[x] for x in arg_sigclip]
        tmp_ind_module_list = np.array(tmp_ind_module_list)[arg_sigclip]
        tmp_ind_fits_name_list = np.array(tmp_ind_fits_name_list)[arg_sigclip]
        arr_spec_2d = arr_spec_2d[:, arg_sigclip, :]
        arr_wht_2d  = arr_wht_2d[:, arg_sigclip, :]
        arr_cov_2d  = arr_cov_2d[:, arg_sigclip, :]
        arr_line_2d = arr_line_2d[:, arg_sigclip, :]
        
        sigma_clip = SigmaClip(sigma=4.)
        arr_spec_2d = sigma_clip(arr_spec_2d, axis=1, masked=False)
        arr_line_2d = sigma_clip(arr_line_2d, axis=1, masked=False)
        arr_wht_2d[np.where(np.isnan(arr_spec_2d))] = 0
        arr_wht_2d[np.where(np.isnan(arr_line_2d))] = 0
        
        tmp_spec_2d = np.nansum(arr_spec_2d * arr_wht_2d, axis=1) / np.nansum(arr_wht_2d, axis=1)
        tmp_wht_2d = np.nansum(arr_wht_2d, axis=1)
        tmp_cov_2d = np.nansum(arr_cov_2d, axis=1)
        tmp_line_2d = np.nansum(arr_line_2d * arr_wht_2d, axis=1) / np.nansum(arr_wht_2d, axis=1)
        
        tmp_tb_cov = Table(names=['index', 'name', 'x0', 'y0', 'module', 'pupil', 'DIFF_X', 'DIFF_Y', 'wave_0', 'wave_1'],
                           data=[range(len(tmp_ind_fits_list)),
                                 [tmp_grism.split('/')[-1][:-16] for tmp_grism in tmp_ind_fits_name_list],
                                 [x[0].header['x0'] for x in tmp_ind_fits_list],
                                 [x[0].header['y0'] for x in tmp_ind_fits_list],
                                 tmp_ind_module_list,
                                 [x[1].header['PUPIL'] for x in tmp_ind_fits_list],
                                 [x[1].header['DIFF_X'] for x in tmp_ind_fits_list],
                                 [x[1].header['DIFF_Y'] for x in tmp_ind_fits_list],
                                 [np.round(wave_sample_c[np.sum(arr_wht_2d, axis=-1)[:, n_] != 0][0], 4) 
                                  for n_ in range(len(tmp_ind_fits_list))],
                                 [np.round(wave_sample_c[np.sum(arr_wht_2d, axis=-1)[:, n_] != 0][-1], 4) 
                                  for n_ in range(len(tmp_ind_fits_list))],
                                ]) 
        
        hdu = fits.PrimaryHDU()
        hdu.header['RA0'] = (tmp_coord.ra.value, 'Reference position RA in direct image')
        hdu.header['DEC0'] = (tmp_coord.dec.value, 'Reference position Dec in direct image')
        hdu.header['N_coadd'] = (len(tmp_tb_cov['pupil']), 'Number of coadded frames')
        hdu.header['N_R'] = (np.sum(tmp_tb_cov['pupil'] == 'R'), 'Number of R grism frames')
        hdu.header['N_C'] = (np.sum(tmp_tb_cov['pupil'] == 'C'), 'Number of C grism frames')
        hdu.header['author'] = ('Zhaoran Liu', 'Author of this file')
        hdu.header['time'] = (time.strftime("%Y/%m/%d %H:%M:%S",  time.localtime()), 'Time of Creation')
        hdu.header['filter'] = (tmp_filter, 'Filter name')
        hdu.header['COMMENTS'] = 'Belows are information taken from input catalog:'
        
        for x in item_sex.colnames:
            if type(item_sex[x]) == np.ma.core.MaskedConstant: continue
            elif type(item_sex[x]) != np.str_ : 
                if np.isnan(item_sex[x]) : hdu.header['HIERARCH ' + x] = 'nan' 
                elif np.isinf(item_sex[x]) : hdu.header['HIERARCH ' + x] = 'inf' 
                else: hdu.header['HIERARCH ' + x] = item_sex[x]
            else: hdu.header['HIERARCH ' + x] = item_sex[x]
        
        hdu_sci = fits.ImageHDU(tmp_spec_2d.T, name='SPEC2D')
        hdu_sci.header['wave_1'] = (wave_sample_c[0], 'Wavelength (um) of first pixel')
        hdu_sci.header['d_wave'] = (wave_sample_c[1] - wave_sample_c[0], 'Wavelength Difference (um) between each pixel')
        hdu_sci.header['comments'] = ('wave = wave_1 + np.arange(0, NAXIS1, 1) * d_wave')
        hdu_sci.header['pixscale'] = (0.0629, 'Pixel scale in undispersed direction (arcsec)')
        hdu_sci.header['aperture'] = (tmp_aper, 'Aperture radius in undispersed direction (pixel)')
        
        hdu_wht = fits.ImageHDU(tmp_wht_2d.T, name='WHT2D')
        hdu_wht.header['comments'] = ('Weight image; ERR = WHT^(-0.5)')
        
        hdu_cov = fits.ImageHDU(tmp_cov_2d.T, name='COV2D')
        hdu_cov.header['comments'] = ('Coverage image')
        
        hdu_line = hdu_sci.copy()
        hdu_line.data = tmp_line_2d.T
        hdu_line.header['extname'] = 'LINE2D'
        hdu_line.header['comments'] = ('Line-only image extracted on continuum-filtered 2D data')
        
        hdu_tab = fits.BinTableHDU(tmp_tb_cov, name='STATS')
        hdu_tab.header['COMMENT'] = 'name:     name of simulated image'
        hdu_tab.header['COMMENT'] = 'x0 / y0:  reference position (i.e., in direct image)'
        hdu_tab.header['COMMENT'] = '   (in reality this is registered to the wcs of grism image)'
        hdu_tab.header['COMMENT'] = 'DIFF_X:   X_(full)-X_(trim)'
        hdu_tab.header['COMMENT'] = 'DIFF_Y:   Y_(full)-Y_(trim)'
        hdu_tab.header['COMMENT'] = 'wave_0:   minimum wavelength (micron) in this coverage'
        hdu_tab.header['COMMENT'] = 'wave_1:   maximum wavelength (micron) in this coverage'
        hdul = fits.HDUList([hdu, hdu_sci, hdu_wht, hdu_cov, hdu_line, hdu_tab])
        
        tmp_spec_2d_name = os.path.join(extract_path, 'spec_2d_%s_%s_ID%s_comb.fits' % (tmp_target, tmp_filter, idx))
        if extract_mode.lower() in ['r', 'c']: 
            tmp_spec_2d_name = tmp_spec_2d_name.replace('_comb.fits', '_%s.fits' % extract_mode.upper())
        hdul.writeto(tmp_spec_2d_name, overwrite=True)
        print(' >> save stacked spec2d; ')

    end = time.time()
    print('run time = %.1fs = %.1fmin' % (end - start, (end - start)/60))




def plot_and_save_extracted_spectra():
    """Function to plot and save the extracted 1D spectra from 2D spectra files."""
    tmp_filter = USER_FILTER

    # Check the filter and set wavelength range
    if tmp_filter == 'F444W':
        WRANGE = np.array([3.8, 5.1])
    elif tmp_filter == 'F322W2':
        WRANGE = np.array([2.4, 4.1])
    elif tmp_filter == 'F356W':
        WRANGE = np.array([3.1, 4.0])
    elif tmp_filter == 'F277W':
        WRANGE = np.array([2.4, 3.1])
        
    list_spec = np.array(glob.glob(os.path.join(USER_OUTPUT_SPECTRA_PATH, 'spec_2d_*_comb.fits')))
    list_spec.sort()
    print(len(list_spec), 'spectra files in total.')

    img_LW = fits.getdata(USER_PATH_LW, 0)
    hd_LW = fits.getheader(USER_PATH_LW, 0)
    band_LW = 'F444W'
    wcs_LW = wcs.WCS(hd_LW)
    pix_LW = wcs.utils.proj_plane_pixel_area(wcs_LW) ** 0.5 * 3600.

    for i, tmp_spec_path in enumerate(list_spec):
        tmp_spec_fits = fits.open(tmp_spec_path)
        tmp_id = tmp_spec_fits[0].header['ID']
        tmp_filter = tmp_spec_fits[0].header['FILTER']
        mag_keyword = 'F444W_mag' if 'F444W_mag' in tmp_spec_fits[0].header else 'F160W_mag'
        tmp_mag = tmp_spec_fits[0].header.get(mag_keyword, 99.0)
        tmp_flux_mJy = 10**(-0.4 * tmp_mag) * 3631e3 if tmp_mag != 99.0 else 0.0
        tmp_N_R = tmp_spec_fits[0].header['N_R']
        tmp_N_C = tmp_spec_fits[0].header['N_C']
        tmp_tb_stats = Table(tmp_spec_fits['STATS'].data)
        tmp_N_A = np.sum(tmp_tb_stats['module'] == 'A')
        tmp_N_B = np.sum(tmp_tb_stats['module'] == 'B')
        tmp_RA = tmp_spec_fits[0].header['RA0']
        tmp_DEC = tmp_spec_fits[0].header['DEC0']
        tmp_x0, tmp_y0 = wcs.utils.skycoord_to_pixel(SkyCoord(tmp_RA, tmp_DEC, unit=(u.deg, u.deg)), wcs_LW)

        is_cont = False
        tmp_spec_2d = tmp_spec_fits['spec2d'].data
        tmp_line_2d = tmp_spec_fits['line2d'].data
        tmp_wht_2d = tmp_spec_fits['wht2d'].data
        tmp_conti_2d  = tmp_spec_fits['spec2d'].data - tmp_spec_fits['line2d'].data
        wave_sample_c = tmp_spec_fits[1].header['WAVE_1'] + np.arange(tmp_spec_fits[1].header['NAXIS1']) * tmp_spec_fits[1].header['D_WAVE']

        if np.sum(np.isnan(np.nansum(tmp_spec_2d, axis=0)) == False) < 200:
            continue

        tmp_spec_ydir = np.nansum(tmp_spec_2d * tmp_wht_2d, axis=1) / np.nansum(tmp_wht_2d, axis=1)
        tmp_yc = 20
        tmp_aper = 5

        if is_cont:
            tmp_spec_1d = np.nansum(tmp_spec_2d[tmp_yc-tmp_aper:tmp_yc+tmp_aper+1], axis=0)
        else:
            tmp_spec_1d = np.nansum(tmp_line_2d[tmp_yc-tmp_aper:tmp_yc+tmp_aper+1], axis=0)
        tmp_unc_1d = np.nansum(tmp_wht_2d[tmp_yc-tmp_aper:tmp_yc+tmp_aper+1]**-1, axis=0)**0.5

        if np.nanmedian(tmp_spec_1d[tmp_spec_1d != 0]) < 0:
            tmp_spec_1d -= np.nanmedian(tmp_spec_1d[tmp_spec_1d != 0])

        plt.close()
        e_fig, b_fig = 0.1, 0.9
        x_fig, y_fig = 16 + e_fig * 3, 7 + e_fig * 5 + b_fig
        fig = plt.figure(figsize=(x_fig / y_fig * 6, 6))
        
        ax_im = fig.add_axes([e_fig/x_fig, (5 + b_fig + 2 * e_fig)/y_fig, 2 / x_fig , 2 / y_fig])  # direct image - 1
        ax_2d = fig.add_axes([(2 + e_fig*2)/x_fig, (5 + b_fig + 2 * e_fig)/y_fig, 14 / x_fig , 2 / y_fig])  # 2d spec (cont)
        ax_con = fig.add_axes([(2 + e_fig*2)/x_fig, (3 + b_fig + 2 * e_fig)/y_fig, 14 / x_fig , 2.0 / y_fig])  # New axis
        ax_li = fig.add_axes([(2 + e_fig*2)/x_fig, (1 + b_fig + 2*e_fig)/y_fig, 14 / x_fig , 2 / y_fig])  # 2d spec (line)
        ax_1d = fig.add_axes([(2 + e_fig*2)/x_fig, 0.9*b_fig/y_fig, 14 / x_fig , 1.2 / y_fig])  # 1d spec
        ax = [ax_2d, ax_con, ax_li, ax_1d]

        vmin_zscale, vmax_zscale = ZScaleInterval().get_limits(tmp_spec_2d[:, 100:-100])
        tmp_aspect = (np.diff(WRANGE) - 0.1) / tmp_spec_fits[1].header['D_WAVE'] / (tmp_spec_2d.shape[0] - 1) / 7
        tmp_xticks = (np.arange(WRANGE[0] + 0.1, WRANGE[1] - 0.05 + 0.01, 0.1) - tmp_spec_fits[1].header['WAVE_1']) / tmp_spec_fits[1].header['D_WAVE']
        
        ax[0].imshow(tmp_spec_2d, aspect = tmp_aspect, vmin = vmin_zscale, vmax = vmax_zscale,
                 cmap = plt.cm.gist_gray_r, origin = 'lower')
        ax[0].set(ylim = (0.5, tmp_spec_2d.shape[0] - 0.5), xticks = [], 
                  aspect = tmp_aspect,
                  xlim = (0.05 / tmp_spec_fits[1].header['D_WAVE'], 
                          (WRANGE[1] - WRANGE[0] - 0.05) / tmp_spec_fits[1].header['D_WAVE']))
        ax[0].set_yticks([tmp_spec_2d.shape[0] / 2.]); ax[0].set_yticklabels([""])
        ax[0].set_yticklabels([""])
        ax[0].set_xticks(tmp_xticks)
        ax[0].set_xticklabels([])
        ax[1].imshow(tmp_conti_2d,  vmin = vmin_zscale/2., vmax = vmax_zscale, cmap = plt.cm.gist_gray_r, origin = 'lower')
        ax[1].set(ylim = (0.5, tmp_spec_2d.shape[0] - 0.5), xticks = [], aspect = tmp_aspect, xlim = (0.05 / tmp_spec_fits[1].header['D_WAVE'], (WRANGE[1] - WRANGE[0] - 0.05) / tmp_spec_fits[1].header['D_WAVE']))
        ax[1].set_yticks([tmp_spec_2d.shape[0] / 2.]); ax[1].set_yticklabels([""])
        ax[1].set_xticks(tmp_xticks);  ax[1].set_xticklabels([])
    
        vmin_zscale, vmax_zscale = ZScaleInterval().get_limits(tmp_line_2d[:,100:-100])
        ax[2].imshow(tmp_line_2d,  vmin = vmin_zscale/2., vmax = vmax_zscale,
                     cmap = plt.cm.gist_gray_r, origin = 'lower')
        ax[2].set(ylim = (0.5, tmp_spec_2d.shape[0] - 0.5), xticks = [], 
                  aspect = tmp_aspect,
                  xlim = (0.05 / tmp_spec_fits[1].header['D_WAVE'], 
                          (WRANGE[1] - WRANGE[0] - 0.05) / tmp_spec_fits[1].header['D_WAVE']))
        ax[2].set_yticks([tmp_spec_2d.shape[0] / 2.]); ax[1].set_yticklabels([""])
        ax[2].set_xticks(tmp_xticks);  ax[1].set_xticklabels([])
        # ax[1].axhline(tmp_yc + tmp_aper + 1.5, color = 'w', ls = '--', dashes = (4, 4))
        # ax[1].axhline(tmp_yc - tmp_aper - 0.5, color = 'w', ls = '--', dashes = (4, 4))
        ax[2].axhline(tmp_yc + tmp_aper + 1.5, color = 'w', ls = '--', dashes = (4, 4))
        ax[2].axhline(tmp_yc - tmp_aper - 0.5, color = 'w', ls = '--', dashes = (4, 4))
        
        
        ax[3].plot(wave_sample_c, ndimage.gaussian_filter1d(tmp_spec_1d, 0.6), color = 'k', zorder = 100, lw = 1.5)
        tmp_max_counts = np.nanpercentile(tmp_spec_1d[(np.isnan(tmp_spec_1d) == False) & (tmp_spec_1d != 0)], 95) * 1.5
        if np.isnan(tmp_max_counts): continue
        ax[3].axhline(0, color = 'grey', ls = '--')
        ax[3].set(xlim = (WRANGE[0] + 0.05, WRANGE[1] - 0.05), 
                  xticks = np.arange(WRANGE[0] + 0.1, WRANGE[1] - 0.05 + 0.01, 0.1),
                  xlabel = 'Observed Wavelength (µm)', ylabel = 'Flux Density [mJy]')
        ax[3].set_ylim(np.clip(vmin_zscale * 1.5, -0.02, 0), np.clip(tmp_max_counts, 0.018, 1e8)) 
            
        corner_text(ax[0], loc = 2, s = 'ID%s' % (tmp_id), weight = 'semibold', color = 'r', fontsize = 20, edge = 5e-3)
        if tmp_N_A == 0: corner_text(ax[0], s = 'modB', loc = 3, color = 'r', fontsize = 14, edge = 5e-3)
        elif tmp_N_B == 0: corner_text(ax[0], s = 'modA', loc = 3, color = 'r', fontsize = 14, edge = 5e-3)
        else: corner_text(ax[0], s = 'modA:%d / modB:%d' % (tmp_N_A, tmp_N_B), loc = 3, color = 'r', fontsize = 14, edge = 5e-3)
        corner_text(ax[1], loc = 3, s = 'Continuum', color = 'r', fontsize = 15, edge = 5e-3)
        corner_text(ax[2], loc = 3, s = 'Continuum Subtracted', color = 'r', fontsize = 15, edge = 5e-3)
        corner_text(ax[1], loc = 4, s = '(%.5f, %.5f)' % (tmp_RA, tmp_DEC), color = 'r', fontsize = 15, edge = 5e-3)
        #corner_text(ax[3], loc = 1, s = '$f_\mathrm{%s}$=%.3f mJy' % (mag_keyword.split('_')[0], tmp_flux_mJy), fontsize = 15, color = 'r', edge = 5e-3)
        corner_text(ax[3], loc = 2, s = 'N(R)=%2d' % tmp_N_R, fontsize = 14, color = 'r', edge = 5e-3)
        
        corner_text(ax[3], loc = 4, s = 'z=%.3f' % tmp_spec_fits[0].header['zspec'], color = 'r', fontsize = 15,
                edge = 5e-3, zorder = 999, bbox = dict(facecolor = 'w', alpha = 0.6, edgecolor = 'none'))
        ax[3].axvline((1 + tmp_spec_fits[0].header['zphot']) * 1.875, color = 'blue', ymin = 0.8, ymax = 1)
        ax[3].axvline((1 + tmp_spec_fits[0].header['zphot']) * 1.282, color = 'blue', ymin = 0.8, ymax = 1)
        ax[3].axvline((1 + tmp_spec_fits[0].header['zphot']) * 1.083, color = 'blue', ymin = 0.8, ymax = 1)
        ax[3].axvline((1 + tmp_spec_fits[0].header['zphot']) * 0.9531, color = 'blue', ymin = 0.8, ymax = 1)
        ax[3].axvline((1 + tmp_spec_fits[0].header['zphot']) * 0.9069, color = 'blue', ymin = 0.8, ymax = 1)
        ax[3].axvline((1 + tmp_spec_fits[0].header['zphot']) * 0.6564, color = 'blue', ymin = 0.8, ymax = 1)
        ax[3].axvline((1 + tmp_spec_fits[0].header['zphot']) * 0.5007, color = 'blue', ymin = 0.8, ymax = 1)
        ax[3].axvline((1 + tmp_spec_fits[0].header['zphot']) * 0.4959, color = 'blue', ymin = 0.8, ymax = 1)
        ax[3].axvline((1 + tmp_spec_fits[0].header['zphot']) * 0.4861, color = 'blue', ymin = 0.8, ymax = 1)
        ax[3].axvline((1 + tmp_spec_fits[0].header['zphot']) * 3.3, color = 'blue', ymin = 0.8, ymax = 1)
        ax[3].axvline((1 + tmp_spec_fits[0].header['zphot']) * 4.052, color = 'blue', ymin = 0.8, ymax = 1)
        ### ax_im: Direct Image
        hf_box = int(0.0629 * (tmp_spec_2d.shape[0] - 1) / 2. / 0.1)
        tmp_img = img_LW[int(tmp_y0)-hf_box-1:int(tmp_y0)+hf_box+2,int(tmp_x0)-hf_box-1:int(tmp_x0)+hf_box+2]
        ax_im.imshow(tmp_img, 
                     vmin = np.nanpercentile(tmp_img, 2.5), vmax = np.nanpercentile(tmp_img, 97.5),
                     cmap = plt.cm.gist_heat, origin = 'lower')
        ax_im.set(xlim = (tmp_x0%1 + 0.5, tmp_x0%1 + 0.5 + hf_box*2), ylim = (tmp_y0%1 + 0.5, tmp_y0%1 + 0.5 + hf_box*2))
        ax_im.set_xticks([])
        ax_im.set_yticks([])
        corner_text(ax_im, loc = 4, s = '(%.1f, %.1f)' % (tmp_x0, tmp_y0), color = 'w', fontsize = 10)
        corner_text(ax_im, loc = 2, s = band_LW, color = 'w', fontsize = 13, weight = 'semibold')    
 
        print(i, tmp_id, f'{mag_keyword}={tmp_mag:.2f}')
        plt.savefig(tmp_spec_path.replace('.fits', '.pdf'), dpi=150)

        tb_1dspec = Table(names=['wavelength_um', 'flux_mJy', 'fluxerr_mJy'], data=[wave_sample_c, tmp_spec_1d, tmp_unc_1d])
        tb_1dspec['wavelength_um'].info.format = '.4f'
        tb_1dspec['flux_mJy'].info.format = '.5f'
        tb_1dspec['fluxerr_mJy'].info.format = '.5f'
        tb_1dspec.meta['comments'] = ['-' * 70]
        for y in [' = '.join(map(str, x[:2])) for x in tmp_spec_fits[0].header.cards][4:]:
            tb_1dspec.meta['comments'].append(y)
        tb_1dspec.meta['comments'].append('-' * 70)
        tb_1dspec.meta['comments'].append(f'1D spectrum was extracted at y_c={tmp_yc:.1f} with full aperture height = {tmp_aper * 2 + 1:.1f}')
        if is_cont:
            tb_1dspec.meta['comments'].append('I only subtracted common grism sky background. Contaminants are not subtracted.')
        else:
            tb_1dspec.meta['comments'].append('Extracted from 2D grism images that have been continuum/background-subtracted.')
        tb_1dspec.meta['comments'].append('Be careful about potential contaminant & aperture loss.')
        tb_1dspec.meta['comments'].append('' * 70)
        tb_1dspec.meta['comments'].append(f'{time.strftime("%Y/%m/%d", time.localtime())})')
        tb_1dspec.meta['comments'].append('-' * 70)
        ascii.write(tb_1dspec, tmp_spec_path.replace('/spec_2d_', '/spec_1d_').replace('.fits', '.dat'), format='commented_header', overwrite=True)

def spectra_extration():
    perform_astrometric_calculations()
    plot_sources_on_footprint()
    generate_pom_catalogs()
    check_and_prepare_spectra_extraction()
    extract_spectra()

def plot_spectra():
    plot_and_save_extracted_spectra()
