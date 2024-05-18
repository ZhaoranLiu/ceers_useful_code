from .config import (
    DEFAULT_MODA_LW_POM_PATH,
    DEFAULT_MODB_LW_POM_PATH,
    DEFAULT_SPEC_COV_DIR,
    DEFAULT_GRISM_NIRCAM_DIR,
    DEFAULT_MATPLT_STYLE,
    DEFAULT_PS_PATH_TEMPLATE,
    DEFAULT_POM_PATH_TEMPLATE,
    DEFAULT_GRISM_RATE_FILES_PATH,
    DEFAULT_FILTER,
    DEFAULT_CALIBRATED_DIR
)

# Astropy modules
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

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

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
from multiprocessing import Pool, Lock

# External packages
import grismconf
import pysiaf
import asdf

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
list_rate = np.array(glob.glob(DEFAULT_GRISM_RATE_FILES_PATH))
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
tmp_filter = DEFAULT_FILTER 
list_rate_this_band = list_rate[(list_filter == tmp_filter) & (list_pupil != 'CLEAR')]
print(len(list_rate_this_band), 'files,\n', list_rate_this_band[0])

calibrated_dir = DEFAULT_CALIBRATED_DIR
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

    for rate_file in list_rate_this_band:
        assignwcs_grism_stage2(rate_file)
    
    all_rate_list = np.array([os.path.join(USER_CALIBRATED_DIR, os.path.basename(x).replace('rate.fits', 'rate_lv1.5.fits'))
                              for x in list_rate_files])
    return all_rate_list
