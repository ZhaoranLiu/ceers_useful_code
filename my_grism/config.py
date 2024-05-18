import pkg_resources
from pathlib import Path
import os
def get_data_file_path(directory, filename):
    resource_path = Path(pkg_resources.resource_filename(__name__, f'data/{directory}/{filename}'))
    if not resource_path.exists():
        raise FileNotFoundError(f"Data file {filename} not found in package data directory.")
    return str(resource_path)

#make sure to prepare those calibrated files before you run the pipeline
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# MODA_LW_POM_FILENAME = 'NIRCAM_LW_POM_ModA_trans.fits'
# MODB_LW_POM_FILENAME = 'NIRCAM_LW_POM_ModB_trans.fits'

DEFAULT_MODA_LW_POM_PATH = os.path.join(BASE_DIR, 'my_grism', 'data', 'GRISM_NIRCAM', 'NIRCAM_LW_POM_ModA_trans.fits')
DEFAULT_MODB_LW_POM_PATH = os.path.join(BASE_DIR, 'my_grism', 'data', 'GRISM_NIRCAM', 'NIRCAM_LW_POM_ModB_trans.fits')

DEFAULT_MATPLT_STYLE = {'font.size': 16}
DEFAULT_PS_PATH_TEMPLATE = os.path.join(BASE_DIR, 'my_grism', 'data', 'FSun_cal', 'FSun_SpecCov_%s_%s_%s.fits')
DEFAULT_POM_PATH_TEMPLATE = os.path.join(BASE_DIR, 'my_grism', 'data', 'GRISM_NIRCAM', 'NIRCAM_LW_POM_Mod%s_trans.fits')

DEFAULT_SPEC_DISPER = os.path.join(BASE_DIR, 'my_grism', 'data', 'FSun_cal', 'disper')
DEFAULT_SPEC_SENSI = os.path.join(BASE_DIR, 'my_grism','data', 'FSun_cal', 'sensitivity')

## grism data
USER_RATE_FILES_PATH = '/Volumes/Extreme_S4/test_my_grism/my_grism/test/goodsn_lw_data/*.fits'
USER_FILTER = 'F444W'
## calibrated data
USER_CALIBRATED_DIR = '/Volumes/Extreme_S4/test_my_grism/my_grism/test/grism_cal/'
## the pipeline will generate _1v1.5.fits in the calibrated directory, we will need to grab them in the pipe
USER_RATE_LV15_FILES_PATH = '/Volumes/Extreme_S4/test_my_grism/my_grism/test/grism_cal/*lv1.5.fits'
## direct image data
USER_SW_RATE_FILES_PATH = '/Volumes/Extreme_S4/test_my_grism/my_grism/test/goodsn_sw_image/'
## your astrometry catalog to correct for distortion and offset (I select bright sources from master catalog for this purpose)
USER_ASTROMETRIC_CATALOG_PATH = '/Volumes/Extreme_S4/test_my_grism/my_grism/test/astrometry/wcs_match_for_goodsn.fits'
## source catalog for spectra extraction
USER_TB_F444W_PATH = '/Volumes/Extreme_S4/test_my_grism/my_grism/test/spectra_extraction/jaden_halpha_candidate.cat'
##  the place for your extracted spectra
USER_OUTPUT_SPECTRA_PATH = '/Volumes/Extreme_S4/test_my_grism/my_grism/test/spectra_extraction/jades_ha_spectra'
# F444W image
USER_PATH_LW = '/Volumes/Extreme_S4/zhaoran_wfss/wfss/hlsp_fresco_jwst_nircam_goods-n_f444w_v1.0_sci.fits'
USER_TARGET = 'GDN'