# MyGrism

A tool for reducing and analyzing JWST NIRCam grism data.

## Installation
~~~~~~~~~~~~~
pip install my_grism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
or 
~~~~~~~~~~~~~
python setup.py install
~~~~~~~~~~~~~~~~~~~~~~

## External dependency
~~~~~~~~~~~~~~~~~~~~~~
pip install git+https://github.com/npirzkal/GRISMCONF.git
~~~~~~~~~~~~~~~~~~~~~~

## Caveats

**Please set your JWST CRDS path by following the instruction [here](https://jwst-pipeline.readthedocs.io/en/latest/jwst/user_documentation/reference_files_crds.html>). Then, try:

~~~~~~~~~~
echo $CRDS
~~~~~~~~~~
This should return the current path you have.

**Please edit config.py to specify the correct path and name for your data**

## If you want to do within 5 steps


    mygrism super_background --data_type GRISMR
    mygrism cont_subtraction --L_box 25 --L_mask 4
    mygrism reduce_sw
    mygrism spectra_extration
    mygrism plot_spectra

 After you finished the first 3 steps, you can easily change the catalog in config.py and only rerun the `spectra_extration` and `plot_spectra`

## If you want to do the reduction step by step
Please check out the cli.py to see the commands or type:
    mygrism --help

## Produced background-subtracted images and continuum-subtracted images:

    ./grism_cal/plots

## Please make sure the project structure is:

    xxx/
    ├── my_grism/
    │   ├── __init__.py
    │   ├── core.py
    │   ├── config.py
    │   ├── cli.py
    │   └── data/
    │       └── GRISM_NIRCAM/
    │       └── FSun_cal/
    │           ├── FSun_SpecCov_*.fits
    │           └── disper/
    │           └── sensitivity/

## Please download the spectral calibration data from the following link (Produced by Dr. Fengwu Sun):
    https://drive.google.com/file/d/1RP7XJvP5-KDrA4F2Ofsz4fQzjcd9oOWS/view?usp=sharing
