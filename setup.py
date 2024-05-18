from setuptools import setup, find_packages

setup(
    name='my_grism',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'my_grism': [
            'data/GRISM_NIRCAM/NIRCAM_LW_POM_ModA_trans.fits',
            'data/GRISM_NIRCAM/NIRCAM_LW_POM_ModB_trans.fits',
            'data/FSun_cal/FSun_SpecCov_F444W_A_R.fits',
            'data/FSun_cal/FSun_SpecCov_F444W_B_R.fits',
        ],
    },
    entry_points={
        'console_scripts': [
            'mygrism=my_grism.cli:main',
        ],
    },
    install_requires=[
        'astropy',
        'photutils',
        'numpy',
        'scipy',
        'matplotlib',
        'tqdm',
        'jwst',
        'crds',
        'tshirt',
        'asdf',
    ],
)
